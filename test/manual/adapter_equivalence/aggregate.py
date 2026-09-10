"""Aggregate exactly the frozen H200 evidence inventory, failing closed."""

from __future__ import annotations

import argparse
import inspect
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adapter_equivalence"

from .compare import (
    comparable_adapter_error,
    comparable_adapter_states,
    compare_bundles,
)
from .policy import (
    BaselineRepetition,
    ComparisonPolicy,
    NumericTolerance,
    ToleranceEnvelope,
)
from .qualification import (
    CURRENT_SCOPE,
    PATH_NAMES,
    DirectoryAnchor,
    QualificationManifest,
    QualificationScope,
    exact_sha,
    outside_artifact_root,
    publish_document,
    read_document,
)
from .scenarios import (
    comparable_lease_observations,
    lifecycle_transition_names,
    validate_lifecycle_observations,
)
from .schema import (
    COMPARABLE_PROVENANCE_HASH_KEYS,
    BundleValidationError,
    RunBundle,
    _require_exact_fields,
    _require_mapping,
    _thaw_json,
    _validate_sha256,
    canonical_sha256,
)


def _require(condition, message):
    if not condition:
        raise BundleValidationError(message)


def _object(value, fields, context):
    value = _require_mapping(value, context)
    _require_exact_fields(value, set(fields), context)
    return value


def _same(left, right):
    return canonical_sha256(left) == canonical_sha256(right)


def _case_launch(bundle, config, seed):
    """The initial lifecycle and every resolved/performance launch attest one case."""
    config = _require_mapping(config, "launch configuration")
    _require(bool(config), "empty resolved launch configuration")
    _require(
        config.get("disable_cuda_graph") is (not bundle.case_key.cuda_graph),
        "CUDA graph case/launch mismatch",
    )
    _require(
        type(config.get("random_seed")) is int
        and config["random_seed"] == seed == bundle.manifest["seed"],
        "seed/procedure mismatch",
    )
    _require(
        type(config.get("tp_size")) is int
        and type(config.get("ep_size")) is int
        and config["tp_size"] > 0
        and config["ep_size"] > 0
        and config["tp_size"] % config["ep_size"] == 0,
        "invalid launch topology",
    )
    _require(
        type(config.get("base_gpu_id")) is int and config["base_gpu_id"] == 1,
        "sender/model placement mismatch",
    )
    if bundle.case_key.mode == "native_lora":
        _require(
            config.get("enable_lora") is True
            and config.get("enable_lora_staging") is True
            and config.get("peft_method") in (None, "lora"),
            "native LoRA launch mismatch",
        )
    elif bundle.case_key.mode == "native_oft":
        _require(
            config.get("peft_method") == "oft"
            and config.get("oft_type") == "oft"
            and (
                config.get("enable_lora") is None or config.get("enable_lora") is False
            ),
            "native OFT launch mismatch",
        )
    expected_precision = {"bf16": None, "fp8": "fp8", "nvfp4": "modelopt_fp4"}[
        bundle.case_key.precision
    ]
    _require(
        config.get("quantization") == expected_precision, "precision/launch mismatch"
    )


def _resolved_startup(bundle, snapshot, phase):
    """Resolved references retain runtime fields; constructor paths are compact."""
    lora = bundle.case_key.mode == "native_lora"
    active = "lora_paths" if lora else "peft_paths"
    for field in ("lora_paths", "peft_paths"):
        _require(field in snapshot, "missing resolved startup paths")
        paths = snapshot[field]
        if phase == "initial" or field != active:
            _require(
                paths is None or (isinstance(paths, (tuple, list)) and not paths),
                "unexpected resolved startup adapters",
            )
            continue
        _require(
            isinstance(paths, (tuple, list)) and len(paths) == 1,
            "resolved startup requires exactly policy-a",
        )
        prefix = "lora" if lora else "adapter"
        version = "version" if lora else "adapter_version"
        reference = _object(
            paths[0],
            (
                f"{prefix}_id",
                f"{prefix}_name",
                f"{prefix}_path",
                "pinned",
                "reloadable",
                version,
            ),
            "resolved startup reference",
        )
        digest = canonical_sha256(
            bundle.provenance["metadata"]["fixture_files"]["policy-a"]
        )
        _require(
            reference[f"{prefix}_name"] == "policy-a"
            and _same(
                reference[f"{prefix}_id"], {"name": "policy-a", "fixture_hash": digest}
            )
            and _same(reference[f"{prefix}_path"], {"fixture_hash": digest}),
            "unbound resolved startup identity",
        )
        _require(
            reference["pinned"] is False
            and reference["reloadable"] is True
            and type(reference[version]) is int
            and reference[version] == (0 if lora else 1),
            "invalid resolved startup state",
        )


def _checkpoint_attestation(bundle):
    checkpoint = _object(
        bundle.provenance["metadata"].get("checkpoint"),
        (
            "id",
            "model",
            "revision",
            "path",
            "layout",
            "index_hash",
            "files",
            "checkpoint_hash",
            "tokenizer_hash",
        ),
        "checkpoint",
    )
    _require(
        checkpoint["model"] == bundle.case_key.model, "checkpoint model/case mismatch"
    )
    _require(
        type(checkpoint["id"]) is str and bool(checkpoint["id"]),
        "invalid checkpoint id",
    )
    _require(
        type(checkpoint["path"]) is str and Path(checkpoint["path"]).is_absolute(),
        "invalid checkpoint path",
    )
    revision = checkpoint["revision"]
    if type(revision) is str and len(revision) == 64:
        _validate_sha256(revision, "checkpoint revision")
    else:
        exact_sha(revision)
    files = _require_mapping(checkpoint["files"], "checkpoint.files")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    _require(required <= files.keys(), "checkpoint file inventory is incomplete")
    for name, digest in files.items():
        _require(
            type(name) is str and Path(name).name == name, "unsafe checkpoint file name"
        )
        _validate_sha256(digest, f"checkpoint.files.{name}")
    if checkpoint["layout"] == "unsharded":
        _require(
            set(files) == required | {"model.safetensors"}
            and checkpoint["index_hash"] is None,
            "invalid unsharded checkpoint inventory",
        )
    elif checkpoint["layout"] == "indexed":
        _validate_sha256(checkpoint["index_hash"], "checkpoint index_hash")
        _require(
            files.get("model.safetensors.index.json") == checkpoint["index_hash"],
            "checkpoint index hash mismatch",
        )
        shards = set(files) - required - {"model.safetensors.index.json"}
        matches = [
            re.fullmatch(r"model-(\d{5})-of-(\d{5})\.safetensors", name)
            for name in shards
        ]
        _require(bool(matches) and all(matches), "invalid indexed shard inventory")
        _require(
            {int(m.group(2)) for m in matches} == {len(matches)}
            and {int(m.group(1)) for m in matches} == set(range(1, len(matches) + 1)),
            "incomplete indexed shard sequence",
        )
    else:
        raise BundleValidationError("invalid checkpoint layout")
    tokenizer_names = {"tokenizer.json", "tokenizer_config.json"}
    _require(
        checkpoint["tokenizer_hash"]
        == canonical_sha256({k: files[k] for k in tokenizer_names}),
        "checkpoint tokenizer hash mismatch",
    )
    _require(
        checkpoint["checkpoint_hash"]
        == canonical_sha256(
            {k: v for k, v in files.items() if k not in tokenizer_names}
        ),
        "checkpoint content hash mismatch",
    )
    return checkpoint


def _procedure_attestation(bundle):
    """Exact Task 9-owned wrappers; resolved runtime maps retain all their fields."""
    procedure = _object(
        bundle.manifest["metadata"].get("performance_procedure"),
        (
            "version",
            "seed",
            "samples",
            "effective_launches",
            "memory_boundary_hash",
            "launch",
            "requests",
            "warmup",
            "workload",
            "sampling",
            "memory",
            "latency",
            "throughput",
            "implementation",
        ),
        "performance procedure",
    )
    _require(
        type(procedure["version"]) is int and procedure["version"] == 1,
        "invalid procedure version",
    )
    _require(
        type(procedure["seed"]) is int and procedure["seed"] == bundle.manifest["seed"],
        "invalid procedure seed",
    )
    _require(
        type(procedure["samples"]) is int and procedure["samples"] == 3,
        "procedure must collect three samples",
    )
    declarations = {
        "warmup": "one factual request on a fresh engine, before every sample",
        "workload": "one fixed batch-32, startup policy-a for native modes",
        "memory": "synchronize/reset each scheduler; request set; synchronize/read each scheduler; sum all TP allocated/reserved peaks",
        "latency": "perf_counter elapsed around async_generate, excluding memory RPCs",
        "throughput": "actual output token IDs across all 32 responses divided by latency",
    }
    _require(
        all(procedure[name] == expected for name, expected in declarations.items()),
        "benchmark procedure declaration mismatch",
    )
    from . import run_case

    _require(procedure["seed"] == run_case.RUN_SEED, "unexpected qualification seed")
    implementation = [
        inspect.getsource(method)
        for method in (
            run_case.launch_engine,
            run_case.ShardRunner._measure_performance,
            run_case.ShardRunner._performance_requests,
            run_case.ShardRunner._scheduler_memory,
            run_case.ShardRunner._kwargs,
            run_case.ShardRunner._generate_one,
            run_case.ShardRunner._async,
        )
    ]
    _require(
        _same(procedure["implementation"], implementation),
        "unattested benchmark implementation",
    )
    sampling = _object(
        procedure["sampling"],
        ("temperature", "top_p", "top_k", "max_new_tokens"),
        "sampling",
    )
    _require(
        type(sampling["temperature"]) is float
        and sampling["temperature"] == 0.0
        and type(sampling["top_p"]) is float
        and sampling["top_p"] == 1.0
        and type(sampling["top_k"]) is int
        and sampling["top_k"] == -1
        and type(sampling["max_new_tokens"]) is int
        and sampling["max_new_tokens"] == 32,
        "invalid deterministic sampling",
    )
    requests = _object(procedure["requests"], ("warmup", "batch"), "procedure.requests")
    adapter_field = (
        "lora_path" if bundle.case_key.mode == "native_lora" else "adapter_path"
    )
    for name, request in requests.items():
        request = _object(
            request,
            (
                "input_ids",
                "sampling_params",
                "return_logprob",
                "top_logprobs_num",
                "stream",
                adapter_field,
            ),
            f"request.{name}",
        )
        _require(
            request["return_logprob"] is True
            and type(request["top_logprobs_num"]) is int
            and request["top_logprobs_num"] == 5
            and request["stream"] is False
            and request[adapter_field] == "policy-a"
            and _same(request["sampling_params"], sampling),
            "invalid benchmark request",
        )
        ids = request["input_ids"]
        _require(
            isinstance(ids, (tuple, list)) and bool(ids), "missing request input IDs"
        )
        rows = (ids,) if name == "warmup" else ids
        _require(name == "warmup" or len(rows) == 32, "benchmark requires batch-32")
        _require(
            all(
                isinstance(row, (tuple, list))
                and bool(row)
                and all(type(token) is int and token >= 0 for token in row)
                for row in rows
            ),
            "invalid request input IDs",
        )
    launches = _object(
        procedure["launch"], ("initial", "preloaded"), "procedure.launch"
    )
    effective = _object(
        procedure["effective_launches"], ("initial", "preloaded"), "effective_launches"
    )
    for phase in launches:
        constructor = _require_mapping(launches[phase], f"launch.{phase}")
        required = {
            "model_path",
            "base_gpu_id",
            "tp_size",
            "ep_size",
            "disable_cuda_graph",
            "mem_fraction_static",
            "max_total_tokens",
            "log_level",
            "random_seed",
        }
        optional = {
            "quantization",
            "moe_runner_backend",
            "max_lora_rank",
            "lora_target_modules",
            "max_oft_block_size",
            "peft_target_modules",
        }
        mode_fields = (
            {"enable_lora", "enable_lora_staging"}
            if bundle.case_key.mode == "native_lora"
            else {"peft_method", "oft_type"}
        )
        paths_key = (
            "lora_paths" if bundle.case_key.mode == "native_lora" else "peft_paths"
        )
        _require(
            required | mode_fields <= constructor.keys()
            and constructor.keys()
            <= required
            | optional
            | mode_fields
            | ({paths_key} if phase == "preloaded" else set()),
            "incomplete/unexpected constructor fields",
        )
        _require(
            type(constructor["mem_fraction_static"]) is float
            and 0 < constructor["mem_fraction_static"] < 1
            and type(constructor["max_total_tokens"]) is int
            and constructor["max_total_tokens"] > 0
            and constructor["log_level"] == "error",
            "invalid launch procedure settings",
        )
        if phase == "preloaded":
            fixtures = bundle.provenance["metadata"]["fixture_files"]
            _require(
                _same(
                    constructor.get(paths_key),
                    [
                        {
                            "name": "policy-a",
                            "fixture_hash": canonical_sha256(fixtures["policy-a"]),
                        }
                    ],
                ),
                "missing preloaded policy-a identity",
            )
        _case_launch(bundle, constructor, procedure["seed"])
        capture = _object(
            effective[phase],
            ("engine", "tokenizer", "schedulers", "requests"),
            f"effective.{phase}",
        )
        _require(
            isinstance(capture["schedulers"], (tuple, list))
            and len(capture["schedulers"]) == 1,
            "one DP1 scheduler attestation required",
        )
        engine = _require_mapping(capture["engine"], "effective engine")
        for snapshot in (engine, capture["tokenizer"], *capture["schedulers"]):
            snapshot = _require_mapping(snapshot, "resolved configuration")
            _case_launch(bundle, snapshot, procedure["seed"])
            _resolved_startup(bundle, snapshot, phase)
            _require(
                engine.keys() <= snapshot.keys(), "incomplete resolved configuration"
            )
            _require(
                all(
                    key in snapshot and _same(value, snapshot[key])
                    for key, value in constructor.items()
                    if key not in ("lora_paths", "peft_paths")
                ),
                "resolved/constructor configuration mismatch",
            )
        observed = _object(
            capture["requests"], ("warmup", "batch"), "effective requests"
        )
        for name, evidence in observed.items():
            evidence = _object(
                evidence, ("request", "sampling"), "resolved request evidence"
            )
            request = _require_mapping(evidence["request"], "normalized request")
            single = name == "warmup"
            _require(
                request.get("is_single") is single
                and type(request.get("batch_size")) is int
                and request["batch_size"] == (1 if single else 32),
                "normalized request batch mismatch",
            )
            for key, value in requests[name].items():
                expected = (
                    [value] * 32
                    if not single
                    and key
                    in (
                        "sampling_params",
                        "return_logprob",
                        "top_logprobs_num",
                        adapter_field,
                    )
                    else value
                )
                _require(
                    key in request and _same(request[key], expected),
                    f"normalized request {key} mismatch",
                )
            samplers = (evidence["sampling"],) if single else evidence["sampling"]
            _require(
                isinstance(samplers, (tuple, list))
                and len(samplers) == (1 if single else 32),
                "resolved sampling batch mismatch",
            )
            for sampler in samplers:
                sampler = _require_mapping(sampler, "resolved sampler")
                _require(
                    bool(sampler) and all(key in sampler for key in sampling),
                    "incomplete resolved sampling",
                )
                _require(
                    type(sampler["top_k"]) is int
                    and sampler["top_k"] == 1
                    and type(sampler["temperature"]) is float
                    and sampler["temperature"] == 1.0
                    and _same(sampler["top_p"], sampling["top_p"])
                    and _same(sampler["max_new_tokens"], sampling["max_new_tokens"]),
                    "resolved greedy sampling mismatch",
                )
    return procedure


def validate_hardware(hardware, tp, ep, base_gpu_id=1, *, gpu_class="H200"):
    """Validate allocation-local topology, including the dedicated sender GPU."""
    hardware = _require_mapping(hardware, "hardware")
    _require_exact_fields(hardware, {"gpus", "visible_devices", "topology"}, "hardware")
    _require(
        type(tp) is int and type(ep) is int and tp > 0 and ep > 0 and tp % ep == 0,
        "invalid runtime topology",
    )
    _require(
        type(base_gpu_id) is int and base_gpu_id == 1, "sender/model placement mismatch"
    )
    gpus, topology = hardware["gpus"], hardware["topology"]
    _require(
        isinstance(gpus, (list, tuple)) and len(gpus) == tp + 1,
        "H200 allocation must contain exactly TP plus sender GPUs",
    )
    _require(
        list(hardware["visible_devices"]) == list(range(len(gpus))),
        "invalid allocation indices",
    )
    identities = set()
    for i, gpu in enumerate(gpus):
        gpu = _require_mapping(gpu, "GPU")
        _require_exact_fields(
            gpu, {"index", "name", "memory_mib", "compute_capability"}, "GPU"
        )
        _require(type(gpu["index"]) is int and gpu["index"] == i, "invalid GPU index")
        _require(
            type(gpu["name"]) is str
            and bool(gpu["name"])
            and (gpu_class is None or gpu_class in gpu["name"].split()),
            "allocated GPU class mismatch",
        )
        _require(
            type(gpu["memory_mib"]) is int and gpu["memory_mib"] > 0,
            "invalid GPU memory",
        )
        _require(
            type(gpu["compute_capability"]) is str
            and bool(gpu["compute_capability"])
            and (gpu_class != "H200" or gpu["compute_capability"] == "9.0"),
            "invalid GPU compute capability",
        )
        identities.add((gpu["name"], gpu["memory_mib"], gpu["compute_capability"]))
    _require(len(identities) == 1, "heterogeneous H200 allocation")
    _require(
        isinstance(topology, (tuple, list)) and len(topology) == len(gpus),
        "incomplete topology",
    )
    for i, row in enumerate(topology):
        _require(
            isinstance(row, (tuple, list)) and len(row) == len(gpus),
            "incomplete topology row",
        )
        for j, link in enumerate(row):
            _require(
                type(link) is str and bool(link) and ((link == "X") == (i == j)),
                "invalid topology link",
            )


def _runtime_identity(bundle):
    """Validate Task 9 metadata and return the portable execution identity.

    Role/repetition and content locations remain immutable diagnostic fields in
    the raw bundle. Execution settings and content/procedure hashes stay exact.
    """
    bundle.validate()
    exact_sha(bundle.case_key.revision)
    metadata = bundle.manifest["metadata"]
    provenance = bundle.provenance["metadata"]
    _object(
        metadata,
        (
            "case_id",
            "role",
            "repetition",
            "performance_procedure",
            "engine_kwargs",
            "initial_engine_kwargs",
        ),
        "manifest metadata",
    )
    _object(
        provenance,
        (
            "runtime_origins",
            "memory_boundary_hash",
            "argv",
            "role",
            "repetition",
            "slurm_job_id",
            "node",
            "environment",
            "hardware",
            "checkpoint",
            "fixture_files",
            "checkpoint_manifest_hash",
            "fixture_manifest_hash",
            "sender_runtime_origins",
        ),
        "provenance metadata",
    )
    _require(
        isinstance(provenance["argv"], (list, tuple))
        and bool(provenance["argv"])
        and all(type(arg) is str for arg in provenance["argv"]),
        "invalid invocation attestation",
    )
    _require(
        type(provenance["node"]) is str and bool(provenance["node"]),
        "invalid node attestation",
    )
    _require(
        provenance["slurm_job_id"] is None
        or (
            type(provenance["slurm_job_id"]) is str and bool(provenance["slurm_job_id"])
        ),
        "invalid allocation attestation",
    )
    for field in ("checkpoint_manifest_hash", "fixture_manifest_hash"):
        _validate_sha256(provenance[field], field)
    from .bundle_capture import expected_runtime_origins
    from .distributed_sender import SENDER_RUNTIME_ORIGINS

    expected_origins = expected_runtime_origins()
    _require(
        provenance.get("runtime_origins") == expected_origins,
        "parent runtime origin attestation mismatch",
    )
    senders = provenance.get("sender_runtime_origins")
    _require(
        isinstance(senders, (tuple, list))
        and bool(senders)
        and all(origin == SENDER_RUNTIME_ORIGINS for origin in senders),
        "sender runtime origin attestation mismatch",
    )
    for name in (
        "case_id",
        "role",
        "repetition",
        "performance_procedure",
        "engine_kwargs",
        "initial_engine_kwargs",
    ):
        _require(name in metadata, f"missing manifest metadata {name}")
    _require(
        type(metadata["case_id"]) is str and bool(metadata["case_id"]),
        "missing case identity",
    )
    _require(metadata["role"] in ("source", "candidate"), "invalid revision role")
    _require(
        type(metadata["repetition"]) is int
        and metadata["repetition"]
        in (range(3) if metadata["role"] == "source" else range(1)),
        "invalid repetition",
    )
    for name in ("role", "repetition"):
        _require(
            type(provenance.get(name)) is type(metadata[name])
            and provenance.get(name) == metadata[name],
            f"provenance {name} mismatch",
        )
    procedure = _procedure_attestation(bundle)
    _require(
        canonical_sha256(procedure) == bundle.performance.procedure_hash,
        "procedure metadata hash mismatch",
    )
    _require(
        type(procedure.get("samples")) is int and procedure["samples"] == 3,
        "procedure must collect three samples",
    )
    boundary = provenance.get("memory_boundary_hash")
    _validate_sha256(boundary, "memory_boundary_hash")
    _require(
        procedure.get("memory_boundary_hash") == boundary,
        "memory boundary metadata/procedure mismatch",
    )
    kwargs = _require_mapping(metadata["engine_kwargs"], "engine kwargs")
    _case_launch(bundle, kwargs, procedure["seed"])
    for metadata_name, hash_name in (
        ("environment", "environment_hash"),
        ("fixture_files", "adapter_hash"),
    ):
        evidence = _require_mapping(provenance.get(metadata_name), metadata_name)
        _require(
            canonical_sha256(evidence) == bundle.provenance[hash_name],
            f"{metadata_name} content hash mismatch",
        )
    checkpoint = _checkpoint_attestation(bundle)
    tp, ep, base = (
        kwargs.get("tp_size"),
        kwargs.get("ep_size"),
        kwargs.get("base_gpu_id"),
    )
    validate_hardware(provenance.get("hardware"), tp, ep, base, gpu_class=None)
    _require(
        canonical_sha256(
            dict(inventory=provenance["hardware"], tp=tp, ep=ep, base_gpu_id=base)
        )
        == bundle.provenance["hardware_hash"],
        "hardware metadata hash mismatch",
    )
    portable_metadata = _thaw_json(metadata)
    portable_metadata.pop("repetition")
    portable_metadata.pop("role")
    # Only these declared locations may differ. Their semantic values are
    # independently bound by the checkpoint/fixture hashes and launch procedure.
    locations = {}
    for key, phase in (
        ("engine_kwargs", "preloaded"),
        ("initial_engine_kwargs", "initial"),
    ):
        raw = portable_metadata[key]
        _require(isinstance(raw, dict), "invalid launch metadata")
        _case_launch(bundle, raw, procedure["seed"])
        launch = _require_mapping(procedure.get("launch"), "procedure.launch")
        expected_launch = _require_mapping(
            launch.get(phase), f"procedure.launch.{phase}"
        )
        for name in ("tp_size", "ep_size", "base_gpu_id"):
            _require(
                raw.get(name) == expected_launch.get(name) == kwargs.get(name),
                "launch topology/procedure mismatch",
            )
        path = raw.get("model_path")
        _require(
            type(path) is str and Path(path).is_absolute(),
            "invalid checkpoint location",
        )
        locations[path] = bundle.provenance["checkpoint_hash"]
        raw["model_path"] = {"checkpoint_hash": bundle.provenance["checkpoint_hash"]}
        for name in ("lora_paths", "peft_paths"):
            if name in raw:
                _require(name in expected_launch, "missing fixture launch identity")
                fixtures = _require_mapping(
                    provenance.get("fixture_files"), "fixture files"
                )
                normalized = []
                _require(isinstance(raw[name], list), "invalid fixture paths")
                for entry in raw[name]:
                    _require(
                        type(entry) is str and "=" in entry,
                        "invalid fixture path entry",
                    )
                    adapter, path = entry.split("=", 1)
                    _require(
                        adapter in fixtures and Path(path).is_absolute(),
                        "unbound fixture location",
                    )
                    content = {
                        "name": adapter,
                        "fixture_hash": canonical_sha256(fixtures[adapter]),
                    }
                    locations[entry] = canonical_sha256(content)
                    normalized.append(content)
                raw[name] = normalized
        _require(_same(raw, expected_launch), "launch metadata differs from procedure")
    args = list(bundle.manifest["server_args"])
    args = [locations.get(arg, arg) for arg in args]
    key = bundle.case_key.to_dict()
    key.pop("revision")
    return dict(
        case_key=key,
        provenance={
            name: bundle.provenance[name] for name in COMPARABLE_PROVENANCE_HASH_KEYS
        },
        metadata=portable_metadata,
        checkpoint={
            name: value for name, value in checkpoint.items() if name != "path"
        },
        server_args=args,
        seed=bundle.manifest["seed"],
        boundary=boundary,
    )


def _bundle_identity(bundle):
    """Final H200 matrix restrictions are additional to generic run identity."""
    identity = _runtime_identity(bundle)
    from .preflight import PINNED_MODEL_REVISIONS

    checkpoint = bundle.provenance["metadata"]["checkpoint"]
    cell_id = bundle.manifest["metadata"]["case_id"].split(".native_")[0]
    _require(
        checkpoint["id"] == cell_id
        and checkpoint["revision"] == PINNED_MODEL_REVISIONS.get(cell_id),
        "checkpoint immutable cell/revision mismatch",
    )
    kwargs = bundle.manifest["metadata"]["engine_kwargs"]
    tp, ep = kwargs["tp_size"], kwargs["ep_size"]
    _require(
        (tp, ep) == ((1, 1) if bundle.case_key.architecture == "dense" else (4, 4)),
        "case topology mismatch",
    )
    _require(
        bundle.case_key.precision in ("bf16", "fp8")
        and bundle.case_key.mode in ("native_lora", "native_oft"),
        "deferred/non-native case cannot qualify",
    )
    validate_hardware(
        bundle.provenance["metadata"]["hardware"], tp, ep, kwargs["base_gpu_id"]
    )
    _require(
        bundle.case_key.scenario == "native-adapter-lifecycle-v2",
        "wrong lifecycle scenario",
    )
    _require(
        bundle.manifest["request_order"]
        == lifecycle_transition_names(bundle.case_key.mode),
        "request order differs from full lifecycle",
    )
    return identity


def derive_reference_envelope(bundles, qualification_hash):
    _validate_sha256(qualification_hash, "qualification_hash")
    _require(
        len(bundles) == 3 and all(isinstance(b, RunBundle) for b in bundles),
        "exactly three source bundles required",
    )
    identities = [_bundle_identity(b) for b in bundles]
    _require(
        all(b.manifest["metadata"]["role"] == "source" for b in bundles),
        "only source evidence can derive tolerance",
    )
    _require(
        {b.manifest["metadata"]["repetition"] for b in bundles} == {0, 1, 2},
        "source repetitions must be exactly 0/1/2",
    )
    _require(
        len({b.digest() for b in bundles}) == 3, "source bundle hashes must be distinct"
    )
    _require(
        len({b.case_key.revision for b in bundles}) == 1, "source revisions differ"
    )
    _require(
        all(identity == identities[0] for identity in identities),
        "source execution identity mismatch",
    )
    bundles = sorted(bundles, key=lambda b: b.manifest["metadata"]["repetition"])
    baseline = bundles[0]
    structural = []
    quantities = []
    for bundle in bundles:
        comparison_observations = comparable_lease_observations(bundle.observations)
        states = comparable_adapter_states(
            {
                name: bundle.observations[name].adapter_state
                for name in bundle.manifest["request_order"]
            }
        )
        shape, values = {}, {"token_logprobs": []}
        for request_id in bundle.manifest["request_order"]:
            observation = comparison_observations[request_id].to_dict()
            observation["error"] = comparable_adapter_error(
                observation["error"], observation["adapter_state"], states[request_id]
            )
            observation["adapter_state"] = states[request_id]
            values["token_logprobs"].extend(observation["token_logprobs"])
            observation["token_logprobs"] = len(observation["token_logprobs"])
            for name, scores in observation["selected_logits"].items():
                values.setdefault(f"selected_logits.{name}", []).extend(scores)
            observation["selected_logits"] = {
                name: len(scores)
                for name, scores in observation["selected_logits"].items()
            }
            shape[request_id] = observation
        structural.append(shape)
        quantities.append(values)
    _require(
        all(shape == structural[0] for shape in structural),
        "source structural/token/text/state/error/shape mismatch",
    )
    for bundle in bundles:
        validate_lifecycle_observations(
            {
                name: bundle.observations[name]
                for name in bundle.manifest["request_order"]
            }
        )
    # Contract ruling: Task 9 hashes repetition and diagnostic locations in each
    # manifest. Rep 0 is the evidence root; retain each original bundle digest.
    tolerances = {
        name: NumericTolerance.create(
            repetitions=tuple(
                BaselineRepetition(
                    baseline.manifest_hash, bundle.digest(), tuple(values[name])
                )
                for bundle, values in zip(bundles, quantities)
            )
        )
        for name, values in quantities[0].items()
        if values
    }
    metadata = {"qualification_hash": qualification_hash}
    envelope = ToleranceEnvelope.create(
        baseline_manifest_hash=baseline.manifest_hash,
        tolerances=tolerances,
        metadata=metadata,
    )
    return envelope.with_policy(
        ComparisonPolicy.create(
            baseline_manifest_hash=baseline.manifest_hash,
            tolerance_envelope_hash=envelope.manifest_hash,
            metadata=metadata,
        )
    )


class _Inventory(dict):
    def __init__(self):
        super().__init__()
        self.failed = {}
        self.blocked = {}


def _read_blocked(manifest, shard, anchor):
    record = _require_mapping(
        read_document(shard.blocked_path, "blocked record", anchor=anchor),
        "blocked record",
    )
    _require_exact_fields(
        record,
        {"status", "reason", "detail", "qualification_hash", "shard", "record_hash"},
        "blocked record",
    )
    _require(
        record["record_hash"]
        == canonical_sha256({k: v for k, v in record.items() if k != "record_hash"}),
        "blocked record hash mismatch",
    )
    _require(
        record["status"] == "blocked"
        and record["qualification_hash"] == manifest.manifest_hash
        and record["shard"] == shard.to_dict(),
        "blocked record identity mismatch",
    )
    _require(
        type(record["detail"]) is str and bool(record["detail"].strip()),
        "blocked record requires evidence detail",
    )
    _require(
        record["reason"] == "required_h200_unavailable"
        or (
            record["reason"] == "reference_unavailable"
            and shard.revision_kind == "source"
        ),
        "invalid blocked reason",
    )
    return record


def _read_exact_inventory(manifest):
    manifest.validate()
    try:
        with DirectoryAnchor(manifest.artifact_root) as anchor:
            inventory = _read_anchored_inventory(manifest, anchor)
            anchor.verify(inventory=True)
            return inventory
    except (OSError, ValueError, TypeError, KeyError) as error:
        inventory = _Inventory()
        inventory.failed.update(
            {shard.identity: str(error) for shard in manifest.shards}
        )
        return inventory


def _read_anchored_inventory(manifest, anchor):
    inventory = _Inventory()
    allowed = {
        Path(getattr(s, f"{name}_path")) for s in manifest.shards for name in PATH_NAMES
    }
    observed = anchor.inventory()
    unexpected = sorted(str(path) for path in observed - allowed)
    for shard in manifest.shards:
        try:
            _require(not unexpected, f"unexpected artifacts: {unexpected}")
            bundle_path, marker_path = Path(shard.bundle_path), Path(
                shard.completion_path
            )
            if Path(shard.blocked_path) in observed:
                _require(
                    bundle_path not in observed and marker_path not in observed,
                    "blocked record conflicts with execution artifacts",
                )
                inventory.blocked[shard.identity] = _read_blocked(
                    manifest, shard, anchor
                )
                continue
            _require(
                bundle_path in observed and marker_path in observed,
                "missing bundle or completion artifact",
            )
            bundle = RunBundle.from_dict(
                read_document(bundle_path, "RunBundle", anchor=anchor)
            )
            marker = read_document(marker_path, "completion marker", anchor=anchor)
            _require(
                marker == {"status": "complete", "bundle_hash": bundle.digest()},
                "completion hash mismatch",
            )
            _require(
                bundle.case_key == shard.case_key, "shard case/SHA identity mismatch"
            )
            _bundle_identity(bundle)
            metadata = bundle.manifest["metadata"]
            _require(
                (metadata["case_id"], metadata["role"], metadata["repetition"])
                == shard.identity,
                "shard role/repetition/case identity mismatch",
            )
            _require(
                (
                    metadata["engine_kwargs"]["tp_size"],
                    metadata["engine_kwargs"]["ep_size"],
                )
                == (shard.tp_size, shard.ep_size),
                "shard topology mismatch",
            )
            _require(shard.identity not in inventory, "duplicate shard identity")
            inventory[shard.identity] = bundle
        except (OSError, ValueError, TypeError, KeyError) as error:
            inventory.failed[shard.identity] = str(error)
    return inventory


def _compare_case(manifest, case_id, inventory):
    shards = [s for s in manifest.shards if s.case_id == case_id]
    failures = [
        dict(shard=s.identity, detail=inventory.failed[s.identity])
        for s in shards
        if s.identity in inventory.failed
    ]
    blocked = [
        inventory.blocked[s.identity] for s in shards if s.identity in inventory.blocked
    ]
    if failures or blocked:
        return dict(
            case_id=case_id,
            status="failed" if failures else "blocked",
            failures=failures,
            blocked=blocked,
            mismatches=[
                dict(kind="inventory_mismatch", detail=item) for item in failures
            ],
        )
    try:
        references = [inventory[case_id, "source", rep] for rep in range(3)]
        candidate = inventory[case_id, "candidate", 0]
        envelope = derive_reference_envelope(references, manifest.manifest_hash)
        envelope.validate()
        _require(
            envelope.metadata.get("qualification_hash") == manifest.manifest_hash
            and envelope.comparison_policy.metadata.get("qualification_hash")
            == manifest.manifest_hash,
            "envelope qualification hash mismatch",
        )
        report = compare_bundles(references[0], candidate, envelope)
        mismatches = [m.to_dict() for m in report.mismatches]
        identity_valid = _bundle_identity(references[0]) == _bundle_identity(candidate)
        if not identity_valid:
            mismatches.append(
                dict(
                    kind="execution_identity_mismatch",
                    detail="source/candidate metadata or memory boundary differs",
                )
            )
        performance = None
        if identity_valid and report.performance is not None:
            performance = dict(
                comparison=report.performance,
                source_medians=[
                    {
                        name: statistics.median(getattr(b.performance, name))
                        for name in (
                            "startup_seconds",
                            "latency_seconds",
                            "throughput_tokens_per_second",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                        )
                    }
                    for b in references
                ],
            )
        else:
            mismatches = [
                m for m in mismatches if m["kind"] != "performance_regression"
            ]
        return dict(
            case_id=case_id,
            status="failed" if mismatches else "passed",
            mismatches=mismatches,
            performance=performance,
            envelope=envelope.to_dict(),
            bundle_hashes=[b.digest() for b in (*references, candidate)],
        )
    except (ValueError, TypeError, KeyError) as error:
        return dict(
            case_id=case_id,
            status="failed",
            mismatches=[dict(kind="invalid_reference_evidence", detail=str(error))],
        )


@dataclass(frozen=True)
class QualificationReport:
    status: str
    scope: QualificationScope
    passed: tuple[str, ...]
    failed: tuple[dict, ...]
    blocked: tuple[dict, ...]
    mismatches: tuple[dict, ...]
    results: tuple[dict, ...]
    manifest_hash: str | None = None
    qualifying: bool = False

    @classmethod
    def from_case_results(cls, results, scope, *, manifest_hash=None, qualifying=False):
        _require(bool(results), "empty case results cannot pass")
        _require(
            all(r.get("status") in ("passed", "failed", "blocked") for r in results),
            "unknown case status",
        )
        passed = tuple(r["case_id"] for r in results if r["status"] == "passed")
        failed = tuple(r for r in results if r["status"] == "failed")
        blocked = tuple(r for r in results if r["status"] == "blocked")
        return cls(
            "failed" if failed or blocked else "passed",
            scope,
            passed,
            failed,
            blocked,
            tuple(m for r in results for m in r.get("mismatches", ())),
            tuple(results),
            manifest_hash,
            qualifying,
        )

    def to_dict(self):
        payload = dict(
            status=self.status,
            scope=self.scope.to_dict(),
            passed=list(self.passed),
            failed=list(self.failed),
            blocked=list(self.blocked),
            mismatches=list(self.mismatches),
            results=list(self.results),
            manifest_hash=self.manifest_hash,
            qualifying=self.qualifying,
            coverage_label=(
                "H200 BF16/FP8 matrix evaluated"
                if self.qualifying
                else "non-qualifying subset"
            ),
            qualification_passed=self.qualifying and self.status == "passed",
            deferred_status="unqualified",
        )
        return dict(payload, report_hash=canonical_sha256(payload))


def aggregate_manifest(path):
    return _aggregate_manifest_object(QualificationManifest.read_json(path))


def _aggregate_manifest_object(manifest):
    inventory = _read_exact_inventory(manifest)
    return QualificationReport.from_case_results(
        tuple(
            _compare_case(manifest, case, inventory)
            for case in sorted({s.case_id for s in manifest.shards})
        ),
        manifest.scope,
        manifest_hash=manifest.manifest_hash,
        qualifying=manifest.qualifying,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Atomic report outside the artifact root",
    )
    args = parser.parse_args(argv)
    try:
        manifest = QualificationManifest.read_json(args.manifest)
        try:
            outside_artifact_root(args.output, manifest.artifact_root)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
        # Containment and evaluation must consume this exact parsed identity.
        report = _aggregate_manifest_object(manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = QualificationReport.from_case_results(
            (
                dict(
                    case_id="<manifest>",
                    status="failed",
                    mismatches=[dict(kind="invalid_manifest", detail=str(error))],
                ),
            ),
            CURRENT_SCOPE,
        )
    try:
        publish_document(args.output, report.to_dict())
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return int(report.status != "passed")


if __name__ == "__main__":
    raise SystemExit(main())

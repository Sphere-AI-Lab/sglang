"""Qualification must reject incomplete or mismatched immutable evidence."""

# ruff: noqa: E402 -- registered CPU tests add the manual harness import root.

import ast
import copy
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence import aggregate, preflight, qualification
from adapter_equivalence.bundle_capture import expected_runtime_origins, publish_bundle
from adapter_equivalence.fixtures import validate_matrix
from adapter_equivalence.schema import (
    PROVENANCE_HASH_KEYS,
    SCHEMA_VERSION,
    PerformanceMetrics,
    RunBundle,
    canonical_sha256,
)
from test_harness_contract import complete_native_observations

MATRIX = Path(qualification.__file__).with_name("matrix.json")
CASE = "qwen3-4b-bf16.native_lora.graph-0"


def manifest_at(tmp_path, selected=(CASE,)):
    return qualification.build_manifest(
        validate_matrix(MATRIX),
        "a" * 40,
        "b" * 40,
        tmp_path / "artifacts",
        selected_case_ids=selected,
    )


def resolved_reference(mode, fixture_files):
    """Source-defined fields/defaults, normalized by the accepted Task 9 producer.

    Avoid importing the GPU/msgspec runtime just to serialize its reference.
    """
    from adapter_equivalence import run_case

    root = Path(__file__).resolve().parents[4]
    relative, class_name, prefix = (
        ("lora/lora_registry.py", "LoRARef", "lora")
        if mode == "native_lora"
        else ("oft/base/registry.py", "AdapterRef", "adapter")
    )
    tree = ast.parse((root / "python/sglang/srt" / relative).read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fields = {
        node.target.id: (
            "ephemeral-id"
            if node.target.id.endswith("_id")
            else ast.literal_eval(node.value)
        )
        for node in definition.body
        if isinstance(node, ast.AnnAssign)
    }
    fields.update(
        {f"{prefix}_name": "policy-a", f"{prefix}_path": "/fixtures/a", "pinned": False}
    )
    return run_case.normalized_effective_config(
        fields,
        SimpleNamespace(
            model_path="/checkpoint", startup_adapters=(("policy-a", "/fixtures/a"),)
        ),
        {"checkpoint_hash": "a" * 64, "metadata": {"fixture_files": fixture_files}},
    )


def with_resolved_references(bundle):
    def change(p):
        procedure = p["manifest"]["metadata"]["performance_procedure"]
        mode = p["case_key"]["mode"]
        active = "lora_paths" if mode == "native_lora" else "peft_paths"
        for phase, capture in procedure["effective_launches"].items():
            for config in (
                capture["engine"],
                capture["tokenizer"],
                *capture["schedulers"],
            ):
                config.update(
                    lora_paths=None, peft_paths=[] if mode == "native_oft" else None
                )
                if phase == "preloaded":
                    config[active] = [
                        resolved_reference(
                            mode, p["provenance"]["metadata"]["fixture_files"]
                        )
                    ]
        digest = canonical_sha256(procedure)
        p["performance"]["procedure_hash"] = p["manifest"][
            "performance_procedure_hash"
        ] = digest

    return rebuild(bundle, change)


def bundle_for(shard, delta=0.0):
    from adapter_equivalence import run_case

    hardware = {
        "gpus": [
            dict(
                index=i, name="NVIDIA H200", memory_mib=143771, compute_capability="9.0"
            )
            for i in range(shard.tp_size + 1)
        ],
        "visible_devices": list(range(shard.tp_size + 1)),
        "topology": [
            ["X" if i == j else "NV18" for j in range(shard.tp_size + 1)]
            for i in range(shard.tp_size + 1)
        ],
    }
    launch = dict(
        tp_size=shard.tp_size,
        ep_size=shard.ep_size,
        base_gpu_id=1,
        disable_cuda_graph=not shard.case_key.cuda_graph,
        random_seed=1729,
        mem_fraction_static=0.7,
        max_total_tokens=32768,
        log_level="error",
    )
    if shard.case_key.mode == "native_lora":
        launch.update(enable_lora=True, enable_lora_staging=True)
    else:
        launch.update(peft_method="oft", oft_type="oft")
    if shard.case_key.precision == "fp8":
        launch["quantization"] = "fp8"
    procedure = dict(
        version=1,
        samples=3,
        seed=1729,
        memory_boundary_hash="c" * 64,
        launch={"initial": launch, "preloaded": launch},
    )
    metadata = dict(
        case_id=shard.case_id,
        role=shard.revision_kind,
        repetition=shard.repetition,
        performance_procedure=procedure,
        engine_kwargs=dict(launch, model_path="/checkpoint"),
        initial_engine_kwargs=dict(launch, model_path="/checkpoint"),
    )
    hashes = {key: str(i) * 64 for i, key in enumerate(PROVENANCE_HASH_KEYS)}
    hashes["environment_hash"] = canonical_sha256({"python": "3.12", "packages": []})
    fixture_files = {
        name: {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}
        for name in ("policy-a", "policy-b", "2", "3")
    }
    hashes["adapter_hash"] = canonical_sha256(fixture_files)
    files = {
        "config.json": "1" * 64,
        "model.safetensors": "2" * 64,
        "tokenizer.json": "3" * 64,
        "tokenizer_config.json": "4" * 64,
    }
    cell_id = shard.case_id.split(".native_")[0]
    checkpoint = dict(
        id=cell_id,
        model=shard.case_key.model,
        revision=preflight.PINNED_MODEL_REVISIONS[cell_id],
        path="/checkpoint",
        layout="unsharded",
        index_hash=None,
        files=files,
        checkpoint_hash=canonical_sha256(
            {k: files[k] for k in ("config.json", "model.safetensors")}
        ),
        tokenizer_hash=canonical_sha256(
            {k: files[k] for k in ("tokenizer.json", "tokenizer_config.json")}
        ),
    )
    hashes["checkpoint_hash"] = canonical_sha256(
        dict(entry={k: v for k, v in checkpoint.items() if k != "path"}, files=files)
    )
    hashes["tokenizer_hash"] = checkpoint["tokenizer_hash"]
    hashes["hardware_hash"] = canonical_sha256(
        dict(inventory=hardware, tp=shard.tp_size, ep=shard.ep_size, base_gpu_id=1)
    )
    for phase in ("initial", "preloaded"):
        procedure["launch"][phase] = dict(
            launch, model_path={"checkpoint_hash": hashes["checkpoint_hash"]}
        )
    adapter_arg = (
        "lora_path" if shard.case_key.mode == "native_lora" else "adapter_path"
    )
    paths_arg = "lora_paths" if shard.case_key.mode == "native_lora" else "peft_paths"
    metadata["engine_kwargs"][paths_arg] = ["policy-a=/fixtures/a"]
    procedure["launch"]["preloaded"][paths_arg] = [
        {
            "name": "policy-a",
            "fixture_hash": canonical_sha256(fixture_files["policy-a"]),
        }
    ]
    sampling = dict(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=32)
    warmup = dict(
        input_ids=[101, 102],
        sampling_params=sampling,
        return_logprob=True,
        top_logprobs_num=5,
        stream=False,
        **{adapter_arg: "policy-a"},
    )
    batch = dict(warmup, input_ids=[[101, 102] for _ in range(32)])
    procedure.update(
        requests=dict(warmup=warmup, batch=batch),
        sampling=sampling,
        warmup="one factual request on a fresh engine, before every sample",
        workload="one fixed batch-32, startup policy-a for native modes",
        memory="synchronize/reset each scheduler; request set; synchronize/read each scheduler; sum all TP allocated/reserved peaks",
        latency="perf_counter elapsed around async_generate, excluding memory RPCs",
        throughput="actual output token IDs across all 32 responses divided by latency",
        implementation=[
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
        ],
    )
    normalized_requests = {}
    for name, request in procedure["requests"].items():
        is_single = name == "warmup"
        normalized = dict(
            copy.deepcopy(request),
            is_single=is_single,
            batch_size=1 if is_single else 32,
        )
        if not is_single:
            for field in (
                "sampling_params",
                "return_logprob",
                "top_logprobs_num",
                adapter_arg,
            ):
                normalized[field] = [copy.deepcopy(request[field]) for _ in range(32)]
        normalized_requests[name] = {
            "request": normalized,
            "sampling": (
                dict(sampling, temperature=1.0, top_k=1, min_new_tokens=0)
                if is_single
                else [
                    dict(sampling, temperature=1.0, top_k=1, min_new_tokens=0)
                    for _ in range(32)
                ]
            ),
        }
    procedure["effective_launches"] = {
        phase: {
            "engine": copy.deepcopy(config),
            "tokenizer": copy.deepcopy(config),
            "schedulers": [copy.deepcopy(config)],
            "requests": copy.deepcopy(normalized_requests),
        }
        for phase, config in procedure["launch"].items()
    }
    reference = resolved_reference(shard.case_key.mode, fixture_files)
    for phase, capture in procedure["effective_launches"].items():
        for config in (capture["engine"], capture["tokenizer"], *capture["schedulers"]):
            config.update(
                lora_paths=None,
                peft_paths=[] if shard.case_key.mode == "native_oft" else None,
            )
            if phase == "preloaded":
                config[paths_arg] = [copy.deepcopy(reference)]
    origins = expected_runtime_origins()
    observations = complete_native_observations(shard.case_key.mode)
    observations = {
        key: replace(
            value, token_logprobs=tuple(x + delta for x in value.token_logprobs)
        )
        for key, value in observations.items()
    }
    return RunBundle.create(
        case_key=shard.case_key,
        manifest=dict(
            schema_version=SCHEMA_VERSION,
            case_key=shard.case_key.to_dict(),
            provenance_hashes=hashes,
            performance_procedure_hash=canonical_sha256(procedure),
            server_args=["--model-path", "/checkpoint"],
            request_order=list(observations),
            seed=1729,
            metadata=metadata,
        ),
        provenance=dict(
            git_sha=shard.case_key.revision,
            dirty=False,
            **hashes,
            metadata=dict(
                role=shard.revision_kind,
                repetition=shard.repetition,
                argv=["run_case.py", "--qualification"],
                node="h200-node",
                slurm_job_id=None,
                checkpoint_manifest_hash="d" * 64,
                fixture_manifest_hash="e" * 64,
                hardware=hardware,
                environment={"python": "3.12", "packages": []},
                fixture_files=fixture_files,
                checkpoint=checkpoint,
                memory_boundary_hash="c" * 64,
                runtime_origins=origins,
                sender_runtime_origins=[
                    {"init_custom_process_group": "python/sglang/srt/utils/common.py"}
                ],
            ),
        ),
        observations=observations,
        performance=PerformanceMetrics(
            canonical_sha256(procedure),
            (1.0, 1.0, 1.0),
            (0.1, 0.1, 0.1),
            (100.0, 100.0, 100.0),
            (1000, 1000, 1000),
            (1200, 1200, 1200),
        ),
        completion=dict(status="complete", exit_code=0, metadata={}),
    )


def write_run(tmp_path, mutate=None, selected=(CASE,)):
    manifest = manifest_at(tmp_path, selected)
    for shard in manifest.shards:
        bundle = bundle_for(shard, shard.repetition * 0.01)
        if mutate:
            bundle = mutate(shard, bundle)
        if bundle is not None:
            Path(shard.bundle_path).parent.mkdir(parents=True, exist_ok=True)
            publish_bundle(bundle, shard.bundle_path, shard.completion_path)
    path = tmp_path / "qualification.json"
    manifest.write_json(path)
    return path, manifest


def rebuild(bundle, transform):
    payload = bundle.to_dict()
    transform(payload)
    payload["manifest_hash"] = canonical_sha256(payload["manifest"])
    payload["provenance"]["manifest_hash"] = payload["manifest_hash"]
    return RunBundle.from_dict(payload)


def test_manifest_expands_to_exactly_64_unique_h200_shards(tmp_path):
    manifest = manifest_at(tmp_path, None)
    assert len(manifest.shards) == len({s.identity for s in manifest.shards}) == 64
    assert Counter(s.revision_kind for s in manifest.shards) == {
        "source": 48,
        "candidate": 16,
    }
    assert {s.gpu_class for s in manifest.shards} == {"H200"}
    assert {s.case_key.precision for s in manifest.shards} == {"bf16", "fp8"}
    assert manifest.scope == qualification.CURRENT_SCOPE
    assert manifest.scope.deferred_coverage == ("dense:nvfp4:B200", "moe:nvfp4:B200")
    assert manifest.qualifying is True
    manifest.write_json(tmp_path / "manifest.json")
    assert (
        qualification.QualificationManifest.read_json(tmp_path / "manifest.json")
        == manifest
    )
    with pytest.raises(FileExistsError):
        manifest.write_json(tmp_path / "manifest.json")


@pytest.mark.parametrize("selected", [(), (CASE, CASE), ("unknown",)])
def test_unknown_empty_duplicate_subset_cannot_hide_required_cases(tmp_path, selected):
    with pytest.raises(ValueError):
        manifest_at(tmp_path, selected)


@pytest.mark.parametrize("sha", ["main", "a" * 39, "A" * 40, "g" * 40, None])
def test_manifest_rejects_nonimmutable_revision(tmp_path, sha):
    with pytest.raises(ValueError):
        qualification.build_manifest(validate_matrix(MATRIX), sha, "b" * 40, tmp_path)


@pytest.mark.parametrize(
    "defect", ["duplicate", "missing", "gpu", "topology", "path", "scope", "qualifying"]
)
def test_rehashed_manifest_cannot_change_frozen_inventory(tmp_path, defect):
    manifest = manifest_at(tmp_path)
    payload = manifest.to_dict()
    if defect == "duplicate":
        payload["shards"].append(payload["shards"][0])
    elif defect == "missing":
        payload["shards"].pop()
    elif defect == "gpu":
        payload["shards"][0]["gpu_class"] = "B200"
    elif defect == "topology":
        payload["shards"][0]["tp_size"] = 2
    elif defect == "path":
        payload["shards"][0]["bundle_path"] = str(tmp_path / "elsewhere.json")
    elif defect == "scope":
        payload["scope"]["deferred_coverage"] = []
    else:
        payload["qualifying"] = True
    payload["manifest_hash"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "manifest_hash"}
    )
    path = tmp_path / "mutant.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        qualification.QualificationManifest.read_json(path)


def test_complete_subset_passes_but_never_claims_final_qualification(tmp_path):
    path, manifest = write_run(tmp_path)
    report = aggregate.aggregate_manifest(path)
    assert report.status == "passed"
    assert report.passed == (CASE,)
    assert report.qualifying is False
    assert report.manifest_hash == manifest.manifest_hash
    assert report.scope == qualification.CURRENT_SCOPE


@pytest.mark.parametrize(
    "defect", ["missing", "marker", "unexpected", "duplicate", "blocked-malformed"]
)
def test_incomplete_or_extra_artifacts_fail(tmp_path, defect):
    path, manifest = write_run(tmp_path)
    shard = manifest.shards[-1]
    if defect == "missing":
        Path(shard.completion_path).unlink()
    elif defect == "marker":
        Path(shard.completion_path).write_text(
            json.dumps(dict(status="complete", bundle_hash="0" * 64))
        )
    elif defect in ("unexpected", "duplicate"):
        Path(shard.bundle_path).with_name("extra.json").write_text(
            Path(shard.bundle_path).read_text()
        )
    else:
        Path(shard.blocked_path).write_text('{"status":"blocked"}')
    report = aggregate.aggregate_manifest(path)
    assert report.status == "failed"
    assert report.failed


@pytest.mark.parametrize(
    "reason", ["required_h200_unavailable", "reference_unavailable"]
)
def test_only_explicit_missing_hardware_or_reference_is_blocked(tmp_path, reason):
    path, manifest = write_run(
        tmp_path, lambda s, b: None if s.revision_kind == "source" else b
    )
    for shard in manifest.shards[:-1]:
        Path(shard.blocked_path).parent.mkdir(parents=True, exist_ok=True)
        preflight.write_blocked_record(
            manifest, shard, reason, "scheduler/reference unavailable"
        )
    report = aggregate.aggregate_manifest(path)
    assert report.status == "failed"
    assert len(report.blocked) == 1
    assert not report.failed


@pytest.mark.parametrize(
    "field",
    ["mode", "sha", "role", "repetition", "hardware", "boundary", "procedure", "order"],
)
def test_shard_identity_mutation_fails(tmp_path, field):
    def mutate(shard, bundle):
        if shard.revision_kind != "candidate":
            return bundle

        def change(p):
            if field in ("mode", "sha"):
                name, value = (
                    ("mode", "native_oft")
                    if field == "mode"
                    else ("revision", "c" * 40)
                )
                p["case_key"][name] = p["manifest"]["case_key"][name] = value
                if field == "mode":
                    for obs in p["observations"].values():
                        obs["adapter_state"]["mode"] = value
                else:
                    p["provenance"]["git_sha"] = value
            elif field in ("role", "repetition"):
                p["provenance"]["metadata"][field] = "source" if field == "role" else 1
            elif field == "hardware":
                p["provenance"]["metadata"]["hardware"]["gpus"][0][
                    "name"
                ] = "NVIDIA H100"
            elif field == "boundary":
                p["provenance"]["metadata"]["memory_boundary_hash"] = "f" * 64
            elif field == "procedure":
                p["manifest"]["metadata"]["performance_procedure"]["samples"] = 1
            else:
                p["manifest"]["request_order"].reverse()

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize(
    "field,kind",
    [
        ("output_ids", "token_mismatch"),
        ("text", "text_mismatch"),
        ("token_logprobs", "numeric_mismatch"),
        ("selected_logits", "numeric_mismatch"),
        ("adapter_state", "adapter_state_mismatch"),
        ("error", "error_mismatch"),
    ],
)
def test_candidate_cannot_widen_envelope_or_hide_exact_mutation(tmp_path, field, kind):
    def mutate(shard, bundle):
        if shard.revision_kind != "candidate":
            return bundle

        def change(p):
            o = p["observations"]["stage.v2.active"]
            if field == "output_ids":
                o[field][0] += 1
            elif field == "text":
                o[field] = "mutant"
            elif field == "token_logprobs":
                o[field][0] = -4.0
            elif field == "selected_logits":
                o[field][next(iter(o[field]))][0] = -4.0
            elif field == "adapter_state":
                o[field]["active"]["version"] = "999"
                o[field]["registered"][0]["version"] = "999"
            else:
                o[field] = dict(
                    kind="product_rejection", code="unexpected", message="mutant"
                )

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    report = aggregate.aggregate_manifest(path)
    assert report.status == "failed"
    assert any(m["kind"] == kind for m in report.mismatches)


def test_reference_roots_preserve_task9_repetition_metadata(tmp_path):
    shards = manifest_at(tmp_path).shards[:3]
    bundles = tuple(bundle_for(s, s.repetition * 0.01) for s in shards)
    assert len({b.manifest_hash for b in bundles}) == 3
    envelope = aggregate.derive_reference_envelope(bundles[::-1], "d" * 64)
    assert envelope.baseline_manifest_hash == bundles[0].manifest_hash
    assert envelope.metadata["qualification_hash"] == "d" * 64
    assert envelope.comparison_policy.metadata["qualification_hash"] == "d" * 64
    reps = envelope.tolerances["token_logprobs"].repetitions
    assert {r.bundle_hash for r in reps} == {b.digest() for b in bundles}
    assert {r.baseline_manifest_hash for r in reps} == {bundles[0].manifest_hash}


@pytest.mark.parametrize(
    "defect", ["count", "duplicate", "candidate", "token", "order", "hardware"]
)
def test_invalid_reference_evidence_never_derives_envelope(tmp_path, defect):
    shards = manifest_at(tmp_path).shards
    bundles = [bundle_for(s, s.repetition * 0.01) for s in shards[:3]]
    if defect == "count":
        bundles.pop()
    elif defect == "duplicate":
        bundles[2] = bundles[1]
    elif defect == "candidate":
        bundles[2] = bundle_for(shards[-1])
    else:

        def change(p):
            if defect == "token":
                p["observations"]["stage.v2.active"]["output_ids"][0] += 1
            elif defect == "order":
                p["manifest"]["request_order"].reverse()
            else:
                p["provenance"]["metadata"]["hardware"]["gpus"][0][
                    "name"
                ] = "NVIDIA H100"

        bundles[2] = rebuild(bundles[2], change)
    with pytest.raises(ValueError):
        aggregate.derive_reference_envelope(bundles, "d" * 64)


@pytest.mark.parametrize(
    "raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "{", "[]"]
)
def test_malformed_manifest_cli_fails_closed_and_hashes_report(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    output = tmp_path / "report.json"
    assert aggregate.main(["--manifest", str(path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report.pop("report_hash") == canonical_sha256(report)


def test_report_publication_is_exclusive(tmp_path):
    path, _ = write_run(tmp_path)
    output = tmp_path / "report.json"
    assert aggregate.main(["--manifest", str(path), "--output", str(output)]) == 0
    original = output.read_bytes()
    assert aggregate.main(["--manifest", str(path), "--output", str(output)]) == 1
    assert output.read_bytes() == original


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def fake_checkout(root):
    """Minimal importable native runtime: test the real isolated process boundary."""
    for package in (
        "sglang",
        "sglang/srt",
        "sglang/srt/entrypoints",
        "sglang/srt/managers",
        "sglang/srt/oft",
    ):
        path = root / "python" / package
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("")
    (root / "python/torch.py").write_text(
        "from types import SimpleNamespace\n"
        "_C = SimpleNamespace()\n"
        "cuda = SimpleNamespace(is_initialized=lambda: False)\n"
    )
    methods = [
        f"{verb}_{adapter}_adapter{suffix}"
        for adapter in ("lora", "oft")
        for verb, suffix in (
            ("load", ""),
            ("load", "_from_tensors"),
            ("load", "_from_distributed"),
            ("unload", ""),
        )
    ]
    (root / "python/sglang/srt/entrypoints/engine.py").write_text(
        "class Engine:\n" + "".join(f"    def {name}(self): pass\n" for name in methods)
    )
    (root / "python/sglang/srt/managers/tokenizer_manager.py").write_text(
        "class TokenizerManager:\n    def update_adapter_from_distributed(self): pass\n"
        "    def activate_adapter_version(self): pass\n"
    )
    for adapter, path in (
        ("LoRA", "managers/io_struct.py"),
        ("OFT", "oft/io_types.py"),
    ):
        names = [
            f"{verb}{adapter}Adapter{suffix}ReqInput"
            for verb, suffix in (
                ("Load", ""),
                ("Load", "FromTensors"),
                ("Load", "FromDistributed"),
                ("Unload", ""),
            )
        ]
        if adapter == "LoRA":
            names += [
                "UpdateAdapterFromDistributedReqInput",
                "ActivateAdapterVersionReqInput",
            ]
        (root / "python/sglang/srt" / path).write_text(
            "".join(f"class {name}: pass\n" for name in names)
        )
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Harness",
        "-c",
        "user.email=harness@example.invalid",
        "commit",
        "-qm",
        "native runtime",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize("mode", ["native_lora", "native_oft"])
def test_native_capability_subprocess_reads_selected_checkout(tmp_path, mode):
    fake_checkout(tmp_path)
    result = preflight.check_native_capability(tmp_path, mode)
    assert result["mode"] == mode
    assert result["cuda_initialized"] is False
    assert (
        result["origins"]["sglang.srt.entrypoints.engine"]
        == "python/sglang/srt/entrypoints/engine.py"
    )


def test_native_capability_stubs_h200_import_probe_without_cuda_initialization(
    tmp_path,
):
    """Engine imports may select H200 code without owning a CUDA context."""
    fake_checkout(tmp_path)
    engine = tmp_path / "python/sglang/srt/entrypoints/engine.py"
    engine.write_text(
        "import torch\n"
        "torch.cuda.is_available = lambda: True\n"
        "device = torch.cuda.current_device()\n"
        "properties = torch.cuda.get_device_properties(device)\n"
        "assert device == 0\n"
        "assert (properties.major, properties.minor) == (9, 0)\n"
        "assert properties.multi_processor_count == 132\n"
        "assert torch.cuda.get_device_capability(device) == (9, 0)\n"
        + engine.read_text()
    )

    result = preflight.check_native_capability(tmp_path, "native_lora")

    assert result["cuda_initialized"] is False


def test_native_capability_accepts_checkout_owned_namespace_package(tmp_path):
    """A namespace package is local when every search path is local."""
    fake_checkout(tmp_path)
    (tmp_path / "python/sglang/srt/__init__.py").unlink()

    result = preflight.check_native_capability(tmp_path, "native_lora")

    assert result["cuda_initialized"] is False


def test_native_capability_prevents_editable_install_fallthrough(tmp_path):
    """Missing generated metadata must not fall through to another checkout."""
    fake_checkout(tmp_path)
    package = tmp_path / "python/sglang"
    foreign = tmp_path.parent / "foreign-version.py"
    foreign.write_text("__version__ = 'foreign'\n")
    (package / "__init__.py").write_text("from sglang.version import __version__\n")
    (package / "version.py").write_text(
        "try:\n"
        "    from sglang._version import __version__\n"
        "except ImportError:\n"
        "    __version__ = 'checkout-fallback'\n"
    )
    (tmp_path / "python/torch.py").write_text(
        "import importlib.util, sys\n"
        "from types import SimpleNamespace\n"
        "_C = SimpleNamespace()\n"
        "cuda = SimpleNamespace(is_initialized=lambda: False)\n"
        "class ForeignVersionFinder:\n"
        "    @staticmethod\n"
        "    def find_spec(fullname, path=None, target=None):\n"
        "        if fullname == 'sglang._version':\n"
        f"            return importlib.util.spec_from_file_location(fullname, {str(foreign)!r})\n"
        "sys.meta_path.append(ForeignVersionFinder)\n"
    )

    result = preflight.check_native_capability(tmp_path, "native_lora")

    assert result["cuda_initialized"] is False
    assert "sglang._version" not in result["origins"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_method",
        "missing_type",
        "foreign",
        "foreign_callable",
        "foreign_package",
        "cuda",
    ],
)
def test_capability_import_fails_closed(tmp_path, defect):
    fake_checkout(tmp_path)
    engine = tmp_path / "python/sglang/srt/entrypoints/engine.py"
    if defect == "missing_method":
        engine.write_text(
            engine.read_text().replace("load_lora_adapter_from_tensors", "absent")
        )
    elif defect == "missing_type":
        types_path = tmp_path / "python/sglang/srt/managers/io_struct.py"
        types_path.write_text(
            types_path.read_text().replace("ActivateAdapterVersionReqInput", "Absent")
        )
    elif defect == "foreign":
        engine.write_text(engine.read_text() + "\n__file__ = '/foreign/engine.py'\n")
    elif defect == "foreign_callable":
        engine.write_text(engine.read_text() + "\nEngine.load_lora_adapter = print\n")
    elif defect == "foreign_package":
        (tmp_path / "python/sglang/srt/__init__.py").unlink()
        foreign = tmp_path.parent / "foreign-runtime"
        foreign.mkdir(exist_ok=True)
        engine.write_text(
            "import sglang.srt\n"
            f"sglang.srt.__path__.append({str(foreign)!r})\n" + engine.read_text()
        )
    else:
        engine.write_text("import torch\ntorch.cuda.init()\n" + engine.read_text())
    with pytest.raises(ValueError):
        preflight.check_native_capability(tmp_path, "native_lora")


@pytest.mark.parametrize(
    "defect",
    [
        "dirty-source",
        "dirty-candidate",
        "sha-source",
        "sha-candidate",
        "boundary",
        "hardware",
    ],
)
def test_preflight_requires_clean_exact_checkouts_and_common_memory_boundary(
    tmp_path, monkeypatch, defect
):
    from adapter_equivalence import bundle_capture

    source, candidate = tmp_path / "source", tmp_path / "candidate"
    source_sha, candidate_sha = fake_checkout(source), fake_checkout(candidate)
    manifest = qualification.build_manifest(
        validate_matrix(MATRIX),
        source_sha,
        candidate_sha,
        tmp_path / "artifacts",
        (CASE,),
    )
    shard = manifest.shards[-1]
    if defect.startswith("dirty"):
        (source if defect.endswith("source") else candidate).joinpath(
            "dirty"
        ).write_text("dirty")
    elif defect.startswith("sha"):
        manifest = qualification.build_manifest(
            validate_matrix(MATRIX),
            "f" * 40 if defect.endswith("source") else source_sha,
            "f" * 40 if defect.endswith("candidate") else candidate_sha,
            tmp_path / "artifacts",
            (CASE,),
        )
        shard = manifest.shards[-1]

    def boundary(checkout, revision):
        assert revision == (source_sha if Path(checkout) == source else candidate_sha)
        return (
            "c" if defect == "boundary" and Path(checkout) == candidate else "d"
        ) * 64

    monkeypatch.setattr(bundle_capture, "memory_boundary_hash", boundary)
    hardware = bundle_for(shard).to_dict()["provenance"]["metadata"]["hardware"]
    if defect == "hardware":
        hardware["gpus"][0]["name"] = "NVIDIA H100"
    with pytest.raises(ValueError):
        preflight.preflight_shard(manifest, shard, source, candidate, hardware=hardware)


def test_absent_hardware_or_reference_is_explicitly_blocked(tmp_path):
    manifest = manifest_at(tmp_path)
    with pytest.raises(preflight.PreflightBlocked, match="reference"):
        preflight.preflight_shard(
            manifest,
            manifest.shards[0],
            tmp_path / "absent",
            tmp_path / "candidate",
            hardware={},
        )


def test_all_64_shards_can_pass_and_report_exact_scope(tmp_path):
    path, _ = write_run(tmp_path, selected=None)
    report = aggregate.aggregate_manifest(path)
    assert report.status == "passed"
    assert len(report.passed) == 16
    assert report.qualifying is True


def test_relocated_bundle_locations_remain_comparable(tmp_path):
    def mutate(shard, bundle):
        def change(p):
            location = f"/{shard.revision_kind}/rep-{shard.repetition}/checkpoint"
            p["manifest"]["server_args"][1] = location
            for name in ("engine_kwargs", "initial_engine_kwargs"):
                p["manifest"]["metadata"][name]["model_path"] = location

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "passed"


@pytest.mark.parametrize(
    "defect", ["runtime", "sender", "extra_gpu", "seed", "launch", "lifecycle"]
)
def test_reference_metadata_and_lifecycle_cannot_be_minimal_or_forged(tmp_path, defect):
    def mutate(shard, bundle):
        def change(p):
            metadata = p["provenance"]["metadata"]
            if defect == "runtime":
                metadata["runtime_origins"] = {}
            elif defect == "sender":
                metadata["sender_runtime_origins"] = []
            elif defect == "extra_gpu":
                hardware = metadata["hardware"]
                hardware["gpus"].append(dict(hardware["gpus"][0], index=2))
                hardware["visible_devices"].append(2)
                hardware["topology"] = [
                    ["X" if i == j else "NV18" for j in range(3)] for i in range(3)
                ]
                digest = canonical_sha256(
                    dict(inventory=hardware, tp=1, ep=1, base_gpu_id=1)
                )
                p["provenance"]["hardware_hash"] = p["manifest"]["provenance_hashes"][
                    "hardware_hash"
                ] = digest
            elif defect == "seed":
                if shard.revision_kind == "candidate":
                    p["manifest"]["seed"] += 1
            elif defect == "launch":
                p["manifest"]["metadata"]["engine_kwargs"]["disable_cuda_graph"] = False
            else:
                p["observations"]["base.restored"]["output_ids"][0] += 1

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "failed"


def test_adapter_path_relocation_is_protected_by_fixture_content(tmp_path):
    def mutate(shard, bundle):
        def change(p):
            fixture = {
                "adapter_config.json": "a" * 64,
                "adapter_model.safetensors": "b" * 64,
            }
            p["provenance"]["metadata"]["fixture_files"]["policy-a"] = fixture
            adapter_hash = canonical_sha256(
                p["provenance"]["metadata"]["fixture_files"]
            )
            p["provenance"]["adapter_hash"] = p["manifest"]["provenance_hashes"][
                "adapter_hash"
            ] = adapter_hash
            entry = f"policy-a=/{shard.revision_kind}/{shard.repetition}/adapter"
            p["manifest"]["server_args"] += ["--lora-paths", entry]
            p["manifest"]["metadata"]["engine_kwargs"]["lora_paths"] = [entry]
            procedure = p["manifest"]["metadata"]["performance_procedure"]
            procedure["launch"]["preloaded"]["lora_paths"] = [
                {"name": "policy-a", "fixture_hash": canonical_sha256(fixture)}
            ]
            digest = canonical_sha256(procedure)
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "passed"


def test_direct_comparison_withholds_all_performance_on_boundary_metadata_mismatch(
    tmp_path,
):
    from adapter_equivalence.compare import compare_bundles

    shards = manifest_at(tmp_path).shards
    references = [bundle_for(s) for s in shards[:3]]
    candidate = bundle_for(shards[-1])

    def change(p):
        p["provenance"]["metadata"]["memory_boundary_hash"] = "f" * 64
        p["performance"]["throughput_tokens_per_second"] = [1.0, 1.0, 1.0]

    candidate = rebuild(candidate, change)
    report = compare_bundles(
        references[0],
        candidate,
        aggregate.derive_reference_envelope(references, "d" * 64),
    )
    assert report.performance is None
    assert all(m.kind != "performance_regression" for m in report.mismatches)
    assert any(m.kind == "performance_identity_mismatch" for m in report.mismatches)


def test_preflight_cli_records_absent_reference_as_blocked(tmp_path):
    manifest = manifest_at(tmp_path)
    path = tmp_path / "manifest.json"
    manifest.write_json(path)
    shard = manifest.shards[0]
    Path(shard.blocked_path).parent.mkdir(parents=True)
    result = preflight.main(
        [
            "--qualification-manifest",
            str(path),
            "--case-id",
            CASE,
            "--revision-kind",
            "source",
            "--repetition",
            "0",
            "--reference-checkout",
            str(tmp_path / "absent"),
            "--candidate-checkout",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "preflight.json"),
        ]
    )
    assert result == 1
    assert (
        json.loads(Path(shard.blocked_path).read_text())["reason"]
        == "reference_unavailable"
    )


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_preflight_rejects_nonfinite_and_duplicate_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        preflight._load_json(path, "preflight input")


@pytest.mark.parametrize(
    "defect", ["graph", "mode", "precision", "checkpoint", "environment", "fixture"]
)
def test_rehashed_metadata_cannot_disagree_with_case_or_content_identity(
    tmp_path, defect
):
    def mutate(shard, bundle):
        def change(p):
            metadata = p["manifest"]["metadata"]
            provenance = p["provenance"]["metadata"]
            if defect in ("graph", "mode", "precision"):
                key, value = {
                    "graph": ("disable_cuda_graph", False),
                    "mode": ("enable_lora", False),
                    "precision": ("quantization", "fp8"),
                }[defect]
                for name, phase in (
                    ("engine_kwargs", "preloaded"),
                    ("initial_engine_kwargs", "initial"),
                ):
                    metadata[name][key] = value
                    metadata["performance_procedure"]["launch"][phase][key] = value
                digest = canonical_sha256(metadata["performance_procedure"])
                p["performance"]["procedure_hash"] = p["manifest"][
                    "performance_procedure_hash"
                ] = digest
            elif defect == "checkpoint":
                provenance["checkpoint"]["model"] = "other/model"
            elif defect == "environment":
                provenance["environment"]["python"] = "3.99"
            else:
                provenance["fixture_files"]["missing"] = {"weights": "f" * 64}

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("disable_cuda_graph", False),
        ("quantization", "fp8"),
        ("random_seed", 999),
        ("enable_lora", False),
    ],
)
def test_initial_phase_cannot_lie_consistently_across_all_shards(
    tmp_path, field, value
):
    def mutate(shard, bundle):
        def change(p):
            metadata = p["manifest"]["metadata"]
            metadata["initial_engine_kwargs"][field] = value
            metadata["performance_procedure"]["launch"]["initial"][field] = value
            effective = (
                metadata["performance_procedure"]
                .get("effective_launches", {})
                .get("initial", {})
            )
            for config in (
                effective.get("engine", {}),
                effective.get("tokenizer", {}),
                *effective.get("schedulers", []),
            ):
                config[field] = value
            digest = canonical_sha256(metadata["performance_procedure"])
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize(
    "field",
    [
        "effective_launches",
        "requests",
        "warmup",
        "workload",
        "sampling",
        "memory",
        "latency",
        "throughput",
        "implementation",
    ],
)
def test_missing_task9_procedure_evidence_never_qualifies(tmp_path, field):
    def mutate(shard, bundle):
        def change(p):
            procedure = p["manifest"]["metadata"]["performance_procedure"]
            procedure.pop(field, None)
            digest = canonical_sha256(procedure)
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "revision",
        "layout",
        "index_hash",
        "files",
        "checkpoint_hash",
        "tokenizer_hash",
    ],
)
def test_missing_checkpoint_attestation_never_qualifies(tmp_path, field):
    def mutate(shard, bundle):
        return rebuild(
            bundle, lambda p: p["provenance"]["metadata"]["checkpoint"].pop(field, None)
        )

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize(
    "name", ["bundle", "completion", "jsonl", "stdout", "stderr", "blocked"]
)
@pytest.mark.parametrize("kind", ["fifo", "hardlink"])
def test_allowed_artifact_objects_are_regular_unaliased_files_without_hanging(
    tmp_path, name, kind
):
    path, manifest = write_run(tmp_path)
    shard = manifest.shards[0]
    target = Path(getattr(shard, name + "_path"))
    content = target.read_bytes() if target.is_file() else b"{}"
    if target.exists():
        target.unlink()
    if name == "blocked":
        Path(shard.bundle_path).unlink()
        Path(shard.completion_path).unlink()
    if kind == "fifo":
        os.mkfifo(target)
    else:
        external = tmp_path / "external.json"
        external.write_bytes(content)
        os.link(external, target)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(aggregate.__file__)),
                "--manifest",
                str(path),
                "--output",
                str(tmp_path / "report.json"),
            ],
            capture_output=True,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("aggregation hung while opening a special artifact")
    assert result.returncode == 1


@pytest.mark.parametrize("mode", ["native_lora", "native_oft"])
def test_direct_native_tp2_comparison_is_not_a_matrix_gate(tmp_path, mode):
    from adapter_equivalence.compare import compare_bundles
    from test_compare import _envelope

    shard = replace(
        manifest_at(tmp_path).shards[0],
        tp_size=2,
        case_key=replace(manifest_at(tmp_path).shards[0].case_key, mode=mode),
    )
    bundle = bundle_for(shard)
    report = compare_bundles(bundle, bundle, _envelope(bundle))
    assert report.passed
    assert report.performance is not None
    with pytest.raises(ValueError, match="case topology"):
        aggregate._bundle_identity(bundle)

    def change(p):
        p["provenance"]["metadata"]["memory_boundary_hash"] = "f" * 64

    mismatch = compare_bundles(bundle, rebuild(bundle, change), _envelope(bundle))
    assert not mismatch.passed and mismatch.performance is None
    assert all(item.kind != "performance_regression" for item in mismatch.mismatches)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_cli_outputs_must_be_canonically_outside_inventory(tmp_path, alias, nested):
    path, manifest = write_run(tmp_path)
    root = Path(manifest.artifact_root)
    if alias:
        link = tmp_path / "alias"
        link.symlink_to(root, target_is_directory=True)
        output = link / "report.json"
    else:
        output = root / "report.json"
    if nested:
        output = output.parent / "nonexistent" / "report.json"
    assert aggregate.main(["--manifest", str(path), "--output", str(output)]) == 1
    assert not output.exists()
    with pytest.raises(ValueError):
        manifest.write_json(output)
    assert aggregate.aggregate_manifest(path).status == "passed"


@pytest.mark.parametrize(
    "failure",
    [
        "first-fsync",
        "second-fsync",
        "third-fsync",
        "link",
        "cleanup",
        "readback",
        "pre-open",
        "post-open",
        "pre-close",
        "post-close",
    ],
)
def test_atomic_document_commit_point_is_unambiguous(tmp_path, monkeypatch, failure):
    path, _ = write_run(tmp_path)
    output = tmp_path / "report.json"
    fsync, link, unlink = (
        qualification.os.fsync,
        qualification.os.link,
        qualification.os.unlink,
    )
    opening, closing, reading = (
        qualification.os.open,
        qualification.os.close,
        qualification.read_document,
    )
    directories, closes = [], []
    calls = []

    def sync(fd):
        calls.append(fd)
        if failure == {1: "first-fsync", 2: "second-fsync", 3: "third-fsync"}.get(
            len(calls)
        ):
            raise OSError("injected durability error")
        return fsync(fd)

    def linking(source, destination):
        if failure == "link":
            raise OSError("injected link error")
        return link(source, destination)

    def cleanup(target):
        assert Path(target) != output
        if failure == "cleanup":
            raise OSError("injected cleanup error")
        return unlink(target)

    def open_directory(target, *args, **kwargs):
        if Path(target) == tmp_path:
            directories.append(target)
            if failure == {1: "pre-open", 2: "post-open"}.get(len(directories)):
                raise OSError("injected directory open error")
        return opening(target, *args, **kwargs)

    def close_directory(fd):
        closing(fd)
        # Reader anchors close their own descriptors before publication starts.
        if not directories:
            return
        closes.append(fd)
        if failure == {1: "pre-close", 2: "post-close"}.get(len(closes)):
            raise OSError("injected directory close error")

    def readback(target, context, **kwargs):
        if failure == "readback" and context == "publication":
            raise ValueError("injected readback error")
        return reading(target, context, **kwargs)

    monkeypatch.setattr(qualification.os, "fsync", sync)
    monkeypatch.setattr(qualification.os, "link", linking)
    monkeypatch.setattr(qualification.os, "unlink", cleanup)
    monkeypatch.setattr(qualification.os, "open", open_directory)
    monkeypatch.setattr(qualification.os, "close", close_directory)
    monkeypatch.setattr(qualification, "read_document", readback)
    result = aggregate.main(["--manifest", str(path), "--output", str(output)])
    committed = failure in ("third-fsync", "cleanup", "post-open", "post-close")
    assert result == (0 if committed else 1)
    assert output.exists() is committed


@pytest.mark.parametrize(
    "defect",
    [
        "procedure-extra",
        "procedure-version-bool",
        "checkpoint-extra",
        "checkpoint-id",
        "checkpoint-revision",
        "checkpoint-files",
        "effective-extra",
        "effective-engine-empty",
        "effective-tokenizer-empty",
        "effective-schedulers-empty",
        "effective-requests-empty",
        "effective-request-extra",
        "effective-sampling-empty",
        "effective-initial-graph",
        "initial-oft-type",
        "constructor-missing",
        "preloaded-missing-adapter",
    ],
)
def test_rehashed_task9_attestations_fail_closed(tmp_path, defect):
    shard = manifest_at(tmp_path).shards[0]
    if defect == "initial-oft-type":
        shard = replace(shard, case_key=replace(shard.case_key, mode="native_oft"))

    def change(p):
        metadata = p["manifest"]["metadata"]
        procedure = metadata["performance_procedure"]
        checkpoint = p["provenance"]["metadata"]["checkpoint"]
        effective = procedure["effective_launches"]["initial"]
        if defect == "procedure-extra":
            procedure["extra"] = True
        elif defect == "procedure-version-bool":
            procedure["version"] = True
        elif defect == "checkpoint-extra":
            checkpoint["extra"] = True
        elif defect == "checkpoint-id":
            checkpoint["id"] = "another-cell"
        elif defect == "checkpoint-revision":
            checkpoint["revision"] = "a" * 40
        elif defect == "checkpoint-files":
            checkpoint["files"]["unattested.bin"] = "a" * 64
        elif defect == "effective-extra":
            effective["extra"] = True
        elif defect.startswith("effective-") and defect.endswith("-empty"):
            field = defect.split("-")[1]
            if field == "sampling":
                effective["requests"]["warmup"]["sampling"] = {}
            else:
                effective[field] = [] if field == "schedulers" else {}
        elif defect == "effective-request-extra":
            effective["requests"]["warmup"]["extra"] = True
        elif defect == "effective-initial-graph":
            for config in (
                effective["engine"],
                effective["tokenizer"],
                *effective["schedulers"],
            ):
                config["disable_cuda_graph"] = False
        elif defect == "initial-oft-type":
            metadata["initial_engine_kwargs"]["enable_lora"] = 0
            procedure["launch"]["initial"]["enable_lora"] = 0
            for config in (
                effective["engine"],
                effective["tokenizer"],
                *effective["schedulers"],
            ):
                config["enable_lora"] = 0
        elif defect == "constructor-missing":
            metadata["initial_engine_kwargs"].pop("log_level", None)
            procedure["launch"]["initial"].pop("log_level", None)
        elif defect == "preloaded-missing-adapter":
            metadata["engine_kwargs"].pop("lora_paths")
            procedure["launch"]["preloaded"].pop("lora_paths")
            for config in (
                procedure["effective_launches"]["preloaded"]["engine"],
                procedure["effective_launches"]["preloaded"]["tokenizer"],
                *procedure["effective_launches"]["preloaded"]["schedulers"],
            ):
                config.pop("lora_paths")
        digest = canonical_sha256(procedure)
        p["performance"]["procedure_hash"] = p["manifest"][
            "performance_procedure_hash"
        ] = digest

    with pytest.raises(ValueError):
        aggregate._bundle_identity(rebuild(bundle_for(shard), change))


def test_rehashed_wrong_initial_graph_cannot_qualify_full_64_shard_matrix(tmp_path):
    def mutate(shard, bundle):
        def change(p):
            metadata = p["manifest"]["metadata"]
            procedure = metadata["performance_procedure"]
            effective = procedure["effective_launches"]["initial"]
            for config in (
                metadata["initial_engine_kwargs"],
                procedure["launch"]["initial"],
                effective["engine"],
                effective["tokenizer"],
                *effective["schedulers"],
            ):
                config["disable_cuda_graph"] = shard.case_key.cuda_graph
            digest = canonical_sha256(procedure)
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, manifest = write_run(tmp_path, mutate, selected=None)
    report = aggregate.aggregate_manifest(path)
    assert len(manifest.shards) == 64 and len(report.failed) == 16
    assert not report.to_dict()["qualification_passed"]


@pytest.mark.parametrize(
    "field",
    [
        "argv",
        "node",
        "slurm_job_id",
        "checkpoint_manifest_hash",
        "fixture_manifest_hash",
    ],
)
def test_stripped_task9_diagnostic_attestation_is_not_evidence(tmp_path, field):
    bundle = bundle_for(manifest_at(tmp_path).shards[0])
    with pytest.raises(ValueError):
        aggregate._bundle_identity(
            rebuild(bundle, lambda p: p["provenance"]["metadata"].pop(field, None))
        )


def test_self_consistent_non_32_token_workload_is_rejected(tmp_path):
    bundle = bundle_for(manifest_at(tmp_path).shards[0])

    def change(p):
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "max_new_tokens":
                        value[key] = 64
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        procedure = p["manifest"]["metadata"]["performance_procedure"]
        walk(procedure)
        digest = canonical_sha256(procedure)
        p["performance"]["procedure_hash"] = p["manifest"][
            "performance_procedure_hash"
        ] = digest

    with pytest.raises(ValueError):
        aggregate._bundle_identity(rebuild(bundle, change))


def test_auxiliary_tokenizer_files_have_distinct_valid_top_level_identity(tmp_path):
    def mutate(shard, bundle):
        def change(p):
            checkpoint = p["provenance"]["metadata"]["checkpoint"]
            # Task 9's recursive tokenizer identity includes auxiliaries; the
            # checkpoint entry intentionally hashes only its required pair.
            files = {
                key: value
                for key, value in checkpoint["files"].items()
                if key.startswith("tokenizer")
            }
            files["special_tokens_map.json"] = "e" * 64
            digest = canonical_sha256(files)
            assert digest != checkpoint["tokenizer_hash"]
            p["provenance"]["tokenizer_hash"] = p["manifest"]["provenance_hashes"][
                "tokenizer_hash"
            ] = digest

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate)
    assert aggregate.aggregate_manifest(path).status == "passed"


@pytest.mark.parametrize("mode", ["native_lora", "native_oft"])
def test_real_task9_resolved_references_are_accepted(tmp_path, mode):
    shard = manifest_at(tmp_path).shards[0]
    shard = replace(shard, case_key=replace(shard.case_key, mode=mode))
    aggregate._bundle_identity(with_resolved_references(bundle_for(shard)))


@pytest.mark.parametrize("oft_type", (None, "canonical_" + "oft"))
def test_old_split_oft_bundles_cannot_qualify_even_with_consistent_hashes(
    tmp_path, oft_type
):
    def mutate(shard, bundle):
        def change(p):
            def replace_mode(value):
                if isinstance(value, dict):
                    if value.get("peft_method") == "oft":
                        if oft_type is None:
                            value.pop("oft_type", None)
                        else:
                            value["oft_type"] = oft_type
                    for child in value.values():
                        replace_mode(child)
                elif isinstance(value, list):
                    for child in value:
                        replace_mode(child)

            replace_mode(p)
            procedure = p["manifest"]["metadata"]["performance_procedure"]
            digest = canonical_sha256(procedure)
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, _ = write_run(
        tmp_path, mutate, selected=("qwen3-4b-bf16.native_oft.graph-0",)
    )
    report = aggregate.aggregate_manifest(path)
    assert report.status == "failed"
    expected_error = (
        "incomplete/unexpected constructor fields"
        if oft_type is None
        else "native OFT launch mismatch"
    )
    assert expected_error in json.dumps(report.to_dict())


@pytest.mark.parametrize("surface", ("constructor", "engine", "tokenizer", "scheduler"))
@pytest.mark.parametrize("phase", ("initial", "preloaded"))
@pytest.mark.parametrize("oft_type", (None, "canonical_" + "oft"))
def test_every_oft_launch_surface_requires_shared_rotation(
    tmp_path, surface, phase, oft_type
):
    shard = manifest_at(tmp_path, ("qwen3-4b-bf16.native_oft.graph-0",)).shards[0]
    bundle = bundle_for(shard)
    procedure = bundle.to_dict()["manifest"]["metadata"]["performance_procedure"]
    capture = procedure["effective_launches"][phase]
    config = (
        procedure["launch"][phase]
        if surface == "constructor"
        else (capture["schedulers"][0] if surface == "scheduler" else capture[surface])
    )
    if oft_type is None:
        config.pop("oft_type")
    else:
        config["oft_type"] = oft_type

    def change(p):
        p["manifest"]["metadata"]["performance_procedure"] = procedure
        digest = canonical_sha256(procedure)
        p["performance"]["procedure_hash"] = p["manifest"][
            "performance_procedure_hash"
        ] = digest

    expected_error = (
        "incomplete/unexpected constructor fields"
        if surface == "constructor" and oft_type is None
        else "native OFT launch mismatch"
    )
    with pytest.raises(ValueError, match=expected_error):
        aggregate._procedure_attestation(rebuild(bundle, change))


@pytest.mark.parametrize("mode", ["native_lora", "native_oft"])
@pytest.mark.parametrize("surface", ["engine", "tokenizer", "scheduler"])
def test_resolved_initial_startup_refs_cannot_lie(tmp_path, mode, surface):
    def mutate(shard, bundle):
        def change(p):
            procedure = p["manifest"]["metadata"]["performance_procedure"]
            capture = procedure["effective_launches"]["initial"]
            config = (
                capture["schedulers"][0] if surface == "scheduler" else capture[surface]
            )
            config["lora_paths" if mode == "native_lora" else "peft_paths"] = [
                resolved_reference(mode, p["provenance"]["metadata"]["fixture_files"])
            ]
            digest = canonical_sha256(procedure)
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, _ = write_run(tmp_path, mutate, selected=(f"qwen3-4b-bf16.{mode}.graph-0",))
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize("kind", ["fifo", "symlink", "hardlink", "device"])
def test_manifest_special_inputs_fail_promptly(tmp_path, kind):
    path, _ = write_run(tmp_path)
    target = tmp_path / "input.json"
    if kind == "fifo":
        os.mkfifo(target)
    elif kind == "symlink":
        target.symlink_to(path)
    elif kind == "hardlink":
        os.link(path, target)
    else:
        target = Path("/dev/null")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(aggregate.__file__)),
                "--manifest",
                str(target),
                "--output",
                str(tmp_path / "report.json"),
            ],
            capture_output=True,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("manifest input blocked on a special file")
    assert result.returncode == 1


@pytest.mark.parametrize("kind", ["manifest", "artifact"])
@pytest.mark.parametrize("stage", ["stat", "read"])
@pytest.mark.parametrize("target", ["parent", "ancestor"])
def test_safe_reader_rejects_directory_substitution(
    tmp_path, monkeypatch, kind, stage, target
):
    directory = tmp_path / "root" / "child"
    directory.mkdir(parents=True)
    path = directory / "evidence.json"
    if kind == "manifest":
        manifest_at(tmp_path).write_json(path)
    else:
        path.write_text('{"valid": true}')
    victim = directory if target == "parent" else directory.parent
    relocated = tmp_path / "relocated"
    done = False
    regular, reading = qualification.regular_artifact, qualification.json.load

    def swap():
        nonlocal done
        if not done:
            done = True
            victim.rename(relocated)
            victim.symlink_to(relocated, target_is_directory=True)

    def checked(*args, **kwargs):
        result = regular(*args, **kwargs)
        if stage == "stat" and Path(args[0]).name == path.name:
            swap()
        return result

    def parsed(*args, **kwargs):
        result = reading(*args, **kwargs)
        if stage == "read":
            swap()
        return result

    monkeypatch.setattr(qualification, "regular_artifact", checked)
    monkeypatch.setattr(qualification.json, "load", parsed)
    with pytest.raises((ValueError, OSError)):
        if kind == "manifest":
            qualification.QualificationManifest.read_json(path)
        else:
            qualification.read_document(path, "evidence", artifact=True)


@pytest.mark.parametrize("target", ["root", "shard"])
@pytest.mark.parametrize("symlink", [False, True])
def test_inventory_anchor_rejects_post_scan_directory_replacement(
    tmp_path, monkeypatch, target, symlink
):
    path, manifest = write_run(tmp_path)
    victim = (
        Path(manifest.artifact_root)
        if target == "root"
        else Path(manifest.shards[0].bundle_path).parent
    )
    relocated = tmp_path / "relocated"
    reading = aggregate.read_document
    done = False

    def replaced(*args, **kwargs):
        nonlocal done
        if not done and args[1] == "RunBundle":
            done = True
            victim.rename(relocated)
            if symlink:
                victim.symlink_to(relocated, target_is_directory=True)
            else:
                shutil.copytree(relocated, victim)
        return reading(*args, **kwargs)

    monkeypatch.setattr(aggregate, "read_document", replaced)
    assert aggregate.aggregate_manifest(path).status == "failed"


@pytest.mark.parametrize("mode", ["native_lora", "native_oft"])
@pytest.mark.parametrize("surface", ["engine", "tokenizer", "scheduler"])
@pytest.mark.parametrize(
    "defect",
    [
        "name",
        "id",
        "path",
        "pinned",
        "reloadable",
        "version",
        "extra",
        "missing",
        "inactive",
    ],
)
def test_resolved_reference_fields_remain_bound(tmp_path, mode, surface, defect):
    shard = manifest_at(tmp_path).shards[0]
    bundle = bundle_for(replace(shard, case_key=replace(shard.case_key, mode=mode)))

    def change(p):
        procedure = p["manifest"]["metadata"]["performance_procedure"]
        capture = procedure["effective_launches"]["preloaded"]
        config = (
            capture["schedulers"][0] if surface == "scheduler" else capture[surface]
        )
        prefix = "lora" if mode == "native_lora" else "adapter"
        active = "lora_paths" if mode == "native_lora" else "peft_paths"
        record = config[active][0]
        if defect in ("name", "id", "path"):
            record[f"{prefix}_{defect}"] = "unbound"
        elif defect == "version":
            record["version" if mode == "native_lora" else "adapter_version"] = True
        elif defect in ("pinned", "reloadable"):
            record[defect] = not record[defect]
        elif defect == "extra":
            record["extra"] = True
        elif defect == "missing":
            record.pop("pinned")
        else:
            config["peft_paths" if active == "lora_paths" else "lora_paths"] = [record]
        digest = canonical_sha256(procedure)
        p["performance"]["procedure_hash"] = p["manifest"][
            "performance_procedure_hash"
        ] = digest

    with pytest.raises(ValueError):
        aggregate._bundle_identity(rebuild(bundle, change))


def test_consistent_initial_resolved_preload_rejects_all_64_shards(tmp_path):
    def mutate(shard, bundle):
        def change(p):
            procedure = p["manifest"]["metadata"]["performance_procedure"]
            active = (
                "lora_paths" if shard.case_key.mode == "native_lora" else "peft_paths"
            )
            capture = procedure["effective_launches"]["initial"]
            for config in (
                capture["engine"],
                capture["tokenizer"],
                *capture["schedulers"],
            ):
                config[active] = [
                    resolved_reference(
                        shard.case_key.mode,
                        p["provenance"]["metadata"]["fixture_files"],
                    )
                ]
            digest = canonical_sha256(procedure)
            p["performance"]["procedure_hash"] = p["manifest"][
                "performance_procedure_hash"
            ] = digest

        return rebuild(bundle, change)

    path, manifest = write_run(tmp_path, mutate, selected=None)
    report = aggregate.aggregate_manifest(path)
    assert len(manifest.shards) == 64 and len(report.failed) == 16
    assert not report.to_dict()["qualification_passed"]


@pytest.mark.parametrize("mode", [stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK])
def test_manifest_special_type_is_rejected_before_leaf_open(
    tmp_path, monkeypatch, mode
):
    path, _ = write_run(tmp_path)
    checking, opening = qualification.os.stat, qualification.os.open

    def checked(target, **kwargs):
        result = checking(target, **kwargs)
        if Path(target).name == path.name:
            fields = list(result)
            fields[0] = mode | 0o600
            return os.stat_result(fields)
        return result

    def opened(target, *args, **kwargs):
        assert Path(target).name != path.name, "special manifest must not be opened"
        return opening(target, *args, **kwargs)

    monkeypatch.setattr(qualification.os, "stat", checked)
    monkeypatch.setattr(qualification.os, "open", opened)
    with pytest.raises(ValueError, match="regular single-link"):
        qualification.QualificationManifest.read_json(path)


@pytest.mark.parametrize("mutation", ["replace", "hardlink", "inplace"])
def test_safe_reader_rechecks_leaf_after_parsing(tmp_path, monkeypatch, mutation):
    path = tmp_path / "evidence.json"
    path.write_text('{"valid": true}')
    reading = qualification.json.load

    def parsed(*args, **kwargs):
        result = reading(*args, **kwargs)
        if mutation == "replace":
            path.rename(tmp_path / "original.json")
            path.write_text('{"valid": true}')
        elif mutation == "hardlink":
            os.link(path, tmp_path / "alias.json")
        else:
            path.write_text('{"valid": false}')
        return result

    monkeypatch.setattr(qualification.json, "load", parsed)
    with pytest.raises(ValueError):
        qualification.read_document(path, "evidence", artifact=True)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_inventory_rejects_entry_swap_after_first_stat(tmp_path, monkeypatch, kind):
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "entry"
    if kind == "directory":
        victim.mkdir()
        (victim / "data.json").write_text("{}")
    else:
        victim.write_text("{}")
    checking = qualification.os.stat
    done = False

    def checked(target, **kwargs):
        nonlocal done
        result = checking(target, **kwargs)
        if target == victim.name and not done:
            done = True
            victim.rename(tmp_path / "old")
            if kind == "directory":
                shutil.copytree(tmp_path / "old", victim)
            else:
                victim.write_text("{}")
        return result

    with qualification.DirectoryAnchor(root) as anchor:
        monkeypatch.setattr(qualification.os, "stat", checked)
        with pytest.raises(ValueError):
            anchor.inventory()


def test_cli_aggregates_the_same_manifest_checked_for_output_containment(
    tmp_path, monkeypatch
):
    path_a, manifest_a = write_run(tmp_path / "a")
    path_b, manifest_b = write_run(tmp_path / "b")
    output = Path(manifest_b.artifact_root) / "report.json"
    read, contain = (
        qualification.QualificationManifest.read_json,
        aggregate.outside_artifact_root,
    )
    reads = []

    def parsed(path):
        reads.append(path)
        return read(path)

    def swap_after_containment(path, root):
        contain(path, root)
        # This is the old CLI boundary between selecting A and rereading B.
        path_a.write_text(path_b.read_text())

    monkeypatch.setattr(
        qualification.QualificationManifest, "read_json", staticmethod(parsed)
    )
    monkeypatch.setattr(aggregate, "outside_artifact_root", swap_after_containment)
    assert aggregate.main(["--manifest", str(path_a), "--output", str(output)]) == 0
    report = qualification.read_document(output, "report")
    assert report["manifest_hash"] == manifest_a.manifest_hash
    assert not output.is_relative_to(manifest_a.artifact_root)
    assert reads == [path_a]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_reference_envelope_accepts_consistent_runtime_id_renaming(tmp_path):
    def rename(bundle, prefix):
        def change(payload):
            for observation in payload["observations"].values():
                state = observation["adapter_state"]
                for record in [
                    *state["registered"],
                    state["active"],
                    state["staged"],
                    *state["tombstoned"],
                ]:
                    if record is not None:
                        record["id"] = prefix + record["id"]
                state["cache_identity"] = {
                    name: prefix + value
                    for name, value in state["cache_identity"].items()
                }

                if observation["error"] and observation["error"]["code"] == "wrong_id":
                    observation["error"]["message"] = (
                        "Requested adapter_id 'wrong-id' does not match expected adapter_id '"
                        + state["active"]["id"]
                        + "'"
                    )

        return rebuild(bundle, change)

    bundles = [
        rename(bundle_for(shard), str(shard.repetition))
        for shard in manifest_at(tmp_path).shards[:3]
    ]
    aggregate.derive_reference_envelope(bundles, "d" * 64)


def test_reference_envelope_ignores_only_valid_lease_chunk_length(tmp_path):
    shards = manifest_at(tmp_path).shards[:3]
    bundles = []
    for index, shard in enumerate(shards):

        def extend(payload):
            obs = payload["observations"]
            # All A-output oracles must agree with the completed request.
            for name in ("switch.a", "switch.a-again", "upsert.lease.complete"):
                item = obs[name]
                item["output_ids"] *= 2
                item["token_logprobs"] *= 2
                item["request_output_lengths"] = [2]
                for field in ("selected_logits", "selected_token_ids"):
                    item[field]["decode.001.top_logprobs"] = list(
                        item[field]["decode.000.top_logprobs"]
                    )
            if index == 1:
                obs["upsert.lease.begin"] = copy.deepcopy(obs["upsert.lease.complete"])

        bundles.append(rebuild(bundle_for(shard), extend))
    raw_before = [b.to_dict() for b in bundles]
    envelope = aggregate.derive_reference_envelope(bundles, "d" * 64)
    assert [b.to_dict() for b in bundles] == raw_before
    assert {
        r.bundle_hash for r in envelope.tolerances["token_logprobs"].repetitions
    } == {b.digest() for b in bundles}

    measured = tuple(
        value
        for name in bundles[0].manifest["request_order"]
        if name != "upsert.lease.begin"
        for value in bundles[0].observations[name].token_logprobs
    )
    assert all(
        repetition.values == measured
        for repetition in envelope.tolerances["token_logprobs"].repetitions
    )

    def shorten(payload):
        payload["observations"]["upsert.lease.begin"] = copy.deepcopy(
            bundles[0].observations["upsert.lease.begin"].to_dict()
        )

    shorter = aggregate.derive_reference_envelope(
        [rebuild(bundle, shorten) for bundle in bundles], "d" * 64
    )
    assert set(shorter.tolerances) == set(envelope.tolerances)
    for quantity in envelope.tolerances:
        assert [r.values for r in shorter.tolerances[quantity].repetitions] == [
            r.values for r in envelope.tolerances[quantity].repetitions
        ]

    def corrupt(payload):
        payload["observations"]["upsert.lease.begin"]["token_logprobs"][0] = -99.0

    corrupted = [rebuild(bundle, corrupt) for bundle in bundles]
    with pytest.raises(ValueError, match="lease prefix"):
        aggregate.derive_reference_envelope(corrupted, "d" * 64)

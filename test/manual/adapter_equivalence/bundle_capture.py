"""Fail-closed evidence capture and publication for adapter qualification."""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from importlib.machinery import PathFinder
from pathlib import Path

from .schema import (
    PERFORMANCE_REPETITIONS,
    PROVENANCE_HASH_KEYS,
    BundleValidationError,
    CaseKey,
    Observation,
    PerformanceMetrics,
    RunBundle,
    canonical_sha256,
    validate_adapter_state,
)
from .server import ControlResult

REPO_ROOT = Path(__file__).resolve().parents[3]

_RUNTIME_MODULES = (
    "sglang.srt.entrypoints.engine",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.tokenizer_control_mixin",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.io_struct",
    "sglang.srt.managers.communicator",
    "sglang.srt.utils.common",
    "sglang.utils",
)
_RUNTIME_NAMESPACE_PACKAGES = frozenset(
    ("sglang.srt", "sglang.srt.entrypoints", "sglang.srt.managers")
)


def expected_runtime_origins():
    """Return the canonical checkout-relative runtime module inventory."""
    origins = {}
    for target in _RUNTIME_MODULES:
        parts = target.split(".")
        for index in range(len(parts)):
            name = ".".join(parts[: index + 1])
            path = Path("python").joinpath(*parts[: index + 1])
            if index == len(parts) - 1:
                path = path.with_suffix(".py")
            elif name not in _RUNTIME_NAMESPACE_PACKAGES:
                path /= "__init__.py"
            origins[name] = path.as_posix()
    return origins


def bind_runtime(checkout=None):
    """Pin imports and spawn's inherited sys.path without executing runtime code."""
    checkout = Path(REPO_ROOT if checkout is None else checkout).resolve(strict=True)
    python_root = checkout / "python"
    if not python_root.is_dir():
        raise BundleValidationError("recorded checkout has no Python runtime root")
    sys.path[:] = [
        str(python_root),
        *(entry for entry in sys.path if entry != str(python_root)),
    ]
    # Never let an already-loaded foreign package defeat the new path order.
    for name, module in tuple(sys.modules.items()):
        if name in ("sglang", "sglang.utils") or name.startswith("sglang.srt"):
            origin = getattr(module, "__file__", None)
            paths = getattr(module, "__path__", ())
            if any(
                not Path(path).resolve().is_relative_to(python_root) for path in paths
            ):
                raise BundleValidationError(
                    f"loaded runtime package path is outside recorded checkout: {name}"
                )
            if origin is not None:
                if not Path(origin).resolve().is_relative_to(python_root):
                    raise BundleValidationError(
                        f"loaded runtime origin is outside recorded checkout: {name}"
                    )
                if (
                    name in _RUNTIME_MODULES
                    and Path(origin).resolve()
                    != python_root.joinpath(*name.split("."))
                    .with_suffix(".py")
                    .resolve()
                ):
                    raise BundleValidationError(
                        f"loaded runtime target differs from recorded checkout: {name}"
                    )
            elif not paths or any(
                not Path(path).resolve().is_relative_to(python_root) for path in paths
            ):
                raise BundleValidationError(
                    f"loaded runtime has no verified checkout origin: {name}"
                )
    expected_origins = expected_runtime_origins()
    origins = {}
    searches = {}
    for target in _RUNTIME_MODULES:
        parts = target.split(".")
        search = None
        for index in range(len(parts)):
            name = ".".join(parts[: index + 1])
            expected = checkout / expected_origins[name]
            is_namespace = name in _RUNTIME_NAMESPACE_PACKAGES
            if not (expected.is_dir() if is_namespace else expected.is_file()):
                raise BundleValidationError(
                    f"recorded checkout is missing runtime target: {name}"
                )
            if name not in origins:
                if is_namespace:
                    valid = not (expected / "__init__.py").exists()
                    next_search = (str(expected.resolve(strict=True)),)
                else:
                    spec = PathFinder.find_spec(name, search)
                    valid = (
                        spec is not None
                        and spec.origin is not None
                        and Path(spec.origin).resolve(strict=True) == expected.resolve()
                    )
                    next_search = (
                        spec.submodule_search_locations if spec is not None else None
                    )
                if not valid:
                    raise BundleValidationError(
                        f"resolved runtime origin is outside recorded checkout: {name}"
                    )
                origins[name] = expected_origins[name]
                searches[name] = next_search
            search = searches[name]
    engine = sys.modules.get("sglang.srt.entrypoints.engine")
    if engine is not None:
        attest_engine_targets(engine.Engine, checkout)
    return origins


def attest_engine_targets(engine_class, checkout=None):
    """Verify the actual factory/TP spawn callables before Engine construction."""
    checkout = Path(REPO_ROOT if checkout is None else checkout).resolve(strict=True)
    scheduler = engine_class.run_scheduler_process_func
    factory = engine_class.init_tokenizer_manager_func
    tokenizer = factory.__globals__.get("TokenizerManager")
    targets = {
        "Engine": (engine_class, "python/sglang/srt/entrypoints/engine.py"),
        "scheduler": (scheduler, "python/sglang/srt/managers/scheduler.py"),
        "tokenizer_factory": (factory, "python/sglang/srt/entrypoints/engine.py"),
        "TokenizerManager": (
            tokenizer,
            "python/sglang/srt/managers/tokenizer_manager.py",
        ),
    }
    for label, (target, relative) in targets.items():
        try:
            origin = Path(inspect.getfile(target)).resolve()
        except (TypeError, OSError) as error:
            raise BundleValidationError(
                f"runtime target has no verified origin: {label}"
            ) from error
        if origin != (checkout / relative).resolve():
            raise BundleValidationError(
                f"runtime target origin is outside recorded checkout: {label}"
            )
    if (
        scheduler.__module__ != "sglang.srt.managers.scheduler"
        or scheduler.__name__ != "run_scheduler_process"
        or getattr(sys.modules.get(scheduler.__module__), scheduler.__name__, None)
        is not scheduler
    ):
        raise BundleValidationError(
            "scheduler spawn target differs from verified runtime module"
        )
    return {label: relative for label, (_, relative) in targets.items()}


_MEMORY_BOUNDARY_NODES = {
    "python/sglang/srt/managers/io_struct.py": (
        "BaseReq",
        "CudaMemoryPeakRankResult",
        "ResetCudaMemoryPeakReqInput",
        "ResetCudaMemoryPeakReqOutput",
        "ReadCudaMemoryPeakReqInput",
        "ReadCudaMemoryPeakReqOutput",
    ),
    "python/sglang/srt/managers/scheduler.py": (
        "Scheduler.reset_cuda_memory_peak",
        "Scheduler.read_cuda_memory_peak",
        "Scheduler._cuda_memory_peak_control",
        "Scheduler.init_request_dispatcher",
        "Scheduler.process_input_requests",
    ),
    "python/sglang/srt/managers/tokenizer_control_mixin.py": (
        "_COMMUNICATOR_SPECS",
        "TokenizerControlMixin.reset_cuda_memory_peak",
        "TokenizerControlMixin.read_cuda_memory_peak",
        "TokenizerControlMixin._cuda_memory_peak_control",
        "TokenizerControlMixin.init_communicators",
        "TokenizerControlMixin.update_control_communicator_fan_out",
    ),
    "python/sglang/srt/managers/tokenizer_manager.py": (
        "TokenizerManager._dispatch_to_scheduler",
    ),
    "python/sglang/srt/managers/communicator.py": tuple(
        "FanOutCommunicator." + name
        for name in (
            "__init__",
            "queueing_call",
            "watching_call",
            "__call__",
            "set_fan_out",
            "handle_recv",
        )
    ),
    "python/sglang/utils.py": ("TypeBasedDispatcher",),
}


def _definition(tree, qualified_name):
    node = tree
    for name in qualified_name.split("."):
        matches = [
            item
            for item in node.body
            if getattr(item, "name", None) == name
            or (
                isinstance(item, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in item.targets
                )
            )
        ]
        if len(matches) != 1:
            raise BundleValidationError(
                f"missing or ambiguous runtime boundary: {qualified_name}"
            )
        node = matches[0]
    return copy.deepcopy(node)


class _ObservationAST(ast.NodeTransformer):
    """Project shared registration tables; ignore formatting and docstrings."""

    def __init__(self, project_registrations=False, project_health=False):
        self.project_registrations = project_registrations
        self.project_health = project_health

    def visit_If(self, node):
        # Only discard the branch guarded by the exact health-request predicate;
        # changes to general dispatch, the predicate, or return transport remain.
        guard = node.test
        if (
            self.project_health
            and not node.orelse
            and isinstance(guard, ast.BoolOp)
            and isinstance(guard.op, ast.And)
            and isinstance(guard.values[0], ast.Call)
            and ast.dump(guard.values[0], include_attributes=False)
            == ast.dump(
                ast.parse("is_health_check_generate_req(recv_req)", mode="eval").body,
                include_attributes=False,
            )
        ):
            return None
        return self.generic_visit(node)

    def visit_List(self, node):
        if (
            self.project_registrations
            and node.elts
            and all(
                isinstance(item, ast.Tuple) and len(item.elts) >= 2
                for item in node.elts
            )
        ):
            node.elts = [
                item
                for item in node.elts
                if (
                    isinstance(item.elts[0], ast.Name)
                    and item.elts[0].id
                    in ("ResetCudaMemoryPeakReqInput", "ReadCudaMemoryPeakReqInput")
                )
                or (
                    isinstance(item.elts[0], ast.Constant)
                    and item.elts[0].value
                    in ("reset_cuda_memory_peak", "read_cuda_memory_peak")
                )
            ]
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def memory_boundary_hash(checkout, revision="HEAD"):
    """Comparable observation-instrument digest, also consumed by preflight.

    Read committed source without importing CUDA-dependent runtime modules.
    Shared dispatch/communicator infrastructure is included, but unrelated
    model methods and non-memory registration entries are excluded.
    """
    contract = {}
    for path, names in _MEMORY_BOUNDARY_NODES.items():
        tree = ast.parse(_command(["git", "show", f"{revision}:{path}"], cwd=checkout))
        nodes = [
            _ObservationAST(
                name
                in (
                    "_COMMUNICATOR_SPECS",
                    "Scheduler.init_request_dispatcher",
                    "TokenizerControlMixin.init_communicators",
                ),
                project_health=name == "Scheduler.process_input_requests",
            ).visit(_definition(tree, name))
            for name in names
        ]
        used = {
            item.id
            for node in nodes
            for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }
        imports = []
        for item in [*tree.body, *(item for node in nodes for item in ast.walk(node))]:
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                aliases = [
                    alias
                    for alias in item.names
                    if (alias.asname or alias.name.split(".")[0]) in used
                ]
                if aliases:
                    imported = copy.deepcopy(item)
                    imported.names = aliases
                    imports.append(ast.dump(imported, include_attributes=False))
        # Imports local to the projected routing method can mention unrelated
        # OFT types. Their bindings are accounted for above only when used.
        for node in nodes:
            for item in ast.walk(node):
                if hasattr(item, "body") and isinstance(item.body, list):
                    item.body = [
                        child
                        for child in item.body
                        if not isinstance(child, (ast.Import, ast.ImportFrom))
                    ]
        contract[path] = {
            "nodes": {
                name: ast.dump(node, include_attributes=False)
                for name, node in zip(names, nodes)
            },
            "imports": sorted(set(imports)),
        }
    return canonical_sha256(contract)


def _command(argv, *, cwd=None):
    return subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    ).stdout


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(root):
    if not root.is_dir() or root.is_symlink():
        raise BundleValidationError(f"immutable directory is absent or linked: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BundleValidationError(f"immutable input is a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _file_hash(path)
    if not result:
        raise BundleValidationError(f"immutable directory is empty: {root}")
    return result


def environment_inventory():
    """Normalize installed versions without checkout paths or editable URLs."""
    import torch

    # setuptools adds its bundled metadata to sys.path on import. Exclude only
    # that private directory; retain custom/PYTHONPATH dependency installations.
    setuptools_spec = PathFinder.find_spec("setuptools")
    vendors = {
        (Path(root) / "_vendor").resolve()
        for root in (getattr(setuptools_spec, "submodule_search_locations", None) or ())
    }
    roots = [root for root in sys.path if Path(root).resolve() not in vendors]
    packages = sorted(
        {
            (
                re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions(path=roots)
        }
    )
    driver = sorted(
        set(
            _command(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ).split()
        )
    )
    if len(driver) != 1 or not torch.version.cuda:
        raise BundleValidationError(
            "CUDA/driver inventory is unavailable or heterogeneous"
        )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "driver": driver[0],
    }


def hardware_inventory():
    """Bind visible allocation GPUs and their topology without physical UUIDs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise BundleValidationError("CUDA_VISIBLE_DEVICES must identify the allocation")
    devices, indices, uuids = [], [], set()
    for token in visible.split(","):
        token = token.strip()
        if not (token.isdecimal() or re.fullmatch(r"GPU-[a-zA-Z0-9-]+", token)):
            raise BundleValidationError("invalid allocation GPU selector")
        raw = _command(
            [
                "nvidia-smi",
                "-i",
                token,
                "--query-gpu=index,uuid,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise BundleValidationError("malformed GPU inventory")
        index, uuid, name, memory, capability = (value.strip() for value in rows[0])
        index = int(index)
        if (
            index in indices
            or uuid in uuids
            or (token.isdecimal() and index != int(token))
            or (token.startswith("GPU-") and not uuid.startswith(token))
        ):
            raise BundleValidationError("invalid or duplicate allocation GPU")
        devices.append(
            {
                "index": len(indices),
                "name": name,
                "memory_mib": int(memory),
                "compute_capability": capability,
            }
        )
        indices.append(index)
        uuids.add(uuid)
    topology = re.sub(
        r"\x1b\[[0-?]*[ -/]*[@-~]", "", _command(["nvidia-smi", "topo", "-m"])
    )
    rows = [line.split() for line in topology.splitlines() if line.split()]
    header = next((row for row in rows if row[0].startswith("GPU")), [])
    labels = [f"GPU{index}" for index in indices]
    links = []
    for label in labels:
        matches = [row for row in rows if row[0] == label and row is not header]
        if len(matches) != 1 or any(other not in header for other in labels):
            raise BundleValidationError("allocation GPU topology is incomplete")
        links.append([matches[0][header.index(other) + 1] for other in labels])
    # Relabel physical indices to allocation-local indices for node comparability.
    return {
        "gpus": devices,
        "visible_devices": list(range(len(indices))),
        "topology": links,
    }


def capture_run_identity(spec, fixtures, argv):
    """Recompute identity from exact local bytes and the current clean checkout."""
    from .preflight import _load_json, hash_checkpoint, load_prompt_manifest
    from .scenarios import lifecycle_steps

    sha = _command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).strip()
    if sha != spec.revision_sha:
        raise BundleValidationError("checkout HEAD does not match revision SHA")
    if _command(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT
    ):
        raise BundleValidationError("qualification checkout is dirty")
    code = _command(["git", "ls-tree", "-r", "--full-tree", "HEAD"], cwd=REPO_ROOT)
    runtime_origins = bind_runtime()
    document = _load_json(spec.checkpoint_manifest, "checkpoint manifest")
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or set(document)
        != {
            "schema_version",
            "matrix_sha256",
            "prompts_sha256",
            "prompt_tokenizer",
            "checkpoints",
        }
    ):
        raise BundleValidationError("invalid checkpoint manifest")
    _validate_sha256(document["matrix_sha256"], "checkpoint manifest matrix hash")
    _validate_sha256(document["prompts_sha256"], "checkpoint manifest prompt hash")
    entries = document.get("checkpoints")
    if not isinstance(entries, list):
        raise BundleValidationError("checkpoint manifest has no entries")
    model_path = Path(spec.server.model_path).resolve(strict=True)
    selected = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == str(model_path)
    ]
    if len(selected) != 1:
        raise BundleValidationError(
            "checkpoint manifest must identify one exact model path"
        )
    entry = selected[0]
    verified = hash_checkpoint(
        model_path, model=entry["model"], revision=entry["revision"]
    )
    exact = dict(asdict(verified), id=entry["id"], path=str(verified.path))
    if entry != exact:
        raise BundleValidationError(
            "checkpoint entry does not match exact checkpoint bytes"
        )
    prompts = load_prompt_manifest(spec.prompts_file)
    if document["prompt_tokenizer"] != {
        "model": prompts.tokenizer_model,
        "revision": prompts.tokenizer_revision,
        "files": prompts.tokenizer_files,
    }:
        raise BundleValidationError(
            "checkpoint manifest prompt tokenizer identity differs"
        )
    prompt_hash = _file_hash(spec.prompts_file)
    if document.get("prompts_sha256") != prompt_hash:
        raise BundleValidationError("prompt bytes do not match checkpoint manifest")
    if prompts.tokenizer_files != {
        name: verified.files[name] for name in prompts.tokenizer_files
    }:
        raise BundleValidationError("prompt tokenizer differs from selected checkpoint")
    supplied = _load_json(spec.fixture_manifest, "fixture manifest")
    if not isinstance(supplied, dict) or supplied != {
        key: str(value) for key, value in fixtures.items()
    }:
        raise BundleValidationError(
            "fixture manifest differs from executed fixture paths"
        )
    adapters = {
        key: _file_inventory(Path(value)) for key, value in sorted(fixtures.items())
    }
    for key, files in adapters.items():
        if not {"adapter_config.json", "adapter_model.safetensors"} <= files.keys():
            raise BundleValidationError(f"adapter fixture is incomplete: {key}")
    environment, hardware = environment_inventory(), hardware_inventory()
    if spec.server.mode != "base" and spec.server.base_gpu_id == 0:
        raise BundleValidationError("native sender GPU 0 overlaps the model GPU range")
    gpus = hardware.get("gpus", [])
    if (
        not gpus
        or not hardware.get("topology")
        or spec.server.base_gpu_id + spec.server.tp_size > len(gpus)
        or any("H200" not in gpu.get("name", "") for gpu in gpus)
        or len(
            {
                (gpu["name"], gpu["memory_mib"], gpu["compute_capability"])
                for gpu in gpus
            }
        )
        != 1
    ):
        raise BundleValidationError(
            "qualification requires homogeneous allocated H200 GPUs and complete topology"
        )
    case = CaseKey(
        entry["model"],
        spec.architecture,
        spec.precision,
        sha,
        spec.server.mode,
        spec.server.cuda_graph,
        "native-adapter-lifecycle-v2",
    )
    case.validate()
    # Capture every auxiliary checkpoint/tokenizer file, not only weight shards.
    checkpoint_files = _file_inventory(model_path)
    tokenizer_files = {
        name: digest
        for name, digest in checkpoint_files.items()
        if Path(name).name.startswith(
            (
                "tokenizer",
                "special_tokens",
                "added_tokens",
                "vocab",
                "merges",
                "chat_template",
            )
        )
        or Path(name).suffix in (".model", ".tiktoken", ".jinja")
    }
    portable_entry = {key: value for key, value in entry.items() if key != "path"}
    hashes = {
        "code_hash": hashlib.sha256(code.encode()).hexdigest(),
        "checkpoint_hash": canonical_sha256(
            {"entry": portable_entry, "files": checkpoint_files}
        ),
        "adapter_hash": canonical_sha256(adapters),
        "tokenizer_hash": canonical_sha256(tokenizer_files),
        "scenario_hash": canonical_sha256(
            {
                "steps": [asdict(step) for step in lifecycle_steps(spec.server.mode)],
                "prompts_sha256": prompt_hash,
            }
        ),
        "environment_hash": canonical_sha256(environment),
        "hardware_hash": canonical_sha256(
            {
                "inventory": hardware,
                "tp": spec.server.tp_size,
                "ep": spec.server.ep_size,
                "base_gpu_id": spec.server.base_gpu_id,
            }
        ),
    }
    metadata = {
        "runtime_origins": runtime_origins,
        "memory_boundary_hash": memory_boundary_hash(REPO_ROOT, sha),
        "argv": list(argv),
        "role": spec.server.revision_kind,
        "repetition": spec.repetition,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "environment": environment,
        "hardware": hardware,
        "checkpoint": entry,
        "fixture_files": adapters,
        "checkpoint_manifest_hash": _file_hash(spec.checkpoint_manifest),
        "fixture_manifest_hash": _file_hash(spec.fixture_manifest),
    }
    return case, build_provenance(case, hashes, metadata)


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise BundleValidationError(f"{key} must be an object")
    return selected


def _required_list(value: Mapping[str, object], key: str) -> list[object]:
    selected = value.get(key)
    if type(selected) is not list:
        raise BundleValidationError(f"{key} must be an array")
    return selected


def _required_string(value: Mapping[str, object], key: str) -> str:
    selected = value.get(key)
    if type(selected) is not str:
        raise BundleValidationError(f"{key} must be a string")
    return selected


def _score_entry(value: object, context: str) -> tuple[float, int]:
    if type(value) not in (list, tuple) or len(value) < 2:
        raise BundleValidationError(
            f"{context} must contain at least a score and token ID"
        )
    score, token_id = value[0], value[1]
    if type(score) is not float or not math.isfinite(score):
        raise BundleValidationError(f"{context} score must be a finite float")
    if type(token_id) is not int:
        raise BundleValidationError(f"{context} token ID must be an integer")
    return score, token_id


def capture_generation(
    response: Mapping[str, object],
    adapter_state: Mapping[str, object],
    *,
    top_k: int,
    error: Mapping[str, object] | None = None,
) -> Observation:
    """Convert one complete engine response into an immutable observation."""

    if not isinstance(response, Mapping):
        raise BundleValidationError("response must be an object")
    if type(top_k) is not int or top_k <= 0:
        raise BundleValidationError("top_k must be a positive integer")

    raw_output_ids = _required_list(response, "output_ids")
    if not raw_output_ids:
        raise BundleValidationError("output_ids must not be empty")
    if any(type(token_id) is not int for token_id in raw_output_ids):
        raise BundleValidationError("output_ids entries must be integers")
    output_ids = tuple(raw_output_ids)

    meta = _required_mapping(response, "meta_info")
    triples = _required_list(meta, "output_token_logprobs")
    if len(triples) != len(output_ids):
        raise BundleValidationError("output token/logprob lengths differ")
    token_logprobs: list[float] = []
    for position, (entry, expected_token_id) in enumerate(zip(triples, output_ids)):
        score, token_id = _score_entry(entry, f"output_token_logprobs[{position}]")
        if token_id != expected_token_id:
            raise BundleValidationError(
                f"output_token_logprobs[{position}] token ID does not match "
                "output_ids"
            )
        token_logprobs.append(score)

    top_rows = _required_list(meta, "output_top_logprobs")
    if len(top_rows) != len(output_ids):
        raise BundleValidationError("selected score rows do not match output IDs")
    selected: dict[str, tuple[float, ...]] = {}
    selected_token_ids: dict[str, tuple[int, ...]] = {}
    for position, raw_row in enumerate(top_rows):
        if type(raw_row) is not list:
            raise BundleValidationError(
                f"output_top_logprobs[{position}] must be an array"
            )
        if len(raw_row) != top_k:
            raise BundleValidationError("selected score width is incomplete")
        entries = [
            _score_entry(entry, f"output_top_logprobs[{position}][{index}]")
            for index, entry in enumerate(raw_row)
        ]
        token_ids = [token_id for _, token_id in entries]
        if len(token_ids) != len(set(token_ids)):
            raise BundleValidationError(
                f"output_top_logprobs[{position}] has duplicate token IDs"
            )
        ordered = sorted(entries, key=lambda entry: entry[1])
        name = f"decode.{position:03d}.top_logprobs"
        selected[name] = tuple(score for score, _ in ordered)
        selected_token_ids[name] = tuple(token_id for _, token_id in ordered)

    validate_adapter_state(adapter_state)
    observation = Observation(
        output_ids=output_ids,
        request_output_lengths=(len(output_ids),),
        request_texts=(_required_string(response, "text"),),
        text=_required_string(response, "text"),
        token_logprobs=tuple(token_logprobs),
        selected_logits=selected,
        selected_token_ids=selected_token_ids,
        adapter_state=dict(adapter_state),
        error=None if error is None else dict(error),
    )
    observation.validate()
    return observation


def _normalized_message(message: object) -> str:
    if type(message) is not str:
        raise BundleValidationError("product rejection message must be a string")
    normalized = " ".join(message.split())
    if not normalized:
        raise BundleValidationError("product rejection message must be non-empty")
    return normalized


def _required_error_code(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise BundleValidationError(f"{context} must be a non-empty string")
    return value


def normalize_expected_error(
    result: ControlResult,
    *,
    expected_code: str,
    returned_code: str,
) -> dict[str, object]:
    """Normalize a declared product rejection only when its code is exact."""

    if not isinstance(result, ControlResult):
        raise BundleValidationError("control result must be a ControlResult")
    expected = _required_error_code(expected_code, "expected error code")
    returned = _required_error_code(returned_code, "returned error code")
    if result.success:
        raise BundleValidationError(
            f"operation succeeded but expected rejection {expected!r}"
        )
    if returned != expected:
        raise BundleValidationError(
            f"returned error code {returned!r} does not match expected {expected!r}"
        )
    return {
        "kind": "product_rejection",
        "code": expected,
        "message": _normalized_message(result.message),
    }


def _validate_control_versions(
    result: ControlResult, adapter_state: Mapping[str, object]
) -> None:
    for result_field, state_field in (
        ("active_version", "active"),
        ("staged_version", "staged"),
    ):
        version = getattr(result, result_field)
        if version is None:
            continue
        if type(version) is not str or not version:
            raise BundleValidationError(
                f"control result {result_field} must be a non-empty string"
            )
        identity = adapter_state[state_field]
        if not isinstance(identity, Mapping) or identity["version"] != version:
            raise BundleValidationError(
                f"control result {result_field} does not match adapter state"
            )


def capture_control(
    result: ControlResult,
    adapter_state: Mapping[str, object],
    *,
    expected_error_code: str | None = None,
    returned_error_code: str | None = None,
) -> Observation:
    """Capture a control transition without treating rejection as success."""

    if not isinstance(result, ControlResult):
        raise BundleValidationError("control result must be a ControlResult")
    validate_adapter_state(adapter_state)
    _validate_control_versions(result, adapter_state)
    if result.success:
        if expected_error_code is not None:
            raise BundleValidationError(
                "control operation succeeded despite a declared expected rejection"
            )
        if returned_error_code is not None:
            raise BundleValidationError(
                "successful control operation supplied a returned error code"
            )
        normalized_error = None
    else:
        if expected_error_code is None:
            raise BundleValidationError(
                "control operation returned an undeclared product rejection"
            )
        if returned_error_code is None:
            raise BundleValidationError(
                "control operation omitted its returned error code"
            )
        normalized_error = normalize_expected_error(
            result,
            expected_code=expected_error_code,
            returned_code=returned_error_code,
        )

    observation = Observation(
        output_ids=(),
        request_output_lengths=(),
        request_texts=(),
        text="",
        token_logprobs=(),
        selected_logits={},
        selected_token_ids={},
        adapter_state=dict(adapter_state),
        error=normalized_error,
    )
    observation.validate()
    return observation


def _validate_finite_float(
    value: object, context: str, *, positive: bool = False
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise BundleValidationError(f"{context} must be a finite float")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise BundleValidationError(f"{context} must be {qualifier}")
    return value


def _validate_non_negative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise BundleValidationError(f"{context} must be a non-negative integer")
    return value


class PerformanceRecorder:
    """Collect the fixed three post-warm-up qualification repetitions."""

    def __init__(self, procedure_hash: str) -> None:
        _validate_sha256(procedure_hash, "performance procedure hash")
        self.procedure_hash = procedure_hash
        self._startup_seconds: list[float] = []
        self._latency_seconds: list[float] = []
        self._throughput: list[float] = []
        self._peak_allocated: list[int] = []
        self._peak_reserved: list[int] = []

    @staticmethod
    def _ensure_room(values: list[object], context: str) -> None:
        if len(values) >= PERFORMANCE_REPETITIONS:
            raise BundleValidationError(
                f"{context} already contains exactly "
                f"{PERFORMANCE_REPETITIONS} repetitions"
            )

    def record_startup(self, seconds: float) -> None:
        self._ensure_room(self._startup_seconds, "startup_seconds")
        self._startup_seconds.append(_validate_finite_float(seconds, "startup_seconds"))

    def record_sample(
        self,
        *,
        latency_seconds: float,
        output_tokens: int,
        peak_allocated_bytes: int,
        peak_reserved_bytes: int,
    ) -> None:
        self._ensure_room(self._latency_seconds, "performance samples")
        latency = _validate_finite_float(
            latency_seconds, "latency_seconds", positive=True
        )
        tokens = _validate_non_negative_integer(output_tokens, "output_tokens")
        if tokens == 0:
            raise BundleValidationError("output_tokens must be positive")
        allocated = _validate_non_negative_integer(
            peak_allocated_bytes, "peak_allocated_bytes"
        )
        reserved = _validate_non_negative_integer(
            peak_reserved_bytes, "peak_reserved_bytes"
        )
        if reserved < allocated:
            raise BundleValidationError(
                "peak_reserved_bytes must be at least peak_allocated_bytes"
            )
        self._latency_seconds.append(latency)
        self._throughput.append(tokens / latency)
        self._peak_allocated.append(allocated)
        self._peak_reserved.append(reserved)

    def build(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics(
            procedure_hash=self.procedure_hash,
            startup_seconds=tuple(self._startup_seconds),
            latency_seconds=tuple(self._latency_seconds),
            throughput_tokens_per_second=tuple(self._throughput),
            peak_allocated_bytes=tuple(self._peak_allocated),
            peak_reserved_bytes=tuple(self._peak_reserved),
        )
        metrics.validate()
        return metrics


def _validate_sha256(value: object, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise BundleValidationError(f"{context} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise BundleValidationError(f"{context} must be a lowercase SHA-256")
    return value


def build_provenance(
    case_key: CaseKey,
    hashes: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    dirty: bool = False,
) -> dict[str, object]:
    """Build the immutable portion of bundle provenance from verified inputs."""

    if not isinstance(case_key, CaseKey):
        raise BundleValidationError("case_key must be a CaseKey")
    case_key.validate()
    if len(case_key.revision) != 40 or any(
        character not in "0123456789abcdef" for character in case_key.revision
    ):
        raise BundleValidationError(
            "case_key.revision must be a lowercase 40-character Git SHA"
        )
    if type(dirty) is not bool or dirty:
        raise BundleValidationError("provenance dirty must be false")
    if not isinstance(hashes, Mapping):
        raise BundleValidationError("provenance hashes must be an object")
    missing = sorted(set(PROVENANCE_HASH_KEYS) - set(hashes))
    unknown = sorted(set(hashes) - set(PROVENANCE_HASH_KEYS))
    if missing:
        raise BundleValidationError(
            "provenance hashes missing fields: " + ", ".join(missing)
        )
    if unknown:
        raise BundleValidationError(
            "provenance hashes contain unknown fields: " + ", ".join(unknown)
        )
    if not isinstance(metadata, Mapping):
        raise BundleValidationError("provenance metadata must be an object")
    canonical_sha256(metadata)
    validated_hashes = {
        key: _validate_sha256(hashes[key], f"provenance.{key}")
        for key in PROVENANCE_HASH_KEYS
    }
    return {
        "git_sha": case_key.revision,
        "dirty": False,
        **validated_hashes,
        "metadata": dict(metadata),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("exclusive evidence write made no progress")
        remaining = remaining[written:]


def _serialize_bundle(descriptor: int, bundle: RunBundle) -> None:
    _write_all(descriptor, _json_bytes(bundle.to_dict()))


def _best_effort_close(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        _best_effort_close(descriptor)


def _assert_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(path)


def _read_matching_bundle(path: Path, expected_digest: str) -> RunBundle:
    bundle = RunBundle.read_json(path)
    if bundle.digest() != expected_digest:
        raise BundleValidationError(f"serialized bundle digest mismatch at {path}")
    return bundle


def _read_matching_marker(path: Path, expected_marker: object) -> None:
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"invalid completion marker at {path}") from error
    if marker != expected_marker:
        raise BundleValidationError(f"completion marker mismatch at {path}")


def publish_bundle(
    bundle: RunBundle,
    bundle_path: str | os.PathLike[str],
    completion_path: str | os.PathLike[str],
) -> None:
    """Publish a validated bundle exclusively, then write its completion marker."""

    if not isinstance(bundle, RunBundle):
        raise BundleValidationError("bundle must be a RunBundle")
    bundle.validate()
    expected_digest = bundle.digest()
    destination = Path(bundle_path)
    completion = Path(completion_path)
    if destination == completion:
        raise BundleValidationError("bundle and completion paths must be distinct")
    destination_parent = destination.parent.resolve(strict=True)
    completion_parent = completion.parent.resolve(strict=True)
    if destination_parent != completion_parent:
        raise BundleValidationError(
            "bundle and completion paths must share one directory"
        )
    _assert_absent(destination)
    _assert_absent(completion)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination_parent
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        _serialize_bundle(descriptor, bundle)
        os.fsync(descriptor)
        descriptor_open = False
        os.close(descriptor)
        _read_matching_bundle(temporary, expected_digest)

        os.link(temporary, destination)
        _fsync_directory(destination_parent)
        temporary.unlink()
        _fsync_directory(destination_parent)
        _read_matching_bundle(destination, expected_digest)

        marker = {"bundle_hash": expected_digest, "status": "complete"}
        marker_descriptor, marker_temporary_name = tempfile.mkstemp(
            prefix=f".{completion.name}.",
            suffix=".tmp",
            dir=destination_parent,
        )
        marker_temporary = Path(marker_temporary_name)
        marker_descriptor_open = True
        try:
            _write_all(marker_descriptor, _json_bytes(marker))
            os.fsync(marker_descriptor)
            marker_descriptor_open = False
            os.close(marker_descriptor)
            _read_matching_marker(marker_temporary, marker)
            _fsync_directory(destination_parent)
            os.link(marker_temporary, completion)
        finally:
            if marker_descriptor_open:
                _best_effort_close(marker_descriptor)
            _best_effort_unlink(marker_temporary)
    finally:
        if descriptor_open:
            _best_effort_close(descriptor)
        _best_effort_unlink(temporary)

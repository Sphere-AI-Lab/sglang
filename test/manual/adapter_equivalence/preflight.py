"""Fail-closed checkpoint and prompt preflight for adapter equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adapter_equivalence"

from .fixtures import FixtureValidationError, validate_matrix
from .schema import canonical_sha256

PINNED_MODEL_REVISIONS = {
    "qwen3-4b-bf16": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "qwen3-4b-fp8": "8591804019c8b22094c3b5b4454e0edc05dffc98",
    "qwen3-4b-nvfp4": "7009563e02c47b3ce728ecdc8cab2f0d9cd52ee4",
    "qwen3-30b-a3b-bf16": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "qwen3-30b-a3b-fp8": "d206ba732169f29bb77fbf80fc2c4b81d4d30782",
    "qwen3-30b-a3b-nvfp4": "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3",
}

_PROMPT_IDS = (
    "factual",
    "arithmetic",
    "code",
    "long-prefix",
    "uneven-mixed",
    "graph-bucket",
)
_BATCH_SIZES = {"batch-1": 1, "batch-2": 2, "batch-8": 8, "batch-32": 32}
# Prompt files may be generated with any active matrix checkpoint's tokenizer.
# Keep the pair pinned; capture additionally checks the selected checkpoint bytes.
_PINNED_TOKENIZER_REVISIONS = {
    "Qwen/Qwen3-4B-Instruct-2507": PINNED_MODEL_REVISIONS["qwen3-4b-bf16"],
    "Qwen/Qwen3-4B-Instruct-2507-FP8": PINNED_MODEL_REVISIONS["qwen3-4b-fp8"],
    "Qwen/Qwen3-30B-A3B": PINNED_MODEL_REVISIONS["qwen3-30b-a3b-bf16"],
    "Qwen/Qwen3-30B-A3B-FP8": PINNED_MODEL_REVISIONS["qwen3-30b-a3b-fp8"],
}
_NUMBERED_SHARD = re.compile(
    r"^model-(?P<number>\d{5})-of-(?P<total>\d{5})\.safetensors$"
)


class PreflightError(ValueError):
    """Raised when immutable checkpoint or prompt evidence is incomplete."""


class PreflightBlocked(PreflightError):
    """Only an absent required allocation/reference can block an active shard."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _run_checked(argv, *, cwd=None):
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError(str(error)) from error
    if result.returncode:
        raise PreflightError(f"preflight subprocess failed: {result.stderr.strip()}")
    return result.stdout.strip()


def check_checkout(checkout, revision, *, role):
    from .qualification import exact_sha

    exact_sha(revision)
    path = Path(checkout)
    if not path.exists():
        if role == "source":
            raise PreflightBlocked(
                "reference_unavailable", f"reference checkout unavailable: {path}"
            )
        raise PreflightError(f"candidate checkout unavailable: {path}")
    if not path.is_absolute() or path != path.resolve():
        raise PreflightError("checkout must be an absolute canonical path")
    if _run_checked(["git", "rev-parse", "--show-toplevel"], cwd=path) != str(path):
        raise PreflightError("checkout is not an exact worktree root")
    if _run_checked(["git", "rev-parse", "HEAD"], cwd=path) != revision:
        raise PreflightError("checkout SHA mismatch")
    if _run_checked(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=path
    ):
        raise PreflightError("qualification checkout is dirty")
    return path


_CAPABILITY_PROBE = r"""
import importlib, inspect, json, pathlib, sys, types
from importlib.machinery import PathFinder
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
mode = sys.argv[2]
sys.path.insert(0, str(root / "python"))
class CheckoutOnlySGLangFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if not fullname.startswith("sglang."):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None:
            raise ModuleNotFoundError("selected checkout has no module " + fullname)
        return spec
sys.meta_path.insert(0, CheckoutOnlySGLangFinder)
import torch
def forbidden(*args, **kwargs):
    raise RuntimeError("native capability preflight must not initialize CUDA")
torch.cuda.init = forbidden
torch.cuda._lazy_init = forbidden
if hasattr(torch._C, "_cuda_init"):
    torch._C._cuda_init = forbidden
# Import-time backend selection asks for the target SM even though this probe
# must not own a CUDA context. Qualification is H200-only, so expose that
# already-validated target while keeping every initialization path forbidden.
torch.cuda.current_device = lambda: 0
properties = types.SimpleNamespace(major=9, minor=0, multi_processor_count=132)
torch.cuda.get_device_properties = lambda *args, **kwargs: properties
torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 0)
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA initialized before runtime probe")
names = ["sglang.srt.entrypoints.engine", "sglang.srt.managers.tokenizer_manager",
         "sglang.srt.managers.io_struct"]
if mode == "native_oft":
    names.append("sglang.srt.oft.io_types")
modules = {name: importlib.import_module(name) for name in names}
engine = modules[names[0]].Engine
manager = modules[names[1]].TokenizerManager
def local_definition(value):
    path = inspect.getsourcefile(value)
    assert path is not None and pathlib.Path(path).resolve(strict=True).is_relative_to(root / "python"), "foreign native definition"
local_definition(engine)
local_definition(manager)
adapter = "lora" if mode == "native_lora" else "oft"
for suffix in ("", "_from_tensors", "_from_distributed"):
    assert callable(getattr(engine, "load_" + adapter + "_adapter" + suffix, None)), "missing native load method"
    local_definition(getattr(engine, "load_" + adapter + "_adapter" + suffix))
assert callable(getattr(engine, "unload_" + adapter + "_adapter", None)), "missing native unload method"
local_definition(getattr(engine, "unload_" + adapter + "_adapter"))
for name in ("update_adapter_from_distributed", "activate_adapter_version"):
    assert callable(getattr(manager, name, None)), "missing native staged method"
    local_definition(getattr(manager, name))
shared = modules["sglang.srt.managers.io_struct"]
for name in ("UpdateAdapterFromDistributedReqInput", "ActivateAdapterVersionReqInput"):
    assert inspect.isclass(getattr(shared, name, None)), "missing native shared request type"
    local_definition(getattr(shared, name))
types = shared if adapter == "lora" else modules["sglang.srt.oft.io_types"]
label = "LoRA" if adapter == "lora" else "OFT"
for name in ("Load" + label + "AdapterReqInput", "Unload" + label + "AdapterReqInput",
             "Load" + label + "AdapterFromTensorsReqInput", "Load" + label + "AdapterFromDistributedReqInput"):
    assert inspect.isclass(getattr(types, name, None)), "missing native request type"
    local_definition(getattr(types, name))
origins = {}
for name, module in tuple(sys.modules.items()):
    if name == "sglang" or name.startswith("sglang."):
        package_paths = tuple(getattr(module, "__path__", ()))
        for package_path in package_paths:
            path = pathlib.Path(package_path).resolve(strict=True)
            assert path.is_relative_to(root / "python"), "foreign package path"
        origin = getattr(module, "__file__", None)
        if origin is None:
            assert package_paths, "runtime module has neither origin nor package path: " + name
            continue
        path = pathlib.Path(origin).resolve(strict=True)
        assert path.is_relative_to(root / "python"), "foreign runtime origin: " + name
        origins[name] = path.relative_to(root).as_posix()
for name, module in modules.items():
    expected = pathlib.Path("python").joinpath(*name.split(".")).with_suffix(".py").as_posix()
    assert origins[name] == expected, "wrong selected runtime module"
assert not torch.cuda.is_initialized(), "CUDA initialized by runtime imports"
print(json.dumps(dict(mode=mode, origins=origins, cuda_initialized=False), sort_keys=True))
"""


def check_native_capability(checkout, mode, *, python_executable=None):
    if mode not in ("native_lora", "native_oft"):
        raise PreflightError("native capability requires a native adapter mode")
    text = _run_checked(
        [
            str(python_executable or sys.executable),
            "-I",
            "-B",
            "-c",
            _CAPABILITY_PROBE,
            str(checkout),
            mode,
        ],
        cwd=checkout,
    )
    try:
        result = json.loads(text, object_pairs_hook=_duplicate_key)
        canonical_sha256(result)
        if (
            result.get("mode") != mode
            or result.get("cuda_initialized") is not False
            or not result.get("origins")
        ):
            raise PreflightError("invalid native capability result")
        return result
    except (ValueError, AttributeError) as error:
        raise PreflightError(f"invalid capability subprocess JSON: {error}") from error


def preflight_shard(
    manifest,
    shard,
    reference_checkout,
    candidate_checkout,
    *,
    hardware=None,
    python_executable=None,
):
    from . import bundle_capture
    from .aggregate import validate_hardware

    manifest.validate()
    if shard not in manifest.shards:
        raise PreflightError("shard does not belong to qualification manifest")
    source = check_checkout(reference_checkout, manifest.reference_sha, role="source")
    candidate = check_checkout(
        candidate_checkout, manifest.candidate_sha, role="candidate"
    )
    boundaries = {
        "source": bundle_capture.memory_boundary_hash(source, manifest.reference_sha),
        "candidate": bundle_capture.memory_boundary_hash(
            candidate, manifest.candidate_sha
        ),
    }
    if boundaries["source"] != boundaries["candidate"]:
        raise PreflightError("source/candidate memory boundary semantics differ")
    _hash(boundaries["source"], "memory boundary")
    capabilities = {
        role: check_native_capability(
            path, shard.case_key.mode, python_executable=python_executable
        )
        for role, path in (("source", source), ("candidate", candidate))
    }
    if hardware is None:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise PreflightBlocked(
                "required_h200_unavailable", "required H200 allocation is absent"
            )
        hardware = bundle_capture.hardware_inventory()
    if isinstance(hardware, Mapping) and hardware.get("gpus") == []:
        raise PreflightBlocked(
            "required_h200_unavailable", "required H200 allocation is absent"
        )
    validate_hardware(hardware, shard.tp_size, shard.ep_size)
    # Imports are isolated; verify neither checkout changed while probing.
    check_checkout(source, manifest.reference_sha, role="source")
    check_checkout(candidate, manifest.candidate_sha, role="candidate")
    return dict(
        qualification_hash=manifest.manifest_hash,
        shard=shard.to_dict(),
        memory_boundary_hash=boundaries["source"],
        capabilities=capabilities,
        hardware=hardware,
    )


def write_blocked_record(manifest, shard, reason, detail):
    from .qualification import publish_document

    manifest.validate()
    if (
        shard not in manifest.shards
        or type(detail) is not str
        or not detail.strip()
        or not (
            reason == "required_h200_unavailable"
            or (reason == "reference_unavailable" and shard.revision_kind == "source")
        )
    ):
        raise PreflightError("invalid explicit blocked record")
    if Path(shard.bundle_path).exists() or Path(shard.completion_path).exists():
        raise PreflightError("blocked record conflicts with execution artifacts")
    record = dict(
        status="blocked",
        reason=reason,
        detail=detail,
        qualification_hash=manifest.manifest_hash,
        shard=shard.to_dict(),
    )
    publish_document(
        shard.blocked_path, dict(record, record_hash=canonical_sha256(record))
    )


@dataclass(frozen=True)
class CheckpointHash:
    model: str
    revision: str
    path: Path
    layout: str
    index_hash: str | None
    files: dict[str, str]
    checkpoint_hash: str
    tokenizer_hash: str


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class PromptRequest:
    id: str
    prompt_id: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class PromptBatch:
    id: str
    temperature: int
    requests: tuple[PromptRequest, ...]


@dataclass(frozen=True)
class PromptManifest:
    tokenizer_model: str
    tokenizer_revision: str
    tokenizer_files: dict[str, str]
    prompts: tuple[Prompt, ...]
    batches: tuple[PromptBatch, ...]


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, context: str) -> object:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_duplicate_key)
            canonical_sha256(value)
            return value
    except PreflightError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read {context}: {error}") from error


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PreflightError(f"{context} must be a JSON object with string keys")
    return value


def _fields(value: Mapping[str, object], required: set[str], context: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise PreflightError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise PreflightError(f"{context} unknown fields: {', '.join(unknown)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _hash(value: object, context: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        type(value) is not str
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        allowed = " or ".join(str(length) for length in lengths)
        raise PreflightError(
            f"{context} must be a lowercase hexadecimal hash of length {allowed}"
        )
    return value


def hash_checkpoint(
    path: str | os.PathLike[str], *, model: str, revision: str
) -> CheckpointHash:
    """Hash one complete absolute checkpoint snapshot without downloading it."""

    if type(model) is not str or not model:
        raise PreflightError("model must be a non-empty string")
    concrete_revision = _hash(revision, "revision", (40, 64))
    supplied = Path(path)
    if not supplied.is_absolute():
        raise PreflightError("checkpoint path must be absolute")
    try:
        snapshot = supplied.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"checkpoint path does not resolve: {error}") from error
    if not snapshot.is_dir():
        raise PreflightError("checkpoint path must resolve to a directory")

    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not _is_regular_file(config_path):
        raise PreflightError("missing or non-regular checkpoint file: config.json")
    _object(_load_json(config_path, "config.json"), "config.json")
    weight_files = {
        candidate.name: candidate
        for candidate in snapshot.iterdir()
        if candidate.name.endswith(".safetensors")
    }
    non_regular = sorted(
        name
        for name, candidate in weight_files.items()
        if not _is_regular_file(candidate)
    )
    if non_regular:
        raise PreflightError(
            f"checkpoint weight files must be regular: {', '.join(non_regular)}"
        )

    if index_path.exists() or index_path.is_symlink():
        if not _is_regular_file(index_path):
            raise PreflightError("model.safetensors.index.json must be a regular file")
        layout = "indexed"
        index = _object(
            _load_json(index_path, "model.safetensors.index.json"),
            "model.safetensors.index.json",
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise PreflightError("safetensors index must have a non-empty weight_map")
        shard_names: set[str] = set()
        for tensor_name, shard_value in weight_map.items():
            if type(tensor_name) is not str or not tensor_name:
                raise PreflightError("safetensors index has an invalid tensor name")
            if type(shard_value) is not str or not shard_value:
                raise PreflightError(
                    f"safetensors index has an invalid shard for {tensor_name}"
                )
            shard_name = shard_value
            shard = Path(shard_name)
            if (
                shard.is_absolute()
                or shard.name != shard_name
                or shard.suffix != ".safetensors"
            ):
                raise PreflightError(f"unsafe shard path: {shard_name}")
            shard_path = snapshot / shard_name
            if not shard_path.exists() and not shard_path.is_symlink():
                raise PreflightError(f"missing shard: {shard_name}")
            if not _is_regular_file(shard_path):
                raise PreflightError(f"non-regular shard: {shard_name}")
            shard_names.add(shard_name)

        actual_shards = set(weight_files)
        unindexed = sorted(actual_shards - shard_names)
        if unindexed:
            raise PreflightError(f"unindexed checkpoint shards: {', '.join(unindexed)}")
        missing_from_directory = sorted(shard_names - actual_shards)
        if missing_from_directory:
            raise PreflightError(f"missing shard: {', '.join(missing_from_directory)}")
        numbered_shards = [
            _NUMBERED_SHARD.fullmatch(shard_name) for shard_name in shard_names
        ]
        if not numbered_shards or not all(numbered_shards):
            raise PreflightError(
                "indexed checkpoint must contain contiguous numbered shards"
            )
        sequence = [
            (int(match.group("number")), int(match.group("total")))
            for match in numbered_shards
            if match is not None
        ]
        totals = {total for _, total in sequence}
        if len(totals) != 1:
            raise PreflightError("checkpoint shards disagree on total shard count")
        total = totals.pop()
        if (
            total <= 0
            or len(sequence) != total
            or {number for number, _ in sequence} != set(range(1, total + 1))
        ):
            raise PreflightError("incomplete shard sequence in safetensors index")
        checkpoint_names = {
            "config.json",
            "model.safetensors.index.json",
            *shard_names,
        }
        index_hash: str | None = _sha256_file(index_path)
    else:
        layout = "unsharded"
        if set(weight_files) != {"model.safetensors"}:
            raise PreflightError(
                "unsharded checkpoint must contain exactly one regular "
                "model.safetensors and no index"
            )
        checkpoint_names = {"config.json", "model.safetensors"}
        index_hash = None

    tokenizer_names = {"tokenizer.json", "tokenizer_config.json"}
    for tokenizer_name in tokenizer_names:
        if not _is_regular_file(snapshot / tokenizer_name):
            raise PreflightError(
                f"missing or non-regular tokenizer file: {tokenizer_name}"
            )
    names = sorted(checkpoint_names | tokenizer_names)
    files = {name: _sha256_file(snapshot / name) for name in names}
    checkpoint_hash = canonical_sha256(
        {name: files[name] for name in sorted(checkpoint_names)}
    )
    tokenizer_hash = canonical_sha256(
        {name: files[name] for name in sorted(tokenizer_names)}
    )
    return CheckpointHash(
        model=model,
        revision=concrete_revision,
        path=snapshot,
        layout=layout,
        index_hash=index_hash,
        files=files,
        checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    )


def _json_lines(path: Path) -> list[Mapping[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"cannot read prompt manifest: {error}") from error
    if not lines or any(not line for line in lines):
        raise PreflightError("prompt manifest must contain non-empty JSONL records")
    records: list[Mapping[str, object]] = []
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line, object_pairs_hook=_duplicate_key)
        except PreflightError:
            raise
        except json.JSONDecodeError as error:
            raise PreflightError(
                f"cannot parse prompt manifest line {number}: {error}"
            ) from error
        records.append(_object(value, f"prompt manifest line {number}"))
        canonical_sha256(value)
    return records


def load_prompt_manifest(path: str | os.PathLike[str]) -> PromptManifest:
    """Load the exact pinned-tokenizer prompts and deterministic batch schedule."""

    records = _json_lines(Path(path))
    if len(records) != 1 + len(_PROMPT_IDS) + len(_BATCH_SIZES):
        raise PreflightError(
            "prompt manifest must contain one metadata, six prompts, and four batches"
        )
    metadata = records[0]
    _fields(metadata, {"kind", "schema_version", "tokenizer"}, "prompt metadata")
    if metadata["kind"] != "metadata" or metadata["schema_version"] != 1:
        raise PreflightError("prompt metadata kind/schema_version does not match")
    tokenizer = _object(metadata["tokenizer"], "prompt tokenizer")
    _fields(
        tokenizer,
        {"model", "revision", "add_special_tokens", "files"},
        "prompt tokenizer",
    )
    model = tokenizer["model"]
    if type(model) is not str or model not in _PINNED_TOKENIZER_REVISIONS:
        raise PreflightError("prompt tokenizer model is not a pinned matrix model")
    if tokenizer["revision"] != _PINNED_TOKENIZER_REVISIONS[model]:
        raise PreflightError("prompt tokenizer revision is not pinned")
    if tokenizer["add_special_tokens"] is not False:
        raise PreflightError("prompt tokenizer must record add_special_tokens=false")
    tokenizer_file_values = _object(tokenizer["files"], "prompt tokenizer files")
    if set(tokenizer_file_values) != {"tokenizer.json", "tokenizer_config.json"}:
        raise PreflightError("prompt tokenizer files are incomplete")
    tokenizer_files = {
        name: _hash(value, f"prompt tokenizer {name}")
        for name, value in sorted(tokenizer_file_values.items())
    }

    prompts: list[Prompt] = []
    for expected_id, record in zip(_PROMPT_IDS, records[1 : 1 + len(_PROMPT_IDS)]):
        _fields(record, {"kind", "id", "text", "input_ids"}, f"prompt {expected_id}")
        if record["kind"] != "prompt" or record["id"] != expected_id:
            raise PreflightError(f"prompt order/id must include {expected_id}")
        text = record["text"]
        input_ids = record["input_ids"]
        if type(text) is not str or not text:
            raise PreflightError(f"prompt {expected_id} text must be non-empty")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(type(token_id) is not int or token_id < 0 for token_id in input_ids)
        ):
            raise PreflightError(f"prompt {expected_id} input_ids are invalid")
        prompts.append(Prompt(expected_id, text, tuple(input_ids)))
    prompt_by_id = {prompt.id: prompt for prompt in prompts}

    batches: list[PromptBatch] = []
    request_ids: set[str] = set()
    batch_records = records[1 + len(_PROMPT_IDS) :]
    for (expected_id, expected_size), record in zip(
        _BATCH_SIZES.items(), batch_records
    ):
        _fields(record, {"kind", "id", "temperature", "requests"}, expected_id)
        if record["kind"] != "batch" or record["id"] != expected_id:
            raise PreflightError(f"batch order/id must include {expected_id}")
        if type(record["temperature"]) is not int or record["temperature"] != 0:
            raise PreflightError(
                f"batch {expected_id} temperature must be integer zero"
            )
        raw_requests = record["requests"]
        if not isinstance(raw_requests, list) or len(raw_requests) != expected_size:
            raise PreflightError(
                f"batch {expected_id} must contain {expected_size} requests"
            )
        requests: list[PromptRequest] = []
        for index, raw_request in enumerate(raw_requests):
            request = _object(raw_request, f"batch {expected_id} request {index}")
            _fields(
                request,
                {"id", "prompt_id"},
                f"batch {expected_id} request {index}",
            )
            expected_request_id = f"{expected_id}-request-{index:02d}"
            if request["id"] != expected_request_id:
                raise PreflightError(
                    f"batch {expected_id} request IDs must be deterministic"
                )
            prompt_id = request["prompt_id"]
            if type(prompt_id) is not str or prompt_id not in prompt_by_id:
                raise PreflightError(
                    f"batch {expected_id} request has unknown prompt_id"
                )
            if expected_request_id in request_ids:
                raise PreflightError(f"duplicate request ID: {expected_request_id}")
            request_ids.add(expected_request_id)
            requests.append(
                PromptRequest(
                    id=expected_request_id,
                    prompt_id=prompt_id,
                    input_ids=prompt_by_id[prompt_id].input_ids,
                )
            )
        batches.append(PromptBatch(expected_id, 0, tuple(requests)))
    return PromptManifest(
        tokenizer_model=model,
        tokenizer_revision=tokenizer["revision"],
        tokenizer_files=tokenizer_files,
        prompts=tuple(prompts),
        batches=tuple(batches),
    )


def build_preflight_manifest(
    matrix_path: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    snapshot_paths: Mapping[str, str | os.PathLike[str]],
    *,
    local_revisions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Resolve and hash every matrix cell, rejecting mutable or partial inputs."""

    try:
        cells = validate_matrix(matrix_path)
    except FixtureValidationError as error:
        raise PreflightError(str(error)) from error
    if not isinstance(snapshot_paths, Mapping) or any(
        type(cell_id) is not str or not cell_id for cell_id in snapshot_paths
    ):
        raise PreflightError("snapshot paths must be a mapping with cell ID keys")
    expected_ids = {cell.id for cell in cells}
    if set(snapshot_paths) != expected_ids:
        missing = sorted(expected_ids - set(snapshot_paths))
        unknown = sorted(set(snapshot_paths) - expected_ids)
        raise PreflightError(
            f"snapshot path keys mismatch; missing={missing}, unknown={unknown}"
        )
    if local_revisions is None:
        local_revisions = {}
    elif not isinstance(local_revisions, Mapping) or any(
        type(cell_id) is not str or not cell_id for cell_id in local_revisions
    ):
        raise PreflightError("local revisions must be a mapping with cell ID keys")
    unknown_local_ids = set(local_revisions) - expected_ids
    if unknown_local_ids:
        raise PreflightError(
            f"unknown local revision cell IDs: {sorted(unknown_local_ids)}"
        )
    checkpoints = []
    for cell in cells:
        revision = PINNED_MODEL_REVISIONS.get(cell.id)
        if revision is None:
            if cell.id not in local_revisions:
                raise PreflightError(
                    f"generated checkpoint {cell.id} needs an immutable local "
                    "revision hash"
                )
            revision = _hash(
                local_revisions[cell.id], f"{cell.id} local revision", (64,)
            )
        elif cell.id in local_revisions:
            raise PreflightError(
                f"upstream cell {cell.id} cannot override its pinned revision"
            )
        snapshot_value = snapshot_paths[cell.id]
        if not isinstance(snapshot_value, (str, os.PathLike)):
            raise PreflightError(f"snapshot path for {cell.id} must be path-like")
        snapshot = Path(snapshot_value)
        if not snapshot.is_absolute():
            raise PreflightError(f"snapshot path for {cell.id} must be absolute")
        if cell.id in PINNED_MODEL_REVISIONS and snapshot.resolve().name != revision:
            raise PreflightError(
                f"snapshot path for {cell.id} does not end in pinned revision "
                f"{revision}"
            )
        checkpoint = hash_checkpoint(snapshot, model=cell.model, revision=revision)
        checkpoints.append(
            {
                "id": cell.id,
                "model": checkpoint.model,
                "revision": checkpoint.revision,
                "path": str(checkpoint.path),
                "layout": checkpoint.layout,
                "index_hash": checkpoint.index_hash,
                "files": checkpoint.files,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "tokenizer_hash": checkpoint.tokenizer_hash,
            }
        )
    prompt_manifest = load_prompt_manifest(prompts_path)
    return {
        "schema_version": 1,
        "matrix_sha256": _sha256_file(Path(matrix_path)),
        "prompts_sha256": _sha256_file(Path(prompts_path)),
        "prompt_tokenizer": {
            "model": prompt_manifest.tokenizer_model,
            "revision": prompt_manifest.tokenizer_revision,
            "files": prompt_manifest.tokenizer_files,
        },
        "checkpoints": checkpoints,
    }


def _assignments(values: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PreflightError(f"{option} value must be CELL=VALUE")
        key, assigned = value.split("=", 1)
        if not key or not assigned or key in result:
            raise PreflightError(f"invalid or duplicate {option} assignment: {value}")
        result[key] = assigned
    return result


def _write_exclusive(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=root / "matrix.json")
    parser.add_argument("--prompts", type=Path, default=root / "prompts.jsonl")
    parser.add_argument("--snapshot", action="append", default=[], metavar="CELL=PATH")
    parser.add_argument(
        "--local-revision", action="append", default=[], metavar="CELL=SHA256"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--revision-kind", choices=("source", "candidate"))
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--reference-checkout", type=Path)
    parser.add_argument("--candidate-checkout", type=Path)
    parser.add_argument("--runtime-python", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.qualification_manifest is not None:
        from .qualification import QualificationManifest, publish_document

        shard = manifest = None
        try:
            if (
                arguments.reference_checkout is None
                or arguments.candidate_checkout is None
            ):
                raise PreflightError("both checkout paths are required")
            manifest = QualificationManifest.read_json(arguments.qualification_manifest)
            selected = [
                s
                for s in manifest.shards
                if s.identity
                == (arguments.case_id, arguments.revision_kind, arguments.repetition)
            ]
            if len(selected) != 1:
                raise PreflightError(
                    "select one exact case/role/repetition from manifest"
                )
            shard = selected[0]
            result = preflight_shard(
                manifest,
                shard,
                arguments.reference_checkout,
                arguments.candidate_checkout,
                python_executable=arguments.runtime_python,
            )
            publish_document(arguments.output, result)
            return 0
        except PreflightBlocked as error:
            try:
                write_blocked_record(manifest, shard, error.reason, str(error))
            except (ValueError, OSError) as publication_error:
                print(str(publication_error), file=sys.stderr)
            print(str(error), file=sys.stderr)
            return 1
        except (ValueError, OSError, TypeError, KeyError) as error:
            print(str(error), file=sys.stderr)
            return 1
    manifest = build_preflight_manifest(
        arguments.matrix,
        arguments.prompts,
        _assignments(arguments.snapshot, "--snapshot"),
        local_revisions=_assignments(arguments.local_revision, "--local-revision"),
    )
    _write_exclusive(arguments.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PINNED_MODEL_REVISIONS",
    "CheckpointHash",
    "PreflightError",
    "PromptManifest",
    "build_preflight_manifest",
    "hash_checkpoint",
    "load_prompt_manifest",
]

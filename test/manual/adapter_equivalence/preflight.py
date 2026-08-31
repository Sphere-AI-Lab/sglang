"""Fail-closed checkpoint and prompt preflight for adapter equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
_TOKENIZER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_TOKENIZER_REVISION = PINNED_MODEL_REVISIONS["qwen3-4b-bf16"]
_NUMBERED_SHARD = re.compile(
    r"^model-(?P<number>\d{5})-of-(?P<total>\d{5})\.safetensors$"
)


class PreflightError(ValueError):
    """Raised when immutable checkpoint or prompt evidence is incomplete."""


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
            return json.load(stream, object_pairs_hook=_duplicate_key)
    except PreflightError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read {context}: {error}") from error


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PreflightError(f"{context} must be a JSON object with string keys")
    return value


def _fields(
    value: Mapping[str, object], required: set[str], context: str
) -> None:
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
            raise PreflightError(
                "model.safetensors.index.json must be a regular file"
            )
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
            raise PreflightError(
                f"unindexed checkpoint shards: {', '.join(unindexed)}"
            )
        missing_from_directory = sorted(shard_names - actual_shards)
        if missing_from_directory:
            raise PreflightError(
                f"missing shard: {', '.join(missing_from_directory)}"
            )
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
    if tokenizer["model"] != _TOKENIZER_MODEL:
        raise PreflightError("prompt tokenizer model does not match the binding model")
    if tokenizer["revision"] != _TOKENIZER_REVISION:
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
    for expected_id, record in zip(
        _PROMPT_IDS, records[1 : 1 + len(_PROMPT_IDS)]
    ):
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
        tokenizer_model=_TOKENIZER_MODEL,
        tokenizer_revision=_TOKENIZER_REVISION,
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
    arguments = parser.parse_args(argv)
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

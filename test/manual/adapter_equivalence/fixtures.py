"""Strict matrix loading and deterministic OFT/LoRA adapter fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from safetensors.numpy import save


DENSE_OFT_BLOCK_SIZE = 128
MOE_OFT_BLOCK_SIZE = 32
LORA_RANK = 8
LORA_ALPHA = 16
ADAPTER_SEEDS = (1729, 2718)

_REQUIRED_MATRIX_FIELDS = {
    "id",
    "model",
    "architecture",
    "precision",
    "gpu",
    "tp",
    "ep",
}
_OPTIONAL_MATRIX_FIELDS = {"quantization", "moe_runner"}
_OFT_TARGET_SUFFIXES = {
    "q_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
_LORA_REQUIRED_SUFFIXES = _OFT_TARGET_SUFFIXES | {"v_proj"}
_LORA_ALLOWED_SUFFIXES = _LORA_REQUIRED_SUFFIXES | {"k_proj"}
_DENSE_TARGET = re.compile(
    r"^model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_MOE_TARGET = re.compile(
    r"^model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj))$"
)


class FixtureValidationError(ValueError):
    """Raised when a matrix or requested fixture is not the binding shape."""


@dataclass(frozen=True)
class MatrixCell:
    id: str
    model: str
    architecture: str
    precision: str
    gpu: str
    tp: int
    ep: int
    quantization: str | None = None
    moe_runner: str | None = None


@dataclass(frozen=True)
class AdapterFixture:
    path: Path
    adapter_id: str
    peft_type: str
    seed: int
    block_size: int | None = None
    rank: int | None = None
    alpha: int | None = None


_EXPECTED_MATRIX = (
    MatrixCell(
        "qwen3-4b-bf16",
        "Qwen/Qwen3-4B-Instruct-2507",
        "dense",
        "bf16",
        "H100",
        1,
        1,
    ),
    MatrixCell(
        "qwen3-4b-fp8",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "dense",
        "fp8",
        "H100",
        1,
        1,
        "fp8",
    ),
    MatrixCell(
        "qwen3-4b-nvfp4",
        "OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4",
        "dense",
        "nvfp4",
        "B200",
        1,
        1,
        "modelopt_fp4",
    ),
    MatrixCell(
        "qwen3-30b-a3b-bf16",
        "Qwen/Qwen3-30B-A3B",
        "moe",
        "bf16",
        "H100",
        4,
        4,
    ),
    MatrixCell(
        "qwen3-30b-a3b-fp8",
        "Qwen/Qwen3-30B-A3B-FP8",
        "moe",
        "fp8",
        "H100",
        4,
        4,
        "fp8",
        "triton",
    ),
    MatrixCell(
        "qwen3-30b-a3b-nvfp4",
        "nvidia/Qwen3-30B-A3B-NVFP4",
        "moe",
        "nvfp4",
        "B200",
        4,
        4,
        "modelopt_fp4",
        "flashinfer_cutlass",
    ),
)


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_matrix(path: str | os.PathLike[str]) -> object:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_duplicate_key)
    except FixtureValidationError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureValidationError(f"cannot read matrix: {error}") from error


def _matrix_cell(row: object, index: int) -> MatrixCell:
    if not isinstance(row, Mapping):
        raise FixtureValidationError(f"matrix row {index} must be an object")
    if any(type(key) is not str for key in row):
        raise FixtureValidationError(f"matrix row {index} keys must be strings")
    fields = set(row)
    missing = sorted(_REQUIRED_MATRIX_FIELDS - fields)
    unknown = sorted(fields - _REQUIRED_MATRIX_FIELDS - _OPTIONAL_MATRIX_FIELDS)
    if missing:
        raise FixtureValidationError(
            f"matrix row {index} missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise FixtureValidationError(
            f"matrix row {index} has unknown fields: {', '.join(unknown)}"
        )
    for name in ("id", "model", "architecture", "precision", "gpu"):
        if type(row[name]) is not str or not row[name]:
            raise FixtureValidationError(
                f"matrix row {index} {name} must be a non-empty string"
            )
    for name in ("tp", "ep"):
        if type(row[name]) is not int or row[name] <= 0:
            raise FixtureValidationError(
                f"matrix row {index} {name} must be an integer greater than zero"
            )
    for name in _OPTIONAL_MATRIX_FIELDS:
        value = row.get(name)
        if value is not None and (type(value) is not str or not value):
            raise FixtureValidationError(
                f"matrix row {index} {name} must be a non-empty string when present"
            )
    return MatrixCell(
        id=row["id"],  # type: ignore[arg-type]
        model=row["model"],  # type: ignore[arg-type]
        architecture=row["architecture"],  # type: ignore[arg-type]
        precision=row["precision"],  # type: ignore[arg-type]
        gpu=row["gpu"],  # type: ignore[arg-type]
        tp=row["tp"],  # type: ignore[arg-type]
        ep=row["ep"],  # type: ignore[arg-type]
        quantization=row.get("quantization"),  # type: ignore[arg-type]
        moe_runner=row.get("moe_runner"),  # type: ignore[arg-type]
    )


def validate_matrix(
    source: str | os.PathLike[str] | Sequence[Mapping[str, object]],
) -> tuple[MatrixCell, ...]:
    """Load and validate the exact, ordered six-cell binding matrix."""

    raw = _load_matrix(source) if isinstance(source, (str, os.PathLike)) else source
    if not isinstance(raw, (list, tuple)):
        raise FixtureValidationError("matrix must be an array")
    cells = tuple(_matrix_cell(row, index) for index, row in enumerate(raw))
    ids = tuple(cell.id for cell in cells)
    if len(ids) != len(set(ids)):
        raise FixtureValidationError("duplicate cell id in matrix")
    expected_ids = tuple(cell.id for cell in _EXPECTED_MATRIX)
    if ids != expected_ids:
        raise FixtureValidationError(
            "matrix order must match the binding six-cell matrix"
        )
    for expected, actual in zip(_EXPECTED_MATRIX, cells):
        if actual != expected:
            raise FixtureValidationError(
                f"matrix cell {actual.id} does not match the binding definition"
            )
    return cells


def _require_seed(seed: object) -> int:
    if type(seed) is not int:
        raise FixtureValidationError("seed must be an integer")
    return seed


def _target_shapes(
    architecture: str,
    target_shapes: Mapping[str, tuple[int, int]],
    block_size: int | None,
    *,
    required_suffixes: set[str],
    allowed_suffixes: set[str],
) -> tuple[tuple[str, int, int], ...]:
    if architecture not in {"dense", "moe"}:
        raise FixtureValidationError("architecture must be dense or moe")
    if not isinstance(target_shapes, Mapping) or not target_shapes:
        raise FixtureValidationError("target_shapes must be a non-empty mapping")
    if any(type(name) is not str or not name for name in target_shapes):
        raise FixtureValidationError("target names must be non-empty module paths")

    resolved: list[tuple[str, int, int]] = []
    suffixes: set[str] = set()
    target_pattern = _DENSE_TARGET if architecture == "dense" else _MOE_TARGET
    for name in sorted(target_shapes):
        if target_pattern.fullmatch(name) is None:
            raise FixtureValidationError(f"unresolved target name: {name}")
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in allowed_suffixes:
            raise FixtureValidationError(f"unresolved target name: {name}")
        shape = target_shapes[name]
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise FixtureValidationError(
                f"target {name} shape must be two positive integers"
            )
        input_features, output_features = shape
        if block_size is not None and input_features % block_size:
            raise FixtureValidationError(
                f"target {name} input width must be divisible by block size "
                f"{block_size}"
            )
        suffixes.add(suffix)
        resolved.append((name, input_features, output_features))
    missing = sorted(required_suffixes - suffixes)
    if missing:
        raise FixtureValidationError(
            f"unresolved required targets: {', '.join(missing)}"
        )
    return tuple(resolved)


def _deterministic_values(
    seed: int, label: str, shape: tuple[int, ...], scale: float
) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    material = bytearray()
    counter = 0
    prefix = f"adapter-equivalence-fixture-v1\0{seed}\0{label}\0".encode()
    while len(material) < count * 4:
        material.extend(hashlib.sha256(prefix + counter.to_bytes(8, "little")).digest())
        counter += 1
    integers = np.frombuffer(material[: count * 4], dtype="<u4").astype(np.float64)
    values = (((integers + 0.5) / 2**32) * 2.0 - 1.0) * scale
    return np.ascontiguousarray(values.astype(np.float32).reshape(shape))


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


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _publish_fixture(
    destination: Path,
    config: Mapping[str, object],
    tensors: Mapping[str, np.ndarray],
) -> Path:
    destination = destination.absolute()
    if destination.exists():
        raise FixtureValidationError(f"fixture directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        config_bytes = _json_bytes(config)
        tensor_bytes = save(dict(sorted(tensors.items())))
        hashes = {
            "adapter_config.json": _sha256(config_bytes),
            "adapter_model.safetensors": _sha256(tensor_bytes),
        }
        _write_exclusive(staging / "adapter_config.json", config_bytes)
        _write_exclusive(staging / "adapter_model.safetensors", tensor_bytes)
        _write_exclusive(staging / "sha256.json", _json_bytes(hashes))
        try:
            os.rename(staging, destination)
        except FileExistsError as error:
            raise FixtureValidationError(
                f"fixture directory already exists: {destination}"
            ) from error
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination.resolve()


def _validate_build_inputs(
    destination: str | os.PathLike[str], adapter_id: str, seed: object
) -> tuple[Path, int]:
    if type(adapter_id) is not str or not adapter_id:
        raise FixtureValidationError("adapter_id must be a non-empty string")
    path = Path(destination)
    return path, _require_seed(seed)


def build_oft_fixture(
    destination: str | os.PathLike[str],
    *,
    adapter_id: str,
    architecture: str,
    seed: int,
    target_shapes: Mapping[str, tuple[int, int]],
) -> AdapterFixture:
    """Create one immutable deterministic compact-OFT fixture."""

    path, validated_seed = _validate_build_inputs(destination, adapter_id, seed)
    block_size = (
        DENSE_OFT_BLOCK_SIZE if architecture == "dense" else MOE_OFT_BLOCK_SIZE
    )
    targets = _target_shapes(
        architecture,
        target_shapes,
        block_size,
        required_suffixes=_OFT_TARGET_SUFFIXES,
        allowed_suffixes=_OFT_TARGET_SUFFIXES,
    )
    compact_size = block_size * (block_size - 1) // 2
    tensors: dict[str, np.ndarray] = {}
    for name, input_features, _ in targets:
        tensor_name = f"base_model.model.{name}.oft_R"
        tensors[tensor_name] = _deterministic_values(
            validated_seed,
            f"{adapter_id}\0{tensor_name}",
            (input_features // block_size, compact_size),
            1e-3,
        )
    config = {
        "inference_mode": True,
        "oft_block_size": block_size,
        "peft_type": "OFT",
        "target_modules": sorted({name.rsplit(".", 1)[-1] for name, _, _ in targets}),
        "task_type": "CAUSAL_LM",
    }
    published = _publish_fixture(path, config, tensors)
    return AdapterFixture(
        path=published,
        adapter_id=adapter_id,
        peft_type="OFT",
        seed=validated_seed,
        block_size=block_size,
    )


def build_lora_fixture(
    destination: str | os.PathLike[str],
    *,
    adapter_id: str,
    architecture: str,
    seed: int,
    target_shapes: Mapping[str, tuple[int, int]],
) -> AdapterFixture:
    """Create one immutable deterministic rank-8, alpha-16 LoRA fixture."""

    path, validated_seed = _validate_build_inputs(destination, adapter_id, seed)
    targets = _target_shapes(
        architecture,
        target_shapes,
        None,
        required_suffixes=_LORA_REQUIRED_SUFFIXES,
        allowed_suffixes=_LORA_ALLOWED_SUFFIXES,
    )
    tensors: dict[str, np.ndarray] = {}
    for name, input_features, output_features in targets:
        prefix = f"base_model.model.{name}"
        a_name = f"{prefix}.lora_A.weight"
        b_name = f"{prefix}.lora_B.weight"
        tensors[a_name] = _deterministic_values(
            validated_seed,
            f"{adapter_id}\0{a_name}",
            (LORA_RANK, input_features),
            1e-2,
        )
        tensors[b_name] = _deterministic_values(
            validated_seed,
            f"{adapter_id}\0{b_name}",
            (output_features, LORA_RANK),
            1e-2,
        )
    config = {
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": LORA_RANK,
        "target_modules": sorted({name.rsplit(".", 1)[-1] for name, _, _ in targets}),
        "task_type": "CAUSAL_LM",
    }
    published = _publish_fixture(path, config, tensors)
    return AdapterFixture(
        path=published,
        adapter_id=adapter_id,
        peft_type="LORA",
        seed=validated_seed,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
    )


__all__ = [
    "ADAPTER_SEEDS",
    "AdapterFixture",
    "FixtureValidationError",
    "MatrixCell",
    "build_lora_fixture",
    "build_oft_fixture",
    "validate_matrix",
]

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

SCHEMA_VERSION = 4

PROVENANCE_HASH_KEYS = (
    "code_hash",
    "checkpoint_hash",
    "adapter_hash",
    "tokenizer_hash",
    "scenario_hash",
    "environment_hash",
    "hardware_hash",
)

COMPARABLE_PROVENANCE_HASH_KEYS = (
    "checkpoint_hash",
    "adapter_hash",
    "tokenizer_hash",
    "scenario_hash",
    "environment_hash",
    "hardware_hash",
)

PERFORMANCE_REPETITIONS = 3

AdapterMode = Literal["base", "native_lora", "native_oft"]
EXECUTABLE_MODES: tuple[AdapterMode, ...] = (
    "base",
    "native_lora",
    "native_oft",
)

_MANIFEST_FIELDS = {
    "schema_version",
    "case_key",
    "performance_procedure_hash",
    "provenance_hashes",
    "server_args",
    "request_order",
    "seed",
    "metadata",
}
_MANIFEST_RESERVED_METADATA_FIELDS = {"server_args", "request_order", "seed"}
_PROVENANCE_FIELDS = {
    "git_sha",
    "dirty",
    "manifest_hash",
    *PROVENANCE_HASH_KEYS,
    "metadata",
}
_COMPLETION_FIELDS = {"status", "exit_code", "metadata"}

_RUN_BUNDLE_FIELDS = {
    "schema_version",
    "case_key",
    "manifest",
    "manifest_hash",
    "provenance",
    "observations",
    "performance",
    "completion",
    "completion_hash",
}

_ADAPTER_STATE_FIELDS = {
    "mode",
    "registered",
    "active",
    "staged",
    "registry_occupancy",
    "quarantined",
    "tombstoned",
    "cache_identity",
}
_REGISTERED_ADAPTER_FIELDS = {
    "name",
    "id",
    "version",
    "registry_slot",
    "pinned",
}
_ACTIVE_ADAPTER_FIELDS = {"name", "id", "version"}
_STAGED_ADAPTER_FIELDS = {"name", "id", "version", "pinned"}
_OBSERVATION_ERROR_FIELDS = {"kind", "code", "message"}


class BundleValidationError(ValueError):
    """Raised when serialized equivalence evidence is incomplete or inconsistent."""


class FrozenDict(dict):
    """A dict-compatible immutable mapping used by the public frozen dataclasses."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("equivalence bundle mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze_json(value: object, context: str = "value") -> object:
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise BundleValidationError(f"{context} must contain only finite floats")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise BundleValidationError(f"{context} keys must be strings")
            frozen[key] = _freeze_json(item, f"{context}.{key}")
        return FrozenDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise BundleValidationError(
        f"{context} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _exact_json_equal(expected: object, actual: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int coercion."""

    if type(expected) is not type(actual):
        return False
    if isinstance(expected, Mapping):
        if set(expected) != set(actual):  # type: ignore[arg-type]
            return False
        return all(
            _exact_json_equal(expected[key], actual[key])  # type: ignore[index]
            for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(  # type: ignore[arg-type]
            _exact_json_equal(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual)  # type: ignore[arg-type]
        )
    return expected == actual


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{context} must be an object")
    if any(type(key) is not str for key in value):
        raise BundleValidationError(f"{context} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, object], required: set[str], context: str
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise BundleValidationError(f"{context}: missing fields: {', '.join(missing)}")
    if unknown:
        raise BundleValidationError(f"{context}: unknown fields: {', '.join(unknown)}")


def _validate_sha256(value: object, context: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise BundleValidationError(f"{context} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise BundleValidationError(f"{context} must be a lowercase SHA-256")


def _require_non_empty_string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise BundleValidationError(f"{context} must be a non-empty string")
    return value


def _require_array(value: object, context: str) -> list[object] | tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise BundleValidationError(f"{context} must be an array")
    return value


def _validate_adapter_identity(
    value: object,
    *,
    required_fields: set[str],
    context: str,
) -> Mapping[str, object]:
    identity = _require_mapping(value, context)
    _require_exact_fields(identity, required_fields, context)
    for key in ("name", "id", "version"):
        _require_non_empty_string(identity[key], f"{context}.{key}")
    if "pinned" in required_fields and type(identity["pinned"]) is not bool:
        raise BundleValidationError(f"{context}.pinned must be a boolean")
    return identity


def validate_adapter_state(value: object, context: str = "adapter_state") -> None:
    """Validate the exact native adapter state emitted by the control layer."""

    state = _require_mapping(value, context)
    _require_exact_fields(state, _ADAPTER_STATE_FIELDS, context)

    mode = state["mode"]
    if type(mode) is not str or mode not in EXECUTABLE_MODES:
        raise BundleValidationError(f"{context}.mode is invalid")

    registered = _require_array(state["registered"], f"{context}.registered")
    registered_by_name: dict[str, Mapping[str, object]] = {}
    registered_ids: set[str] = set()
    registered_names: list[str] = []
    for index, raw_record in enumerate(registered):
        record_context = f"{context}.registered[{index}]"
        record = _validate_adapter_identity(
            raw_record,
            required_fields=_REGISTERED_ADAPTER_FIELDS,
            context=record_context,
        )
        slot = record["registry_slot"]
        if type(slot) is not int or slot != index:
            raise BundleValidationError(
                f"{record_context}.registry_slot must equal {index}"
            )
        name = record["name"]
        adapter_id = record["id"]
        assert isinstance(name, str) and isinstance(adapter_id, str)
        if name in registered_by_name:
            raise BundleValidationError(f"{context}.registered names must be unique")
        if adapter_id in registered_ids:
            raise BundleValidationError(f"{context}.registered IDs must be unique")
        registered_by_name[name] = record
        registered_ids.add(adapter_id)
        registered_names.append(name)
    if registered_names != sorted(registered_names):
        raise BundleValidationError(f"{context}.registered must be sorted by name")

    occupancy = state["registry_occupancy"]
    if type(occupancy) is not int or occupancy != len(registered):
        raise BundleValidationError(
            f"{context}.registry_occupancy must equal registered length"
        )

    active = state["active"]
    if active is not None:
        active_identity = _validate_adapter_identity(
            active,
            required_fields=_ACTIVE_ADAPTER_FIELDS,
            context=f"{context}.active",
        )
        registered_identity = registered_by_name.get(active_identity["name"])
        if registered_identity is None or any(
            active_identity[key] != registered_identity[key]
            for key in ("name", "id", "version")
        ):
            raise BundleValidationError(
                f"{context}.active must match an exact registered identity"
            )

    staged = state["staged"]
    if staged is not None:
        staged_identity = _validate_adapter_identity(
            staged,
            required_fields=_STAGED_ADAPTER_FIELDS,
            context=f"{context}.staged",
        )
        registered_identity = registered_by_name.get(staged_identity["name"])
        if (
            registered_identity is not None
            and staged_identity["id"] != registered_identity["id"]
        ):
            raise BundleValidationError(
                f"{context}.staged ID does not match registered identity"
            )

    quarantined = _require_array(state["quarantined"], f"{context}.quarantined")
    if any(type(entry) is not str or not entry for entry in quarantined):
        raise BundleValidationError(
            f"{context}.quarantined entries must be non-empty strings"
        )
    if list(quarantined) != sorted(set(quarantined)):
        raise BundleValidationError(f"{context}.quarantined must be sorted and unique")

    tombstoned = _require_array(state["tombstoned"], f"{context}.tombstoned")
    tombstone_names = []
    tombstone_ids = set()
    for index, raw_record in enumerate(tombstoned):
        record = _validate_adapter_identity(
            raw_record,
            required_fields=_STAGED_ADAPTER_FIELDS,
            context=f"{context}.tombstoned[{index}]",
        )
        name, adapter_id = record["name"], record["id"]
        if adapter_id in tombstone_ids:
            raise BundleValidationError(f"{context}.tombstoned IDs must be unique")
        tombstone_names.append(name)
        tombstone_ids.add(adapter_id)
        if name in registered_by_name or adapter_id in registered_ids:
            raise BundleValidationError(
                f"{context}.tombstoned identity must be absent from registered"
            )
        if staged is not None and (
            staged["name"] == name or staged["id"] == adapter_id
        ):
            raise BundleValidationError(
                f"{context}.tombstoned identity must be absent from staged"
            )
    if tombstone_names != sorted(set(tombstone_names)):
        raise BundleValidationError(
            f"{context}.tombstoned names must be sorted and unique"
        )

    cache_identity = _require_mapping(
        state["cache_identity"], f"{context}.cache_identity"
    )
    for record in tombstoned:
        if cache_identity.get(record["name"]) != record["id"]:
            raise BundleValidationError(
                f"{context}.tombstoned ID must match cache_identity"
            )
    for name, adapter_id in cache_identity.items():
        _require_non_empty_string(name, f"{context}.cache_identity key")
        _require_non_empty_string(adapter_id, f"{context}.cache_identity.{name}")
        registered_identity = registered_by_name.get(name)
        if registered_identity is not None and registered_identity["id"] != adapter_id:
            raise BundleValidationError(
                f"{context}.cache_identity.{name} does not match registered ID"
            )

    if mode == "base" and (
        registered
        or active is not None
        or staged is not None
        or occupancy != 0
        or state["quarantined"]
        or state["tombstoned"]
        or cache_identity
    ):
        raise BundleValidationError(
            f"{context}: base adapter state must be completely empty"
        )


def validate_observation_error(value: object, context: str = "error") -> None:
    """Validate the exact expected product-rejection shape."""

    error = _require_mapping(value, context)
    _require_exact_fields(error, _OBSERVATION_ERROR_FIELDS, context)
    if error["kind"] != "product_rejection":
        raise BundleValidationError(f"{context}.kind must equal 'product_rejection'")
    _require_non_empty_string(error["code"], f"{context}.code")
    _require_non_empty_string(error["message"], f"{context}.message")


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of the canonical JSON representation of ``value``."""

    frozen = _freeze_json(value)
    encoded = json.dumps(
        _thaw_json(frozen),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BundleValidationError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _read_json_document(path: str | os.PathLike[str], context: str) -> object:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_reject_duplicate_object_keys)
    except BundleValidationError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"cannot read {context}: {error}") from error


def _write_json_document(path: str | os.PathLike[str], value: object) -> None:
    destination = Path(path)
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class CaseKey:
    model: str
    architecture: Literal["dense", "moe"]
    precision: Literal["bf16", "fp8", "nvfp4"]
    revision: str
    mode: AdapterMode
    cuda_graph: bool
    scenario: str

    def validate(self) -> None:
        for field_name in ("model", "revision", "scenario"):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise BundleValidationError(f"case_key.{field_name} must be non-empty")
        if type(self.architecture) is not str or self.architecture not in (
            "dense",
            "moe",
        ):
            raise BundleValidationError("case_key.architecture is invalid")
        if type(self.precision) is not str or self.precision not in (
            "bf16",
            "fp8",
            "nvfp4",
        ):
            raise BundleValidationError("case_key.precision is invalid")
        if type(self.mode) is not str or self.mode not in EXECUTABLE_MODES:
            raise BundleValidationError("case_key.mode is invalid")
        if type(self.cuda_graph) is not bool:
            raise BundleValidationError("case_key.cuda_graph must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "architecture": self.architecture,
            "precision": self.precision,
            "revision": self.revision,
            "mode": self.mode,
            "cuda_graph": self.cuda_graph,
            "scenario": self.scenario,
        }

    @classmethod
    def from_dict(cls, value: object) -> CaseKey:
        mapping = _require_mapping(value, "case_key")
        _require_exact_fields(
            mapping,
            {
                "model",
                "architecture",
                "precision",
                "revision",
                "mode",
                "cuda_graph",
                "scenario",
            },
            "case_key",
        )
        case_key = cls(**mapping)  # type: ignore[arg-type]
        case_key.validate()
        return case_key


@dataclass(frozen=True)
class Observation:
    output_ids: tuple[int, ...]
    text: str
    token_logprobs: tuple[float, ...]
    selected_logits: dict[str, tuple[float, ...]]
    selected_token_ids: dict[str, tuple[int, ...]]
    adapter_state: dict[str, object]
    error: dict[str, object] | None
    request_output_lengths: tuple[int, ...]
    request_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_output_lengths", tuple(self.request_output_lengths)
        )
        object.__setattr__(self, "request_texts", tuple(self.request_texts))
        object.__setattr__(self, "output_ids", tuple(self.output_ids))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        logits = {name: tuple(values) for name, values in self.selected_logits.items()}
        object.__setattr__(self, "selected_logits", FrozenDict(logits))
        token_ids = {
            name: tuple(values) for name, values in self.selected_token_ids.items()
        }
        object.__setattr__(self, "selected_token_ids", FrozenDict(token_ids))
        object.__setattr__(
            self, "adapter_state", _freeze_json(self.adapter_state, "adapter_state")
        )
        if self.error is not None:
            object.__setattr__(self, "error", _freeze_json(self.error, "error"))

    def validate(self, context: str = "observation") -> None:
        if any(type(token_id) is not int for token_id in self.output_ids):
            raise BundleValidationError(
                f"{context}.output_ids entries must be integers"
            )
        if type(self.text) is not str:
            raise BundleValidationError(f"{context}.text must be a string")
        if any(type(value) is not float for value in self.token_logprobs):
            raise BundleValidationError(
                f"{context}.token_logprobs entries must have float dtype"
            )
        if any(not math.isfinite(value) for value in self.token_logprobs):
            raise BundleValidationError(
                f"{context}.token_logprobs entries must be finite"
            )
        if len(self.token_logprobs) != len(self.output_ids):
            raise BundleValidationError(
                f"{context}.token_logprobs shape must match output_ids shape"
            )
        expected_selected_names = {
            f"decode.{position:03d}.top_logprobs"
            for position in range(len(self.output_ids))
        }
        selected_logit_names = set(self.selected_logits)
        selected_token_names = set(self.selected_token_ids)
        if selected_logit_names != expected_selected_names:
            raise BundleValidationError(
                f"{context}.selected_logits rows must exactly match output_ids"
            )
        if selected_token_names != expected_selected_names:
            raise BundleValidationError(
                f"{context}.selected_token_ids rows must exactly match output_ids"
            )

        selected_width: int | None = None
        for name in sorted(expected_selected_names):
            values = self.selected_logits[name]
            if type(name) is not str or not name:
                raise BundleValidationError(
                    f"{context}.selected_logits names must be non-empty strings"
                )
            if any(type(value) is not float for value in values):
                raise BundleValidationError(
                    f"{context}.selected_logits.{name} entries must have float dtype"
                )
            if any(not math.isfinite(value) for value in values):
                raise BundleValidationError(
                    f"{context}.selected_logits.{name} entries must be finite"
                )
            token_ids = self.selected_token_ids[name]
            if any(type(token_id) is not int for token_id in token_ids):
                raise BundleValidationError(
                    f"{context}.selected_token_ids.{name} entries must be integers"
                )
            if tuple(sorted(set(token_ids))) != token_ids:
                raise BundleValidationError(
                    f"{context}.selected_token_ids.{name} entries must be unique "
                    "and sorted"
                )
            if len(values) != len(token_ids):
                raise BundleValidationError(
                    f"{context}.selected score and token ID widths must match for "
                    f"{name}"
                )
            if not values:
                raise BundleValidationError(
                    f"{context}.selected score rows must not be empty"
                )
            if selected_width is None:
                selected_width = len(values)
            elif len(values) != selected_width:
                raise BundleValidationError(
                    f"{context}.selected score rows must have one stable width"
                )
        if any(
            type(length) is not int or length <= 0
            for length in self.request_output_lengths
        ):
            raise BundleValidationError(
                f"{context}.request_output_lengths must contain positive integers"
            )
        if sum(self.request_output_lengths) != len(self.output_ids):
            raise BundleValidationError(
                f"{context}.request_output_lengths must sum to output_ids length"
            )
        if len(self.request_texts) != len(self.request_output_lengths) or any(
            type(text) is not str for text in self.request_texts
        ):
            raise BundleValidationError(
                f"{context}.request_texts must contain one string per request"
            )
        validate_adapter_state(self.adapter_state, f"{context}.adapter_state")
        if self.error is not None:
            validate_observation_error(self.error, f"{context}.error")
        _freeze_json(self.adapter_state, f"{context}.adapter_state")
        _freeze_json(self.error, f"{context}.error")

    def to_dict(self) -> dict[str, object]:
        return {
            "output_ids": list(self.output_ids),
            "request_output_lengths": list(self.request_output_lengths),
            "request_texts": list(self.request_texts),
            "text": self.text,
            "token_logprobs": list(self.token_logprobs),
            "selected_logits": {
                name: list(values) for name, values in self.selected_logits.items()
            },
            "selected_token_ids": {
                name: list(values) for name, values in self.selected_token_ids.items()
            },
            "adapter_state": _thaw_json(self.adapter_state),
            "error": _thaw_json(self.error),
        }

    @classmethod
    def from_dict(cls, value: object, context: str = "observation") -> Observation:
        mapping = _require_mapping(value, context)
        _require_exact_fields(
            mapping,
            {
                "output_ids",
                "request_output_lengths",
                "request_texts",
                "text",
                "token_logprobs",
                "selected_logits",
                "selected_token_ids",
                "adapter_state",
                "error",
            },
            context,
        )
        output_ids = mapping["output_ids"]
        for field in ("request_output_lengths", "request_texts"):
            if type(mapping[field]) is not list:
                raise BundleValidationError(f"{context}.{field} must be an array")
        token_logprobs = mapping["token_logprobs"]
        selected_logits = _require_mapping(
            mapping["selected_logits"], f"{context}.selected_logits"
        )
        selected_token_ids = _require_mapping(
            mapping["selected_token_ids"], f"{context}.selected_token_ids"
        )
        if type(output_ids) is not list:
            raise BundleValidationError(f"{context}.output_ids must be an array")
        if type(token_logprobs) is not list:
            raise BundleValidationError(f"{context}.token_logprobs must be an array")
        logits: dict[str, tuple[float, ...]] = {}
        for name, values in selected_logits.items():
            if type(values) is not list:
                raise BundleValidationError(
                    f"{context}.selected_logits.{name} must be an array"
                )
            logits[name] = tuple(values)  # type: ignore[arg-type]
        token_ids: dict[str, tuple[int, ...]] = {}
        for name, values in selected_token_ids.items():
            if type(values) is not list:
                raise BundleValidationError(
                    f"{context}.selected_token_ids.{name} must be an array"
                )
            token_ids[name] = tuple(values)  # type: ignore[arg-type]
        observation = cls(
            request_output_lengths=tuple(mapping["request_output_lengths"]),
            request_texts=tuple(mapping["request_texts"]),
            output_ids=tuple(output_ids),  # type: ignore[arg-type]
            text=mapping["text"],  # type: ignore[arg-type]
            token_logprobs=tuple(token_logprobs),  # type: ignore[arg-type]
            selected_logits=logits,
            selected_token_ids=token_ids,
            adapter_state=dict(
                _require_mapping(mapping["adapter_state"], f"{context}.adapter_state")
            ),
            error=(
                None
                if mapping["error"] is None
                else dict(_require_mapping(mapping["error"], f"{context}.error"))
            ),
        )
        observation.validate(context)
        return observation


@dataclass(frozen=True)
class PerformanceMetrics:
    procedure_hash: str
    startup_seconds: tuple[float, ...]
    latency_seconds: tuple[float, ...]
    throughput_tokens_per_second: tuple[float, ...]
    peak_allocated_bytes: tuple[int, ...]
    peak_reserved_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "startup_seconds",
            "latency_seconds",
            "throughput_tokens_per_second",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    def validate(self) -> None:
        _validate_sha256(self.procedure_hash, "performance.procedure_hash")
        for field_name in (
            "startup_seconds",
            "latency_seconds",
            "throughput_tokens_per_second",
        ):
            samples = getattr(self, field_name)
            if len(samples) != PERFORMANCE_REPETITIONS:
                raise BundleValidationError(
                    f"performance.{field_name} must contain exactly "
                    f"{PERFORMANCE_REPETITIONS} post-warm-up repetitions"
                )
            if any(type(sample) is not float for sample in samples):
                raise BundleValidationError(
                    f"performance.{field_name} entries must have float dtype"
                )
            if any(not math.isfinite(sample) or sample < 0 for sample in samples):
                raise BundleValidationError(
                    f"performance.{field_name} entries must be finite and non-negative"
                )
        for field_name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            samples = getattr(self, field_name)
            if len(samples) != PERFORMANCE_REPETITIONS:
                raise BundleValidationError(
                    f"performance.{field_name} must contain exactly "
                    f"{PERFORMANCE_REPETITIONS} post-warm-up repetitions"
                )
            if any(type(sample) is not int or sample < 0 for sample in samples):
                raise BundleValidationError(
                    f"performance.{field_name} entries must be non-negative integers"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "procedure_hash": self.procedure_hash,
            "startup_seconds": list(self.startup_seconds),
            "latency_seconds": list(self.latency_seconds),
            "throughput_tokens_per_second": list(self.throughput_tokens_per_second),
            "peak_allocated_bytes": list(self.peak_allocated_bytes),
            "peak_reserved_bytes": list(self.peak_reserved_bytes),
        }

    @classmethod
    def from_dict(cls, value: object) -> PerformanceMetrics:
        mapping = _require_mapping(value, "performance")
        _require_exact_fields(
            mapping,
            {
                "procedure_hash",
                "startup_seconds",
                "latency_seconds",
                "throughput_tokens_per_second",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            },
            "performance",
        )
        for field_name in (
            "startup_seconds",
            "latency_seconds",
            "throughput_tokens_per_second",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            if type(mapping[field_name]) is not list:
                raise BundleValidationError(
                    f"performance.{field_name} must be an array"
                )
        metrics = cls(
            procedure_hash=mapping["procedure_hash"],  # type: ignore[arg-type]
            startup_seconds=tuple(mapping["startup_seconds"]),  # type: ignore[arg-type]
            latency_seconds=tuple(mapping["latency_seconds"]),  # type: ignore[arg-type]
            throughput_tokens_per_second=tuple(
                mapping["throughput_tokens_per_second"]  # type: ignore[arg-type]
            ),
            peak_allocated_bytes=tuple(  # type: ignore[arg-type]
                mapping["peak_allocated_bytes"]
            ),
            peak_reserved_bytes=tuple(  # type: ignore[arg-type]
                mapping["peak_reserved_bytes"]
            ),
        )
        metrics.validate()
        return metrics


@dataclass(frozen=True)
class RunBundle:
    schema_version: int
    case_key: CaseKey
    manifest: dict[str, object]
    manifest_hash: str
    provenance: dict[str, object]
    observations: dict[str, Observation]
    performance: PerformanceMetrics
    completion: dict[str, object]
    completion_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", _freeze_json(self.manifest, "manifest"))
        object.__setattr__(
            self, "provenance", _freeze_json(self.provenance, "provenance")
        )
        object.__setattr__(self, "observations", FrozenDict(dict(self.observations)))
        object.__setattr__(
            self, "completion", _freeze_json(self.completion, "completion")
        )

    @classmethod
    def create(
        cls,
        *,
        case_key: CaseKey,
        manifest: Mapping[str, object],
        provenance: Mapping[str, object],
        observations: Mapping[str, Observation],
        performance: PerformanceMetrics,
        completion: Mapping[str, object],
    ) -> RunBundle:
        manifest_hash = canonical_sha256(manifest)
        provenance_copy = dict(provenance)
        recorded_manifest_hash = provenance_copy.get("manifest_hash")
        if recorded_manifest_hash not in (None, manifest_hash):
            raise BundleValidationError(
                "provenance.manifest_hash does not match the manifest"
            )
        provenance_copy["manifest_hash"] = manifest_hash
        bundle = cls(
            schema_version=SCHEMA_VERSION,
            case_key=case_key,
            manifest=dict(manifest),
            manifest_hash=manifest_hash,
            provenance=provenance_copy,
            observations=dict(observations),
            performance=performance,
            completion=dict(completion),
            completion_hash=canonical_sha256(completion),
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise BundleValidationError(f"schema_version must equal {SCHEMA_VERSION}")
        if not isinstance(self.case_key, CaseKey):
            raise BundleValidationError("case_key must be a CaseKey")
        self.case_key.validate()
        manifest = _require_mapping(self.manifest, "manifest")
        _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != self.schema_version
        ):
            raise BundleValidationError(
                "manifest.schema_version does not match schema_version"
            )
        manifest_case_key = CaseKey.from_dict(_thaw_json(manifest["case_key"]))
        if not _exact_json_equal(manifest_case_key.to_dict(), self.case_key.to_dict()):
            raise BundleValidationError("manifest.case_key does not match case_key")

        server_args = manifest["server_args"]
        if type(server_args) is not tuple or any(
            type(argument) is not str for argument in server_args
        ):
            raise BundleValidationError(
                "manifest.server_args must be an array of strings"
            )
        request_order = manifest["request_order"]
        if type(request_order) is not tuple:
            raise BundleValidationError("manifest.request_order must be an array")
        if any(
            type(request_id) is not str or not request_id
            for request_id in request_order
        ):
            raise BundleValidationError(
                "manifest.request_order entries must be non-empty strings"
            )
        if len(request_order) != len(set(request_order)):
            raise BundleValidationError("manifest.request_order entries must be unique")
        if type(manifest["seed"]) is not int:
            raise BundleValidationError("manifest.seed must be an integer")
        metadata = _require_mapping(manifest["metadata"], "manifest.metadata")
        reserved_metadata = sorted(
            _MANIFEST_RESERVED_METADATA_FIELDS.intersection(metadata)
        )
        if reserved_metadata:
            raise BundleValidationError(
                "manifest.metadata contains reserved contract fields: "
                + ", ".join(reserved_metadata)
            )

        provenance = _require_mapping(self.provenance, "provenance")
        _require_exact_fields(provenance, _PROVENANCE_FIELDS, "provenance")
        if (
            type(provenance["git_sha"]) is not str
            or provenance["git_sha"] != self.case_key.revision
        ):
            raise BundleValidationError(
                "provenance.git_sha does not match case_key.revision"
            )
        if type(provenance["dirty"]) is not bool or provenance["dirty"] is not False:
            raise BundleValidationError("provenance.dirty must be false")
        _require_mapping(provenance["metadata"], "provenance.metadata")
        for key in (*PROVENANCE_HASH_KEYS, "manifest_hash"):
            _validate_sha256(provenance[key], f"provenance.{key}")

        manifest_hashes = _require_mapping(
            manifest["provenance_hashes"], "manifest.provenance_hashes"
        )
        _require_exact_fields(
            manifest_hashes, set(PROVENANCE_HASH_KEYS), "manifest.provenance_hashes"
        )
        for key in PROVENANCE_HASH_KEYS:
            if manifest_hashes[key] != provenance[key]:
                raise BundleValidationError(
                    f"manifest.provenance_hashes.{key} does not match provenance"
                )

        _validate_sha256(self.manifest_hash, "manifest_hash")
        calculated_manifest_hash = canonical_sha256(manifest)
        if self.manifest_hash != calculated_manifest_hash:
            raise BundleValidationError(
                "manifest_hash does not match the canonical manifest"
            )
        if provenance["manifest_hash"] != self.manifest_hash:
            raise BundleValidationError(
                "provenance.manifest_hash does not match manifest_hash"
            )

        if not isinstance(self.performance, PerformanceMetrics):
            raise BundleValidationError("performance must be PerformanceMetrics")
        self.performance.validate()
        _validate_sha256(
            manifest["performance_procedure_hash"],
            "manifest.performance_procedure_hash",
        )
        if self.performance.procedure_hash != manifest["performance_procedure_hash"]:
            raise BundleValidationError(
                "performance.procedure_hash does not match manifest"
            )

        observations = _require_mapping(self.observations, "observations")
        if not observations:
            raise BundleValidationError("observations must not be empty")
        for request_id, observation in observations.items():
            if type(request_id) is not str or not request_id:
                raise BundleValidationError("observation request IDs must be non-empty")
            if not isinstance(observation, Observation):
                raise BundleValidationError(
                    f"observations.{request_id} must be an Observation"
                )
            observation.validate(f"observations.{request_id}")
            if observation.adapter_state["mode"] != self.case_key.mode:
                raise BundleValidationError(
                    f"observations.{request_id}.adapter_state.mode does not match "
                    "case_key.mode"
                )
        if set(request_order) != set(observations):
            raise BundleValidationError(
                "manifest.request_order must name every observation exactly once"
            )

        completion = _require_mapping(self.completion, "completion")
        _require_exact_fields(completion, _COMPLETION_FIELDS, "completion")
        _require_mapping(completion["metadata"], "completion.metadata")
        _validate_sha256(self.completion_hash, "completion_hash")
        if self.completion_hash != canonical_sha256(completion):
            raise BundleValidationError(
                "completion_hash does not match the canonical completion marker"
            )
        if type(completion["status"]) is not str or completion["status"] != "complete":
            raise BundleValidationError("completion.status must be 'complete'")
        if type(completion["exit_code"]) is not int or completion["exit_code"] != 0:
            raise BundleValidationError("completion.exit_code must be 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_key": self.case_key.to_dict(),
            "manifest": _thaw_json(self.manifest),
            "manifest_hash": self.manifest_hash,
            "provenance": _thaw_json(self.provenance),
            "observations": {
                request_id: observation.to_dict()
                for request_id, observation in self.observations.items()
            },
            "performance": self.performance.to_dict(),
            "completion": _thaw_json(self.completion),
            "completion_hash": self.completion_hash,
        }

    @classmethod
    def from_dict(cls, value: object) -> RunBundle:
        mapping = _require_mapping(value, "RunBundle")
        _require_exact_fields(mapping, _RUN_BUNDLE_FIELDS, "RunBundle")
        observations_data = _require_mapping(mapping["observations"], "observations")
        observations = {
            request_id: Observation.from_dict(observation, f"observations.{request_id}")
            for request_id, observation in observations_data.items()
        }
        bundle = cls(
            schema_version=mapping["schema_version"],  # type: ignore[arg-type]
            case_key=CaseKey.from_dict(mapping["case_key"]),
            manifest=dict(_require_mapping(mapping["manifest"], "manifest")),
            manifest_hash=mapping["manifest_hash"],  # type: ignore[arg-type]
            provenance=dict(_require_mapping(mapping["provenance"], "provenance")),
            observations=observations,
            performance=PerformanceMetrics.from_dict(mapping["performance"]),
            completion=dict(_require_mapping(mapping["completion"], "completion")),
            completion_hash=mapping["completion_hash"],  # type: ignore[arg-type]
        )
        bundle.validate()
        return bundle

    def write_json(self, path: str | os.PathLike[str]) -> None:
        _write_json_document(path, self.to_dict())

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> RunBundle:
        return cls.from_dict(_read_json_document(path, "RunBundle"))

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


from .policy import (  # noqa: E402  (re-export the public schema contract)
    BaselineRepetition as BaselineRepetition,
)
from .policy import ComparisonPolicy as ComparisonPolicy
from .policy import NumericTolerance as NumericTolerance
from .policy import ToleranceEnvelope as ToleranceEnvelope

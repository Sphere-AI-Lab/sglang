from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Mapping

from .schema import (
    SCHEMA_VERSION,
    BundleValidationError,
    FrozenDict,
    _freeze_json,
    _read_json_document,
    _require_exact_fields,
    _require_mapping,
    _thaw_json,
    _validate_sha256,
    _write_json_document,
    canonical_sha256,
)


@dataclass(frozen=True)
class BaselineRepetition:
    """One immutable unchanged-baseline result for a named numeric quantity."""

    baseline_manifest_hash: str
    bundle_hash: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def validate(self, context: str) -> None:
        _validate_sha256(
            self.baseline_manifest_hash,
            f"{context}.baseline_manifest_hash",
        )
        _validate_sha256(self.bundle_hash, f"{context}.bundle_hash")
        if not self.values:
            raise BundleValidationError(f"{context}.values must not be empty")
        if any(type(value) is not float for value in self.values):
            raise BundleValidationError(
                f"{context}.values entries must have float dtype"
            )
        if any(not math.isfinite(value) for value in self.values):
            raise BundleValidationError(f"{context}.values entries must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_manifest_hash": self.baseline_manifest_hash,
            "bundle_hash": self.bundle_hash,
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, value: object, context: str) -> BaselineRepetition:
        mapping = _require_mapping(value, context)
        _require_exact_fields(
            mapping,
            {"baseline_manifest_hash", "bundle_hash", "values"},
            context,
        )
        if type(mapping["values"]) is not list:
            raise BundleValidationError(f"{context}.values must be an array")
        repetition = cls(
            baseline_manifest_hash=mapping[  # type: ignore[arg-type]
                "baseline_manifest_hash"
            ],
            bundle_hash=mapping["bundle_hash"],  # type: ignore[arg-type]
            values=tuple(mapping["values"]),  # type: ignore[arg-type]
        )
        repetition.validate(context)
        return repetition


def _derive_observed_bounds(
    repetitions: tuple[BaselineRepetition, ...],
) -> tuple[float, float]:
    max_absolute = 0.0
    max_relative = 0.0
    for left_index, left in enumerate(repetitions):
        for right in repetitions[left_index + 1 :]:
            for left_value, right_value in zip(left.values, right.values):
                absolute = abs(left_value - right_value)
                scale = max(abs(left_value), abs(right_value))
                relative = 0.0 if absolute == 0.0 else absolute / scale
                max_absolute = max(max_absolute, absolute)
                max_relative = max(max_relative, relative)
    return max_absolute, max_relative


@dataclass(frozen=True)
class NumericTolerance:
    """Observed bounds derived from immutable unchanged-baseline repetitions."""

    repetitions: tuple[BaselineRepetition, ...]
    observed_atol: float
    observed_rtol: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "repetitions", tuple(self.repetitions))

    @classmethod
    def create(
        cls, *, repetitions: tuple[BaselineRepetition, ...]
    ) -> NumericTolerance:
        frozen_repetitions = tuple(repetitions)
        observed_atol, observed_rtol = _derive_observed_bounds(frozen_repetitions)
        tolerance = cls(
            repetitions=frozen_repetitions,
            observed_atol=observed_atol,
            observed_rtol=observed_rtol,
        )
        tolerance.validate("<unbound>")
        return tolerance

    def validate(self, name: str) -> None:
        context = f"tolerances.{name}"
        if len(self.repetitions) < 2:
            raise BundleValidationError(
                f"{context} requires at least two unchanged-baseline repetitions"
            )
        bundle_hashes: set[str] = set()
        expected_shape: int | None = None
        for index, repetition in enumerate(self.repetitions):
            if not isinstance(repetition, BaselineRepetition):
                raise BundleValidationError(
                    f"{context}.repetitions[{index}] must be a BaselineRepetition"
                )
            repetition.validate(f"{context}.repetitions[{index}]")
            if repetition.bundle_hash in bundle_hashes:
                raise BundleValidationError(
                    f"{context}.repetitions bundle_hash values must be distinct"
                )
            bundle_hashes.add(repetition.bundle_hash)
            if expected_shape is None:
                expected_shape = len(repetition.values)
            elif len(repetition.values) != expected_shape:
                raise BundleValidationError(
                    f"{context}.repetitions values must have one shared shape"
                )
        for field_name in ("observed_atol", "observed_rtol"):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise BundleValidationError(
                    f"{context}.{field_name} must be a finite non-negative float"
                )
        derived_atol, derived_rtol = _derive_observed_bounds(self.repetitions)
        if self.observed_atol != derived_atol:
            raise BundleValidationError(
                f"{context}.observed_atol does not match unchanged-baseline evidence"
            )
        if self.observed_rtol != derived_rtol:
            raise BundleValidationError(
                f"{context}.observed_rtol does not match unchanged-baseline evidence"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "repetitions": [
                repetition.to_dict() for repetition in self.repetitions
            ],
            "observed_atol": self.observed_atol,
            "observed_rtol": self.observed_rtol,
        }

    @classmethod
    def from_dict(cls, value: object, name: str) -> NumericTolerance:
        context = f"tolerances.{name}"
        mapping = _require_mapping(value, context)
        _require_exact_fields(
            mapping,
            {"repetitions", "observed_atol", "observed_rtol"},
            context,
        )
        repetitions = mapping["repetitions"]
        if type(repetitions) is not list:
            raise BundleValidationError(f"{context}.repetitions must be an array")
        tolerance = cls(
            repetitions=tuple(
                BaselineRepetition.from_dict(
                    repetition, f"{context}.repetitions[{index}]"
                )
                for index, repetition in enumerate(repetitions)
            ),
            observed_atol=mapping["observed_atol"],  # type: ignore[arg-type]
            observed_rtol=mapping["observed_rtol"],  # type: ignore[arg-type]
        )
        tolerance.validate(name)
        return tolerance


@dataclass(frozen=True)
class ToleranceEnvelope:
    schema_version: int
    baseline_manifest_hash: str
    tolerances: dict[str, NumericTolerance]
    metadata: dict[str, object]
    manifest_hash: str
    comparison_policy: ComparisonPolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerances", FrozenDict(dict(self.tolerances)))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @classmethod
    def create(
        cls,
        *,
        baseline_manifest_hash: str,
        tolerances: Mapping[str, NumericTolerance],
        metadata: Mapping[str, object] | None = None,
    ) -> ToleranceEnvelope:
        metadata_value = dict(metadata or {})
        payload = cls._manifest_payload(
            SCHEMA_VERSION, baseline_manifest_hash, tolerances, metadata_value
        )
        envelope = cls(
            schema_version=SCHEMA_VERSION,
            baseline_manifest_hash=baseline_manifest_hash,
            tolerances=dict(tolerances),
            metadata=metadata_value,
            manifest_hash=canonical_sha256(payload),
        )
        envelope._validate_evidence()
        return envelope

    def with_policy(self, policy: ComparisonPolicy) -> ToleranceEnvelope:
        """Bind separately reviewed policy material to this evidence envelope."""

        reviewed = replace(self, comparison_policy=policy)
        reviewed.validate()
        return reviewed

    @staticmethod
    def _manifest_payload(
        schema_version: int,
        baseline_manifest_hash: str,
        tolerances: Mapping[str, NumericTolerance],
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": schema_version,
            "baseline_manifest_hash": baseline_manifest_hash,
            "tolerances": {
                name: tolerance.to_dict()
                for name, tolerance in sorted(tolerances.items())
            },
            "metadata": _thaw_json(metadata),
        }

    def _validate_evidence(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise BundleValidationError(
                f"ToleranceEnvelope.schema_version must equal {SCHEMA_VERSION}"
            )
        _validate_sha256(self.baseline_manifest_hash, "baseline_manifest_hash")
        _require_mapping(self.metadata, "ToleranceEnvelope.metadata")
        for name, tolerance in self.tolerances.items():
            if type(name) is not str or not name:
                raise BundleValidationError(
                    "tolerance names must be non-empty strings"
                )
            if not isinstance(tolerance, NumericTolerance):
                raise BundleValidationError(
                    f"tolerances.{name} must be a NumericTolerance"
                )
            tolerance.validate(name)
            for index, repetition in enumerate(tolerance.repetitions):
                if repetition.baseline_manifest_hash != self.baseline_manifest_hash:
                    raise BundleValidationError(
                        f"tolerances.{name}.repetitions[{index}]."
                        "baseline_manifest_hash does not match envelope baseline"
                    )
        _validate_sha256(self.manifest_hash, "ToleranceEnvelope.manifest_hash")
        expected_hash = canonical_sha256(
            self._manifest_payload(
                self.schema_version,
                self.baseline_manifest_hash,
                self.tolerances,
                self.metadata,
            )
        )
        if self.manifest_hash != expected_hash:
            raise BundleValidationError(
                "ToleranceEnvelope.manifest_hash does not match its canonical payload"
            )

    def validate(self) -> None:
        self._validate_evidence()
        policy = self.comparison_policy
        if not isinstance(policy, ComparisonPolicy):
            raise BundleValidationError(
                "ToleranceEnvelope requires a reviewed ComparisonPolicy"
            )
        policy.validate()
        if policy.baseline_manifest_hash != self.baseline_manifest_hash:
            raise BundleValidationError(
                "ComparisonPolicy.baseline_manifest_hash does not match envelope"
            )
        if policy.tolerance_envelope_hash != self.manifest_hash:
            raise BundleValidationError(
                "ComparisonPolicy.tolerance_envelope_hash does not match envelope"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        policy = self.comparison_policy
        assert isinstance(policy, ComparisonPolicy)
        payload = self._manifest_payload(
            self.schema_version,
            self.baseline_manifest_hash,
            self.tolerances,
            self.metadata,
        )
        payload["manifest_hash"] = self.manifest_hash
        payload["comparison_policy"] = policy.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ToleranceEnvelope:
        mapping = _require_mapping(value, "ToleranceEnvelope")
        _require_exact_fields(
            mapping,
            {
                "schema_version",
                "baseline_manifest_hash",
                "tolerances",
                "metadata",
                "manifest_hash",
                "comparison_policy",
            },
            "ToleranceEnvelope",
        )
        tolerance_data = _require_mapping(mapping["tolerances"], "tolerances")
        envelope = cls(
            schema_version=mapping["schema_version"],  # type: ignore[arg-type]
            baseline_manifest_hash=mapping["baseline_manifest_hash"],  # type: ignore[arg-type]
            tolerances={
                name: NumericTolerance.from_dict(tolerance, name)
                for name, tolerance in tolerance_data.items()
            },
            metadata=dict(
                _require_mapping(mapping["metadata"], "ToleranceEnvelope.metadata")
            ),
            manifest_hash=mapping["manifest_hash"],  # type: ignore[arg-type]
            comparison_policy=ComparisonPolicy.from_dict(
                mapping["comparison_policy"]
            ),
        )
        envelope.validate()
        return envelope

    def write_json(self, path: str | os.PathLike[str]) -> None:
        _write_json_document(path, self.to_dict())

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> ToleranceEnvelope:
        return cls.from_dict(_read_json_document(path, "ToleranceEnvelope"))


@dataclass(frozen=True)
class ComparisonPolicy:
    """Separately reviewed root that pins one baseline and tolerance envelope."""

    schema_version: int
    baseline_manifest_hash: str
    tolerance_envelope_hash: str
    metadata: dict[str, object]
    manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @classmethod
    def create(
        cls,
        *,
        baseline_manifest_hash: str,
        tolerance_envelope_hash: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ComparisonPolicy:
        metadata_value = dict(metadata or {})
        payload = cls._manifest_payload(
            SCHEMA_VERSION,
            baseline_manifest_hash,
            tolerance_envelope_hash,
            metadata_value,
        )
        policy = cls(
            schema_version=SCHEMA_VERSION,
            baseline_manifest_hash=baseline_manifest_hash,
            tolerance_envelope_hash=tolerance_envelope_hash,
            metadata=metadata_value,
            manifest_hash=canonical_sha256(payload),
        )
        policy.validate()
        return policy

    @staticmethod
    def _manifest_payload(
        schema_version: int,
        baseline_manifest_hash: str,
        tolerance_envelope_hash: str,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": schema_version,
            "baseline_manifest_hash": baseline_manifest_hash,
            "tolerance_envelope_hash": tolerance_envelope_hash,
            "metadata": _thaw_json(metadata),
        }

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise BundleValidationError(
                f"ComparisonPolicy.schema_version must equal {SCHEMA_VERSION}"
            )
        _validate_sha256(
            self.baseline_manifest_hash,
            "ComparisonPolicy.baseline_manifest_hash",
        )
        _validate_sha256(
            self.tolerance_envelope_hash,
            "ComparisonPolicy.tolerance_envelope_hash",
        )
        _require_mapping(self.metadata, "ComparisonPolicy.metadata")
        _validate_sha256(self.manifest_hash, "ComparisonPolicy.manifest_hash")
        expected_hash = canonical_sha256(
            self._manifest_payload(
                self.schema_version,
                self.baseline_manifest_hash,
                self.tolerance_envelope_hash,
                self.metadata,
            )
        )
        if self.manifest_hash != expected_hash:
            raise BundleValidationError(
                "ComparisonPolicy.manifest_hash does not match its canonical payload"
            )

    def to_dict(self) -> dict[str, object]:
        payload = self._manifest_payload(
            self.schema_version,
            self.baseline_manifest_hash,
            self.tolerance_envelope_hash,
            self.metadata,
        )
        payload["manifest_hash"] = self.manifest_hash
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ComparisonPolicy:
        mapping = _require_mapping(value, "ComparisonPolicy")
        _require_exact_fields(
            mapping,
            {
                "schema_version",
                "baseline_manifest_hash",
                "tolerance_envelope_hash",
                "metadata",
                "manifest_hash",
            },
            "ComparisonPolicy",
        )
        policy = cls(
            schema_version=mapping["schema_version"],  # type: ignore[arg-type]
            baseline_manifest_hash=mapping["baseline_manifest_hash"],  # type: ignore[arg-type]
            tolerance_envelope_hash=mapping["tolerance_envelope_hash"],  # type: ignore[arg-type]
            metadata=dict(
                _require_mapping(mapping["metadata"], "ComparisonPolicy.metadata")
            ),
            manifest_hash=mapping["manifest_hash"],  # type: ignore[arg-type]
        )
        policy.validate()
        return policy

    def write_json(self, path: str | os.PathLike[str]) -> None:
        _write_json_document(path, self.to_dict())

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> ComparisonPolicy:
        return cls.from_dict(_read_json_document(path, "ComparisonPolicy"))

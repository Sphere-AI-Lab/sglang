from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import (
    COMPARABLE_PROVENANCE_HASH_KEYS,
    BundleValidationError,
    CaseKey,
    NumericTolerance,
    RunBundle,
    ToleranceEnvelope,
)


PERFORMANCE_RATIO_LIMIT = 1.05
THROUGHPUT_FLOOR_RATIO = 0.95
_BUNDLE_REQUEST_ID = "<bundle>"
_MISSING = "<missing>"


@dataclass(frozen=True)
class ComparisonMismatch:
    kind: str
    case_key: CaseKey
    request_id: str
    position: str
    expected: object
    actual: object
    envelope: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "case_key": self.case_key.to_dict(),
            "request_id": self.request_id,
            "position": self.position,
            "expected": self.expected,
            "actual": self.actual,
            "envelope": self.envelope,
        }


@dataclass(frozen=True)
class ComparisonReport:
    expected_case_key: CaseKey
    actual_case_key: CaseKey
    mismatches: tuple[ComparisonMismatch, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "expected_case_key": self.expected_case_key.to_dict(),
            "actual_case_key": self.actual_case_key.to_dict(),
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
        }


def _envelope_dict(envelope: ToleranceEnvelope) -> dict[str, object]:
    try:
        return envelope.to_dict()
    except Exception:
        return {"manifest_hash": getattr(envelope, "manifest_hash", _MISSING)}


def _mismatch(
    *,
    kind: str,
    case_key: CaseKey,
    request_id: str,
    position: str,
    expected: object,
    actual: object,
    envelope: ToleranceEnvelope,
) -> ComparisonMismatch:
    return ComparisonMismatch(
        kind=kind,
        case_key=case_key,
        request_id=request_id,
        position=position,
        expected=expected,
        actual=actual,
        envelope=_envelope_dict(envelope),
    )


def _validation_report(
    expected: RunBundle,
    actual: RunBundle,
    envelope: ToleranceEnvelope,
    *,
    kind: str,
    position: str,
    error: BundleValidationError,
) -> ComparisonReport:
    mismatch = _mismatch(
        kind=kind,
        case_key=actual.case_key,
        request_id=_BUNDLE_REQUEST_ID,
        position=position,
        expected="valid immutable evidence",
        actual=str(error),
        envelope=envelope,
    )
    return ComparisonReport(expected.case_key, actual.case_key, (mismatch,))


def _exact_differences(
    expected: object, actual: object, position: str
) -> list[tuple[str, object, object]]:
    if type(expected) is not type(actual):
        return [(position, expected, actual)]
    if isinstance(expected, Mapping):
        differences: list[tuple[str, object, object]] = []
        expected_keys = set(expected)
        actual_keys = set(actual)  # type: ignore[arg-type]
        for key in sorted(expected_keys | actual_keys):
            child_position = f"{position}.{key}"
            if key not in expected:
                differences.append(
                    (child_position, _MISSING, actual[key])  # type: ignore[index]
                )
            elif key not in actual:
                differences.append((child_position, expected[key], _MISSING))
            else:
                differences.extend(
                    _exact_differences(
                        expected[key],
                        actual[key],  # type: ignore[index]
                        child_position,
                    )
                )
        return differences
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):  # type: ignore[arg-type]
            return [
                (
                    f"{position}.shape",
                    [len(expected)],
                    [len(actual)],  # type: ignore[arg-type]
                )
            ]
        differences = []
        for index, expected_value in enumerate(expected):
            differences.extend(
                _exact_differences(
                    expected_value,
                    actual[index],  # type: ignore[index]
                    f"{position}[{index}]",
                )
            )
        return differences
    if expected != actual:
        return [(position, expected, actual)]
    return []


def _numeric_equal(
    expected: float,
    actual: float,
    tolerance: NumericTolerance | None,
) -> bool:
    if tolerance is None or (tolerance.atol == 0.0 and tolerance.rtol == 0.0):
        return expected == actual
    return math.isclose(
        expected,
        actual,
        abs_tol=tolerance.atol,
        rel_tol=tolerance.rtol,
    )


def _median(samples: Sequence[float]) -> float:
    return float(statistics.median(samples))


def compare_bundles(
    expected: RunBundle,
    actual: RunBundle,
    envelope: ToleranceEnvelope,
) -> ComparisonReport:
    """Compare validated bundles, with exact identity before numeric tolerance."""

    try:
        expected.validate()
    except BundleValidationError as error:
        return _validation_report(
            expected,
            actual,
            envelope,
            kind="invalid_expected_bundle",
            position="validation",
            error=error,
        )
    try:
        actual.validate()
    except BundleValidationError as error:
        return _validation_report(
            expected,
            actual,
            envelope,
            kind="invalid_actual_bundle",
            position="validation",
            error=error,
        )
    try:
        envelope.validate()
    except BundleValidationError as error:
        position = "manifest_hash" if "manifest_hash" in str(error) else "validation"
        return _validation_report(
            expected,
            actual,
            envelope,
            kind="invalid_envelope",
            position=position,
            error=error,
        )

    if envelope.baseline_manifest_hash != expected.manifest_hash:
        mismatch = _mismatch(
            kind="envelope_baseline_mismatch",
            case_key=actual.case_key,
            request_id=_BUNDLE_REQUEST_ID,
            position="baseline_manifest_hash",
            expected=expected.manifest_hash,
            actual=envelope.baseline_manifest_hash,
            envelope=envelope,
        )
        return ComparisonReport(expected.case_key, actual.case_key, (mismatch,))

    mismatches: list[ComparisonMismatch] = []

    # Code and revision are deliberately allowed to differ. All inputs and the
    # hardware/environment identities that make comparison meaningful are not.
    for key in COMPARABLE_PROVENANCE_HASH_KEYS:
        expected_hash = expected.provenance[key]
        actual_hash = actual.provenance[key]
        if expected_hash != actual_hash:
            mismatches.append(
                _mismatch(
                    kind="provenance_mismatch",
                    case_key=actual.case_key,
                    request_id=_BUNDLE_REQUEST_ID,
                    position=f"provenance.{key}",
                    expected=expected_hash,
                    actual=actual_hash,
                    envelope=envelope,
                )
            )

    for field_name in (
        "model",
        "architecture",
        "precision",
        "cuda_graph",
        "scenario",
    ):
        expected_value = getattr(expected.case_key, field_name)
        actual_value = getattr(actual.case_key, field_name)
        if expected_value != actual_value:
            mismatches.append(
                _mismatch(
                    kind="case_key_mismatch",
                    case_key=actual.case_key,
                    request_id=_BUNDLE_REQUEST_ID,
                    position=f"case_key.{field_name}",
                    expected=expected_value,
                    actual=actual_value,
                    envelope=envelope,
                )
            )

    expected_request_ids = set(expected.observations)
    actual_request_ids = set(actual.observations)
    if expected_request_ids != actual_request_ids:
        mismatches.append(
            _mismatch(
                kind="request_set_mismatch",
                case_key=actual.case_key,
                request_id=_BUNDLE_REQUEST_ID,
                position="observations",
                expected=sorted(expected_request_ids),
                actual=sorted(actual_request_ids),
                envelope=envelope,
            )
        )

    common_request_ids = sorted(expected_request_ids & actual_request_ids)

    # Structural, token, state, and error equality always precede floating-point
    # comparison. A tolerance can never hide a token or state divergence.
    for request_id in common_request_ids:
        expected_observation = expected.observations[request_id]
        actual_observation = actual.observations[request_id]

        if len(expected_observation.output_ids) != len(actual_observation.output_ids):
            mismatches.append(
                _mismatch(
                    kind="shape_mismatch",
                    case_key=actual.case_key,
                    request_id=request_id,
                    position="output_ids.shape",
                    expected=[len(expected_observation.output_ids)],
                    actual=[len(actual_observation.output_ids)],
                    envelope=envelope,
                )
            )
        else:
            for position, (expected_token, actual_token) in enumerate(
                zip(expected_observation.output_ids, actual_observation.output_ids)
            ):
                if expected_token != actual_token:
                    mismatches.append(
                        _mismatch(
                            kind="token_mismatch",
                            case_key=actual.case_key,
                            request_id=request_id,
                            position=f"output_ids[{position}]",
                            expected=expected_token,
                            actual=actual_token,
                            envelope=envelope,
                        )
                    )

        if expected_observation.text != actual_observation.text:
            mismatches.append(
                _mismatch(
                    kind="text_mismatch",
                    case_key=actual.case_key,
                    request_id=request_id,
                    position="text",
                    expected=expected_observation.text,
                    actual=actual_observation.text,
                    envelope=envelope,
                )
            )

        if len(expected_observation.token_logprobs) != len(
            actual_observation.token_logprobs
        ):
            mismatches.append(
                _mismatch(
                    kind="shape_mismatch",
                    case_key=actual.case_key,
                    request_id=request_id,
                    position="token_logprobs.shape",
                    expected=[len(expected_observation.token_logprobs)],
                    actual=[len(actual_observation.token_logprobs)],
                    envelope=envelope,
                )
            )

        expected_logit_names = set(expected_observation.selected_logits)
        actual_logit_names = set(actual_observation.selected_logits)
        if expected_logit_names != actual_logit_names:
            mismatches.append(
                _mismatch(
                    kind="shape_mismatch",
                    case_key=actual.case_key,
                    request_id=request_id,
                    position="selected_logits.keys",
                    expected=sorted(expected_logit_names),
                    actual=sorted(actual_logit_names),
                    envelope=envelope,
                )
            )
        for name in sorted(expected_logit_names & actual_logit_names):
            expected_values = expected_observation.selected_logits[name]
            actual_values = actual_observation.selected_logits[name]
            if len(expected_values) != len(actual_values):
                mismatches.append(
                    _mismatch(
                        kind="shape_mismatch",
                        case_key=actual.case_key,
                        request_id=request_id,
                        position=f"selected_logits.{name}.shape",
                        expected=[len(expected_values)],
                        actual=[len(actual_values)],
                        envelope=envelope,
                    )
                )

        for position, expected_value, actual_value in _exact_differences(
            expected_observation.adapter_state,
            actual_observation.adapter_state,
            "adapter_state",
        ):
            mismatches.append(
                _mismatch(
                    kind="adapter_state_mismatch",
                    case_key=actual.case_key,
                    request_id=request_id,
                    position=position,
                    expected=expected_value,
                    actual=actual_value,
                    envelope=envelope,
                )
            )

        for position, expected_value, actual_value in _exact_differences(
            expected_observation.error,
            actual_observation.error,
            "error",
        ):
            mismatches.append(
                _mismatch(
                    kind="error_mismatch",
                    case_key=actual.case_key,
                    request_id=request_id,
                    position=position,
                    expected=expected_value,
                    actual=actual_value,
                    envelope=envelope,
                )
            )

    used_tolerances: set[str] = set()
    for request_id in common_request_ids:
        expected_observation = expected.observations[request_id]
        actual_observation = actual.observations[request_id]

        tolerance_name = "token_logprobs"
        tolerance = envelope.tolerances.get(tolerance_name)
        if tolerance is not None:
            used_tolerances.add(tolerance_name)
        if len(expected_observation.token_logprobs) == len(
            actual_observation.token_logprobs
        ):
            for position, (expected_value, actual_value) in enumerate(
                zip(
                    expected_observation.token_logprobs,
                    actual_observation.token_logprobs,
                )
            ):
                if not _numeric_equal(expected_value, actual_value, tolerance):
                    mismatches.append(
                        _mismatch(
                            kind="numeric_mismatch",
                            case_key=actual.case_key,
                            request_id=request_id,
                            position=f"token_logprobs[{position}]",
                            expected=expected_value,
                            actual=actual_value,
                            envelope=envelope,
                        )
                    )

        for name in sorted(
            set(expected_observation.selected_logits)
            & set(actual_observation.selected_logits)
        ):
            tolerance_name = f"selected_logits.{name}"
            tolerance = envelope.tolerances.get(tolerance_name)
            if tolerance is not None:
                used_tolerances.add(tolerance_name)
            expected_values = expected_observation.selected_logits[name]
            actual_values = actual_observation.selected_logits[name]
            if len(expected_values) != len(actual_values):
                continue
            for position, (expected_value, actual_value) in enumerate(
                zip(expected_values, actual_values)
            ):
                if not _numeric_equal(expected_value, actual_value, tolerance):
                    mismatches.append(
                        _mismatch(
                            kind="numeric_mismatch",
                            case_key=actual.case_key,
                            request_id=request_id,
                            position=f"selected_logits.{name}[{position}]",
                            expected=expected_value,
                            actual=actual_value,
                            envelope=envelope,
                        )
                    )

    for tolerance_name in sorted(set(envelope.tolerances) - used_tolerances):
        mismatches.append(
            _mismatch(
                kind="unused_tolerance",
                case_key=actual.case_key,
                request_id=_BUNDLE_REQUEST_ID,
                position=f"tolerances.{tolerance_name}",
                expected="a numeric quantity present in both bundles",
                actual=tolerance_name,
                envelope=envelope,
            )
        )

    median_metrics = (
        ("startup_seconds", False),
        ("latency_seconds", False),
        ("throughput_tokens_per_second", True),
    )
    for metric_name, higher_is_better in median_metrics:
        expected_samples = getattr(expected.performance, metric_name)
        actual_samples = getattr(actual.performance, metric_name)
        if bool(expected_samples) != bool(actual_samples):
            mismatches.append(
                _mismatch(
                    kind="performance_shape_mismatch",
                    case_key=actual.case_key,
                    request_id=_BUNDLE_REQUEST_ID,
                    position=f"performance.{metric_name}",
                    expected=len(expected_samples),
                    actual=len(actual_samples),
                    envelope=envelope,
                )
            )
            continue
        if not expected_samples:
            continue
        expected_median = _median(expected_samples)
        actual_median = _median(actual_samples)
        if higher_is_better:
            regressed = actual_median < expected_median * THROUGHPUT_FLOOR_RATIO
        else:
            regressed = actual_median > expected_median * PERFORMANCE_RATIO_LIMIT
        if regressed:
            mismatches.append(
                _mismatch(
                    kind="performance_regression",
                    case_key=actual.case_key,
                    request_id=_BUNDLE_REQUEST_ID,
                    position=f"performance.{metric_name}.median",
                    expected=expected_median,
                    actual=actual_median,
                    envelope=envelope,
                )
            )

    for metric_name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        expected_value = getattr(expected.performance, metric_name)
        actual_value = getattr(actual.performance, metric_name)
        if actual_value > expected_value * PERFORMANCE_RATIO_LIMIT:
            mismatches.append(
                _mismatch(
                    kind="performance_regression",
                    case_key=actual.case_key,
                    request_id=_BUNDLE_REQUEST_ID,
                    position=f"performance.{metric_name}",
                    expected=expected_value,
                    actual=actual_value,
                    envelope=envelope,
                )
            )

    return ComparisonReport(
        expected_case_key=expected.case_key,
        actual_case_key=actual.case_key,
        mismatches=tuple(mismatches),
    )

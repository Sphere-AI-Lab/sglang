from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from .scenarios import ScenarioContractError, comparable_lease_observations
from .schema import (
    COMPARABLE_PROVENANCE_HASH_KEYS,
    BundleValidationError,
    CaseKey,
    NumericTolerance,
    RunBundle,
    ToleranceEnvelope,
    canonical_sha256,
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
    performance: dict[str, object] | None = None

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "expected_case_key": self.expected_case_key.to_dict(),
            "actual_case_key": self.actual_case_key.to_dict(),
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "performance": self.performance,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


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


def comparable_adapter_states(states):
    """Rename opaque IDs once per run, preserving every identity relationship.

    Call with states in declared request order. A new ID gets a new label even
    when its adapter name was seen before; reuse and churn remain observable.
    Raw evidence is never modified.
    """
    ids = {}

    def label(value):
        if value not in ids:
            ids[value] = f"runtime-id-{len(ids)}"
        return ids[value]

    def record(value):
        return None if value is None else dict(value, id=label(value["id"]))

    result = {}
    for request, state in states.items():
        normalized = dict(state)
        normalized["registered"] = [record(value) for value in state["registered"]]
        normalized["active"] = record(state["active"])
        normalized["staged"] = record(state["staged"])
        normalized["tombstoned"] = [record(value) for value in state["tombstoned"]]
        normalized["cache_identity"] = {
            name: label(state["cache_identity"][name])
            for name in sorted(state["cache_identity"])
        }
        result[request] = normalized
    return result


def comparable_adapter_error(error, state, comparable_state):
    """Align only the known expected ID in the native wrong-ID diagnostic."""
    active = state["active"]
    if (
        error is not None
        and active is not None
        and (error["kind"] == "product_rejection" and error["code"] == "wrong_id")
    ):
        prefix = "Requested adapter_id 'wrong-id' does not match expected adapter_id '"
        if error["message"] == prefix + active["id"] + "'":
            return dict(
                error, message=("verified-wrong-id", comparable_state["active"]["id"])
            )
    return error


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
    if tolerance is None or (
        tolerance.observed_atol == 0.0 and tolerance.observed_rtol == 0.0
    ):
        return expected == actual
    return math.isclose(
        expected,
        actual,
        abs_tol=tolerance.observed_atol,
        rel_tol=tolerance.observed_rtol,
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
        expected_observations = comparable_lease_observations(expected.observations)
    except (BundleValidationError, ScenarioContractError) as error:
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
        actual_observations = comparable_lease_observations(actual.observations)
    except (BundleValidationError, ScenarioContractError) as error:
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
    performance_identity_valid = True

    # Code and revision are deliberately allowed to differ. All inputs and the
    # hardware/environment identities that make comparison meaningful are not.
    for key in COMPARABLE_PROVENANCE_HASH_KEYS:
        expected_hash = expected.provenance[key]
        actual_hash = actual.provenance[key]
        if expected_hash != actual_hash:
            performance_identity_valid = False
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
        "mode",
        "cuda_graph",
        "scenario",
    ):
        expected_value = getattr(expected.case_key, field_name)
        actual_value = getattr(actual.case_key, field_name)
        if expected_value != actual_value:
            performance_identity_valid = False
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

    if expected.manifest["request_order"] != actual.manifest["request_order"]:
        performance_identity_valid = False
        mismatches.append(
            _mismatch(
                kind="request_order_mismatch",
                case_key=actual.case_key,
                request_id=_BUNDLE_REQUEST_ID,
                position="manifest.request_order",
                expected=list(expected.manifest["request_order"]),
                actual=list(actual.manifest["request_order"]),
                envelope=envelope,
            )
        )

    expected_request_ids = set(expected.observations)
    actual_request_ids = set(actual.observations)
    if expected_request_ids != actual_request_ids:
        performance_identity_valid = False
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
    expected_states = comparable_adapter_states(
        {
            name: expected.observations[name].adapter_state
            for name in expected.manifest["request_order"]
        }
    )
    actual_states = comparable_adapter_states(
        {
            name: actual.observations[name].adapter_state
            for name in actual.manifest["request_order"]
        }
    )

    # Structural, token, state, and error equality always precede floating-point
    # comparison. A tolerance can never hide a token or state divergence.
    for request_id in common_request_ids:
        expected_observation = expected_observations[request_id]
        actual_observation = actual_observations[request_id]

        for field in ("request_output_lengths", "request_texts"):
            expected_value = getattr(expected_observation, field)
            actual_value = getattr(actual_observation, field)
            if expected_value != actual_value:
                mismatches.append(
                    _mismatch(
                        kind=(
                            "shape_mismatch"
                            if field == "request_output_lengths"
                            else "text_mismatch"
                        ),
                        case_key=actual.case_key,
                        request_id=request_id,
                        position=field,
                        expected=list(expected_value),
                        actual=list(actual_value),
                        envelope=envelope,
                    )
                )

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
                continue
            expected_token_ids = expected_observation.selected_token_ids[name]
            actual_token_ids = actual_observation.selected_token_ids[name]
            for position, (expected_token, actual_token) in enumerate(
                zip(expected_token_ids, actual_token_ids)
            ):
                if expected_token != actual_token:
                    mismatches.append(
                        _mismatch(
                            kind="token_mismatch",
                            case_key=actual.case_key,
                            request_id=request_id,
                            position=f"selected_token_ids.{name}[{position}]",
                            expected=expected_token,
                            actual=actual_token,
                            envelope=envelope,
                        )
                    )

        for position, expected_value, actual_value in _exact_differences(
            expected_states[request_id],
            actual_states[request_id],
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
            comparable_adapter_error(
                expected_observation.error,
                expected_observation.adapter_state,
                expected_states[request_id],
            ),
            comparable_adapter_error(
                actual_observation.error,
                actual_observation.adapter_state,
                actual_states[request_id],
            ),
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
        expected_observation = expected_observations[request_id]
        actual_observation = actual_observations[request_id]

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

    if expected.performance.procedure_hash != actual.performance.procedure_hash:
        performance_identity_valid = False
        mismatches.append(
            _mismatch(
                kind="performance_identity_mismatch",
                case_key=actual.case_key,
                request_id=_BUNDLE_REQUEST_ID,
                position="performance.procedure_hash",
                expected=expected.performance.procedure_hash,
                actual=actual.performance.procedure_hash,
                envelope=envelope,
            )
        )

    # Ratios are evidence only when case, provenance, placement, and benchmark
    # procedure identities all match. Identity mismatches above remain failures.
    if any(
        "performance_procedure" in bundle.manifest["metadata"]
        for bundle in (expected, actual)
    ):
        from .aggregate import _runtime_identity

        try:
            if expected.case_key.mode == actual.case_key.mode == "base":
                # Base/component diagnostics remain comparable outside the
                # native-only final inventory (including dense TP2 evidence).
                identity_matches = all(
                    canonical_sha256(
                        bundle.manifest["metadata"]["performance_procedure"]
                    )
                    == bundle.performance.procedure_hash
                    and bundle.manifest["metadata"]["performance_procedure"][
                        "memory_boundary_hash"
                    ]
                    == bundle.provenance["metadata"]["memory_boundary_hash"]
                    for bundle in (expected, actual)
                )
            else:
                identity_matches = _runtime_identity(expected) == _runtime_identity(
                    actual
                )
            detail = "execution metadata differs"
        except (ValueError, TypeError, KeyError) as error:
            identity_matches, detail = False, str(error)
        if not identity_matches:
            performance_identity_valid = False
            mismatches.append(
                _mismatch(
                    kind="performance_identity_mismatch",
                    case_key=actual.case_key,
                    request_id=_BUNDLE_REQUEST_ID,
                    position="metadata.execution_identity",
                    expected="matching validated execution identity",
                    actual=detail,
                    envelope=envelope,
                )
            )
    performance_evidence = None
    if performance_identity_valid:
        performance_evidence = {}
        median_metrics = (
            ("startup_seconds", False),
            ("latency_seconds", False),
            ("throughput_tokens_per_second", True),
        )
        for metric_name, higher_is_better in median_metrics:
            expected_median = _median(getattr(expected.performance, metric_name))
            actual_median = _median(getattr(actual.performance, metric_name))
            performance_evidence[metric_name] = {
                "reference": expected_median,
                "candidate": actual_median,
                "ratio": (
                    None if expected_median == 0 else actual_median / expected_median
                ),
            }
            if higher_is_better:
                regressed = actual_median < expected_median * THROUGHPUT_FLOOR_RATIO
            else:
                regressed = (
                    False  # Startup/latency are evidence, with no approved gate.
                )
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
            expected_value = _median(getattr(expected.performance, metric_name))
            actual_value = _median(getattr(actual.performance, metric_name))
            performance_evidence[metric_name] = {
                "reference": expected_value,
                "candidate": actual_value,
                "ratio": None if expected_value == 0 else actual_value / expected_value,
            }
            if actual_value > expected_value * PERFORMANCE_RATIO_LIMIT:
                mismatches.append(
                    _mismatch(
                        kind="performance_regression",
                        case_key=actual.case_key,
                        request_id=_BUNDLE_REQUEST_ID,
                        position=f"performance.{metric_name}.median",
                        expected=expected_value,
                        actual=actual_value,
                        envelope=envelope,
                    )
                )

    return ComparisonReport(
        expected_case_key=expected.case_key,
        actual_case_key=actual.case_key,
        mismatches=tuple(mismatches),
        performance=performance_evidence,
    )

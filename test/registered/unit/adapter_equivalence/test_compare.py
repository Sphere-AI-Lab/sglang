import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence.compare import compare_bundles
from adapter_equivalence.schema import (
    PROVENANCE_HASH_KEYS,
    SCHEMA_VERSION,
    BaselineRepetition,
    BundleValidationError,
    CaseKey,
    ComparisonPolicy,
    NumericTolerance,
    Observation,
    PerformanceMetrics,
    RunBundle,
    ToleranceEnvelope,
    canonical_sha256,
)


def _sha(character: str) -> str:
    return character * 64


def _case() -> CaseKey:
    return CaseKey(
        model="Qwen/Qwen3-4B-Instruct-2507",
        architecture="dense",
        precision="bf16",
        revision="a" * 40,
        mode="canonical_oft",
        cuda_graph=True,
        scenario="dynamic-load-switch",
    )


def _observation() -> Observation:
    return Observation(
        output_ids=(101, 202, 303),
        text="alpha beta gamma",
        token_logprobs=(-0.1, -0.2, -0.3),
        selected_logits={"decoder.output": (1.0, 2.0, 3.0)},
        adapter_state={
            "active": {"id": "adapter-a", "version": 7},
            "staged": {"adapter-b": 8},
        },
        error=None,
    )


def _provenance(case: CaseKey) -> dict[str, object]:
    return {
        "git_sha": case.revision,
        "dirty": False,
        "code_hash": _sha("0"),
        "checkpoint_hash": _sha("1"),
        "adapter_hash": _sha("2"),
        "tokenizer_hash": _sha("3"),
        "scenario_hash": _sha("4"),
        "environment_hash": _sha("5"),
        "hardware_hash": _sha("6"),
        "metadata": {"packages": ["torch==2.8.0"]},
    }


def _bundle() -> RunBundle:
    case = _case()
    provenance = _provenance(case)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_key": case.to_dict(),
        "provenance_hashes": {
            key: provenance[key] for key in PROVENANCE_HASH_KEYS
        },
        "performance_procedure_hash": _sha("7"),
        "metadata": {
            "server_args": ["--peft-method", "oft"],
            "request_order": ["req-0"],
            "seed": 1729,
        },
    }
    return RunBundle.create(
        case_key=case,
        manifest=manifest,
        provenance=provenance,
        observations={"req-0": _observation()},
        performance=PerformanceMetrics(
            procedure_hash=_sha("7"),
            startup_seconds=(10.0, 10.2, 9.8),
            latency_seconds=(0.20, 0.21, 0.19),
            throughput_tokens_per_second=(100.0, 101.0, 99.0),
            peak_allocated_bytes=(1_000, 990, 995),
            peak_reserved_bytes=(1_200, 1_190, 1_195),
        ),
        completion={
            "status": "complete",
            "exit_code": 0,
            "metadata": {"job_id": "17.0"},
        },
    )


def _round_trip(tmp_path, name: str, bundle: RunBundle) -> RunBundle:
    path = tmp_path / f"{name}.json"
    bundle.write_json(path)
    return RunBundle.read_json(path)


def _envelope(bundle: RunBundle, **tolerances: NumericTolerance) -> ToleranceEnvelope:
    return ToleranceEnvelope.create(
        baseline_manifest_hash=bundle.manifest_hash,
        tolerances=tolerances,
    )


def _policy(
    bundle: RunBundle, envelope: ToleranceEnvelope | None = None
) -> ComparisonPolicy:
    envelope = envelope or _envelope(bundle)
    return ComparisonPolicy.create(
        baseline_manifest_hash=bundle.manifest_hash,
        tolerance_envelope_hash=envelope.manifest_hash,
    )


def _comparison_inputs(
    bundle: RunBundle, **tolerances: NumericTolerance
) -> tuple[ToleranceEnvelope, ComparisonPolicy]:
    envelope = _envelope(bundle, **tolerances)
    return envelope, _policy(bundle, envelope)


def _baseline_repetitions(
    baseline_manifest_hash: str, *values: tuple[float, ...]
) -> tuple[BaselineRepetition, ...]:
    return tuple(
        BaselineRepetition(
            baseline_manifest_hash=baseline_manifest_hash,
            bundle_hash=_sha(str(index + 1)),
            values=samples,
        )
        for index, samples in enumerate(values)
    )


def _tolerance(bundle: RunBundle, *values: tuple[float, ...]) -> NumericTolerance:
    return NumericTolerance.create(
        repetitions=_baseline_repetitions(bundle.manifest_hash, *values)
    )


def _replace_observation(bundle: RunBundle, **changes: object) -> RunBundle:
    observations = dict(bundle.observations)
    observations["req-0"] = replace(observations["req-0"], **changes)
    return replace(bundle, observations=observations)


def _replace_provenance_hash(
    bundle: RunBundle, key: str, value: str
) -> RunBundle:
    provenance = dict(bundle.provenance)
    provenance.pop("manifest_hash")
    provenance[key] = value
    manifest = dict(bundle.manifest)
    manifest["provenance_hashes"] = {
        hash_key: provenance[hash_key] for hash_key in PROVENANCE_HASH_KEYS
    }
    return RunBundle.create(
        case_key=bundle.case_key,
        manifest=manifest,
        provenance=provenance,
        observations=bundle.observations,
        performance=bundle.performance,
        completion=bundle.completion,
    )


def _rebuild_bundle(
    bundle: RunBundle,
    *,
    case_key: CaseKey | None = None,
    provenance_changes: dict[str, object] | None = None,
    performance: PerformanceMetrics | None = None,
) -> RunBundle:
    case_key = case_key or bundle.case_key
    provenance = dict(bundle.provenance)
    provenance.pop("manifest_hash")
    provenance.update(provenance_changes or {})
    manifest = dict(bundle.manifest)
    manifest["case_key"] = case_key.to_dict()
    manifest["provenance_hashes"] = {
        hash_key: provenance[hash_key] for hash_key in PROVENANCE_HASH_KEYS
    }
    selected_performance = performance or bundle.performance
    manifest["performance_procedure_hash"] = selected_performance.procedure_hash
    return RunBundle.create(
        case_key=case_key,
        manifest=manifest,
        provenance=provenance,
        observations=bundle.observations,
        performance=selected_performance,
        completion=bundle.completion,
    )


def test_identical_serialized_bundles_compare_equal(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert report.passed
    assert report.mismatches == ()


def test_token_mismatch_reports_request_and_position(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, output_ids=(101, 999, 303))
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "token_mismatch"
    assert mismatch.case_key == actual.case_key
    assert mismatch.request_id == "req-0"
    assert mismatch.position == "output_ids[1]"
    assert mismatch.expected == 202
    assert mismatch.actual == 999
    assert mismatch.envelope["manifest_hash"] == _envelope(expected).manifest_hash


def test_token_shape_mismatch_is_exact(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected,
        output_ids=(101, 202, 303, 404),
        token_logprobs=(-0.1, -0.2, -0.3, -0.4),
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "shape_mismatch"
    assert mismatch.position == "output_ids.shape"
    assert mismatch.expected == [3]
    assert mismatch.actual == [4]


def test_numeric_dtype_mismatch_is_rejected_from_serialized_bundle(tmp_path):
    bundle = _bundle()
    path = tmp_path / "dtype-mismatch.json"
    bundle.write_json(path)
    payload = json.loads(path.read_text())
    payload["observations"]["req-0"]["token_logprobs"][0] = 0
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="token_logprobs.*float"):
        RunBundle.read_json(path)


def test_adapter_version_mismatch_reports_state_path(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    state = {
        "active": {"id": "adapter-a", "version": 8},
        "staged": {"adapter-b": 8},
    }
    actual = _round_trip(
        tmp_path,
        "actual",
        _replace_observation(expected, adapter_state=state),
    )

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "adapter_state_mismatch"
    assert mismatch.position == "adapter_state.active.version"
    assert mismatch.expected == 7
    assert mismatch.actual == 8


def test_error_mismatch_reports_error_path(tmp_path):
    expected = _replace_observation(
        _bundle(), error={"code": "invalid_adapter", "status": 400}
    )
    actual = _replace_observation(
        expected, error={"code": "adapter_not_found", "status": 400}
    )
    expected = _round_trip(tmp_path, "expected", expected)
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "error_mismatch"
    assert mismatch.position == "error.code"
    assert mismatch.expected == "invalid_adapter"
    assert mismatch.actual == "adapter_not_found"


def test_missing_provenance_is_rejected_from_serialized_bundle(tmp_path):
    path = tmp_path / "missing-provenance.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    del payload["provenance"]
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="missing fields.*provenance"):
        RunBundle.read_json(path)


def test_manifest_hash_mismatch_is_rejected_from_serialized_bundle(tmp_path):
    path = tmp_path / "bad-manifest-hash.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["metadata"]["seed"] = 2718
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="manifest_hash"):
        RunBundle.read_json(path)


def test_unknown_top_level_schema_field_is_rejected(tmp_path):
    path = tmp_path / "unknown-field.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["future_field"] = "must not be silently accepted"
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="unknown fields.*future_field"):
        RunBundle.read_json(path)


def test_unknown_nested_schema_field_is_rejected_from_serialized_bundle(tmp_path):
    path = tmp_path / "unknown-nested-field.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["performance"]["future_field"] = "must live in typed metadata"
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError, match="performance: unknown fields.*future_field"
    ):
        RunBundle.read_json(path)


def test_unknown_manifest_field_is_rejected_even_when_rehashed(tmp_path):
    path = tmp_path / "unknown-manifest-field.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["future_field"] = "not metadata"
    payload["manifest_hash"] = canonical_sha256(payload["manifest"])
    payload["provenance"]["manifest_hash"] = payload["manifest_hash"]
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError, match="manifest: unknown fields.*future_field"
    ):
        RunBundle.read_json(path)


def test_duplicate_json_object_key_is_rejected_before_validation(tmp_path):
    path = tmp_path / "duplicate-key.json"
    _bundle().write_json(path)
    serialized = path.read_text()
    serialized = serialized.replace(
        '  "schema_version": 1\n}',
        '  "schema_version": 1,\n  "schema_version": 1\n}',
        1,
    )
    path.write_text(serialized)

    with pytest.raises(BundleValidationError, match="duplicate object key.*schema_version"):
        RunBundle.read_json(path)


def test_boolean_is_not_accepted_as_an_integer_in_serialized_bundle(tmp_path):
    path = tmp_path / "boolean-token-id.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["observations"]["req-0"]["output_ids"][0] = True
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="output_ids entries must be integers"):
        RunBundle.read_json(path)


def test_incomplete_completion_marker_is_rejected(tmp_path):
    path = tmp_path / "incomplete.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["completion"]["status"] = "failed"
    payload["completion_hash"] = canonical_sha256(payload["completion"])
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="completion.status.*complete"):
        RunBundle.read_json(path)


def test_provenance_hash_mismatch_precedes_observation_comparison(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_provenance_hash(expected, "checkpoint_hash", _sha("9"))
    actual = _replace_observation(actual, output_ids=(999, 202, 303))
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    assert report.mismatches[0].kind == "provenance_mismatch"
    assert report.mismatches[0].position == "provenance.checkpoint_hash"


def test_revision_mode_and_code_hash_differences_are_allowed(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    candidate_case = replace(
        expected.case_key,
        revision="b" * 40,
        mode="native_lora",
    )
    actual = _rebuild_bundle(
        expected,
        case_key=candidate_case,
        provenance_changes={
            "git_sha": candidate_case.revision,
            "code_hash": _sha("9"),
        },
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert report.passed


def test_multiple_mismatch_order_and_report_serialization_are_deterministic(
    tmp_path,
):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual_case = replace(expected.case_key, model="Qwen/other-model")
    actual = _rebuild_bundle(
        expected,
        case_key=actual_case,
        provenance_changes={"checkpoint_hash": _sha("9")},
    )
    actual = _replace_observation(
        actual,
        output_ids=(101, 999, 303),
        text="different text",
        token_logprobs=(-0.1, -0.9, -0.3),
        selected_logits={"decoder.output": (1.0, 9.0, 3.0)},
        adapter_state={
            "active": {"id": "adapter-a", "version": 8},
            "staged": {"adapter-b": 8},
        },
        error={"code": "unexpected", "status": 500},
    )
    actual = _round_trip(tmp_path, "actual", actual)
    envelope, policy = _comparison_inputs(expected)

    first = compare_bundles(expected, actual, envelope, policy)
    second = compare_bundles(expected, actual, envelope, policy)

    assert [mismatch.position for mismatch in first.mismatches] == [
        "provenance.checkpoint_hash",
        "case_key.model",
        "output_ids[1]",
        "text",
        "adapter_state.active.version",
        "error",
        "token_logprobs[1]",
        "selected_logits.decoder.output[1]",
    ]
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_logprob_mismatch_fails_when_envelope_is_exact(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, token_logprobs=(-0.1, -0.2001, -0.3)
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "numeric_mismatch"
    assert mismatch.position == "token_logprobs[1]"


def test_selected_logit_shape_mismatch_is_exact(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, selected_logits={"decoder.output": (1.0, 2.0)}
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    assert report.mismatches[0].kind == "shape_mismatch"
    assert report.mismatches[0].position == "selected_logits.decoder.output.shape"


def test_selected_logit_dtype_mismatch_is_rejected_from_serialized_bundle(tmp_path):
    path = tmp_path / "selected-logit-dtype.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["observations"]["req-0"]["selected_logits"]["decoder.output"][0] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="selected_logits.*float"):
        RunBundle.read_json(path)


def test_selected_logit_uses_exact_then_reviewed_tolerance_path(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, selected_logits={"decoder.output": (1.0, 2.0001, 3.0)}
    )
    actual = _round_trip(tmp_path, "actual", actual)

    exact_report = compare_bundles(
        expected, actual, *_comparison_inputs(expected)
    )
    envelope, policy = _comparison_inputs(
        expected,
        **{
            "selected_logits.decoder.output": _tolerance(
                expected,
                (1.0, 2.0, 3.0),
                (1.0, 2.0001, 3.0),
                (1.0, 2.0, 3.0),
            )
        },
    )
    tolerant_report = compare_bundles(expected, actual, envelope, policy)

    assert not exact_report.passed
    assert exact_report.mismatches[0].position == (
        "selected_logits.decoder.output[1]"
    )
    assert tolerant_report.passed


def test_named_unchanged_baseline_tolerance_allows_numeric_drift(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, token_logprobs=(-0.1, -0.2001, -0.3)
    )
    actual = _round_trip(tmp_path, "actual", actual)
    envelope, policy = _comparison_inputs(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )

    report = compare_bundles(expected, actual, envelope, policy)

    assert report.passed


def test_token_divergence_fails_even_with_wide_numeric_tolerance(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, output_ids=(101, 999, 303))
    actual = _round_trip(tmp_path, "actual", actual)
    envelope, policy = _comparison_inputs(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (100.0, 100.0, 100.0),
            (-100.0, -100.0, -100.0),
        ),
    )

    report = compare_bundles(expected, actual, envelope, policy)

    assert not report.passed
    assert report.mismatches[0].kind == "token_mismatch"


def test_peak_memory_regression_fails_at_five_percent_ratio(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(
        expected.performance, peak_allocated_bytes=(1_051, 1_040, 1_030)
    )
    actual = _round_trip(
        tmp_path,
        "actual",
        _rebuild_bundle(expected, performance=performance),
    )

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "performance_regression"
    assert mismatch.position == "performance.peak_allocated_bytes.maximum"
    assert mismatch.expected == 1_000
    assert mismatch.actual == 1_051


def test_throughput_at_five_percent_floor_passes(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(
        expected.performance,
        throughput_tokens_per_second=(95.0, 95.0, 95.0),
    )
    actual = _round_trip(
        tmp_path, "actual", replace(expected, performance=performance)
    )

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert report.passed


def test_throughput_below_five_percent_floor_fails(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(
        expected.performance,
        throughput_tokens_per_second=(94.9, 94.9, 94.9),
    )
    actual = _round_trip(
        tmp_path, "actual", replace(expected, performance=performance)
    )

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    assert report.mismatches[0].position == (
        "performance.throughput_tokens_per_second.median"
    )


def test_one_performance_repetition_is_rejected_from_serialized_bundle(tmp_path):
    path = tmp_path / "one-performance-repetition.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["performance"]["latency_seconds"] = [0.2]
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="performance.latency_seconds must contain exactly 3 post-warm-up repetitions",
    ):
        RunBundle.read_json(path)


def test_performance_procedure_identity_is_manifest_bound(tmp_path):
    path = tmp_path / "unbound-performance-procedure.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["performance"]["procedure_hash"] = _sha("8")
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="performance.procedure_hash does not match manifest",
    ):
        RunBundle.read_json(path)


def test_performance_procedure_mismatch_withholds_regression_ratios(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(
        expected.performance,
        procedure_hash=_sha("8"),
        throughput_tokens_per_second=(1.0, 1.0, 1.0),
        peak_allocated_bytes=(10_000, 10_000, 10_000),
    )
    actual = _round_trip(
        tmp_path,
        "actual",
        _rebuild_bundle(expected, performance=performance),
    )

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert [mismatch.kind for mismatch in report.mismatches] == [
        "performance_identity_mismatch"
    ]
    assert report.mismatches[0].position == "performance.procedure_hash"


@pytest.mark.parametrize("identity_key", ["environment_hash", "hardware_hash"])
def test_placement_identity_mismatch_withholds_regression_ratios(
    tmp_path, identity_key
):
    expected = _round_trip(tmp_path, "expected", _bundle())
    regressed_performance = replace(
        expected.performance,
        throughput_tokens_per_second=(1.0, 1.0, 1.0),
        peak_allocated_bytes=(10_000, 10_000, 10_000),
    )
    actual = _rebuild_bundle(
        expected,
        provenance_changes={identity_key: _sha("9")},
        performance=regressed_performance,
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, *_comparison_inputs(expected))

    assert not report.passed
    assert report.mismatches[0].kind == "provenance_mismatch"
    assert report.mismatches[0].position == f"provenance.{identity_key}"
    assert all(
        mismatch.kind != "performance_regression"
        for mismatch in report.mismatches
    )


def test_widened_tolerance_with_stale_envelope_hash_invalidates_comparison(
    tmp_path,
):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    envelope, policy = _comparison_inputs(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )
    widened = replace(
        envelope,
        tolerances={
            "token_logprobs": _tolerance(
                expected,
                (-0.1, -0.2, -0.3),
                (1.0, 1.0, 1.0),
                (-1.0, -1.0, -1.0),
            )
        },
    )

    report = compare_bundles(expected, actual, widened, policy)

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "invalid_envelope"
    assert mismatch.position == "manifest_hash"


def test_serialized_tolerance_rejects_bounds_not_derived_from_evidence(tmp_path):
    bundle = _bundle()
    envelope = _envelope(
        bundle,
        token_logprobs=_tolerance(
            bundle,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )
    path = tmp_path / "fabricated-bounds.json"
    envelope.write_json(path)
    payload = json.loads(path.read_text())
    payload["tolerances"]["token_logprobs"]["observed_atol"] = 100.0
    manifest_payload = dict(payload)
    manifest_payload.pop("manifest_hash")
    payload["manifest_hash"] = canonical_sha256(manifest_payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="observed_atol does not match unchanged-baseline evidence",
    ):
        ToleranceEnvelope.read_json(path)


def test_serialized_tolerance_rejects_repetition_from_another_manifest(tmp_path):
    bundle = _bundle()
    envelope = _envelope(
        bundle,
        token_logprobs=_tolerance(
            bundle,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )
    path = tmp_path / "changed-baseline-repetition.json"
    envelope.write_json(path)
    payload = json.loads(path.read_text())
    repetition = payload["tolerances"]["token_logprobs"]["repetitions"][0]
    repetition["baseline_manifest_hash"] = _sha("f")
    manifest_payload = dict(payload)
    manifest_payload.pop("manifest_hash")
    payload["manifest_hash"] = canonical_sha256(manifest_payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="baseline_manifest_hash does not match envelope baseline",
    ):
        ToleranceEnvelope.read_json(path)


def test_reviewed_policy_rejects_fabricated_repetition_evidence(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    reviewed_envelope = _envelope(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )
    reviewed_policy = _policy(expected, reviewed_envelope)
    fabricated = ToleranceEnvelope.create(
        baseline_manifest_hash=expected.manifest_hash,
        tolerances={
            "token_logprobs": NumericTolerance.create(
                repetitions=(
                    BaselineRepetition(
                        expected.manifest_hash,
                        _sha("a"),
                        (-0.1, -0.2, -0.3),
                    ),
                    BaselineRepetition(
                        expected.manifest_hash,
                        _sha("b"),
                        (-0.1, -0.2001, -0.3),
                    ),
                    BaselineRepetition(
                        expected.manifest_hash,
                        _sha("c"),
                        (-0.1, -0.2, -0.3),
                    ),
                )
            )
        },
    )
    envelope_path = tmp_path / "fabricated-envelope.json"
    policy_path = tmp_path / "reviewed-policy.json"
    fabricated.write_json(envelope_path)
    reviewed_policy.write_json(policy_path)

    report = compare_bundles(
        expected,
        actual,
        ToleranceEnvelope.read_json(envelope_path),
        ComparisonPolicy.read_json(policy_path),
    )

    assert not report.passed
    assert report.mismatches[0].kind == "policy_envelope_mismatch"
    assert report.mismatches[0].position == "tolerance_envelope_hash"


def test_reviewed_policy_rejects_freshly_rehashed_widened_envelope(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, token_logprobs=(-0.1, -0.9, -0.3)
    )
    actual = _round_trip(tmp_path, "actual", actual)
    reviewed_envelope = _envelope(expected)
    reviewed_policy = _policy(expected, reviewed_envelope)
    widened = _envelope(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (-0.1, -10.0, -0.3),
            (-0.1, 10.0, -0.3),
        ),
    )
    envelope_path = tmp_path / "widened-envelope.json"
    policy_path = tmp_path / "reviewed-exact-policy.json"
    widened.write_json(envelope_path)
    reviewed_policy.write_json(policy_path)

    report = compare_bundles(
        expected,
        actual,
        ToleranceEnvelope.read_json(envelope_path),
        ComparisonPolicy.read_json(policy_path),
    )

    assert not report.passed
    assert [mismatch.kind for mismatch in report.mismatches] == [
        "policy_envelope_mismatch"
    ]


def test_envelope_for_another_baseline_manifest_is_rejected(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    envelope = ToleranceEnvelope.create(
        baseline_manifest_hash=_sha("f"), tolerances={}
    )

    report = compare_bundles(expected, actual, envelope, _policy(expected, envelope))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "envelope_baseline_mismatch"
    assert mismatch.position == "baseline_manifest_hash"

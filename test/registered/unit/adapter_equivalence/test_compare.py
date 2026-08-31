import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence.compare import compare_bundles
from adapter_equivalence.schema import (
    SCHEMA_VERSION,
    PROVENANCE_HASH_KEYS,
    BundleValidationError,
    CaseKey,
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
        "server_args": ["--peft-method", "oft"],
        "request_order": ["req-0"],
        "seed": 1729,
    }
    return RunBundle.create(
        case_key=case,
        manifest=manifest,
        provenance=provenance,
        observations={"req-0": _observation()},
        performance=PerformanceMetrics(
            startup_seconds=(10.0, 10.2, 9.8),
            latency_seconds=(0.20, 0.21, 0.19),
            throughput_tokens_per_second=(100.0, 101.0, 99.0),
            peak_allocated_bytes=1_000,
            peak_reserved_bytes=1_200,
        ),
        completion={"status": "complete", "exit_code": 0, "job_id": "17.0"},
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


def test_identical_serialized_bundles_compare_equal(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())

    report = compare_bundles(expected, actual, _envelope(expected))

    assert report.passed
    assert report.mismatches == ()


def test_token_mismatch_reports_request_and_position(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, output_ids=(101, 999, 303))
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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
    payload["manifest"]["seed"] = 2718
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

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    assert report.mismatches[0].kind == "provenance_mismatch"
    assert report.mismatches[0].position == "provenance.checkpoint_hash"


def test_logprob_mismatch_fails_when_envelope_is_exact(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, token_logprobs=(-0.1, -0.2001, -0.3)
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "numeric_mismatch"
    assert mismatch.position == "token_logprobs[1]"


def test_named_unchanged_baseline_tolerance_allows_numeric_drift(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, token_logprobs=(-0.1, -0.2001, -0.3)
    )
    actual = _round_trip(tmp_path, "actual", actual)
    envelope = _envelope(
        expected,
        token_logprobs=NumericTolerance(
            atol=0.001, rtol=0.0, baseline_repetitions=3
        ),
    )

    report = compare_bundles(expected, actual, envelope)

    assert report.passed


def test_token_divergence_fails_even_with_wide_numeric_tolerance(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, output_ids=(101, 999, 303))
    actual = _round_trip(tmp_path, "actual", actual)
    envelope = _envelope(
        expected,
        token_logprobs=NumericTolerance(
            atol=100.0, rtol=100.0, baseline_repetitions=3
        ),
    )

    report = compare_bundles(expected, actual, envelope)

    assert not report.passed
    assert report.mismatches[0].kind == "token_mismatch"


def test_peak_memory_regression_fails_at_five_percent_ratio(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(expected.performance, peak_allocated_bytes=1_051)
    actual = _round_trip(
        tmp_path, "actual", replace(expected, performance=performance)
    )

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "performance_regression"
    assert mismatch.position == "performance.peak_allocated_bytes"
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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    assert report.mismatches[0].position == (
        "performance.throughput_tokens_per_second.median"
    )


def test_widened_tolerance_with_stale_envelope_hash_invalidates_comparison(
    tmp_path,
):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    envelope = _envelope(
        expected,
        token_logprobs=NumericTolerance(
            atol=0.001, rtol=0.0, baseline_repetitions=3
        ),
    )
    widened = replace(
        envelope,
        tolerances={
            "token_logprobs": NumericTolerance(
                atol=1.0, rtol=1.0, baseline_repetitions=3
            )
        },
    )

    report = compare_bundles(expected, actual, widened)

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "invalid_envelope"
    assert mismatch.position == "manifest_hash"


def test_envelope_for_another_baseline_manifest_is_rejected(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    envelope = ToleranceEnvelope.create(
        baseline_manifest_hash=_sha("f"), tolerances={}
    )

    report = compare_bundles(expected, actual, envelope)

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "envelope_baseline_mismatch"
    assert mismatch.position == "baseline_manifest_hash"

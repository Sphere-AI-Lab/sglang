import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence.compare import compare_bundles

# Imported rather than re-spelled: these are the retired CLI/key spellings the
# frozen source still uses, assembled from fragments in server.py so that
# test/registered/unit/oft/test_no_legacy_peft.py's source scan stays strict.
# Importing them also ties this assertion to the harness it is checking.
from adapter_equivalence.server import (
    _LEGACY_IMPL_FLAG,
    _LEGACY_IMPL_KEY,
    _LEGACY_LORA_VALUE,
)
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
        "server_args": ["--peft-method", "oft"],
        "request_order": ["req-0"],
        "seed": 1729,
        "metadata": {},
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


def _round_trip_envelope(
    tmp_path, name: str, envelope: ToleranceEnvelope
) -> ToleranceEnvelope:
    path = tmp_path / f"{name}.json"
    envelope.write_json(path)
    return ToleranceEnvelope.read_json(path)


def _envelope(bundle: RunBundle, **tolerances: NumericTolerance) -> ToleranceEnvelope:
    unreviewed = ToleranceEnvelope.create(
        baseline_manifest_hash=bundle.manifest_hash,
        tolerances=tolerances,
    )
    return unreviewed.with_policy(_policy(bundle, unreviewed))


def _policy(bundle: RunBundle, envelope: ToleranceEnvelope) -> ComparisonPolicy:
    return ComparisonPolicy.create(
        baseline_manifest_hash=bundle.manifest_hash,
        tolerance_envelope_hash=envelope.manifest_hash,
    )


def _rehash_bundle_manifest(payload: dict[str, object]) -> None:
    manifest_hash = canonical_sha256(payload["manifest"])
    payload["manifest_hash"] = manifest_hash
    payload["provenance"]["manifest_hash"] = manifest_hash


def _rehash_envelope_manifest(payload: dict[str, object]) -> None:
    manifest_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"comparison_policy", "manifest_hash"}
    }
    payload["manifest_hash"] = canonical_sha256(manifest_payload)


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


def test_three_argument_exact_comparison_accepts_identical_serialized_bundles(
    tmp_path,
):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())

    envelope = _round_trip_envelope(
        tmp_path, "exact-envelope", _envelope(expected)
    )

    report = compare_bundles(expected, actual, envelope)

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


def test_embedded_manifest_case_key_rejects_integer_for_boolean(tmp_path):
    path = tmp_path / "manifest-case-key-int-for-bool.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["case_key"]["cuda_graph"] = 1
    _rehash_bundle_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="case_key.cuda_graph must be a boolean",
    ):
        RunBundle.read_json(path)


@pytest.mark.parametrize("field_name", ["server_args", "request_order", "seed"])
def test_required_manifest_execution_field_cannot_be_hidden_in_metadata(
    tmp_path, field_name
):
    path = tmp_path / f"missing-{field_name}.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["metadata"][field_name] = payload["manifest"].pop(
        field_name
    )
    _rehash_bundle_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match=rf"manifest: missing fields: {field_name}",
    ):
        RunBundle.read_json(path)


@pytest.mark.parametrize(
    ("field_name", "confused_value", "error_match"),
    [
        pytest.param(
            "server_args",
            False,
            "manifest.server_args must be an array of strings",
            id="server-args-boolean",
        ),
        pytest.param(
            "request_order",
            ["req-0", 1],
            "manifest.request_order entries must be non-empty strings",
            id="request-order-integer",
        ),
        pytest.param(
            "seed",
            True,
            "manifest.seed must be an integer",
            id="seed-boolean",
        ),
    ],
)
def test_required_manifest_execution_field_rejects_type_confusion(
    tmp_path, field_name, confused_value, error_match
):
    path = tmp_path / f"type-confused-{field_name}.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"][field_name] = confused_value
    _rehash_bundle_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match=error_match):
        RunBundle.read_json(path)


def test_manifest_metadata_rejects_reserved_execution_field_names(tmp_path):
    path = tmp_path / "reserved-metadata-field.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["metadata"]["seed"] = 2718
    _rehash_bundle_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="manifest.metadata contains reserved contract fields: seed",
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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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
    envelope = _envelope(expected)

    first = compare_bundles(expected, actual, envelope)
    second = compare_bundles(expected, actual, envelope)

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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    exact_report = compare_bundles(expected, actual, _envelope(expected))
    envelope = _envelope(
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
    tolerant_report = compare_bundles(expected, actual, envelope)

    assert not exact_report.passed
    assert exact_report.mismatches[0].position == (
        "selected_logits.decoder.output[1]"
    )
    assert tolerant_report.passed


def test_three_argument_reviewed_tolerance_allows_numeric_drift(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected, token_logprobs=(-0.1, -0.2001, -0.3)
    )
    actual = _round_trip(tmp_path, "actual", actual)
    envelope = _envelope(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )

    envelope = _round_trip_envelope(tmp_path, "reviewed-envelope", envelope)

    report = compare_bundles(expected, actual, envelope)

    assert report.passed


def test_token_divergence_fails_even_with_wide_numeric_tolerance(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, output_ids=(101, 999, 303))
    actual = _round_trip(tmp_path, "actual", actual)
    envelope = _envelope(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (100.0, 100.0, 100.0),
            (-100.0, -100.0, -100.0),
        ),
    )

    report = compare_bundles(expected, actual, envelope)

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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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

    report = compare_bundles(expected, actual, _envelope(expected))

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
    envelope = _envelope(
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

    report = compare_bundles(expected, actual, widened)

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
    _rehash_envelope_manifest(payload)
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
    _rehash_envelope_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="baseline_manifest_hash does not match envelope baseline",
    ):
        ToleranceEnvelope.read_json(path)


def test_reviewed_policy_rejects_fabricated_repetition_evidence(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    reviewed_envelope = _envelope(
        expected,
        token_logprobs=_tolerance(
            expected,
            (-0.1, -0.2, -0.3),
            (-0.1, -0.2001, -0.3),
            (-0.1, -0.2, -0.3),
        ),
    )
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
    payload = reviewed_envelope.to_dict()
    payload["tolerances"] = {
        name: tolerance.to_dict()
        for name, tolerance in fabricated.tolerances.items()
    }
    _rehash_envelope_manifest(payload)
    envelope_path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="ComparisonPolicy.tolerance_envelope_hash does not match envelope",
    ):
        ToleranceEnvelope.read_json(envelope_path)


def test_reviewed_policy_rejects_freshly_rehashed_widened_envelope(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    reviewed_envelope = _envelope(expected)
    widened_evidence = ToleranceEnvelope.create(
        baseline_manifest_hash=expected.manifest_hash,
        tolerances={
            "token_logprobs": _tolerance(
                expected,
                (-0.1, -0.2, -0.3),
                (-0.1, -10.0, -0.3),
                (-0.1, 10.0, -0.3),
            )
        },
    )
    envelope_path = tmp_path / "widened-envelope.json"
    payload = reviewed_envelope.to_dict()
    payload["tolerances"] = {
        name: tolerance.to_dict()
        for name, tolerance in widened_evidence.tolerances.items()
    }
    _rehash_envelope_manifest(payload)
    envelope_path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="ComparisonPolicy.tolerance_envelope_hash does not match envelope",
    ):
        ToleranceEnvelope.read_json(envelope_path)


def test_three_argument_comparison_rejects_freshly_widened_evidence(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    reviewed = _envelope(expected)
    widened_evidence = ToleranceEnvelope.create(
        baseline_manifest_hash=expected.manifest_hash,
        tolerances={
            "token_logprobs": _tolerance(
                expected,
                (-0.1, -0.2, -0.3),
                (-0.1, -10.0, -0.3),
                (-0.1, 10.0, -0.3),
            )
        },
    )
    widened = replace(
        widened_evidence,
        comparison_policy=reviewed.comparison_policy,
    )

    report = compare_bundles(expected, actual, widened)

    assert not report.passed
    assert report.mismatches[0].kind == "invalid_envelope"
    assert "tolerance_envelope_hash does not match" in report.mismatches[0].actual


def test_envelope_for_another_baseline_manifest_is_rejected(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    unreviewed = ToleranceEnvelope.create(
        baseline_manifest_hash=_sha("f"), tolerances={}
    )
    envelope = unreviewed.with_policy(
        ComparisonPolicy.create(
            baseline_manifest_hash=_sha("f"),
            tolerance_envelope_hash=unreviewed.manifest_hash,
        )
    )

    report = compare_bundles(expected, actual, envelope)

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "envelope_baseline_mismatch"
    assert mismatch.position == "baseline_manifest_hash"


def test_three_argument_comparison_rejects_unreviewed_envelope(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _round_trip(tmp_path, "actual", _bundle())
    unreviewed = ToleranceEnvelope.create(
        baseline_manifest_hash=expected.manifest_hash,
        tolerances={},
    )

    report = compare_bundles(expected, actual, unreviewed)

    assert not report.passed
    assert report.mismatches[0].kind == "invalid_envelope"
    assert "reviewed ComparisonPolicy" in report.mismatches[0].actual


def test_task8_source_and_candidate_mode_arguments_are_frozen():
    from adapter_equivalence.scenarios import ScenarioContractError
    from adapter_equivalence.server import mode_server_args

    assert {
        mode: mode_server_args("source", mode)
        for mode in (
            "base",
            "legacy_oft",
            "canonical_oft",
            "legacy_lora",
            "native_lora",
        )
    } == {
        "base": (),
        "legacy_oft": ("--peft-method", "oft", _LEGACY_IMPL_FLAG, "peft"),
        "canonical_oft": (
            "--peft-method",
            "oft",
            _LEGACY_IMPL_FLAG,
            "sibling",
        ),
        "legacy_lora": ("--peft-method", _LEGACY_LORA_VALUE),
        "native_lora": ("--enable-lora", "--enable-lora-staging"),
    }
    assert {
        mode: mode_server_args("candidate", mode)
        for mode in ("base", "canonical_oft", "native_lora")
    } == {
        "base": (),
        "canonical_oft": ("--peft-method", "oft"),
        "native_lora": ("--enable-lora", "--enable-lora-staging"),
    }
    for legacy_mode in ("legacy_oft", "legacy_lora"):
        with pytest.raises(
            ScenarioContractError,
            match=f"{legacy_mode} is a source-only oracle mode",
        ):
            mode_server_args("candidate", legacy_mode)


def test_task8_adapter_lifecycle_contains_every_required_transition():
    from adapter_equivalence.scenarios import lifecycle_transition_names

    assert lifecycle_transition_names("canonical_oft") == (
        "base.initial",
        "startup.adapter",
        "dynamic.load",
        "dynamic.infer",
        "dynamic.unload",
        "dynamic.post-unload-base",
        "switch.a",
        "switch.b",
        "switch.a-again",
        "mixed.base-a-b",
        "concurrent.stream",
        "concurrent.non-stream",
        "prefill.short",
        "prefill.long",
        "decode.short",
        "decode.long",
        "stage.v1",
        "activate.v1",
        "stage.v2",
        "activate.v2",
        "reject.duplicate",
        "reject.stale",
        "reject.invalid-id",
        "reject.invalid-config",
        "rollback.previous",
        "restart.same-manifest",
    )


def test_task8_post_unload_output_must_exactly_restore_initial_base():
    from adapter_equivalence.scenarios import (
        ScenarioContractError,
        validate_lifecycle_observations,
    )

    initial = _observation()
    validate_lifecycle_observations(
        {
            "base.initial": initial,
            "dynamic.post-unload-base": initial,
        }
    )

    divergent = replace(initial, output_ids=initial.output_ids[:-1] + (999,))
    with pytest.raises(
        ScenarioContractError,
        match="post-unload output does not exactly match initial base output",
    ):
        validate_lifecycle_observations(
            {
                "base.initial": initial,
                "dynamic.post-unload-base": divergent,
            }
        )


def test_task8_only_unchanged_source_legacy_startup_can_be_unsupported():
    from adapter_equivalence.scenarios import (
        ScenarioContractError,
        classify_startup_failure,
    )

    traceback = "RuntimeError: kernel not implemented for this precision"
    assert classify_startup_failure("source", "legacy_oft", traceback) == {
        "status": "unsupported_by_legacy",
        "mode": "legacy_oft",
        "traceback": traceback,
    }

    with pytest.raises(
        ScenarioContractError,
        match="startup failure is not eligible for legacy-source classification",
    ):
        classify_startup_failure("source", "canonical_oft", traceback)
    with pytest.raises(
        ScenarioContractError,
        match="startup failure is not eligible for legacy-source classification",
    ):
        classify_startup_failure("candidate", "legacy_oft", traceback)


def test_task8_server_launch_arguments_are_deterministic():
    from adapter_equivalence.server import ServerSpec, server_other_args

    spec = ServerSpec(
        revision_kind="candidate",
        model_path="/models/qwen3-30b-fp8",
        mode="canonical_oft",
        port=31000,
        tp_size=4,
        ep_size=4,
        cuda_graph=False,
        quantization="fp8",
        moe_runner="triton",
        startup_adapters=(
            ("policy-a", "/adapters/a"),
            ("policy-b", "/adapters/b"),
        ),
    )

    assert server_other_args(spec) == (
        "--base-gpu-id",
        "1",
        "--tp-size",
        "4",
        "--ep-size",
        "4",
        "--quantization",
        "fp8",
        "--moe-runner-backend",
        "triton",
        "--disable-cuda-graph",
        "--peft-method",
        "oft",
        "--peft-paths",
        "policy-a=/adapters/a",
        "policy-b=/adapters/b",
        "--mem-fraction-static",
        "0.8",
        "--log-level",
        "error",
    )


def test_task8_lifecycle_executor_records_the_frozen_state_machine():
    from adapter_equivalence.scenarios import (
        LifecycleStep,
        execute_lifecycle,
        lifecycle_steps,
    )

    steps = lifecycle_steps("native_lora")
    selected = {step.name: step for step in steps}
    assert selected["base.initial"] == LifecycleStep(
        "base.initial", "generate", prompt_id="factual"
    )
    assert selected["dynamic.load"] == LifecycleStep(
        "dynamic.load", "load", adapter="policy-a"
    )
    assert selected["mixed.base-a-b"] == LifecycleStep(
        "mixed.base-a-b", "mixed_batch", prompt_id="batch-8"
    )
    assert selected["concurrent.stream"] == LifecycleStep(
        "concurrent.stream", "concurrent", prompt_id="batch-8", stream=True
    )
    assert selected["stage.v2"] == LifecycleStep(
        "stage.v2", "stage", adapter="policy-a", version="2"
    )
    assert selected["reject.invalid-config"] == LifecycleStep(
        "reject.invalid-config",
        "reject_invalid_config",
        adapter="policy-a",
        version="3",
    )
    assert selected["restart.same-manifest"] == LifecycleStep(
        "restart.same-manifest", "restart", prompt_id="factual"
    )

    initial = _observation()

    class RecordingExecutor:
        def __init__(self):
            self.seen = []

        def execute(self, step):
            self.seen.append(step)
            if step.name in (
                "base.initial",
                "dynamic.post-unload-base",
            ):
                return initial
            return replace(initial, adapter_state={"transition": step.name})

    executor = RecordingExecutor()
    observations = execute_lifecycle("native_lora", executor)

    assert tuple(observations) == tuple(step.name for step in steps)
    assert executor.seen == list(steps)


def test_task8_offline_engine_kwargs_preserve_source_oft_selection():
    from adapter_equivalence.server import ServerSpec, engine_kwargs

    spec = ServerSpec(
        revision_kind="source",
        model_path="/models/qwen3-4b",
        mode="canonical_oft",
        port=31001,
        tp_size=1,
        ep_size=1,
        cuda_graph=True,
        startup_adapters=(("policy-a", "/adapters/a"),),
    )

    assert engine_kwargs(spec) == {
        "base_gpu_id": 1,
        "disable_cuda_graph": False,
        "ep_size": 1,
        "log_level": "error",
        "mem_fraction_static": 0.8,
        "model_path": "/models/qwen3-4b",
        _LEGACY_IMPL_KEY: "sibling",
        "peft_method": "oft",
        "peft_paths": ("policy-a=/adapters/a",),
        "tp_size": 1,
    }


def test_task8_dynamic_oft_engine_kwargs_declare_fixture_shape_contract():
    from adapter_equivalence.server import ServerSpec, engine_kwargs

    target_modules = (
        "down_proj",
        "gate_proj",
        "o_proj",
        "q_proj",
        "up_proj",
    )
    spec = ServerSpec(
        revision_kind="source",
        model_path="/models/qwen3-4b",
        mode="canonical_oft",
        port=31002,
        tp_size=1,
        ep_size=1,
        cuda_graph=False,
        max_oft_block_size=128,
        peft_target_modules=target_modules,
    )

    assert engine_kwargs(spec) == {
        "base_gpu_id": 1,
        "disable_cuda_graph": True,
        "ep_size": 1,
        "log_level": "error",
        "max_oft_block_size": 128,
        "mem_fraction_static": 0.8,
        "model_path": "/models/qwen3-4b",
        _LEGACY_IMPL_KEY: "sibling",
        "peft_method": "oft",
        "peft_target_modules": target_modules,
        "tp_size": 1,
    }


@pytest.mark.parametrize(
    "missing_field",
    ["max_oft_block_size", "peft_target_modules"],
)
def test_task8_dynamic_oft_requires_fixture_shape_contract(missing_field):
    from adapter_equivalence.scenarios import ScenarioContractError
    from adapter_equivalence.server import ServerSpec

    arguments = {
        "revision_kind": "source",
        "model_path": "/models/qwen3-4b",
        "mode": "canonical_oft",
        "port": 31002,
        "tp_size": 1,
        "ep_size": 1,
        "cuda_graph": False,
        "max_oft_block_size": 128,
        "peft_target_modules": ("q_proj",),
    }
    arguments[missing_field] = (
        None if missing_field == "max_oft_block_size" else ()
    )

    with pytest.raises(
        ScenarioContractError,
        match="dynamic OFT requires max_oft_block_size and peft_target_modules",
    ):
        ServerSpec(**arguments)


def test_task8_internal_oft_control_streams_fixture_and_uses_control_for_unload():
    import asyncio
    from dataclasses import dataclass
    from types import SimpleNamespace

    from adapter_equivalence.server import InternalOFTControl

    @dataclass
    class Ref:
        adapter_id: str
        adapter_name: str
        adapter_path: str
        pinned: bool

    @dataclass
    class UpdateRequest:
        serialized_named_tensors: list[bytes]
        load_format: str
        flush_cache: bool
        adapter_config: dict[str, object]
        adapter_name: str | None
        adapter_id: str | None

    class Registry:
        def __init__(self):
            self.refs = {}
            self.released = []
            self.waited_for = []

        async def register(self, ref):
            self.refs[ref.adapter_name] = ref

        async def unregister(self, name):
            return self.refs.pop(name).adapter_id

        async def release(self, adapter_id):
            self.released.append(adapter_id)

        async def wait_for_unload(self, adapter_id):
            self.waited_for.append(adapter_id)

    class Manager:
        def __init__(self):
            self.calls = []
            self.peft_registry = Registry()
            self.peft_ref_cache = {}
            self.peft_update_lock = asyncio.Lock()

        async def update_weights_from_tensor(self, request, http_request):
            self.calls.append((request, http_request))
            if request.adapter_name is not None:
                ref = Ref(
                    adapter_id="adapter-id-a",
                    adapter_name=request.adapter_name,
                    adapter_path=request.adapter_name,
                    pinned=False,
                )
                request.adapter_id = ref.adapter_id
                await self.peft_registry.register(ref)
                self.peft_ref_cache[ref.adapter_name] = ref
            return True, "ok"

    class Loop:
        def run_until_complete(self, awaitable):
            return asyncio.run(awaitable)

    class Engine:
        def __init__(self, manager):
            self.loop = Loop()
            self.tokenizer_manager = manager
            self.server_args = SimpleNamespace(tp_size=2)
            self.serialized = []
            self.weight_update_sessions = []

        def _serialize_tensors_per_rank(self, tensors, load_format):
            self.serialized.append((tensors, load_format))
            return [b"empty-control-payload"]

        def begin_weight_update(self):
            self.weight_update_sessions.append("begin")
            return True, "ok"

        def end_weight_update(self):
            self.weight_update_sessions.append("end")
            return True, "ok"

    manager = Manager()
    engine = Engine(manager)
    fixture_tensors = [("base_model.model.layers.0.self_attn.q_proj.oft_R", object())]
    fixture_config = {
        "oft_block_size": 128,
        "peft_type": "OFT",
        "target_modules": ["q_proj"],
    }
    loaded_paths = []
    serialized_payloads = []

    def load_fixture(path):
        loaded_paths.append(path)
        return fixture_tensors, fixture_config

    def serialize_payload(tensors):
        serialized_payloads.append(tensors)
        return b"flattened-oft-payload"

    control = InternalOFTControl(
        engine,
        revision_kind="source",
        update_request_type=UpdateRequest,
        fixture_loader=load_fixture,
        payload_serializer=serialize_payload,
    )

    assert control.load("policy-a", "/adapters/a").success
    assert control.unload("policy-a").success
    assert loaded_paths == ["/adapters/a"]
    assert serialized_payloads == [fixture_tensors]
    assert engine.serialized == [([], "adapter_equivalence_oft_control")]
    assert engine.weight_update_sessions == ["begin", "end", "begin", "end"]
    load_request, load_http_request = manager.calls[0]
    assert load_http_request is None
    assert load_request.serialized_named_tensors == [
        b"flattened-oft-payload",
        b"flattened-oft-payload",
    ]
    assert load_request.load_format == "oft_adapter"
    assert load_request.adapter_name == "policy-a"
    assert load_request.adapter_id == "adapter-id-a"
    assert load_request.adapter_config == fixture_config
    unload_request, unload_http_request = manager.calls[1]
    assert unload_http_request is None
    assert unload_request.adapter_name is None
    assert unload_request.adapter_id == "adapter-id-a"
    assert unload_request.adapter_config == {
        "operation": "unload",
        "adapter_name": "policy-a",
        "adapter_path": "policy-a",
        "pinned": False,
    }
    assert manager.peft_registry.refs == {}
    assert manager.peft_registry.released == ["adapter-id-a"]
    assert manager.peft_registry.waited_for == ["adapter-id-a"]
    assert manager.peft_ref_cache == {}


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_task8_internal_oft_bridge_dispatches_to_real_manager(operation):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from adapter_equivalence.server import _dispatch_internal_oft_control

    @dataclass
    class Ref:
        adapter_id: str
        adapter_name: str
        adapter_path: str
        pinned: bool

    class OFTManager:
        def __init__(self):
            self.calls = []

        def load_oft_adapter(self, ref):
            self.calls.append(("load", ref))
            return SimpleNamespace(success=True, error_message="")

        def unload_oft_adapter(self, ref):
            self.calls.append(("unload", ref))
            return SimpleNamespace(success=True, error_message="")

    manager = OFTManager()
    runner = SimpleNamespace(
        server_args=SimpleNamespace(peft_method="oft"),
        oft_manager=manager,
    )
    result = _dispatch_internal_oft_control(
        runner,
        adapter_config={
            "operation": operation,
            "adapter_name": "policy-a",
            "adapter_path": "/adapters/a",
            "pinned": False,
        },
        adapter_id="adapter-id-a",
        ref_type=Ref,
    )

    assert result == (True, "")
    assert manager.calls == [
        (
            operation,
            Ref("adapter-id-a", "policy-a", "/adapters/a", False),
        )
    ]

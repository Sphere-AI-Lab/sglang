# ruff: noqa: E402 -- registered CPU tests add the manual harness import root.
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
        mode="native_oft",
        cuda_graph=True,
        scenario="dynamic-load-switch",
    )


def _adapter_state(
    *, active_version: str = "7", staged_version: str = "8"
) -> dict[str, object]:
    return {
        "mode": "native_oft",
        "registered": [
            {
                "name": "adapter-a",
                "id": "id-a",
                "version": active_version,
                "registry_slot": 0,
                "pinned": False,
            }
        ],
        "active": {
            "name": "adapter-a",
            "id": "id-a",
            "version": active_version,
        },
        "staged": {
            "name": "adapter-b",
            "id": "id-b",
            "version": staged_version,
            "pinned": False,
        },
        "registry_occupancy": 1,
        "quarantined": [],
        "tombstoned": [],
        "cache_identity": {"adapter-a": "id-a"},
    }


def _observation() -> Observation:
    return Observation(
        request_output_lengths=(3,),
        request_texts=("alpha beta gamma",),
        output_ids=(101, 202, 303),
        text="alpha beta gamma",
        token_logprobs=(-0.1, -0.2, -0.3),
        selected_logits={
            "decode.000.top_logprobs": (0.1, 0.2, 0.3),
            "decode.001.top_logprobs": (1.0, 2.0, 3.0),
            "decode.002.top_logprobs": (2.1, 2.2, 2.3),
        },
        selected_token_ids={
            "decode.000.top_logprobs": (10, 11, 12),
            "decode.001.top_logprobs": (20, 21, 22),
            "decode.002.top_logprobs": (30, 31, 32),
        },
        adapter_state=_adapter_state(),
        error=None,
    )


@pytest.mark.parametrize("field", ("request_output_lengths", "request_texts"))
def test_request_boundaries_compare_exactly_even_with_identical_flat_payload(field):
    """Defect: tolerances conceal different request partitioning or per-request text."""
    expected = _bundle()
    base = expected.observations["req-0"]
    expected = _replace_observation(
        expected, request_output_lengths=(1, 2), request_texts=("a", "bc")
    )
    if field == "request_output_lengths":
        actual = _replace_observation(expected, request_output_lengths=(2, 1))
    else:
        actual = _replace_observation(expected, request_texts=("ab", "c"))
    assert actual.observations["req-0"].output_ids == base.output_ids
    report = compare_bundles(expected, actual, _envelope(expected))
    assert any(mismatch.position == field for mismatch in report.mismatches)


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
        "provenance_hashes": {key: provenance[key] for key in PROVENANCE_HASH_KEYS},
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
    # Three independent artifacts may contain identical numeric measurements.
    if len(values) == 2:
        values = (*values, values[0])
    return NumericTolerance.create(
        repetitions=_baseline_repetitions(bundle.manifest_hash, *values)
    )


def _replace_observation(bundle: RunBundle, **changes: object) -> RunBundle:
    observations = dict(bundle.observations)
    observations["req-0"] = replace(observations["req-0"], **changes)
    return replace(bundle, observations=observations)


def _replace_provenance_hash(bundle: RunBundle, key: str, value: str) -> RunBundle:
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

    envelope = _round_trip_envelope(tmp_path, "exact-envelope", _envelope(expected))

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
    selected_logits = dict(expected.observations["req-0"].selected_logits)
    selected_logits["decode.003.top_logprobs"] = (3.1, 3.2, 3.3)
    selected_token_ids = dict(expected.observations["req-0"].selected_token_ids)
    selected_token_ids["decode.003.top_logprobs"] = (40, 41, 42)
    actual = _replace_observation(
        expected,
        output_ids=(101, 202, 303, 404),
        request_output_lengths=(4,),
        token_logprobs=(-0.1, -0.2, -0.3, -0.4),
        selected_logits=selected_logits,
        selected_token_ids=selected_token_ids,
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    mismatch = next(
        item for item in report.mismatches if item.position == "output_ids.shape"
    )
    assert mismatch.kind == "shape_mismatch"
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
    state = _adapter_state(active_version="8")
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
    assert mismatch.expected == "7"
    assert mismatch.actual == "8"


def test_error_mismatch_reports_error_path(tmp_path):
    expected = _replace_observation(
        _bundle(),
        error={
            "kind": "product_rejection",
            "code": "invalid_adapter",
            "message": "request rejected",
        },
    )
    actual = _replace_observation(
        expected,
        error={
            "kind": "product_rejection",
            "code": "adapter_not_found",
            "message": "request rejected",
        },
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
    payload["manifest"]["metadata"][field_name] = payload["manifest"].pop(field_name)
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


def test_manifest_request_order_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate-request-order.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["request_order"] = ["req-0", "req-0"]
    _rehash_bundle_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="manifest.request_order entries must be unique",
    ):
        RunBundle.read_json(path)


def test_manifest_request_order_must_name_each_observation_exactly_once(tmp_path):
    path = tmp_path / "missing-observation-request-order.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["manifest"]["request_order"] = ["different-request"]
    _rehash_bundle_manifest(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError,
        match="manifest.request_order must name every observation exactly once",
    ):
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
        f'  "schema_version": {SCHEMA_VERSION}\n}}',
        f'  "schema_version": {SCHEMA_VERSION},\n'
        f'  "schema_version": {SCHEMA_VERSION}\n}}',
        1,
    )
    path.write_text(serialized)

    with pytest.raises(
        BundleValidationError, match="duplicate object key.*schema_version"
    ):
        RunBundle.read_json(path)


def test_boolean_is_not_accepted_as_an_integer_in_serialized_bundle(tmp_path):
    path = tmp_path / "boolean-token-id.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["observations"]["req-0"]["output_ids"][0] = True
    path.write_text(json.dumps(payload))

    with pytest.raises(
        BundleValidationError, match="output_ids entries must be integers"
    ):
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


def test_revision_and_code_hash_differences_are_allowed(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    candidate_case = replace(
        expected.case_key,
        revision="b" * 40,
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
        selected_logits={
            "decode.000.top_logprobs": (0.1, 0.2, 0.3),
            "decode.001.top_logprobs": (1.0, 9.0, 3.0),
            "decode.002.top_logprobs": (2.1, 2.2, 2.3),
        },
        adapter_state=_adapter_state(active_version="8"),
        error={
            "kind": "product_rejection",
            "code": "unexpected",
            "message": "request rejected",
        },
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
        "adapter_state.registered[0].version",
        "error",
        "token_logprobs[1]",
        "selected_logits.decode.001.top_logprobs[1]",
    ]
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_logprob_mismatch_fails_when_envelope_is_exact(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, token_logprobs=(-0.1, -0.2001, -0.3))
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    mismatch = report.mismatches[0]
    assert mismatch.kind == "numeric_mismatch"
    assert mismatch.position == "token_logprobs[1]"


def test_selected_logit_shape_mismatch_is_exact(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(
        expected,
        selected_logits={
            name: values[:2]
            for name, values in expected.observations["req-0"].selected_logits.items()
        },
        selected_token_ids={
            name: values[:2]
            for name, values in expected.observations[
                "req-0"
            ].selected_token_ids.items()
        },
    )
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    assert report.mismatches[0].kind == "shape_mismatch"
    assert report.mismatches[0].position == (
        "selected_logits.decode.000.top_logprobs.shape"
    )


def test_selected_logit_dtype_mismatch_is_rejected_from_serialized_bundle(tmp_path):
    path = tmp_path / "selected-logit-dtype.json"
    _bundle().write_json(path)
    payload = json.loads(path.read_text())
    payload["observations"]["req-0"]["selected_logits"]["decode.001.top_logprobs"][
        0
    ] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="selected_logits.*float"):
        RunBundle.read_json(path)


def test_selected_logit_uses_exact_then_reviewed_tolerance_path(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    selected_logits = dict(expected.observations["req-0"].selected_logits)
    selected_logits["decode.001.top_logprobs"] = (1.0, 2.0001, 3.0)
    actual = _replace_observation(expected, selected_logits=selected_logits)
    actual = _round_trip(tmp_path, "actual", actual)

    exact_report = compare_bundles(expected, actual, _envelope(expected))
    envelope = _envelope(
        expected,
        **{
            "selected_logits.decode.001.top_logprobs": _tolerance(
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
        "selected_logits.decode.001.top_logprobs[1]"
    )
    assert tolerant_report.passed


def test_selected_token_identity_mismatch_is_never_tolerated(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    selected_token_ids = dict(expected.observations["req-0"].selected_token_ids)
    selected_token_ids["decode.001.top_logprobs"] = (20, 21, 999)
    actual = _replace_observation(expected, selected_token_ids=selected_token_ids)
    actual = _round_trip(tmp_path, "actual", actual)

    report = compare_bundles(expected, actual, _envelope(expected))

    assert not report.passed
    assert report.mismatches[0].kind == "token_mismatch"
    assert report.mismatches[0].position == (
        "selected_token_ids.decode.001.top_logprobs[2]"
    )


def test_three_argument_reviewed_tolerance_allows_numeric_drift(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    actual = _replace_observation(expected, token_logprobs=(-0.1, -0.2001, -0.3))
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
        expected.performance, peak_allocated_bytes=(1_060, 1_051, 1_030)
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
    assert mismatch.position == "performance.peak_allocated_bytes.median"
    assert mismatch.expected == 995
    assert mismatch.actual == 1_051


def test_one_memory_outlier_does_not_replace_the_three_sample_median():
    expected = _bundle()
    actual = replace(
        expected,
        performance=replace(
            expected.performance, peak_allocated_bytes=(9000, 1000, 995)
        ),
    )
    assert compare_bundles(expected, actual, _envelope(expected)).passed


def test_startup_and_latency_have_evidence_but_no_unapproved_threshold():
    expected = _bundle()
    actual = replace(
        expected,
        performance=replace(
            expected.performance,
            startup_seconds=(100.0, 100.0, 100.0),
            latency_seconds=(20.0, 20.0, 20.0),
        ),
    )
    report = compare_bundles(expected, actual, _envelope(expected))
    assert report.passed
    assert report.performance["startup_seconds"]["candidate"] == 100.0
    assert report.performance["latency_seconds"]["candidate"] == 20.0


def test_two_reference_repetitions_cannot_authorize_tolerance():
    with pytest.raises(BundleValidationError, match="three"):
        NumericTolerance.create(
            repetitions=_baseline_repetitions(_bundle().manifest_hash, (-0.1,), (-0.2,))
        )


def test_zero_reference_timings_have_defined_evidence_ratios():
    bundle = _bundle()
    bundle = replace(
        bundle,
        performance=replace(
            bundle.performance,
            startup_seconds=(0.0, 0.0, 0.0),
            latency_seconds=(0.0, 0.0, 0.0),
            throughput_tokens_per_second=(0.0, 0.0, 0.0),
        ),
    )
    report = compare_bundles(bundle, bundle, _envelope(bundle))
    assert report.passed
    assert report.performance["startup_seconds"]["ratio"] is None


def test_request_order_is_exact_and_withholds_performance():
    payload = _bundle().to_dict()
    payload["observations"]["req-1"] = payload["observations"]["req-0"]
    payload["manifest"]["request_order"] = ["req-0", "req-1"]
    _rehash_bundle_manifest(payload)
    expected = RunBundle.from_dict(payload)
    payload["manifest"]["request_order"].reverse()
    _rehash_bundle_manifest(payload)
    actual = RunBundle.from_dict(payload)
    report = compare_bundles(expected, actual, _envelope(expected))
    assert any(m.position == "manifest.request_order" for m in report.mismatches)
    assert report.performance is None


def test_mode_is_exact_even_when_outputs_match():
    expected = _bundle()
    payload = expected.to_dict()
    payload["case_key"]["mode"] = payload["manifest"]["case_key"]["mode"] = (
        "native_lora"
    )
    payload["observations"]["req-0"]["adapter_state"]["mode"] = "native_lora"
    _rehash_bundle_manifest(payload)
    report = compare_bundles(
        expected, RunBundle.from_dict(payload), _envelope(expected)
    )
    assert any(m.position == "case_key.mode" for m in report.mismatches)
    assert report.performance is None


def test_throughput_at_five_percent_floor_passes(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(
        expected.performance,
        throughput_tokens_per_second=(95.0, 95.0, 95.0),
    )
    actual = _round_trip(tmp_path, "actual", replace(expected, performance=performance))

    report = compare_bundles(expected, actual, _envelope(expected))

    assert report.passed


def test_throughput_below_five_percent_floor_fails(tmp_path):
    expected = _round_trip(tmp_path, "expected", _bundle())
    performance = replace(
        expected.performance,
        throughput_tokens_per_second=(94.9, 94.9, 94.9),
    )
    actual = _round_trip(tmp_path, "actual", replace(expected, performance=performance))

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
        mismatch.kind != "performance_regression" for mismatch in report.mismatches
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
        name: tolerance.to_dict() for name, tolerance in fabricated.tolerances.items()
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


def test_task8_server_launch_arguments_are_deterministic():
    from adapter_equivalence.server import ServerSpec, server_other_args

    spec = ServerSpec(
        revision_kind="candidate",
        model_path="/models/qwen3-30b-fp8",
        mode="native_oft",
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
        "--oft-type",
        "oft",
        "--peft-paths",
        "policy-a=/adapters/a",
        "policy-b=/adapters/b",
        "--mem-fraction-static",
        "0.8",
        "--max-total-tokens",
        "32768",
        "--log-level",
        "error",
    )


@pytest.mark.parametrize("revision_kind", ("source", "candidate"))
def test_task8_offline_engine_kwargs_preserve_source_oft_selection(revision_kind):
    from adapter_equivalence.server import ServerSpec, engine_kwargs

    spec = ServerSpec(
        revision_kind=revision_kind,
        model_path="/models/qwen3-4b",
        mode="native_oft",
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
        "max_total_tokens": 32768,
        "model_path": "/models/qwen3-4b",
        "peft_method": "oft",
        "oft_type": "oft",
        "peft_paths": ["policy-a=/adapters/a"],
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
        mode="native_oft",
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
        "max_total_tokens": 32768,
        "model_path": "/models/qwen3-4b",
        "peft_method": "oft",
        "oft_type": "oft",
        "peft_target_modules": list(target_modules),
        "tp_size": 1,
    }


def test_dynamic_lora_engine_kwargs_declare_fixture_shape_contract():
    from adapter_equivalence.server import ServerSpec, engine_kwargs, server_other_args

    target_modules = ("q_proj", "embed_tokens", "lm_head")
    spec = ServerSpec(
        revision_kind="source",
        model_path="/models/qwen3-4b",
        mode="native_lora",
        port=31003,
        tp_size=1,
        ep_size=1,
        cuda_graph=False,
        max_lora_rank=8,
        lora_target_modules=target_modules,
    )

    assert engine_kwargs(spec) == {
        "base_gpu_id": 1,
        "disable_cuda_graph": True,
        "enable_lora": True,
        "enable_lora_staging": True,
        "ep_size": 1,
        "log_level": "error",
        "lora_target_modules": ["embed_tokens", "lm_head", "q_proj"],
        "max_lora_rank": 8,
        "mem_fraction_static": 0.8,
        "max_total_tokens": 32768,
        "model_path": "/models/qwen3-4b",
        "tp_size": 1,
    }
    assert "--max-lora-rank" in server_other_args(spec)
    assert "--lora-target-modules" in server_other_args(spec)


@pytest.mark.parametrize("missing_field", ["max_lora_rank", "lora_target_modules"])
def test_dynamic_lora_requires_fixture_shape_contract(missing_field):
    from adapter_equivalence.scenarios import ScenarioContractError
    from adapter_equivalence.server import ServerSpec

    arguments = {
        "revision_kind": "source",
        "model_path": "/models/qwen3-4b",
        "mode": "native_lora",
        "port": 31003,
        "tp_size": 1,
        "ep_size": 1,
        "cuda_graph": False,
        "max_lora_rank": 8,
        "lora_target_modules": ("q_proj",),
    }
    arguments[missing_field] = None if missing_field == "max_lora_rank" else ()

    with pytest.raises(
        ScenarioContractError,
        match="dynamic LoRA requires max_lora_rank and lora_target_modules",
    ):
        ServerSpec(**arguments)


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
        "mode": "native_oft",
        "port": 31002,
        "tp_size": 1,
        "ep_size": 1,
        "cuda_graph": False,
        "max_oft_block_size": 128,
        "peft_target_modules": ("q_proj",),
    }
    arguments[missing_field] = None if missing_field == "max_oft_block_size" else ()

    with pytest.raises(
        ScenarioContractError,
        match="dynamic OFT requires max_oft_block_size and peft_target_modules",
    ):
        ServerSpec(**arguments)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_runtime_id_comparison_preserves_identity_history():
    from copy import deepcopy

    from adapter_equivalence.compare import comparable_adapter_states

    def states(first, second):
        result = {}
        for request, adapter_id in (("first", first), ("second", second)):
            state = _adapter_state()
            state["registered"][0]["id"] = adapter_id
            state["active"]["id"] = adapter_id
            state["cache_identity"]["adapter-a"] = adapter_id
            result[request] = state
        return result

    baseline = states("source-a", "source-a")
    original = deepcopy(baseline)
    assert comparable_adapter_states(baseline) == comparable_adapter_states(
        states("candidate-a", "candidate-a")
    )
    assert baseline == original
    assert comparable_adapter_states(baseline) != comparable_adapter_states(
        states("candidate-a", "changed-id")
    )
    assert comparable_adapter_states(
        states("source-a", "new-registration")
    ) != comparable_adapter_states(states("candidate-a", "candidate-a"))


def test_compare_accepts_bijective_id_renaming():
    expected = _bundle()
    state = _adapter_state()
    for record in [*state["registered"], state["active"], state["staged"]]:
        record["id"] = "new-" + record["id"]
    state["cache_identity"] = {
        name: "new-" + value for name, value in state["cache_identity"].items()
    }
    actual = _replace_observation(expected, adapter_state=state)
    assert compare_bundles(expected, actual, _envelope(expected)).passed


@pytest.mark.parametrize(
    "message",
    [
        "changed diagnostic",
        "Requested adapter_id 'wrong-id' does not match expected adapter_id 'unobserved'",
        "Requested adapter_id 'different-request' does not match expected adapter_id 'id-a'",
    ],
)
def test_runtime_id_error_normalization_preserves_unverified_diagnostics(message):
    from adapter_equivalence.compare import (
        comparable_adapter_error,
        comparable_adapter_states,
    )

    state = _adapter_state()
    error = {"kind": "product_rejection", "code": "wrong_id", "message": message}
    normalized = comparable_adapter_states({"request": state})["request"]
    assert comparable_adapter_error(error, state, normalized) == error


def test_unverified_error_cannot_impersonate_a_normalized_runtime_id():
    from adapter_equivalence.compare import (
        comparable_adapter_error,
        comparable_adapter_states,
    )

    state = _adapter_state()
    normalized = comparable_adapter_states({"request": state})["request"]
    prefix = "Requested adapter_id 'wrong-id' does not match expected adapter_id '"
    valid = {
        "kind": "product_rejection",
        "code": "wrong_id",
        "message": prefix + "id-a'",
    }
    forged = dict(valid, message=prefix + "runtime-id-0'")
    assert comparable_adapter_error(
        valid, state, normalized
    ) != comparable_adapter_error(forged, state, normalized)


def _lease_prefix_bundle(length, *, partial_changes=None, final_changes=None):
    base = _bundle()
    final = replace(_observation(), **(final_changes or {}))
    partial = replace(
        final,
        output_ids=final.output_ids[:length],
        token_logprobs=final.token_logprobs[:length],
        request_output_lengths=(length,),
        text="alpha",
        request_texts=("alpha",),
        selected_logits={
            k: v
            for k, v in final.selected_logits.items()
            if int(k.split(".")[1]) < length
        },
        selected_token_ids={
            k: v
            for k, v in final.selected_token_ids.items()
            if int(k.split(".")[1]) < length
        },
    )
    partial = replace(partial, **(partial_changes or {}))
    observations = {"upsert.lease.begin": partial, "upsert.lease.complete": final}
    manifest = dict(base.manifest)
    manifest["request_order"] = list(observations)
    provenance = dict(base.provenance)
    provenance.pop("manifest_hash")
    return RunBundle.create(
        case_key=base.case_key,
        manifest=manifest,
        provenance=provenance,
        observations=observations,
        performance=base.performance,
        completion=base.completion,
    )


def test_lease_first_chunk_length_can_differ_with_exact_completed_response():
    expected, actual = _lease_prefix_bundle(2), _lease_prefix_bundle(1)
    before = actual.to_dict()
    report = compare_bundles(expected, actual, _envelope(expected))
    assert report.passed, report.mismatches
    assert actual.to_dict() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"output_ids": (999,)},
        {"token_logprobs": (-9.0,)},
        {"selected_logits": {"decode.000.top_logprobs": (9.0, 0.2, 0.3)}},
        {"selected_token_ids": {"decode.000.top_logprobs": (9, 11, 12)}},
    ],
)
def test_lease_corrupt_prefix_is_rejected_even_when_identical_across_runs(changes):
    corrupt = _lease_prefix_bundle(1, partial_changes=changes)
    report = compare_bundles(corrupt, corrupt, _envelope(corrupt))
    assert not report.passed


def test_lease_final_response_drift_remains_rejected():
    expected = _lease_prefix_bundle(1)
    actual = _lease_prefix_bundle(1, final_changes={"output_ids": (101, 202, 999)})
    report = compare_bundles(expected, actual, _envelope(expected))
    assert not report.passed

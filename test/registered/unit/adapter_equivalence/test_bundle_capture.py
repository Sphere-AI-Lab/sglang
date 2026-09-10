import copy
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence import bundle_capture
from adapter_equivalence.bundle_capture import (
    PerformanceRecorder,
    build_provenance,
    capture_control,
    capture_generation,
    normalize_expected_error,
    publish_bundle,
)
from adapter_equivalence.schema import (
    PROVENANCE_HASH_KEYS,
    SCHEMA_VERSION,
    BundleValidationError,
    CaseKey,
    Observation,
    PerformanceMetrics,
    RunBundle,
    validate_adapter_state,
)
from adapter_equivalence.server import ControlResult


def test_request_boundary_metadata_survives_capture_and_roundtrip():
    """Defect: serialized observations cannot recover individual request outputs."""
    observation = capture_generation(_complete_response(), _complete_state(), top_k=2)
    assert observation.request_output_lengths == (2,)
    assert observation.request_texts == ("ok",)
    encoded = json.loads(json.dumps(observation.to_dict()))
    assert encoded["request_output_lengths"] == [2]
    assert encoded["request_texts"] == ["ok"]
    assert Observation.from_dict(encoded) == observation
    control = capture_control(
        ControlResult(True, "ok", {}, None, None), _complete_state()
    )
    assert control.request_output_lengths == control.request_texts == ()
    assert SCHEMA_VERSION == 4


@pytest.mark.parametrize(
    "lengths,texts",
    (
        ([], []),
        ([0, 2], ["", "ok"]),
        ([1], ["ok"]),
        ([True, 1], ["a", "b"]),
        ([2], []),
        ([2], [3]),
    ),
)
def test_malformed_request_boundaries_are_rejected(lengths, texts):
    """Defect: boundary arrays admit empty requests, gaps, wrong dtypes or text counts."""
    value = capture_generation(
        _complete_response(), _complete_state(), top_k=2
    ).to_dict()
    value.update(request_output_lengths=lengths, request_texts=texts)
    with pytest.raises(BundleValidationError, match="request_"):
        Observation.from_dict(value)


@pytest.mark.parametrize("field", ("request_output_lengths", "request_texts"))
def test_serialized_request_boundaries_cannot_be_omitted(field):
    """Defect: missing serialized boundaries silently become a guessed single request."""
    value = capture_generation(
        _complete_response(), _complete_state(), top_k=2
    ).to_dict()
    value.pop(field)
    with pytest.raises(BundleValidationError, match=field):
        Observation.from_dict(value)


def _complete_state(mode: str = "native_oft") -> dict[str, object]:
    return {
        "mode": mode,
        "registered": [
            {
                "name": "policy-a",
                "id": "id-a",
                "version": "1",
                "registry_slot": 0,
                "pinned": False,
            }
        ],
        "active": {"name": "policy-a", "id": "id-a", "version": "1"},
        "staged": None,
        "registry_occupancy": 1,
        "quarantined": [],
        "tombstoned": [],
        "cache_identity": {"policy-a": "id-a"},
    }


def _complete_response() -> dict[str, object]:
    return {
        "output_ids": [11, 12],
        "text": "ok",
        "meta_info": {
            "output_token_logprobs": [(-0.1, 11, "o"), (-0.2, 12, "k")],
            "output_top_logprobs": [
                [(-0.1, 11, "o"), (-1.1, 7, "x")],
                [(-0.2, 12, "k"), (-1.2, 8, "y")],
            ],
        },
    }


def _tombstone_state(mode="native_oft"):
    state = _complete_state(mode)
    state.update(registered=[], active=None, registry_occupancy=0)
    state["tombstoned"] = [
        {"name": "policy-a", "id": "id-a", "version": "3", "pinned": True},
        {"name": "policy-z", "id": "id-z", "version": "9", "pinned": False},
    ]
    state["cache_identity"]["policy-z"] = "id-z"
    return state


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_exact_tombstones_roundtrip_without_activation_quarantine(mode):
    """Failed unload alone blocks admission; exact references survive freezing."""
    state = _tombstone_state(mode)
    observation = capture_control(
        ControlResult(True, "inspected", {}, None, None), state
    )
    assert observation.adapter_state["tombstoned"] == (
        {"name": "policy-a", "id": "id-a", "version": "3", "pinned": True},
        {"name": "policy-z", "id": "id-z", "version": "9", "pinned": False},
    )
    encoded = json.loads(json.dumps(observation.to_dict()))
    assert encoded["adapter_state"] == state
    assert Observation.from_dict(encoded) == observation
    empty = capture_control(
        ControlResult(True, "ok", {}, None, None), _complete_state(mode)
    )
    assert empty.adapter_state["tombstoned"] == ()
    assert empty.to_dict()["adapter_state"]["tombstoned"] == []


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize(
    "defect",
    (
        "legacy_names",
        "missing_field",
        "extra_field",
        "empty_name",
        "empty_id",
        "numeric_version",
        "numeric_pinned",
        "non_record",
        "not_array",
        "unsorted",
        "duplicate_name",
        "duplicate_id",
        "registered",
        "staged_name",
        "staged_id",
        "registered_id",
        "missing_cache",
        "wrong_cache",
    ),
)
def test_tombstone_schema_rejects_lossy_or_inconsistent_evidence(mode, defect):
    """Malformed identities or serving/cache overlap must fail closed."""
    state = _tombstone_state(mode)
    record = state["tombstoned"][0]
    if defect == "legacy_names":
        state["tombstoned"] = ["policy-a", "policy-z"]
    elif defect == "missing_field":
        record.pop("pinned")
    elif defect == "extra_field":
        record["registry_slot"] = 0
    elif defect in ("empty_name", "empty_id"):
        record[defect.removeprefix("empty_")] = ""
    elif defect == "numeric_version":
        record["version"] = 3
    elif defect == "numeric_pinned":
        record["pinned"] = 1
    elif defect == "non_record":
        state["tombstoned"][0] = None
    elif defect == "not_array":
        state["tombstoned"] = {"policy-a": record}
    elif defect == "unsorted":
        state["tombstoned"].reverse()
    elif defect == "duplicate_name":
        state["tombstoned"][1]["name"] = "policy-a"
    elif defect == "duplicate_id":
        state["tombstoned"][1]["id"] = "id-a"
        state["cache_identity"]["policy-z"] = "id-a"
    elif defect in ("registered", "registered_id"):
        name = "policy-a" if defect == "registered" else "policy-b"
        state["registered"] = [dict(record, name=name, registry_slot=0)]
        state["registry_occupancy"] = 1
        state["cache_identity"][name] = "id-a"
    elif defect in ("staged_name", "staged_id"):
        name = "policy-a" if defect == "staged_name" else "policy-b"
        state["staged"] = dict(record, name=name, version="4")
    elif defect == "missing_cache":
        state["cache_identity"].pop("policy-a")
    else:
        state["cache_identity"]["policy-a"] = "wrong-id"
    with pytest.raises(BundleValidationError, match="tombstoned"):
        validate_adapter_state(state)


def test_schema_v3_bundle_is_rejected_without_coercion(valid_bundle):
    payload = valid_bundle.to_dict()
    payload["schema_version"] = 3
    with pytest.raises(BundleValidationError, match="schema_version must equal 4"):
        RunBundle.from_dict(payload)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_tombstone_rejects_staged_same_name_with_different_id(mode):
    """Checking only the staged ID would admit a tombstoned name again."""
    state = _tombstone_state(mode)
    state["staged"] = {
        "name": "policy-a",
        "id": "replacement-a",
        "version": "4",
        "pinned": False,
    }
    with pytest.raises(BundleValidationError, match="absent from staged"):
        validate_adapter_state(state)


def _case() -> CaseKey:
    return CaseKey(
        model="Qwen/Qwen3-4B-Instruct-2507",
        architecture="dense",
        precision="bf16",
        revision="a" * 40,
        mode="native_oft",
        cuda_graph=False,
        scenario="native-adapter-lifecycle-v2",
    )


def _hashes() -> dict[str, str]:
    return {
        key: chr(ord("1") + index) * 64
        for index, key in enumerate(PROVENANCE_HASH_KEYS)
    }


@pytest.fixture
def valid_bundle() -> RunBundle:
    case = _case()
    hashes = _hashes()
    procedure_hash = "f" * 64
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_key": case.to_dict(),
        "performance_procedure_hash": procedure_hash,
        "provenance_hashes": hashes,
        "server_args": ["--peft-method", "oft"],
        "request_order": ["base.initial"],
        "seed": 1729,
        "metadata": {},
    }
    return RunBundle.create(
        case_key=case,
        manifest=manifest,
        provenance=build_provenance(case, hashes, {"role": "candidate"}),
        observations={
            "base.initial": capture_generation(
                _complete_response(), _complete_state(), top_k=2
            )
        },
        performance=PerformanceMetrics(
            procedure_hash=procedure_hash,
            startup_seconds=(1.0, 1.1, 0.9),
            latency_seconds=(0.1, 0.11, 0.09),
            throughput_tokens_per_second=(100.0, 101.0, 99.0),
            peak_allocated_bytes=(100, 101, 99),
            peak_reserved_bytes=(120, 121, 119),
        ),
        completion={"status": "complete", "exit_code": 0, "metadata": {}},
    )


def test_complete_response_becomes_observation() -> None:
    observation = capture_generation(_complete_response(), _complete_state(), top_k=2)

    assert observation.output_ids == (11, 12)
    assert observation.text == "ok"
    assert observation.token_logprobs == (-0.1, -0.2)
    assert observation.selected_logits == {
        "decode.000.top_logprobs": (-1.1, -0.1),
        "decode.001.top_logprobs": (-1.2, -0.2),
    }
    assert observation.selected_token_ids == {
        "decode.000.top_logprobs": (7, 11),
        "decode.001.top_logprobs": (8, 12),
    }
    assert observation.to_dict()["adapter_state"] == _complete_state()
    assert observation.error is None


def test_selected_token_ids_are_part_of_generation_evidence() -> None:
    expected_response = _complete_response()
    actual_response = _complete_response()
    actual_response["meta_info"]["output_top_logprobs"][0][1] = (-1.1, 6, "x")

    expected = capture_generation(expected_response, _complete_state(), top_k=2)
    actual = capture_generation(actual_response, _complete_state(), top_k=2)

    assert actual.selected_logits == expected.selected_logits
    assert actual.selected_token_ids != expected.selected_token_ids


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda response: response.pop("output_ids"), "output_ids"),
        (lambda response: response.pop("meta_info"), "meta_info"),
        (
            lambda response: response["meta_info"].pop("output_token_logprobs"),
            "output_token_logprobs",
        ),
        (
            lambda response: response["meta_info"].update(
                {"output_token_logprobs": [(-0.1, 11, "o")]}
            ),
            "token/logprob lengths",
        ),
        (
            lambda response: response["meta_info"].pop("output_top_logprobs"),
            "output_top_logprobs",
        ),
        (
            lambda response: response["meta_info"].update(
                {"output_top_logprobs": [[(-0.1, 11, "o"), (-1.1, 7, "x")]]}
            ),
            "selected score rows",
        ),
        (
            lambda response: response["meta_info"]["output_top_logprobs"][0].pop(),
            "selected score width",
        ),
    ),
)
def test_generation_capture_rejects_incomplete_response(mutation, match: str) -> None:
    response = _complete_response()
    mutation(response)

    with pytest.raises(BundleValidationError, match=match):
        capture_generation(response, _complete_state(), top_k=2)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda response: response["output_ids"].__setitem__(0, "11"), "integers"),
        (
            lambda response: response["meta_info"]["output_token_logprobs"].__setitem__(
                0, (-0.1, 99, "o")
            ),
            "token ID",
        ),
        (
            lambda response: response["meta_info"]["output_token_logprobs"].__setitem__(
                0, (float("nan"), 11, "o")
            ),
            "finite float",
        ),
        (
            lambda response: response["meta_info"]["output_top_logprobs"][
                0
            ].__setitem__(0, (float("inf"), 11, "o")),
            "finite float",
        ),
        (
            lambda response: response["meta_info"]["output_top_logprobs"][
                0
            ].__setitem__(1, (-1.1, 11, "x")),
            "duplicate token IDs",
        ),
    ),
)
def test_generation_capture_rejects_malformed_scores(mutation, match: str) -> None:
    response = _complete_response()
    mutation(response)

    with pytest.raises(BundleValidationError, match=match):
        capture_generation(response, _complete_state(), top_k=2)


@pytest.mark.parametrize("top_k", (0, -1, True, 1.5))
def test_generation_capture_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(BundleValidationError, match="top_k"):
        capture_generation(_complete_response(), _complete_state(), top_k=top_k)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda state: state.pop("active"), "missing fields.*active"),
        (lambda state: state.update({"extra": True}), "unknown fields.*extra"),
        (
            lambda state: state.update({"registry_occupancy": 0}),
            "registry_occupancy",
        ),
        (
            lambda state: state["active"].update({"id": "wrong-id"}),
            "active.*registered",
        ),
        (
            lambda state: state["registered"][0].update({"version": 1}),
            "version.*string",
        ),
        (
            lambda state: state["cache_identity"].update({"policy-a": "wrong-id"}),
            "cache_identity.*registered",
        ),
        (
            lambda state: state.update(
                {
                    "staged": {
                        "name": "policy-a",
                        "id": "wrong-id",
                        "version": "2",
                        "pinned": False,
                    }
                }
            ),
            "staged.*registered",
        ),
    ),
)
def test_generation_capture_rejects_incomplete_or_inconsistent_state(
    mutation, match: str
) -> None:
    state = _complete_state()
    mutation(state)

    with pytest.raises(BundleValidationError, match=match):
        capture_generation(_complete_response(), state, top_k=2)


def test_base_state_must_be_completely_empty() -> None:
    state = _complete_state("base")

    with pytest.raises(BundleValidationError, match="base adapter state"):
        capture_generation(_complete_response(), state, top_k=2)


def test_control_capture_records_success_with_empty_model_output() -> None:
    result = ControlResult(True, "loaded", {"policy-a": "id-a"}, "1", None)

    observation = capture_control(result, _complete_state())

    assert observation.output_ids == ()
    assert observation.text == ""
    assert observation.token_logprobs == ()
    assert observation.selected_logits == {}
    assert observation.selected_token_ids == {}
    assert observation.to_dict()["adapter_state"] == _complete_state()
    assert observation.error is None


def test_expected_control_rejection_is_normalized_exactly() -> None:
    result = ControlResult(False, "  stale\n adapter   version  ", {}, "1", None)

    error = normalize_expected_error(
        result,
        expected_code="stale_version",
        returned_code="stale_version",
    )
    observation = capture_control(
        result,
        _complete_state(),
        expected_error_code="stale_version",
        returned_error_code="stale_version",
    )

    assert error == {
        "kind": "product_rejection",
        "code": "stale_version",
        "message": "stale adapter version",
    }
    assert observation.error == error


@pytest.mark.parametrize(
    "result,expected_code,returned_code,match",
    (
        (
            ControlResult(False, "failed", {}, None, None),
            None,
            "unexpected",
            "undeclared",
        ),
        (
            ControlResult(False, "failed", {}, None, None),
            "stale_version",
            None,
            "omitted",
        ),
        (
            ControlResult(False, "failed", {}, None, None),
            "stale_version",
            "wrong_id",
            "does not match",
        ),
        (
            ControlResult(True, "ok", {}, None, None),
            "stale_version",
            "stale_version",
            "succeeded",
        ),
    ),
)
def test_control_capture_fails_closed_on_wrong_rejection(
    result: ControlResult,
    expected_code: str | None,
    returned_code: str | None,
    match: str,
) -> None:
    with pytest.raises(BundleValidationError, match=match):
        capture_control(
            result,
            _complete_state(),
            expected_error_code=expected_code,
            returned_error_code=returned_code,
        )


def test_control_capture_does_not_swallow_state_exception() -> None:
    state = _complete_state()
    state["registered"][0]["pinned"] = "no"

    with pytest.raises(BundleValidationError, match="pinned"):
        capture_control(ControlResult(True, "ok", {}, None, None), state)


@pytest.mark.parametrize(
    "result,match",
    (
        (
            ControlResult(True, "activated", {}, "2", None),
            "active_version.*state",
        ),
        (
            ControlResult(True, "staged", {}, "1", "2"),
            "staged_version.*state",
        ),
    ),
)
def test_control_capture_rejects_result_versions_missing_from_state(
    result: ControlResult, match: str
) -> None:
    with pytest.raises(BundleValidationError, match=match):
        capture_control(result, _complete_state())


def test_control_capture_accepts_exact_staged_result_version() -> None:
    state = _complete_state()
    state["staged"] = {
        "name": "policy-a",
        "id": "id-a",
        "version": "2",
        "pinned": False,
    }

    observation = capture_control(ControlResult(True, "staged", {}, "1", "2"), state)

    assert observation.adapter_state["staged"]["version"] == "2"


def test_performance_recorder_builds_exactly_three_real_samples() -> None:
    recorder = PerformanceRecorder("f" * 64)
    for startup in (9.0, 8.0, 7.0):
        recorder.record_startup(startup)
    recorder.record_sample(
        latency_seconds=0.5,
        output_tokens=50,
        peak_allocated_bytes=100,
        peak_reserved_bytes=120,
    )
    recorder.record_sample(
        latency_seconds=0.25,
        output_tokens=50,
        peak_allocated_bytes=101,
        peak_reserved_bytes=121,
    )
    recorder.record_sample(
        latency_seconds=0.2,
        output_tokens=50,
        peak_allocated_bytes=99,
        peak_reserved_bytes=119,
    )

    metrics = recorder.build()

    assert metrics.startup_seconds == (9.0, 8.0, 7.0)
    assert metrics.latency_seconds == (0.5, 0.25, 0.2)
    assert metrics.throughput_tokens_per_second == (100.0, 200.0, 250.0)
    assert metrics.peak_allocated_bytes == (100, 101, 99)
    assert metrics.peak_reserved_bytes == (120, 121, 119)


def test_performance_recorder_rejects_missing_or_extra_samples() -> None:
    recorder = PerformanceRecorder("f" * 64)
    for startup in (9.0, 8.0):
        recorder.record_startup(startup)
    for _ in range(3):
        recorder.record_sample(
            latency_seconds=0.5,
            output_tokens=50,
            peak_allocated_bytes=100,
            peak_reserved_bytes=120,
        )

    with pytest.raises(BundleValidationError, match="startup_seconds"):
        recorder.build()

    recorder.record_startup(7.0)
    with pytest.raises(BundleValidationError, match="exactly 3"):
        recorder.record_startup(6.0)


@pytest.mark.parametrize(
    "kwargs,match",
    (
        (
            {
                "latency_seconds": 0.0,
                "output_tokens": 50,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 120,
            },
            "latency_seconds",
        ),
        (
            {
                "latency_seconds": 0.5,
                "output_tokens": True,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 120,
            },
            "output_tokens",
        ),
        (
            {
                "latency_seconds": 0.5,
                "output_tokens": 50,
                "peak_allocated_bytes": -1,
                "peak_reserved_bytes": 120,
            },
            "peak_allocated_bytes",
        ),
    ),
)
def test_performance_recorder_rejects_invalid_samples(kwargs, match: str) -> None:
    recorder = PerformanceRecorder("f" * 64)

    with pytest.raises(BundleValidationError, match=match):
        recorder.record_sample(**kwargs)


def test_build_provenance_uses_exact_case_and_hash_identity() -> None:
    provenance = build_provenance(
        _case(), _hashes(), {"argv": ["run_case.py"], "role": "candidate"}
    )

    assert provenance == {
        "git_sha": "a" * 40,
        "dirty": False,
        **_hashes(),
        "metadata": {"argv": ["run_case.py"], "role": "candidate"},
    }


def test_build_provenance_rejects_missing_hash_or_dirty_checkout() -> None:
    hashes = _hashes()
    hashes.pop("hardware_hash")

    with pytest.raises(BundleValidationError, match="hardware_hash"):
        build_provenance(_case(), hashes, {})
    with pytest.raises(BundleValidationError, match="dirty"):
        build_provenance(_case(), _hashes(), {}, dirty=True)


def test_completion_is_written_only_after_valid_bundle(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"

    publish_bundle(valid_bundle, bundle_path, completion_path)

    assert RunBundle.read_json(bundle_path).digest() == valid_bundle.digest()
    marker = json.loads(completion_path.read_text())
    assert marker == {
        "bundle_hash": valid_bundle.digest(),
        "status": "complete",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_refuses_existing_bundle(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"
    bundle_path.write_text("occupied")

    with pytest.raises(FileExistsError):
        publish_bundle(valid_bundle, bundle_path, completion_path)

    assert bundle_path.read_text() == "occupied"
    assert not completion_path.exists()


def test_publish_refuses_existing_marker_before_writing_bundle(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"
    completion_path.write_text("occupied")

    with pytest.raises(FileExistsError):
        publish_bundle(valid_bundle, bundle_path, completion_path)

    assert not bundle_path.exists()
    assert completion_path.read_text() == "occupied"


def test_concurrent_publish_has_one_winner_and_one_valid_marker(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"

    def publish() -> Exception | None:
        try:
            publish_bundle(valid_bundle, bundle_path, completion_path)
        except Exception as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish(), range(2)))

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert RunBundle.read_json(bundle_path).digest() == valid_bundle.digest()
    assert json.loads(completion_path.read_text()) == {
        "bundle_hash": valid_bundle.digest(),
        "status": "complete",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_bundle_creates_no_files(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    invalid = replace(valid_bundle, manifest_hash="0" * 64)
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"

    with pytest.raises(BundleValidationError, match="manifest_hash"):
        publish_bundle(invalid, bundle_path, completion_path)

    assert not bundle_path.exists()
    assert not completion_path.exists()


def test_corrupt_temporary_serialization_creates_no_publication(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    def corrupt(descriptor: int, bundle: RunBundle) -> None:
        del bundle
        with open(descriptor, "w", closefd=False) as stream:
            stream.write("{}\n")
            stream.flush()

    monkeypatch.setattr(bundle_capture, "_serialize_bundle", corrupt)
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"

    with pytest.raises(BundleValidationError):
        publish_bundle(valid_bundle, bundle_path, completion_path)

    assert not bundle_path.exists()
    assert not completion_path.exists()


def test_corrupt_published_readback_withholds_completion(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    original = RunBundle.read_json.__func__
    reads = 0

    def read_json(cls, path):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise BundleValidationError("corrupt published read-back")
        return original(cls, path)

    monkeypatch.setattr(RunBundle, "read_json", classmethod(read_json))
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"

    with pytest.raises(BundleValidationError, match="published read-back"):
        publish_bundle(valid_bundle, bundle_path, completion_path)

    assert bundle_path.exists()
    assert not completion_path.exists()


def test_marker_race_never_deletes_another_publishers_marker(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"
    original_link = bundle_capture.os.link
    calls = 0

    def competing_link(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            completion_path.write_text("other publisher")
        original_link(source, destination)

    monkeypatch.setattr(bundle_capture.os, "link", competing_link)

    with pytest.raises(FileExistsError):
        publish_bundle(valid_bundle, bundle_path, completion_path)

    assert completion_path.read_text() == "other publisher"


def test_marker_preparation_durability_failure_withholds_completion(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"
    original = bundle_capture._fsync_directory
    calls = 0

    def fail_marker_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("directory fsync failed")
        original(path)

    monkeypatch.setattr(bundle_capture, "_fsync_directory", fail_marker_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        publish_bundle(valid_bundle, bundle_path, completion_path)

    assert bundle_path.exists()
    assert not completion_path.exists()


def test_private_marker_cleanup_failure_does_not_retract_completion(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    bundle_path = tmp_path / "case.bundle.json"
    completion_path = tmp_path / "case.complete.json"
    original_unlink = Path.unlink

    def fail_marker_temp_cleanup(path: Path, *args, **kwargs) -> None:
        if path.name.startswith(f".{completion_path.name}."):
            raise OSError("private marker cleanup failed")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_marker_temp_cleanup)

    publish_bundle(valid_bundle, bundle_path, completion_path)

    assert json.loads(completion_path.read_text()) == {
        "bundle_hash": valid_bundle.digest(),
        "status": "complete",
    }


def test_cleanup_close_failure_does_not_replace_primary_error(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    def fail_serialization(descriptor: int, bundle: RunBundle) -> None:
        del descriptor, bundle
        raise RuntimeError("primary serialization failure")

    def fail_close(descriptor: int) -> None:
        del descriptor
        raise OSError("cleanup close failed")

    monkeypatch.setattr(bundle_capture, "_serialize_bundle", fail_serialization)
    monkeypatch.setattr(bundle_capture.os, "close", fail_close)

    with pytest.raises(RuntimeError, match="primary serialization failure"):
        publish_bundle(
            valid_bundle,
            tmp_path / "case.bundle.json",
            tmp_path / "case.complete.json",
        )


def test_explicit_close_failure_is_not_retried(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    original_close = bundle_capture.os.close
    close_attempts: list[int] = []

    def close_then_report_error(descriptor: int) -> None:
        close_attempts.append(descriptor)
        original_close(descriptor)
        raise OSError("explicit close failed")

    monkeypatch.setattr(bundle_capture.os, "close", close_then_report_error)

    with pytest.raises(OSError, match="explicit close failed"):
        publish_bundle(
            valid_bundle,
            tmp_path / "case.bundle.json",
            tmp_path / "case.complete.json",
        )

    assert len(close_attempts) == 1


def test_explicit_marker_close_failure_is_not_retried(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    original_mkstemp = bundle_capture.tempfile.mkstemp
    original_close = bundle_capture.os.close
    marker_descriptor: int | None = None
    marker_close_attempts = 0
    temporary_files = 0

    def recording_mkstemp(*args, **kwargs):
        nonlocal marker_descriptor, temporary_files
        descriptor, path = original_mkstemp(*args, **kwargs)
        temporary_files += 1
        if temporary_files == 2:
            marker_descriptor = descriptor
        return descriptor, path

    def close_marker_then_report_error(descriptor: int) -> None:
        nonlocal marker_close_attempts
        if descriptor == marker_descriptor:
            marker_close_attempts += 1
        original_close(descriptor)
        if descriptor == marker_descriptor:
            raise OSError("explicit marker close failed")

    monkeypatch.setattr(bundle_capture.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(bundle_capture.os, "close", close_marker_then_report_error)

    completion_path = tmp_path / "case.complete.json"
    with pytest.raises(OSError, match="explicit marker close failed"):
        publish_bundle(
            valid_bundle,
            tmp_path / "case.bundle.json",
            completion_path,
        )

    assert marker_close_attempts == 1
    assert not completion_path.exists()


def test_cleanup_unlink_failure_does_not_replace_primary_error(
    monkeypatch, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    def fail_serialization(descriptor: int, bundle: RunBundle) -> None:
        del descriptor, bundle
        raise RuntimeError("primary serialization failure")

    def fail_unlink(path: Path, *args, **kwargs) -> None:
        del path, args, kwargs
        raise OSError("cleanup unlink failed")

    monkeypatch.setattr(bundle_capture, "_serialize_bundle", fail_serialization)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError, match="primary serialization failure"):
        publish_bundle(
            valid_bundle,
            tmp_path / "case.bundle.json",
            tmp_path / "case.complete.json",
        )


def test_schema_rejects_state_or_error_corruption_on_readback(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    state_path = tmp_path / "state.json"
    valid_bundle.write_json(state_path)
    state_payload = json.loads(state_path.read_text())
    del state_payload["observations"]["base.initial"]["adapter_state"]["active"]
    state_path.write_text(json.dumps(state_payload))

    with pytest.raises(BundleValidationError, match="missing fields.*active"):
        RunBundle.read_json(state_path)

    error_bundle = replace(
        valid_bundle,
        observations={
            "base.initial": replace(
                valid_bundle.observations["base.initial"],
                error={"kind": "product_rejection", "code": "stale"},
            )
        },
    )
    error_path = tmp_path / "error.json"
    payload = copy.deepcopy(error_bundle.to_dict())
    error_path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="missing fields.*message"):
        RunBundle.read_json(error_path)


def test_schema_rejects_missing_selected_rows_for_generation(
    tmp_path: Path, valid_bundle: RunBundle
) -> None:
    path = tmp_path / "missing-selected-rows.json"
    payload = copy.deepcopy(valid_bundle.to_dict())
    observation = payload["observations"]["base.initial"]
    observation["selected_logits"] = {}
    observation["selected_token_ids"] = {}
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match="selected.*rows"):
        RunBundle.read_json(path)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda observation: observation["selected_token_ids"].pop(
                "decode.001.top_logprobs"
            ),
            "selected_token_ids rows",
        ),
        (
            lambda observation: observation["selected_token_ids"][
                "decode.000.top_logprobs"
            ].__setitem__(0, True),
            "selected_token_ids.*integers",
        ),
        (
            lambda observation: observation["selected_token_ids"].__setitem__(
                "decode.000.top_logprobs", [11, 7]
            ),
            "selected_token_ids.*unique and sorted",
        ),
        (
            lambda observation: observation["selected_token_ids"][
                "decode.000.top_logprobs"
            ].pop(),
            "score and token ID widths",
        ),
    ),
)
def test_schema_rejects_corrupt_selected_token_rows(
    mutation, match: str, tmp_path: Path, valid_bundle: RunBundle
) -> None:
    path = tmp_path / "corrupt-selected-token-rows.json"
    payload = copy.deepcopy(valid_bundle.to_dict())
    mutation(payload["observations"]["base.initial"])
    path.write_text(json.dumps(payload))

    with pytest.raises(BundleValidationError, match=match):
        RunBundle.read_json(path)


def test_bundle_rejects_adapter_state_from_another_mode(
    valid_bundle: RunBundle,
) -> None:
    wrong_mode = capture_generation(
        _complete_response(), _complete_state("native_lora"), top_k=2
    )
    bundle = replace(
        valid_bundle,
        observations={"base.initial": wrong_mode},
    )

    with pytest.raises(
        BundleValidationError, match="adapter_state.mode.*case_key.mode"
    ):
        bundle.validate()


def qualification_inputs(tmp_path, monkeypatch):
    """Real immutable files and Git checkout; only machine inventory is external."""
    from adapter_equivalence.preflight import PINNED_MODEL_REVISIONS, hash_checkpoint
    from adapter_equivalence.run_case import RunSpec
    from adapter_equivalence.server import ServerSpec

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "code.py").write_text("pass\n")
    runtime_files = (
        "sglang/utils.py",
        "sglang/srt/entrypoints/engine.py",
        "sglang/srt/server_args.py",
        "sglang/srt/utils/common.py",
        "sglang/srt/managers/io_struct.py",
        "sglang/srt/managers/scheduler.py",
        "sglang/srt/managers/tokenizer_manager.py",
        "sglang/srt/managers/tokenizer_control_mixin.py",
        "sglang/srt/managers/communicator.py",
    )
    for relative in runtime_files:
        target = repo / "python" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            Path(__file__).resolve().parents[4] / "python" / relative, target
        )
    for package in (
        "sglang",
        "sglang/srt/utils",
    ):
        (repo / "python" / package / "__init__.py").write_text("")
    for args in (
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Harness",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    model = tmp_path / "model"
    model.mkdir()
    for name, content in {
        "config.json": "{}",
        "model.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
    }.items():
        (model / name).write_text(content)
    checkpoint = hash_checkpoint(model, model="test/model", revision="b" * 40)
    entry = dict(vars(checkpoint), id="cell", path=str(model))
    checkpoints = tmp_path / "checkpoints.json"
    prompts = tmp_path / "prompts.jsonl"
    tokenizer_files = {
        name: checkpoint.files[name]
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    records = [
        {
            "kind": "metadata",
            "schema_version": 1,
            "tokenizer": {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "revision": PINNED_MODEL_REVISIONS["qwen3-4b-bf16"],
                "add_special_tokens": False,
                "files": tokenizer_files,
            },
        }
    ]
    names = (
        "factual",
        "arithmetic",
        "code",
        "long-prefix",
        "uneven-mixed",
        "graph-bucket",
    )
    records += [
        {"kind": "prompt", "id": name, "text": name, "input_ids": [10 + i]}
        for i, name in enumerate(names)
    ]
    records += [
        {
            "kind": "batch",
            "id": f"batch-{size}",
            "temperature": 0,
            "requests": [
                {"id": f"batch-{size}-request-{i:02d}", "prompt_id": names[i % 6]}
                for i in range(size)
            ],
        }
        for size in (1, 2, 8, 32)
    ]
    prompts.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    checkpoints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matrix_sha256": "c" * 64,
                "checkpoints": [entry],
                "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                "prompt_tokenizer": {
                    key: records[0]["tokenizer"][key]
                    for key in ("model", "revision", "files")
                },
            }
        )
    )
    fixtures_file = tmp_path / "fixtures.json"
    fixtures_file.write_text("{}")
    spec = RunSpec(
        ServerSpec("candidate", str(model), "base", 30000, 2, 1, False),
        "cell",
        sha,
        "dense",
        "bf16",
        checkpoints,
        prompts,
        fixtures_file,
        tmp_path / "bundle.json",
        tmp_path / "complete.json",
    )
    environment = {
        "python": "3.11.9",
        "packages": [["torch", "2.7.0"]],
        "pytorch": "2.7.0",
        "cuda": "12.8",
        "driver": "570.1",
    }
    hardware = {
        "gpus": [
            {
                "index": i,
                "name": "NVIDIA H200",
                "memory_mib": 143771,
                "compute_capability": "9.0",
            }
            for i in range(3)
        ],
        "topology": "GPU0 X NV18 NV18\nGPU1 NV18 X NV18\nGPU2 NV18 NV18 X",
        "visible_devices": [0, 1, 2],
    }
    monkeypatch.setattr(bundle_capture, "REPO_ROOT", repo, raising=False)
    monkeypatch.setattr(
        bundle_capture, "environment_inventory", lambda: environment, raising=False
    )
    monkeypatch.setattr(
        bundle_capture, "hardware_inventory", lambda: hardware, raising=False
    )
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    _clear_runtime_modules(monkeypatch)
    return SimpleNamespace(
        spec=spec,
        repo=repo,
        entry=entry,
        environment=environment,
        hardware=hardware,
        records=records,
    )


@pytest.mark.parametrize(
    "defect", ("sha", "dirty", "missing", "mutated", "prompts", "hardware")
)
def test_run_provenance_rejects_unverified_inputs(tmp_path, monkeypatch, defect):
    """Wrong checkout, changed bytes, or incompatible GPU evidence cannot qualify."""
    inputs = qualification_inputs(tmp_path, monkeypatch)
    spec = inputs.spec
    if defect == "sha":
        spec = replace(spec, revision_sha="f" * 40)
    elif defect == "dirty":
        (inputs.repo / "code.py").write_text("changed")
    elif defect == "missing":
        (Path(spec.server.model_path) / "model.safetensors").unlink()
    elif defect == "mutated":
        (Path(spec.server.model_path) / "model.safetensors").write_text("mutated")
    elif defect == "prompts":
        spec.prompts_file.write_text(spec.prompts_file.read_text() + "\n")
    else:
        inputs.hardware["gpus"][1]["name"] = "NVIDIA H100"
    with pytest.raises((BundleValidationError, ValueError, OSError)):
        bundle_capture.capture_run_identity(spec, {}, ["run_case.py"])


def test_run_provenance_records_exact_identity(tmp_path, monkeypatch):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    case, provenance = bundle_capture.capture_run_identity(
        inputs.spec, {}, ["runner", "--mode", "base"]
    )
    assert case.model == "test/model"
    assert case.revision == inputs.spec.revision_sha
    assert provenance["dirty"] is False
    assert provenance["metadata"]["argv"] == ["runner", "--mode", "base"]
    assert provenance["metadata"]["role"] == "candidate"
    assert provenance["metadata"]["repetition"] == 0
    assert provenance["metadata"]["slurm_job_id"] == "1234"
    assert set(PROVENANCE_HASH_KEYS) <= provenance.keys()
    assert provenance["tokenizer_hash"] == inputs.entry["tokenizer_hash"]
    again = bundle_capture.capture_run_identity(
        replace(
            inputs.spec, server=replace(inputs.spec.server, revision_kind="source")
        ),
        {},
        ["different"],
    )[1]
    assert all(provenance[key] == again[key] for key in PROVENANCE_HASH_KEYS)


@pytest.mark.parametrize("mismatched_bytes", [False, True])
def test_run_identity_accepts_pinned_moe_tokenizer_only_with_exact_bytes(
    tmp_path, monkeypatch, mismatched_bytes
):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    spec = inputs.spec
    records = [json.loads(line) for line in spec.prompts_file.read_text().splitlines()]
    tokenizer = records[0]["tokenizer"]
    tokenizer.update(
        model="Qwen/Qwen3-30B-A3B",
        revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    )
    if mismatched_bytes:
        tokenizer["files"]["tokenizer.json"] = "f" * 64
    spec.prompts_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    document = json.loads(spec.checkpoint_manifest.read_text())
    document["prompt_tokenizer"] = {
        key: tokenizer[key] for key in ("model", "revision", "files")
    }
    document["prompts_sha256"] = bundle_capture._file_hash(spec.prompts_file)
    spec.checkpoint_manifest.write_text(json.dumps(document))
    if mismatched_bytes:
        with pytest.raises(BundleValidationError, match="prompt tokenizer differs"):
            bundle_capture.capture_run_identity(spec, {}, [])
    else:
        bundle_capture.capture_run_identity(spec, {}, [])


@pytest.mark.parametrize(
    "defect",
    (
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_files",
        "unknown_field",
        "schema_bool",
        "matrix_hash",
        "duplicate_entry",
    ),
)
def test_run_identity_validates_checkpoint_manifest_consistency(
    tmp_path, monkeypatch, defect
):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    document = json.loads(inputs.spec.checkpoint_manifest.read_text())
    if defect.startswith("tokenizer_"):
        field = defect.removeprefix("tokenizer_")
        document["prompt_tokenizer"][field] = {} if field == "files" else "wrong"
    elif defect == "unknown_field":
        document["ignored"] = True
    elif defect == "schema_bool":
        document["schema_version"] = True
    elif defect == "matrix_hash":
        document["matrix_sha256"] = "unknown"
    else:
        document["checkpoints"].append(document["checkpoints"][0])
    inputs.spec.checkpoint_manifest.write_text(json.dumps(document))
    with pytest.raises(BundleValidationError):
        bundle_capture.capture_run_identity(inputs.spec, {}, [])


@pytest.mark.parametrize(
    "filename",
    ("adapter_config.json", "adapter_model.safetensors", "fixture_metadata.json"),
)
def test_native_provenance_hash_covers_every_fixture_byte(
    tmp_path, monkeypatch, filename
):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    fixtures = {}
    for name in ("policy-a", "policy-b"):
        folder = tmp_path / name
        folder.mkdir()
        for file in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "fixture_metadata.json",
        ):
            (folder / file).write_text("{}" if file.endswith("json") else "tensor")
        fixtures[name] = folder
    fixtures.update({"2": fixtures["policy-a"], "3": fixtures["policy-b"]})
    inputs.spec.fixture_manifest.write_text(
        json.dumps({key: str(value) for key, value in fixtures.items()})
    )
    spec = replace(
        inputs.spec,
        server=replace(
            inputs.spec.server,
            mode="native_lora",
            startup_adapters=(("policy-a", str(fixtures["policy-a"])),),
        ),
    )
    before = bundle_capture.capture_run_identity(spec, fixtures, [])[1]
    (fixtures["policy-b"] / filename).write_text("changed bytes")
    after = bundle_capture.capture_run_identity(spec, fixtures, [])[1]
    assert before["adapter_hash"] != after["adapter_hash"]
    assert before["checkpoint_hash"] == after["checkpoint_hash"]


def test_tokenizer_hash_covers_auxiliary_tokenizer_bytes(tmp_path, monkeypatch):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    template = Path(inputs.spec.server.model_path) / "chat_template.jinja"
    template.write_text("template one")
    before = bundle_capture.capture_run_identity(inputs.spec, {}, [])[1]
    template.write_text("template two")
    after = bundle_capture.capture_run_identity(inputs.spec, {}, [])[1]
    assert before["tokenizer_hash"] != after["tokenizer_hash"]


def test_hardware_inventory_resolves_allocation_without_parent_cuda(
    monkeypatch,
):
    """Inventory must not initialize CUDA contexts on model or sender GPUs."""

    def command(argv):
        if argv == ["nvidia-smi", "topo", "-m"]:
            return "        GPU0 GPU1 GPU2 CPU Affinity NUMA Affinity GPU NUMA ID\nGPU0 X NV18 SYS 0-31 0 N/A\nGPU1 NV18 X SYS 0-31 0 N/A\nGPU2 SYS SYS X 32-63 1 N/A\n"
        assert argv[:2] == ["nvidia-smi", "-i"]
        assert argv[3:] == [
            "--query-gpu=index,uuid,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
        return {
            "GPU-b": "1, GPU-b, NVIDIA H200 B, 143771, 9.0\n",
            "0": "0, GPU-a, NVIDIA H200 A, 143771, 9.0\n",
        }[argv[2]]

    monkeypatch.setattr(bundle_capture, "_command", command)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,0")
    inventory = bundle_capture.hardware_inventory()
    assert inventory["gpus"] == [
        {"index": i, "name": name, "memory_mib": 143771, "compute_capability": "9.0"}
        for i, name in enumerate(("NVIDIA H200 B", "NVIDIA H200 A"))
    ]
    assert inventory["topology"] == [["X", "NV18"], ["NV18", "X"]]
    assert "GPU-a" not in json.dumps(inventory)
    assert "GPU-c" not in json.dumps(inventory)


def test_hardware_inventory_accepts_ansi_formatted_topology_header(monkeypatch):
    def command(argv):
        if argv == ["nvidia-smi", "topo", "-m"]:
            return (
                "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\t"
                "CPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
                "GPU1\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t"
                "0-77\t0\tN/A\n"
                "GPU2\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\t"
                "0-77\t0\tN/A\n"
            )
        return {
            "1": "1, GPU-one, NVIDIA H200, 143771, 9.0\n",
            "2": "2, GPU-two, NVIDIA H200, 143771, 9.0\n",
        }[argv[2]]

    monkeypatch.setattr(bundle_capture, "_command", command)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")

    inventory = bundle_capture.hardware_inventory()

    assert inventory["visible_devices"] == [0, 1]
    assert inventory["topology"] == [["X", "NV18"], ["NV18", "X"]]


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_qualification_rejects_sender_model_gpu_overlap(tmp_path, monkeypatch, mode):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    spec = replace(
        inputs.spec,
        server=replace(
            inputs.spec.server,
            mode=mode,
            base_gpu_id=0,
            startup_adapters=(("policy-a", "/fixture"),),
        ),
    )
    with pytest.raises(BundleValidationError, match="sender|overlap"):
        bundle_capture.capture_run_identity(spec, {}, [])


def test_environment_inventory_ignores_import_added_vendor_search_path(
    tmp_path, monkeypatch
):
    installed = tmp_path / "site-packages"
    vendor = installed / "setuptools" / "_vendor"

    def metadata(root, name, version):
        info = root / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")

    metadata(installed, "packaging", "26.1")
    metadata(vendor, "packaging", "26.0")
    monkeypatch.setattr(sys, "path", [str(installed)])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.7.0", version=SimpleNamespace(cuda="12.8")),
    )
    monkeypatch.setattr(bundle_capture, "_command", lambda argv: "570.1\n")
    before = bundle_capture.environment_inventory()
    sys.path.append(str(vendor))
    after = bundle_capture.environment_inventory()
    assert before == after
    assert after["packages"] == [("packaging", "26.1")]
    metadata(installed, "new-package", "1.0")
    assert bundle_capture.environment_inventory() != before
    custom = tmp_path / "custom-dependencies"
    metadata(custom, "external-package", "2.0")
    sys.path.append(str(custom))
    assert ("external-package", "2.0") in bundle_capture.environment_inventory()[
        "packages"
    ]


def test_environment_inventory_normalizes_packages_without_install_paths(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.7.0", version=SimpleNamespace(cuda="12.8")),
    )
    monkeypatch.setattr(bundle_capture, "_command", lambda argv: "570.1\n570.1\n")
    monkeypatch.setattr(
        bundle_capture.importlib.metadata,
        "distributions",
        lambda **kwargs: [
            SimpleNamespace(metadata={"Name": "Zed_Package"}, version="1.0"),
            SimpleNamespace(metadata={"Name": "A-Package"}, version="2.0"),
        ],
    )
    inventory = bundle_capture.environment_inventory()
    assert inventory["packages"] == [("a-package", "2.0"), ("zed-package", "1.0")]
    assert inventory["driver"] == "570.1"
    assert inventory["cuda"] == "12.8"


@pytest.mark.parametrize(
    "mutation",
    (
        "semantic",
        "formatting",
        "unrelated",
        "routing",
        "struct",
        "registration",
        "measurement_tuple",
        "health_branch",
        "return_transport",
    ),
)
def test_memory_boundary_digest_tracks_only_observation_semantics(
    tmp_path, monkeypatch, mutation
):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    if mutation == "measurement_tuple":
        path = inputs.repo / "python/sglang/srt/managers/scheduler.py"
        path.write_text(
            path.read_text().replace(
                "def _cuda_memory_peak_control(self, recv_req, operation):",
                "def _cuda_memory_peak_control(self, recv_req, operation):\n        observed = [('allocated', 1)]",
            )
        )
        subprocess.run(["git", "-C", str(inputs.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(inputs.repo),
                "-c",
                "user.name=Harness",
                "-c",
                "user.email=harness@example.invalid",
                "commit",
                "-qm",
                "tuple observation",
            ],
            check=True,
        )
        inputs.spec = replace(
            inputs.spec,
            revision_sha=subprocess.check_output(
                ["git", "-C", str(inputs.repo), "rev-parse", "HEAD"], text=True
            ).strip(),
        )
    before = bundle_capture.memory_boundary_hash(inputs.repo, inputs.spec.revision_sha)
    relative = "python/sglang/srt/managers/" + {
        "struct": "io_struct.py",
        "registration": "tokenizer_control_mixin.py",
    }.get(mutation, "scheduler.py")
    path = inputs.repo / relative
    source = path.read_text()
    if mutation == "health_branch":
        source = source.replace("for_health_check=True", "for_health_check=False")
    elif mutation == "return_transport":
        source = source.replace(
            "self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)",
            "self.ipc_channels.send_to_tokenizer.send_output(recv_req, output)",
        )
    elif mutation == "measurement_tuple":
        source = source.replace(
            "observed = [('allocated', 1)]", "observed = [('allocated', 2)]"
        )
    elif mutation == "semantic":
        source = source.replace(
            "torch.cuda.max_memory_allocated(device)",
            "torch.cuda.memory_allocated(device)",
        )
    elif mutation == "routing":
        source = source.replace(
            "(ResetCudaMemoryPeakReqInput, self.reset_cuda_memory_peak)",
            "(ResetCudaMemoryPeakReqInput, self.read_cuda_memory_peak)",
        )
    elif mutation == "registration":
        source = source.replace(
            '("reset_cuda_memory_peak", ResetCudaMemoryPeakReqOutput)',
            '("reset_cuda_memory_peak", ReadCudaMemoryPeakReqOutput)',
        )
    elif mutation == "struct":
        source = source.replace(
            'operation: Literal["reset", "read"]',
            'operation: Literal["reset", "read", "other"]',
        )
    elif mutation == "formatting":
        source = source.replace(
            "torch.cuda.max_memory_allocated(device)",
            "torch.cuda.max_memory_allocated( device )  # formatting only",
        )
    else:
        source += "\n\ndef unrelated_new_feature():\n    return 42\n"
    path.write_text(source)
    subprocess.run(["git", "-C", str(inputs.repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(inputs.repo),
            "-c",
            "user.name=Harness",
            "-c",
            "user.email=harness@example.invalid",
            "commit",
            "-qm",
            "mutation",
        ],
        check=True,
    )
    after = bundle_capture.memory_boundary_hash(inputs.repo, "HEAD")
    assert (before == after) is (
        mutation in ("formatting", "unrelated", "health_branch")
    )


def _runtime_tree(root):
    """Tiny importable runtime used only for origin/spawn probes, never metrics."""
    for package in (
        "sglang",
        "sglang/srt/utils",
    ):
        folder = root / "python" / package
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    for package in ("sglang/srt/entrypoints", "sglang/srt/managers"):
        (root / "python" / package).mkdir(parents=True, exist_ok=True)
    for name in ("io_struct", "tokenizer_control_mixin", "communicator"):
        (root / "python/sglang/srt/managers" / f"{name}.py").write_text("")
    (root / "python/sglang/utils.py").write_text("")
    (root / "python/sglang/srt/utils/common.py").write_text("")
    (root / "python/sglang/srt/managers/tokenizer_manager.py").write_text(
        "class TokenizerManager:\n    pass\n"
    )
    (root / "python/sglang/srt/managers/scheduler.py").write_text(
        "def run_scheduler_process(queue):\n    queue.put(__file__)\n"
    )
    (root / "python/sglang/srt/entrypoints/engine.py").write_text(
        "from sglang.srt.managers.scheduler import run_scheduler_process\n"
        "from sglang.srt.managers.tokenizer_manager import TokenizerManager\n"
        "def init_tokenizer_manager():\n    return TokenizerManager()\n"
        "class Engine:\n    run_scheduler_process_func = staticmethod(run_scheduler_process)\n"
        "    init_tokenizer_manager_func = staticmethod(init_tokenizer_manager)\n"
    )


def _clear_runtime_modules(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    names = set(sys.modules)
    for target in bundle_capture._RUNTIME_MODULES:
        parts = target.split(".")
        names.update(".".join(parts[: index + 1]) for index in range(len(parts)))
    for name in names:
        if name == "sglang" or name == "sglang.utils" or name.startswith("sglang.srt"):
            # Record absent names too: imports by an origin probe must not leak
            # a tiny runtime package into the later real-backend contract tests.
            monkeypatch.setitem(sys.modules, name, None)
            monkeypatch.delitem(sys.modules, name)


def test_runtime_origin_accepts_checkout_owned_namespace(tmp_path, monkeypatch):
    local = tmp_path / "checkout"
    _runtime_tree(local)
    _clear_runtime_modules(monkeypatch)

    origins = bundle_capture.bind_runtime(local)

    assert origins["sglang.srt"] == "python/sglang/srt"
    assert origins["sglang.srt.entrypoints"] == "python/sglang/srt/entrypoints"
    assert origins["sglang.srt.managers"] == "python/sglang/srt/managers"


@pytest.mark.parametrize(
    "foreign", ("loaded", "pythonpath", "package_path", "wrong_local_file")
)
def test_runtime_origin_rejects_foreign_imports(tmp_path, monkeypatch, foreign):
    local, other = tmp_path / "source", tmp_path / "candidate"
    _runtime_tree(local)
    _runtime_tree(other)
    _clear_runtime_modules(monkeypatch)
    name = "sglang.srt.entrypoints.engine"
    path = other / "python/sglang/srt/entrypoints/engine.py"
    if foreign in ("package_path", "wrong_local_file"):
        name = (
            "sglang" if foreign == "package_path" else "sglang.srt.managers.io_struct"
        )
        module = ModuleType(name)
        module.__file__ = str(local / "python/sglang/__init__.py")
        if foreign == "package_path":
            module.__path__ = [str(other / "python/sglang")]
        monkeypatch.setitem(sys.modules, name, module)
    elif foreign == "loaded":
        module = ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
    else:
        (local / "python/sglang/srt/entrypoints/engine.py").unlink()
        monkeypatch.setenv("PYTHONPATH", str(other / "python"))
        sys.path.insert(0, str(other / "python"))
    with pytest.raises(BundleValidationError, match="runtime|checkout|origin"):
        bundle_capture.bind_runtime(local)


def test_runtime_origin_ignores_foreign_editable_finder(tmp_path, monkeypatch):
    import importlib.util

    local, other = tmp_path / "source", tmp_path / "candidate"
    _runtime_tree(local)
    _runtime_tree(other)
    _clear_runtime_modules(monkeypatch)
    name = "sglang.srt.entrypoints.engine"

    class ForeignEditable:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == name:
                return importlib.util.spec_from_file_location(
                    fullname, str(other / "python/sglang/srt/entrypoints/engine.py")
                )

    monkeypatch.setattr(sys, "meta_path", [ForeignEditable(), *sys.meta_path])

    origins = bundle_capture.bind_runtime(local)

    assert origins[name] == "python/sglang/srt/entrypoints/engine.py"


def test_identity_collection_rejects_foreign_loaded_runtime(tmp_path, monkeypatch):
    inputs = qualification_inputs(tmp_path, monkeypatch)
    module = ModuleType("sglang.srt.managers.scheduler")
    module.__file__ = str(tmp_path / "foreign/scheduler.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(BundleValidationError, match="loaded runtime origin"):
        bundle_capture.capture_run_identity(inputs.spec, {}, [])


def test_engine_attestation_rejects_foreign_spawn_target(tmp_path, monkeypatch):
    import importlib

    local = tmp_path / "source"
    _runtime_tree(local)
    _clear_runtime_modules(monkeypatch)
    bundle_capture.bind_runtime(local)
    engine = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    foreign = {}
    exec(
        compile(
            "def run_scheduler_process(queue): pass",
            str(tmp_path / "foreign.py"),
            "exec",
        ),
        foreign,
    )
    monkeypatch.setattr(
        engine,
        "run_scheduler_process_func",
        staticmethod(foreign["run_scheduler_process"]),
    )
    with pytest.raises(BundleValidationError, match="runtime target origin"):
        bundle_capture.attest_engine_targets(engine, local)


def test_separate_worktrees_share_dependencies_and_spawn_exact_runtime(tmp_path):
    harness = Path(__file__).resolve().parents[3] / "manual"
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys,multiprocessing,json\n"
        f"sys.path.insert(0,{str(harness)!r})\n"
        "from adapter_equivalence.bundle_capture import bind_runtime,attest_engine_targets\n"
        "if __name__ == '__main__':\n"
        "    origins=bind_runtime(sys.argv[1])\n"
        "    from sglang.srt.entrypoints.engine import Engine\n"
        "    attest_engine_targets(Engine,sys.argv[1])\n"
        "    ctx=multiprocessing.get_context('spawn')\n"
        "    queue=ctx.Queue()\n"
        "    process=ctx.Process(target=Engine.run_scheduler_process_func,args=(queue,))\n"
        "    process.start()\n"
        "    actual=queue.get(timeout=10)\n"
        "    process.join(10)\n"
        "    assert process.exitcode==0\n"
        "    print(json.dumps({'origins':origins,'scheduler':actual}))\n"
    )
    roots = [tmp_path / "source", tmp_path / "candidate"]
    outputs = []
    for root in roots:
        _runtime_tree(root)
    import os

    for root, foreign in zip(roots, reversed(roots)):
        result = subprocess.run(
            [sys.executable, str(script), str(root)],
            env=dict(os.environ, PYTHONPATH=str(foreign / "python")),
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        output = json.loads(result.stdout)
        assert (
            Path(output["scheduler"]).resolve()
            == root / "python/sglang/srt/managers/scheduler.py"
        )
        outputs.append(output["origins"])
    assert outputs[0] == outputs[1]
    assert all(not Path(origin).is_absolute() for origin in outputs[0].values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

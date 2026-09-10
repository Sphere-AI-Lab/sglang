"""Failed lifecycle runs retain diagnostic outputs without certifying success."""

import json
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

import test_run_case
from adapter_equivalence import run_case
from adapter_equivalence.schema import canonical_sha256

make_runner = test_run_case.make_runner


def fail_validation(runner, monkeypatch):
    error = run_case.ScenarioContractError("concurrent outputs differ")
    error.details = {
        "kind": "concurrent-output-mismatch",
        "observations": {"stream": {"text": "é", "output_ids": [1]}},
        "changed_fields": ["output_ids"],
    }
    observation = runner._state_observation(runner.state())
    monkeypatch.setattr(runner, "execute", lambda step: observation)

    def fail(observations):
        raise error

    monkeypatch.setattr(run_case, "validate_lifecycle_observations", fail)
    return error


def test_failed_lifecycle_persists_hashed_details_before_cleanup(
    make_runner, monkeypatch
):
    runner = make_runner()
    error = fail_validation(runner, monkeypatch)
    destination = runner.spec.bundle_output.with_name("failure-details.json")
    original_close = runner.close
    saved_before_close = []

    def close():
        saved_before_close.append(destination.exists())
        original_close()

    monkeypatch.setattr(runner, "close", close)
    with pytest.raises(run_case.ScenarioContractError) as caught:
        runner.run_selected("full")
    assert caught.value is error
    assert saved_before_close == [True]
    artifact = json.loads(destination.read_text())
    assert artifact["details"] == error.details
    assert artifact["revision_sha"] == runner.spec.revision_sha
    assert artifact["case_id"] == runner.spec.case_id
    assert artifact["status"] == "failed"
    events = [json.loads(line) for line in runner.diagnostics.getvalue().splitlines()]
    event = next(e for e in events if e["event"] == "failure.details.saved")
    assert event["artifact_hash"] == canonical_sha256(artifact)
    assert event["path"] == str(destination)
    assert not runner.spec.bundle_output.exists()
    assert not runner.spec.completion_output.exists()
    assert runner.sender.closed and runner.closed


@pytest.mark.parametrize("failure", ["existing-artifact", "invalid-json"])
def test_diagnostic_write_failure_preserves_original_error_and_cleanup(
    make_runner, monkeypatch, failure
):
    runner = make_runner()
    error = fail_validation(runner, monkeypatch)
    destination = runner.spec.bundle_output.with_name("failure-details.json")
    if failure == "existing-artifact":
        destination.write_text("earlier evidence")
    else:
        error.details["unsupported"] = object()
    with pytest.raises(run_case.ScenarioContractError) as caught:
        runner.run_selected("full")
    assert caught.value is error
    if failure == "existing-artifact":
        assert destination.read_text() == "earlier evidence"
    else:
        assert not destination.exists()
    assert runner.sender.closed and runner.closed
    events = [json.loads(line) for line in runner.diagnostics.getvalue().splitlines()]
    assert any(e["event"] == "failure.details.failed" for e in events)
    assert not runner.spec.completion_output.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

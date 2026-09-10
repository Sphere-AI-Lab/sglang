"""Stress deadline attribution and progress must survive unsuccessful runs."""

import threading
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

import pytest
from test_run_case import make_stress_spec, stress_api

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def instrument(spec, events, **changes):
    # Give the old implementation the proposed callback without failing at its
    # constructor, so RED exercises the missing telemetry/deadline behavior.
    values = dict(vars(spec))
    values.update(
        changes, emit=lambda event, **fields: events.append(dict(event=event, **fields))
    )
    return SimpleNamespace(**values)


def test_stress_reports_all_cycles_and_acknowledged_cancellation():
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    events = []
    result = api.run_stress(instrument(spec, events))
    completed = [e for e in events if e["event"] == "stress.cycle.completed"]
    assert [e["cycle"] for e in completed] == list(range(100))
    assert completed[-1]["cycles_completed"] == result.cycles_completed == 100
    assert completed[-1]["requests_completed"] == result.requests_completed == 1000
    starts = [e for e in events if e["event"] == "stress.operation.started"]
    ends = [e for e in events if e["event"] == "stress.operation.completed"]
    assert [(e["cycle"], e["operation"], e["adapter"]) for e in starts] == [
        (e["cycle"], e["operation"], e["adapter"]) for e in ends
    ]
    cancelled = [e for e in ends if e["outcome"] == "cancelled"]
    assert [e["cycle"] for e in cancelled] == list(range(9, 100, 10))
    assert all(
        e["operation"] == "upsert" and e["adapter"] == "policy-b" for e in cancelled
    )
    assert all(e["duration_seconds"] >= 0 and e["elapsed_seconds"] >= 0 for e in ends)


@pytest.mark.parametrize("scope", ["operation", "whole-job"])
def test_stress_identifies_expired_budget_and_inflight_operation(scope):
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    events = []
    release = threading.Event()
    spec.control.load_tensors = lambda *args: release.wait()
    changes = dict(
        operation_timeout=0.02 if scope == "operation" else 1,
        job_timeout=0.02 if scope == "whole-job" else 1,
    )
    try:
        with pytest.raises(
            TimeoutError, match=rf"{scope} deadline.*cycle=0.*operation=load"
        ):
            api.run_stress(instrument(spec, events, **changes))
        timeout = next(e for e in events if e["event"] == "stress.operation.timeout")
        assert timeout["timeout_scope"] == scope
        assert (
            timeout["cycle"]
            == timeout["cycles_completed"]
            == timeout["requests_completed"]
            == 0
        )
        assert timeout["operation"] == "load" and timeout["adapter"] == "policy-a"
        assert timeout["operation_submitted"] is True
        assert timeout["duration_seconds"] >= 0
        assert not any(e["event"] == "stress.operation.completed" for e in events)
    finally:
        release.set()


def test_stress_records_completed_progress_when_budget_expires_between_operations(
    monkeypatch,
):
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    clock = [0.0]
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    events = []
    wrapped = instrument(spec, events)

    def emit(event, **fields):
        events.append(dict(event=event, **fields))
        if event == "stress.cycle.completed":
            clock[0] = spec.job_timeout + 1

    wrapped.emit = emit
    with pytest.raises(TimeoutError, match="whole-job deadline"):
        api.run_stress(wrapped)
    timeout = events[-1]
    assert timeout["event"] == "stress.operation.timeout"
    assert timeout["operation_submitted"] is False
    assert timeout["cycle"] == timeout["cycles_completed"] == 1
    assert timeout["requests_completed"] == 10


@pytest.mark.parametrize("failure_type", [RuntimeError, TimeoutError])
def test_stress_preserves_native_exception_and_does_not_mislabel_timeout(failure_type):
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    events = []
    original = failure_type("native operation failed")

    def fail(*args):
        raise original

    spec.control.load_tensors = fail
    with pytest.raises(failure_type) as caught:
        api.run_stress(instrument(spec, events))
    assert caught.value is original
    event = events[-1]
    assert event["event"] == "stress.operation.failed"
    assert event["error_type"] == failure_type.__name__
    assert event["operation"] == "load" and event["cycle"] == 0
    assert not any(e["event"] == "stress.operation.timeout" for e in events)


def test_stress_diagnostic_exception_does_not_replace_native_failure():
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    original = RuntimeError("native failure retained")

    def fail(*args):
        raise original

    spec.control.load_tensors = fail
    wrapped = instrument(spec, [])

    def emit(event, **fields):
        if event == "stress.operation.failed":
            raise OSError("diagnostic destination failed")

    wrapped.emit = emit
    with pytest.raises(RuntimeError) as caught:
        api.run_stress(wrapped)
    assert caught.value is original


def test_stress_resolves_completion_racing_wait_timeout(monkeypatch):
    from adapter_equivalence import run_case

    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    original_worker = run_case._Worker

    class RaceWorker(original_worker):
        used = False

        def submit(self, function):
            future = super().submit(function)
            if self.used:
                return future
            self.used = True

            class CompletedDuringTimeout:
                def result(self, timeout=None):
                    value = future.result(timeout=timeout)
                    if timeout is not None:
                        raise FutureTimeout("wait boundary raced completion")
                    return value

                def done(self):
                    return future.done()

            return CompletedDuringTimeout()

    monkeypatch.setattr(run_case, "_Worker", RaceWorker)
    events = []
    result = api.run_stress(instrument(spec, events))
    assert result.cycles_completed == 100
    assert not any(e["event"] == "stress.operation.timeout" for e in events)


@pytest.mark.parametrize("scope", ["operation", "whole-job"])
def test_start_telemetry_cannot_extend_deadline_or_start_expired_work(
    monkeypatch, scope
):
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    events = []
    clock = [0.0]
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    wrapped = instrument(
        spec,
        events,
        operation_timeout=1 if scope == "operation" else 10,
        job_timeout=1 if scope == "whole-job" else 10,
    )
    original_load = spec.control.load_tensors
    invoked = []

    def load(*args):
        invoked.append(True)
        return original_load(*args)

    spec.control.load_tensors = load

    def emit(event, **fields):
        events.append(dict(event=event, **fields))
        if event == "stress.operation.started":
            clock[0] = 2.0

    wrapped.emit = emit
    with pytest.raises(TimeoutError, match=rf"{scope} deadline"):
        api.run_stress(wrapped)
    assert not invoked
    assert events[-1]["event"] == "stress.operation.timeout"
    assert events[-1]["operation_submitted"] is False


@pytest.mark.parametrize(
    "broken_event",
    [
        "stress.cycle.started",
        "stress.operation.started",
        "stress.operation.completed",
        "stress.cycle.completed",
    ],
)
def test_all_telemetry_is_best_effort_without_changing_success(broken_event):
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    wrapped = instrument(spec, [])
    fired = []

    def emit(event, **fields):
        if event == broken_event:
            fired.append(True)
            raise OSError("diagnostic write failed")

    wrapped.emit = emit
    result = api.run_stress(wrapped)
    assert fired
    assert result.cycles_completed == 100 and result.requests_completed == 1000


def test_returned_failed_control_is_not_logged_as_success():
    api = stress_api()
    spec, _, _ = make_stress_spec(api, fail="upsert")
    events = []
    with pytest.raises(ValueError, match="real update failed"):
        api.run_stress(instrument(spec, events))
    returned = [
        e
        for e in events
        if e["event"] == "stress.operation.completed" and e["operation"] == "upsert"
    ]
    assert len(returned) == 1 and returned[0]["outcome"] == "returned"
    assert not any(e["event"] == "stress.cycle.completed" for e in events)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

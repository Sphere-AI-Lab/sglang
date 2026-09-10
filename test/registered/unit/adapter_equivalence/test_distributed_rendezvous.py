"""The sender must own the rendezvous port before workers connect."""

import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence import distributed_sender as sender
from test_distributed_sender import _Child, _Engine, _Process, _runtime, _spec


def _bound_record(port=29501):
    return {
        "event": "rendezvous.ready",
        "request_id": None,
        "master_port": port,
        "runtime_origins": sender.SENDER_RUNTIME_ORIGINS,
    }


@pytest.mark.parametrize("requested_port", (0, 29501))
def test_sender_announces_owned_store_before_group_initialization(requested_port):
    events = []
    output = io.StringIO()
    runtime = _runtime(events)
    store = type("Store", (), {"port": 29501})()

    def create_store(**kwargs):
        assert kwargs["port"] == requested_port
        assert kwargs["host_name"] == "127.0.0.1"
        assert kwargs["is_master"] is True
        assert kwargs["wait_for_workers"] is False
        assert kwargs["world_size"] == 3
        return store

    runtime.torch.distributed.TCPStore = create_store
    runtime.torch.distributed.PrefixStore = lambda prefix, raw: (prefix, raw)

    def init_group(**kwargs):
        assert [json.loads(line) for line in output.getvalue().splitlines()] == [
            _bound_record()
        ]
        assert kwargs["store"] == ("adapter-eq", store)
        assert "init_method" not in kwargs
        assert kwargs["rank"] == 0
        assert kwargs["world_size"] == 3
        return "group"

    runtime.init_custom_process_group = init_group
    code = sender.run_protocol(
        replace(_spec(), master_port=requested_port),
        [sender.SenderCommand("close", "close", None)],
        output=output,
        runtime=runtime,
    )

    assert code == 0, output.getvalue()
    assert [json.loads(line)["event"] for line in output.getvalue().splitlines()] == [
        "rendezvous.ready",
        "ready",
        "close.complete",
    ]


def test_store_failure_is_reported_before_group_initialization():
    events = []
    output = io.StringIO()
    runtime = _runtime(events)

    def occupied(**kwargs):
        raise OSError("address already in use")

    runtime.torch.distributed.TCPStore = occupied
    code = sender.run_protocol(
        _spec(),
        [sender.SenderCommand("close", "close", None)],
        output=output,
        runtime=runtime,
    )

    assert code == 1
    assert not any(event[0] == "init" for event in events)
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {
            "event": "error",
            "request_id": None,
            "error_type": "OSError",
            "message": "address already in use",
        }
    ]


def test_sender_ready_consumes_validated_bound_port_first(monkeypatch):
    child = sender.DistributedSender(_spec(), _Process())
    responses = iter(
        [
            _bound_record(),
            {
                "event": "ready",
                "request_id": None,
                "runtime_origins": sender.SENDER_RUNTIME_ORIGINS,
            },
        ]
    )
    monkeypatch.setattr(child, "_response", lambda timeout: next(responses))

    child.wait_ready(1)

    assert child.rendezvous_port == 29501
    assert child._ready


@pytest.mark.parametrize(
    "changes",
    [
        {"master_port": 0},
        {"master_port": 65536},
        {"master_port": True},
        {"master_port": "29501"},
        {"master_port": 29502},
        {"request_id": "unrequested"},
        {"runtime_origins": {}},
        {"extra": "unknown"},
    ],
)
def test_parent_rejects_invalid_rendezvous_before_ready(monkeypatch, changes):
    child = sender.DistributedSender(_spec(), _Process())
    response = {**_bound_record(), **changes}
    monkeypatch.setattr(child, "_response", lambda timeout: response)

    with pytest.raises(sender.SenderProtocolError):
        child.wait_ready(1)
    assert not child._ready


class _BoundChild(_Child):
    def __init__(self, spec, events, *, fail=False):
        super().__init__(spec, events)
        self.fail = fail

    def wait_rendezvous(self, timeout):
        assert not any(event[0] == "engine.init" for event in self.events)
        self.events.append(("sender.bound", 29501))
        if self.fail:
            raise sender.SenderProcessError("store bind failed")
        return 29501

    def wait_ready(self, timeout):
        if self.fail:
            raise sender.SenderProcessError("store bind failed")
        super().wait_ready(timeout)


@pytest.mark.parametrize("requested_port", (None, 29501))
def test_session_starts_workers_only_after_sender_owns_port(
    monkeypatch, requested_port
):
    events = []
    child = _BoundChild(_spec(), events)

    def spawn(spec, timeout):
        assert spec.master_port == (0 if requested_port is None else requested_port)
        return child

    monkeypatch.setattr(sender.DistributedSender, "spawn", spawn)

    session = sender.DistributedSession.open(
        _Engine(events), tp_size=2, timeout=5, master_port=requested_port
    )
    session.close(timeout=5)

    assert [event[0] for event in events[:3]] == [
        "sender.bound",
        "engine.init",
        "sender.ready",
    ]
    assert events[1][1]["master_port"] == 29501


def test_session_store_failure_never_initializes_workers(monkeypatch):
    events = []
    child = _BoundChild(_spec(), events, fail=True)
    monkeypatch.setattr(sender.DistributedSender, "spawn", lambda spec, timeout: child)

    with pytest.raises(sender.DistributedSessionError, match="store bind failed"):
        sender.DistributedSession.open(
            _Engine(events), tp_size=2, timeout=5, master_port=29501
        )

    assert not any(event[0].startswith("engine.") for event in events)
    assert events[-1] == ("sender.terminate",)


def test_rendezvous_and_group_ready_share_one_deadline(monkeypatch):
    child = sender.DistributedSender(_spec(), _Process())
    records = iter(
        [
            _bound_record(),
            {
                "event": "ready",
                "request_id": None,
                "runtime_origins": sender.SENDER_RUNTIME_ORIGINS,
            },
        ]
    )
    ticks = iter((10.0, 10.75))
    timeouts = []
    monkeypatch.setattr(sender.time, "monotonic", lambda: next(ticks))

    def response(timeout):
        timeouts.append(timeout)
        return next(records)

    monkeypatch.setattr(child, "_response", response)
    child.wait_ready(1)
    assert timeouts == [1, 0.25]


def test_expired_rendezvous_deadline_does_not_initialize_workers(monkeypatch):
    events = []
    child = _BoundChild(_spec(), events)
    monkeypatch.setattr(sender.DistributedSender, "spawn", lambda spec, timeout: child)
    ticks = iter((10.0, 10.1, 11.1))
    monkeypatch.setattr(sender.time, "monotonic", lambda: next(ticks))

    with pytest.raises(sender.DistributedSessionError, match="timed out"):
        sender.DistributedSession.open(
            _Engine(events), tp_size=2, timeout=1, master_port=29501
        )
    assert not any(event[0].startswith("engine.") for event in events)
    assert events[-1] == ("sender.terminate",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

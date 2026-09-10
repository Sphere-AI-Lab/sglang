import io
import json
import subprocess
import sys
from pathlib import Path
from threading import Timer
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence import distributed_sender as sender
from adapter_equivalence.server import DistributedPayload


class _Tensor:
    def __init__(self, name: str, shape: tuple[int, ...], events: list) -> None:
        self.name = name
        self.shape = shape
        self.dtype = "torch.float32"
        self._events = events

    def to(self, device: str):
        self._events.append(("to", self.name, device))
        return self


def _runtime(events: list, *, fail_broadcast: bool = False):
    def set_device(device):
        events.append(("device", device))

    def synchronize():
        events.append(("synchronize",))

    def broadcast(tensor, *, src, group):
        events.append(("broadcast", tensor.name, src, group))
        if fail_broadcast:
            raise RuntimeError("collective failed")

    def destroy(group):
        events.append(("destroy", group))

    def init_group(**kwargs):
        events.append(("init", kwargs))
        return "group"

    def load_file(path, *, device):
        assert Path(path).name == "adapter_model.safetensors"
        assert device == "cpu"
        return {
            "z.weight": _Tensor("z.weight", (2, 4), events),
            "a.weight": _Tensor("a.weight", (1, 8), events),
        }

    torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=set_device, synchronize=synchronize),
        distributed=SimpleNamespace(
            broadcast=broadcast,
            destroy_process_group=destroy,
            TCPStore=lambda **kwargs: SimpleNamespace(port=kwargs["port"] or 29501),
            PrefixStore=lambda prefix, store: (prefix, store),
        ),
    )
    return SimpleNamespace(
        torch=torch,
        init_custom_process_group=init_group,
        load_file=load_file,
        runtime_origins={
            "init_custom_process_group": "python/sglang/srt/utils/common.py"
        },
    )


def _fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "adapter_model.safetensors").write_bytes(b"fixture")
    (fixture / "adapter_config.json").write_text(
        json.dumps({"peft_type": "OFT", "target_modules": ["q_proj"]})
    )
    return fixture


def _spec() -> sender.SenderSpec:
    return sender.SenderSpec(
        master_address="127.0.0.1",
        master_port=29501,
        rank=0,
        world_size=3,
        group_name="adapter-eq",
        device=0,
    )


def test_sender_broadcasts_sorted_payload_and_destroys_group(
    tmp_path: Path,
) -> None:
    events = []
    output = io.StringIO()
    fixture = _fixture(tmp_path)

    exit_code = sender.run_protocol(
        _spec(),
        (
            sender.SenderCommand("req-1", "broadcast", str(fixture)),
            sender.SenderCommand("close-1", "close", None),
        ),
        output=output,
        runtime=_runtime(events),
    )

    assert exit_code == 0
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["event"] for record in records] == [
        "rendezvous.ready",
        "ready",
        "broadcast.complete",
        "close.complete",
    ]
    assert records[2] == {
        "event": "broadcast.complete",
        "payload": {
            "config": {"peft_type": "OFT", "target_modules": ["q_proj"]},
            "dtypes": ["float32", "float32"],
            "names": ["a.weight", "z.weight"],
            "shapes": [[1, 8], [2, 4]],
        },
        "request_id": "req-1",
    }
    assert [event[1] for event in events if event[0] == "broadcast"] == [
        "a.weight",
        "z.weight",
    ]
    assert events[-1] == ("destroy", "group")


def test_sender_collective_failure_is_reported_and_group_is_destroyed(
    tmp_path: Path,
) -> None:
    events = []
    output = io.StringIO()

    exit_code = sender.run_protocol(
        _spec(),
        (sender.SenderCommand("req-1", "broadcast", str(_fixture(tmp_path))),),
        output=output,
        runtime=_runtime(events, fail_broadcast=True),
    )

    assert exit_code == 1
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[-1] == {
        "error_type": "RuntimeError",
        "event": "error",
        "message": "collective failed",
        "request_id": "req-1",
    }
    assert events[-1] == ("destroy", "group")


@pytest.mark.parametrize(
    "payload",
    (
        "not-json",
        '{"request_id":"req-1","action":"unknown","fixture_path":null}',
        '{"request_id":"req-1","action":"close"}',
    ),
)
def test_sender_command_rejects_malformed_input(payload: str) -> None:
    with pytest.raises(sender.SenderProtocolError):
        sender.SenderCommand.from_json(payload)


def test_response_reader_reports_timeout_and_terminates_cleanly() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with pytest.raises(sender.SenderTimeout, match="timed out"):
            sender.read_response_line(process, timeout=0.01)
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_response_reader_reports_nonzero_child_exit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait(timeout=5)

    with pytest.raises(sender.SenderProcessError, match="status 7"):
        sender.read_response_line(process, timeout=1)


def test_response_reader_retains_back_to_back_lines_while_child_is_alive():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys; os.write(1, b'first\\nsecond\\n'); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert sender.read_response_line(process, timeout=1) == "first"
        assert sender.read_response_line(process, timeout=0.1) == "second"
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_response_reader_times_out_on_partial_line_without_blocking():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys; os.write(1, b'partial'); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Bound even the old blocking readline implementation during RED.
    watchdog = Timer(1, process.kill)
    watchdog.start()
    try:
        with pytest.raises(sender.SenderTimeout):
            sender.read_response_line(process, timeout=0.1)
    finally:
        watchdog.cancel()
        process.terminate()
        process.communicate(timeout=5)


@pytest.mark.parametrize("terminated", (False, True))
def test_response_reader_requires_newline_even_at_eof(terminated):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, "
            + repr(b"complete\n" if terminated else b"partial")
            + ")",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait(timeout=5)
    try:
        if terminated:
            assert sender.read_response_line(process, timeout=1) == "complete"
        else:
            with pytest.raises(sender.SenderProtocolError, match="newline|incomplete"):
                sender.read_response_line(process, timeout=1)
    finally:
        process.communicate(timeout=5)


def test_response_reader_preserves_partial_utf8_across_a_timeout():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys; os.write(1, b'begin\\n\\xc3'); "
            "sys.stdin.read(1); os.write(1, b'\\xa9\\n'); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert sender.read_response_line(process, timeout=1) == "begin"
        with pytest.raises(sender.SenderTimeout):
            sender.read_response_line(process, timeout=0.1)
        process.stdin.write("x")
        process.stdin.flush()
        assert sender.read_response_line(process, timeout=1) == "é"
    finally:
        process.terminate()
        process.communicate(timeout=5)


@pytest.mark.parametrize("payload", (b"\xff\n", b"not-json\n"))
def test_response_reader_rejects_invalid_utf8_or_json(payload):
    process = subprocess.Popen(
        [sys.executable, "-c", "import os; os.write(1, " + repr(payload) + ")"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait(timeout=5)
    try:
        child = sender.DistributedSender(_spec(), process)
        with pytest.raises(sender.SenderProtocolError):
            child._response(1)
    finally:
        process.communicate(timeout=5)


class _Input:
    def __init__(self) -> None:
        self.lines = []

    def write(self, value: str) -> None:
        self.lines.append(value)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Process:
    def __init__(self) -> None:
        self.stdin = _Input()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_sender_inherits_stderr_so_runtime_logs_cannot_fill_a_pipe(
    monkeypatch,
) -> None:
    process = _Process()
    captured = {}

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setattr(sender.subprocess, "Popen", popen)

    sender.DistributedSender.spawn(_spec(), timeout=1)

    assert captured["stderr"] is None


def test_parent_validates_ready_broadcast_and_close_responses(
    monkeypatch, tmp_path: Path
) -> None:
    process = _Process()
    fixture = _fixture(tmp_path)
    responses = iter(
        (
            '{"event":"rendezvous.ready","request_id":null,"master_port":29501,"runtime_origins":{"init_custom_process_group":"python/sglang/srt/utils/common.py"}}',
            '{"event":"ready","request_id":null,"runtime_origins":{"init_custom_process_group":"python/sglang/srt/utils/common.py"}}',
            json.dumps(
                {
                    "event": "broadcast.complete",
                    "request_id": "req-1",
                    "payload": {
                        "names": ["a.weight"],
                        "dtypes": ["float32"],
                        "shapes": [[1, 8]],
                        "config": {"peft_type": "OFT"},
                    },
                }
            ),
            '{"event":"close.complete","request_id":"close-1"}',
        )
    )
    monkeypatch.setattr(sender.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        sender,
        "read_response_line",
        lambda process, timeout: next(responses),
    )

    child = sender.DistributedSender.start(_spec(), timeout=1)
    payload = child.broadcast_fixture("req-1", fixture, timeout=1)
    child.close(request_id="close-1", timeout=1)

    assert payload == DistributedPayload(
        names=("a.weight",),
        dtypes=("float32",),
        shapes=((1, 8),),
        config={"peft_type": "OFT"},
    )
    commands = [json.loads(line) for line in process.stdin.lines]
    assert commands == [
        {
            "action": "broadcast",
            "fixture_path": str(fixture.resolve()),
            "request_id": "req-1",
        },
        {
            "action": "close",
            "fixture_path": None,
            "request_id": "close-1",
        },
    ]
    assert process.returncode == 0


def test_parent_close_after_sender_error_terminates_child(
    monkeypatch, tmp_path: Path
) -> None:
    process = _Process()
    responses = iter(
        (
            '{"event":"rendezvous.ready","request_id":null,"master_port":29501,"runtime_origins":{"init_custom_process_group":"python/sglang/srt/utils/common.py"}}',
            '{"event":"ready","request_id":null,"runtime_origins":{"init_custom_process_group":"python/sglang/srt/utils/common.py"}}',
            '{"event":"error","request_id":"req-1",'
            '"error_type":"RuntimeError","message":"collective failed"}',
        )
    )
    monkeypatch.setattr(sender.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        sender,
        "read_response_line",
        lambda process, timeout: next(responses),
    )
    child = sender.DistributedSender.start(_spec(), timeout=1)

    with pytest.raises(sender.SenderProcessError, match="collective failed"):
        child.broadcast_fixture("req-1", _fixture(tmp_path), timeout=1)
    child.close(request_id="close-1", timeout=1)

    assert process.terminated


class _Engine:
    def __init__(self, events: list, *, init_success=True, destroy_success=True):
        self.events = events
        self.init_success = init_success
        self.destroy_success = destroy_success

    def init_weights_update_group(self, **kwargs):
        self.events.append(("engine.init", kwargs))
        return self.init_success, "init result"

    def destroy_weights_update_group(self, group_name):
        self.events.append(("engine.destroy", group_name))
        return self.destroy_success, "destroy result"


class _Child:
    def __init__(self, spec, events):
        self.spec = spec
        self.events = events

    def wait_rendezvous(self, timeout):
        return self.spec.master_port or 29501

    def wait_ready(self, timeout):
        assert any(event[0] == "engine.init" for event in self.events)
        self.events.append(("sender.ready", timeout))

    def begin_close(self, *, request_id):
        self.events.append(("sender.close-request", request_id))

    def finish_close(self, *, timeout):
        self.events.append(("sender.close", timeout))

    def terminate(self):
        self.events.append(("sender.terminate",))


def test_distributed_session_opens_concurrently_and_closes_both_sides(
    monkeypatch,
) -> None:
    events = []
    children = []

    def spawn(spec, timeout):
        child = _Child(spec, events)
        children.append(child)
        return child

    monkeypatch.setattr(sender.DistributedSender, "spawn", spawn)
    engine = _Engine(events)

    session = sender.DistributedSession.open(
        engine,
        tp_size=2,
        timeout=5,
        master_port=29501,
        group_name="adapter-eq",
    )
    session.close(timeout=5)

    assert children[0].spec == _spec()
    assert events[0] == (
        "engine.init",
        {
            "master_address": "127.0.0.1",
            "master_port": 29501,
            "rank_offset": 1,
            "world_size": 3,
            "group_name": "adapter-eq",
            "backend": "nccl",
        },
    )
    assert [event[0] for event in events[-3:]] == [
        "sender.close-request",
        "engine.destroy",
        "sender.close",
    ]


def test_session_init_failure_terminates_sender(monkeypatch) -> None:
    events = []
    child = _Child(_spec(), events)
    monkeypatch.setattr(
        sender.DistributedSender,
        "spawn",
        lambda spec, timeout: child,
    )

    with pytest.raises(sender.DistributedSessionError, match="init result"):
        sender.DistributedSession.open(
            _Engine(events, init_success=False),
            tp_size=2,
            timeout=5,
            master_port=29501,
            group_name="adapter-eq",
        )

    assert events[-1] == ("sender.terminate",)


def test_session_close_attempts_sender_after_engine_teardown_failure(
    monkeypatch,
) -> None:
    events = []
    child = _Child(_spec(), events)
    monkeypatch.setattr(
        sender.DistributedSender,
        "spawn",
        lambda spec, timeout: child,
    )
    session = sender.DistributedSession.open(
        _Engine(events, destroy_success=False),
        tp_size=2,
        timeout=5,
        master_port=29501,
        group_name="adapter-eq",
    )

    with pytest.raises(sender.DistributedSessionError, match="destroy result"):
        session.close(timeout=5)

    assert events[-1][0] == "sender.close"


@pytest.mark.parametrize(
    "mode", ("local", "noisy_runtime", "foreign_pythonpath", "foreign_editable")
)
def test_fresh_exec_sender_attests_recorded_runtime(tmp_path, monkeypatch, capfd, mode):
    from adapter_equivalence import bundle_capture
    from test_bundle_capture import _runtime_tree

    local, foreign, dependencies = (
        tmp_path / "source",
        tmp_path / "foreign",
        tmp_path / "dependencies",
    )
    for root in (local, foreign):
        _runtime_tree(root)
        package = root / "python/sglang/srt/utils"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text(
            "from .common import init_custom_process_group\n"
        )
        (package / "common.py").write_text(
            "def init_custom_process_group(**kwargs): return object()\n"
        )
    if mode == "noisy_runtime":
        (local / "python/sglang/srt/utils/common.py").write_text(
            "import os\n"
            "print('runtime import diagnostic', flush=True)\n"
            "def init_custom_process_group(**kwargs):\n"
            "    os.write(1, b'NCCL version 2.29.7+cuda13.2\\n')\n"
            "    return object()\n"
        )
    dependencies.mkdir()
    (dependencies / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "cuda=SimpleNamespace(set_device=lambda device: None)\n"
        "distributed=SimpleNamespace(\n"
        "    destroy_process_group=lambda group: None,\n"
        "    TCPStore=lambda **kw: SimpleNamespace(port=kw['port'] or 29501),\n"
        "    PrefixStore=lambda prefix, store: (prefix, store),\n"
        ")\n"
    )
    (dependencies / "safetensors").mkdir()
    (dependencies / "safetensors/__init__.py").write_text("")
    (dependencies / "safetensors/torch.py").write_text(
        "def load_file(*args, **kwargs): return {}\n"
    )
    (dependencies / "safetensors/numpy.py").write_text(
        "def save(*args, **kwargs): return b''\n"
    )
    if mode == "foreign_pythonpath":
        (local / "python/sglang/srt/utils/common.py").unlink()
    elif mode == "foreign_editable":
        (dependencies / "sitecustomize.py").write_text(
            "import sys,importlib.util\nclass ForeignEditable:\n"
            "    @staticmethod\n    def find_spec(name,path=None,target=None):\n"
            "        if name == 'sglang.srt.utils.common':\n"
            f"            return importlib.util.spec_from_file_location(name,{str(foreign / 'python/sglang/srt/utils/common.py')!r})\n"
            "sys.meta_path.insert(0,ForeignEditable())\n"
        )
    monkeypatch.setattr(bundle_capture, "REPO_ROOT", local)
    monkeypatch.setenv("PYTHONPATH", str(dependencies) + ":" + str(foreign / "python"))
    child = sender.DistributedSender.spawn(_spec(), timeout=5)
    try:
        if mode in {"local", "noisy_runtime"}:
            child.wait_ready(5)
            assert child.runtime_origins == {
                "init_custom_process_group": "python/sglang/srt/utils/common.py"
            }
            child.close(request_id="close", timeout=5)
            if mode == "noisy_runtime":
                captured = capfd.readouterr()
                assert "runtime import diagnostic" in captured.err
                assert "NCCL version 2.29.7+cuda13.2" in captured.err
        else:
            with pytest.raises(
                sender.SenderProcessError, match="runtime|checkout|origin"
            ):
                child.wait_ready(5)
    finally:
        child.terminate()


@pytest.mark.parametrize(
    "origins", (None, {}, {"init_custom_process_group": "/foreign/common.py"})
)
def test_parent_rejects_missing_or_foreign_sender_attestation(monkeypatch, origins):
    child = sender.DistributedSender(_spec(), _Process())
    child.rendezvous_port = 29501
    response = {"event": "ready", "request_id": None}
    if origins is not None:
        response["runtime_origins"] = origins
    monkeypatch.setattr(child, "_response", lambda timeout: response)
    with pytest.raises(sender.SenderProtocolError, match="ready|origin"):
        child.wait_ready(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_session_requests_peer_shutdown_before_engine_collective_destroy():
    events = []

    class CollectiveEngine(_Engine):
        def destroy_weights_update_group(self, group_name):
            assert (
                "sender.close-requested",
            ) in events, "peer cannot enter collective teardown"
            return super().destroy_weights_update_group(group_name)

    class Peer(_Child):
        def begin_close(self, *, request_id):
            events.append(("sender.close-requested",))

        def finish_close(self, *, timeout):
            assert ("engine.destroy", "adapter-eq") in events
            events.append(("sender.close-finished",))

    session = sender.DistributedSession(
        CollectiveEngine(events), Peer(_spec(), events), "adapter-eq"
    )
    session.close(timeout=1)
    assert [event[0] for event in events] == [
        "sender.close-requested",
        "engine.destroy",
        "sender.close-finished",
    ]


@pytest.mark.parametrize(
    "response,exit_code",
    [
        ({"event": "close.complete", "request_id": "wrong"}, 0),
        ({"event": "close.complete", "request_id": "close-1"}, 7),
    ],
)
def test_split_close_still_requires_exact_ack_and_successful_exit(
    monkeypatch, response, exit_code
):
    process = _Process()
    child = sender.DistributedSender(_spec(), process)
    child.begin_close(request_id="close-1")
    monkeypatch.setattr(child, "_response", lambda timeout: response)
    monkeypatch.setattr(process, "wait", lambda timeout: exit_code)
    with pytest.raises((sender.SenderProtocolError, sender.SenderProcessError)):
        child.finish_close(timeout=1)
    assert child._failed
    assert process.terminated


def test_split_close_response_and_exit_share_one_deadline(monkeypatch):
    process = _Process()
    child = sender.DistributedSender(_spec(), process)
    child.begin_close(request_id="close-1")
    ticks = iter((10.0, 10.75))
    monkeypatch.setattr(sender.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        child,
        "_response",
        lambda timeout: {"event": "close.complete", "request_id": "close-1"},
    )
    waits = []
    monkeypatch.setattr(process, "wait", lambda timeout: waits.append(timeout) or 0)
    child.finish_close(timeout=1)
    assert waits == [0.25]

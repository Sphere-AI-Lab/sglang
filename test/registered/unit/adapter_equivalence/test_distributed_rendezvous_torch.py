"""Real TCPStore ownership; NCCL collectives are covered by native GPU suites."""

import errno
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence import distributed_sender as sender
from test_distributed_sender import _spec


class _Records:
    def __init__(self):
        self.queue = Queue()

    def write(self, value):
        self.queue.put(json.loads(value))

    def flush(self):
        pass


def _runtime(torch, initialize):
    return SimpleNamespace(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda device: None),
            distributed=SimpleNamespace(
                TCPStore=torch.distributed.TCPStore,
                PrefixStore=torch.distributed.PrefixStore,
                destroy_process_group=lambda group: None,
            ),
        ),
        init_custom_process_group=initialize,
        runtime_origins=sender.SENDER_RUNTIME_ORIGINS,
    )


def test_sender_owns_auto_port_through_real_client_rendezvous_and_releases_it():
    import torch

    output = _Records()

    def initialize(**kwargs):
        # Exercise the exact store that run_protocol supplies to the native
        # helper. A separately recreated store or missing prefix cannot join.
        store = kwargs["store"]
        store.set("server", "owned")
        assert store.get("client") == b"joined"
        return object()

    spec = replace(_spec(), master_port=0, world_size=2, group_timeout_seconds=10)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            sender.run_protocol,
            spec,
            [sender.SenderCommand("close", "close", None)],
            output=output,
            runtime=_runtime(torch, initialize),
        )
        bound = output.queue.get(timeout=10)
        assert bound["event"] == "rendezvous.ready", bound
        port = bound["master_port"]
        assert type(port) is int and 1 <= port <= 65535
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
            with pytest.raises(OSError) as error:
                competitor.bind(("127.0.0.1", port))
            assert error.value.errno == errno.EADDRINUSE

        rendezvous = torch.distributed.rendezvous(
            f"tcp://127.0.0.1:{port}",
            rank=1,
            world_size=2,
            timeout=timedelta(seconds=10),
        )
        client, rank, world_size = next(rendezvous)
        assert (rank, world_size) == (1, 2)
        peer = torch.distributed.PrefixStore("adapter-eq", client)
        assert peer.get("server") == b"owned"
        peer.set("client", "joined")
        assert future.result(timeout=10) == 0
        assert output.queue.get(timeout=1)["event"] == "ready"
        assert output.queue.get(timeout=1) == {
            "event": "close.complete",
            "request_id": "close",
        }
        rendezvous.close()
        del peer, client, rendezvous

    # Reusing the transport's actual listener proves teardown released the
    # port, without confusing accepted TCP connections in TIME_WAIT with a leak.
    replacement = torch.distributed.TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=1,
        is_master=True,
        timeout=timedelta(seconds=10),
        wait_for_workers=False,
    )
    assert replacement.port == port


def test_explicit_occupied_port_reports_failure_without_group_initialization():
    import torch

    output = _Records()
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs)
        return object()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        spec = replace(
            _spec(),
            master_port=occupied.getsockname()[1],
            world_size=2,
            group_timeout_seconds=10,
        )
        code = sender.run_protocol(
            spec,
            [sender.SenderCommand("close", "close", None)],
            output=output,
            runtime=_runtime(torch, initialize),
        )

    assert code == 1
    assert not calls
    failure = output.queue.get(timeout=1)
    assert failure["event"] == "error", failure
    assert "address already in use" in failure["message"].lower(), failure
    assert output.queue.empty()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

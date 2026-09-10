#!/usr/bin/env python3
"""External rank for native adapter broadcasts over SGLang's update group."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import selectors
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adapter_equivalence"

SENDER_RUNTIME_ORIGINS = {
    "init_custom_process_group": "python/sglang/srt/utils/common.py"
}


class SenderProtocolError(ValueError):
    """Raised when either side violates the line-delimited JSON protocol."""


class SenderTimeout(TimeoutError):
    """Raised when the child does not produce a complete response in time."""


class SenderProcessError(RuntimeError):
    """Raised when the child exits or reports a broadcast failure."""


class DistributedSessionError(RuntimeError):
    """Raised when group rendezvous or two-sided teardown is incomplete."""


@dataclass(frozen=True)
class SenderSpec:
    master_address: str
    master_port: int
    rank: int
    world_size: int
    group_name: str
    device: int
    group_timeout_seconds: float = field(default=300.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.master_address) is not str or not self.master_address:
            raise SenderProtocolError("master_address must be a non-empty string")
        if type(self.master_port) is not int or not 0 <= self.master_port <= 65535:
            raise SenderProtocolError("master_port must be an integer from 0 to 65535")
        if type(self.rank) is not int or self.rank < 0:
            raise SenderProtocolError("rank must be a non-negative integer")
        if self.master_port == 0 and self.rank != 0:
            raise SenderProtocolError("only rank zero can select an automatic port")
        if type(self.world_size) is not int or self.world_size <= self.rank:
            raise SenderProtocolError("world_size must be greater than rank")
        if type(self.group_name) is not str or not self.group_name:
            raise SenderProtocolError("group_name must be a non-empty string")
        if type(self.device) is not int or self.device < 0:
            raise SenderProtocolError("device must be a non-negative integer")
        if (
            type(self.group_timeout_seconds) not in (int, float)
            or self.group_timeout_seconds <= 0
        ):
            raise SenderProtocolError("group timeout must be positive")


@dataclass(frozen=True)
class SenderCommand:
    request_id: str
    action: str
    fixture_path: str | None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise SenderProtocolError("request_id must be a non-empty string")
        if self.action not in {"broadcast", "close"}:
            raise SenderProtocolError(f"unknown sender action: {self.action}")
        if self.action == "broadcast":
            if type(self.fixture_path) is not str or not self.fixture_path:
                raise SenderProtocolError("broadcast requires fixture_path")
        elif self.fixture_path is not None:
            raise SenderProtocolError("close fixture_path must be null")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "fixture_path": self.fixture_path,
        }

    @classmethod
    def from_json(cls, raw: str) -> SenderCommand:
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, SenderProtocolError) as error:
            raise SenderProtocolError(f"invalid sender command: {error}") from error
        if not isinstance(value, Mapping):
            raise SenderProtocolError("sender command must be an object")
        expected = {"request_id", "action", "fixture_path"}
        if set(value) != expected:
            raise SenderProtocolError("sender command fields are incomplete or unknown")
        return cls(
            request_id=value["request_id"],
            action=value["action"],
            fixture_path=value["fixture_path"],
        )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SenderProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_runtime(runtime_root) -> SimpleNamespace:
    from .bundle_capture import bind_runtime

    root = Path(runtime_root).resolve(strict=True)
    bind_runtime(root)
    import torch
    from safetensors.torch import load_file

    from sglang.srt.utils import init_custom_process_group

    bind_runtime(root)
    if (
        Path(inspect.getfile(init_custom_process_group)).resolve()
        != root / SENDER_RUNTIME_ORIGINS["init_custom_process_group"]
    ):
        raise SenderProtocolError(
            "sender runtime helper origin differs from recorded checkout"
        )

    return SimpleNamespace(
        torch=torch,
        init_custom_process_group=init_custom_process_group,
        load_file=load_file,
        runtime_origins=dict(SENDER_RUNTIME_ORIGINS),
    )


def _load_config(fixture: Path) -> dict[str, object]:
    config_path = fixture / "adapter_config.json"
    try:
        value = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise SenderProtocolError(f"cannot read adapter config: {error}") from error
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SenderProtocolError("adapter config must be an object")
    return dict(value)


def _broadcast_fixture(
    fixture_path: str, group: object, runtime: SimpleNamespace
) -> dict[str, object]:
    fixture = Path(fixture_path)
    if not fixture.is_dir():
        raise SenderProtocolError("fixture_path must name an existing directory")
    tensors = runtime.load_file(
        str(fixture / "adapter_model.safetensors"), device="cpu"
    )
    if not isinstance(tensors, Mapping) or not tensors:
        raise SenderProtocolError("adapter tensor file must be non-empty")
    names = sorted(tensors)
    dtypes = []
    shapes = []
    for name in names:
        tensor = tensors[name]
        dtypes.append(str(tensor.dtype).removeprefix("torch."))
        shapes.append([int(dimension) for dimension in tensor.shape])
        cuda_tensor = tensor.to("cuda")
        runtime.torch.distributed.broadcast(cuda_tensor, src=0, group=group)
    runtime.torch.cuda.synchronize()
    return {
        "names": names,
        "dtypes": dtypes,
        "shapes": shapes,
        "config": _load_config(fixture),
    }


def _emit(output: TextIO, event: str, **fields: object) -> None:
    record = {"event": event, **fields}
    output.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    output.flush()


def run_protocol(
    spec: SenderSpec,
    commands: Iterable[SenderCommand],
    *,
    output: TextIO = sys.stdout,
    runtime: SimpleNamespace | None = None,
    runtime_root: str | None = None,
) -> int:
    """Join the update group, execute commands, and always destroy the group."""

    group = None
    store = None
    current_request_id = None
    close_request_id = None
    failure = None
    try:
        runtime = _load_runtime(runtime_root) if runtime is None else runtime
        if runtime.runtime_origins != SENDER_RUNTIME_ORIGINS:
            raise SenderProtocolError("sender runtime origin attestation is invalid")
        runtime.torch.cuda.set_device(spec.device)
        # Bind the actual rendezvous listener before announcing its port. A
        # bind-and-close probe leaves the port unowned while this child imports.
        store = runtime.torch.distributed.TCPStore(
            host_name=spec.master_address,
            port=spec.master_port,
            world_size=spec.world_size,
            is_master=spec.rank == 0,
            timeout=timedelta(seconds=spec.group_timeout_seconds),
            wait_for_workers=False,
        )
        _emit(
            output,
            "rendezvous.ready",
            request_id=None,
            master_port=store.port,
            runtime_origins=runtime.runtime_origins,
        )
        group = runtime.init_custom_process_group(
            backend="nccl",
            store=runtime.torch.distributed.PrefixStore(spec.group_name, store),
            timeout=timedelta(seconds=spec.group_timeout_seconds),
            world_size=spec.world_size,
            rank=spec.rank,
            group_name=spec.group_name,
        )
        _emit(output, "ready", request_id=None, runtime_origins=runtime.runtime_origins)
        for command in commands:
            current_request_id = getattr(command, "request_id", None)
            if not isinstance(command, SenderCommand):
                raise SenderProtocolError("command iterator produced an invalid value")
            if command.action == "broadcast":
                payload = _broadcast_fixture(command.fixture_path, group, runtime)
                _emit(
                    output,
                    "broadcast.complete",
                    request_id=command.request_id,
                    payload=payload,
                )
                current_request_id = None
                continue
            close_request_id = command.request_id
            current_request_id = None
            break
        else:
            raise SenderProtocolError("command stream ended before close")
    except Exception as error:  # noqa: BLE001 - child reports any bounded failure
        failure = error
    finally:
        if group is not None:
            try:
                runtime.torch.distributed.destroy_process_group(group)
            except Exception as error:  # noqa: BLE001 - teardown is part of verdict
                if failure is None:
                    failure = error
        group = None
        store = None

    if failure is not None:
        _emit(
            output,
            "error",
            request_id=current_request_id,
            error_type=type(failure).__name__,
            message=str(failure),
        )
        return 1
    _emit(output, "close.complete", request_id=close_request_id)
    return 0


def _child_exit_error(process: subprocess.Popen) -> SenderProcessError:
    return_code = process.poll()
    stderr = ""
    if return_code is not None and process.stderr is not None:
        stderr = process.stderr.read().strip()
    detail = f": {stderr}" if stderr else ""
    return SenderProcessError(
        f"distributed sender exited with status {return_code} before response{detail}"
    )


def read_response_line(process: subprocess.Popen, timeout: float) -> str:
    """Read one complete child response without trusting a still-running PID."""

    if type(timeout) not in (int, float) or timeout <= 0:
        raise SenderTimeout("sender response timed out")
    if process.stdout is None:
        raise SenderProcessError("distributed sender stdout is unavailable")
    deadline = time.monotonic() + timeout
    # TextIO.readline may prefetch the following record, which then becomes
    # invisible to select(), or block past the deadline on a partial record.
    # This pipe has one consumer; keep unconsumed raw bytes with its process.
    buffer = getattr(process, "_adapter_sender_response_buffer", None)
    if buffer is None:
        buffer = bytearray()
        process._adapter_sender_response_buffer = buffer
    descriptor = process.stdout.fileno()
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    try:
        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                try:
                    return line.decode("utf-8").removesuffix("\r")
                except UnicodeDecodeError as error:
                    raise SenderProtocolError(
                        "sender response is not valid UTF-8"
                    ) from error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SenderTimeout("sender response timed out")
            if not selector.select(remaining):
                if process.poll() is not None:
                    raise _child_exit_error(process)
                raise SenderTimeout("sender response timed out")
            chunk = os.read(descriptor, 65536)
            if not chunk:
                if buffer:
                    raise SenderProtocolError("sender response ended without a newline")
                raise _child_exit_error(process)
            buffer.extend(chunk)
    finally:
        selector.close()


class DistributedSender:
    def __init__(self, spec: SenderSpec, process: subprocess.Popen) -> None:
        self.spec = spec
        self.process = process
        self._used_request_ids = set()
        self._ready = False
        self._failed = False
        self._closed = False
        self._close_request_id = None
        self.runtime_origins = None
        self.rendezvous_port = None

    @classmethod
    def spawn(cls, spec: SenderSpec, timeout: float) -> DistributedSender:
        from .bundle_capture import REPO_ROOT

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--runtime-root",
            str(REPO_ROOT.resolve(strict=True)),
            "--master-address",
            spec.master_address,
            "--master-port",
            str(spec.master_port),
            "--rank",
            str(spec.rank),
            "--world-size",
            str(spec.world_size),
            "--group-name",
            spec.group_name,
            "--device",
            str(spec.device),
            "--group-timeout",
            str(spec.group_timeout_seconds),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        return cls(spec, process)

    @classmethod
    def start(cls, spec: SenderSpec, timeout: float) -> DistributedSender:
        sender = cls.spawn(spec, timeout)
        try:
            sender.wait_ready(timeout)
        except Exception:
            sender.terminate()
            raise
        return sender

    def _response(self, timeout: float) -> dict[str, object]:
        try:
            raw = read_response_line(self.process, timeout)
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, SenderProtocolError) as error:
            raise SenderProtocolError(f"invalid sender response: {error}") from error
        if not isinstance(value, Mapping):
            raise SenderProtocolError("sender response must be an object")
        response = dict(value)
        if response.get("event") == "error":
            self._failed = True
            raise SenderProcessError(str(response.get("message", "sender failed")))
        return response

    def wait_rendezvous(self, timeout: float) -> int:
        if self.rendezvous_port is not None:
            return self.rendezvous_port
        response = self._response(timeout)
        port = response.get("master_port")
        if (
            response
            != {
                "event": "rendezvous.ready",
                "request_id": None,
                "master_port": port,
                "runtime_origins": SENDER_RUNTIME_ORIGINS,
            }
            or type(port) is not int
            or not 1 <= port <= 65535
            or self.spec.master_port not in (0, port)
        ):
            raise SenderProtocolError(
                "sender did not return an exact rendezvous ready record"
            )
        self.rendezvous_port = port
        self.runtime_origins = dict(response["runtime_origins"])
        return port

    def wait_ready(self, timeout: float) -> None:
        if self._ready:
            return
        deadline = time.monotonic() + timeout
        self.wait_rendezvous(timeout)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SenderTimeout("sender group initialization timed out")
        response = self._response(remaining)
        if response != {
            "event": "ready",
            "request_id": None,
            "runtime_origins": SENDER_RUNTIME_ORIGINS,
        }:
            raise SenderProtocolError("sender did not return the exact ready record")
        self.runtime_origins = dict(response["runtime_origins"])
        self._ready = True

    def _send(self, command: SenderCommand) -> None:
        if self._closed or self._failed:
            raise SenderProcessError("distributed sender is not available")
        if command.request_id in self._used_request_ids:
            raise SenderProtocolError("sender request_id must be unique")
        if self.process.poll() is not None:
            raise _child_exit_error(self.process)
        if self.process.stdin is None:
            raise SenderProcessError("distributed sender stdin is unavailable")
        self._used_request_ids.add(command.request_id)
        try:
            self.process.stdin.write(
                json.dumps(command.to_dict(), allow_nan=False, sort_keys=True) + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise SenderProcessError(f"cannot write sender command: {error}") from error

    def broadcast_fixture(self, request_id: str, fixture: Path, *, timeout: float):
        from .server import DistributedPayload

        command = SenderCommand(request_id, "broadcast", str(fixture.resolve()))
        try:
            self._send(command)
            response = self._response(timeout)
            if response.get("event") != "broadcast.complete":
                raise SenderProtocolError("sender returned the wrong broadcast event")
            if response.get("request_id") != request_id:
                raise SenderProtocolError("sender response request_id does not match")
            payload = response.get("payload")
            if not isinstance(payload, Mapping):
                raise SenderProtocolError("sender broadcast payload must be an object")
            expected = {"names", "dtypes", "shapes", "config"}
            if set(payload) != expected:
                raise SenderProtocolError("sender broadcast payload fields are invalid")
            names = payload["names"]
            dtypes = payload["dtypes"]
            shapes = payload["shapes"]
            config = payload["config"]
            if (
                type(names) is not list
                or any(type(name) is not str or not name for name in names)
                or names != sorted(names)
                or type(dtypes) is not list
                or any(type(dtype) is not str or not dtype for dtype in dtypes)
                or type(shapes) is not list
                or any(
                    type(shape) is not list
                    or not shape
                    or any(type(size) is not int or size <= 0 for size in shape)
                    for shape in shapes
                )
                or not len(names) == len(dtypes) == len(shapes)
                or not isinstance(config, Mapping)
            ):
                raise SenderProtocolError("sender broadcast payload is malformed")
            return DistributedPayload(
                names=tuple(names),
                dtypes=tuple(dtypes),
                shapes=tuple(tuple(shape) for shape in shapes),
                config=dict(config),
            )
        except Exception:
            self._failed = True
            self.terminate()
            raise

    def begin_close(self, *, request_id: str) -> None:
        """Let the peer enter collective teardown before waiting on either rank."""
        if self._closed:
            return
        if self._failed or self.process.poll() is not None:
            self.terminate()
            return
        if self._close_request_id is not None:
            if request_id != self._close_request_id:
                raise SenderProtocolError("sender close already requested")
            return
        try:
            self._send(SenderCommand(request_id, "close", None))
            self._close_request_id = request_id
        except Exception:
            self._failed = True
            self.terminate()
            raise

    def finish_close(self, *, timeout: float) -> None:
        if self._closed:
            return
        if self._close_request_id is None:
            raise SenderProtocolError("sender close was not requested")
        deadline = time.monotonic() + timeout
        try:
            response = self._response(timeout)
            if response != {
                "event": "close.complete",
                "request_id": self._close_request_id,
            }:
                raise SenderProtocolError("sender did not acknowledge exact close")
            return_code = self.process.wait(timeout=max(0, deadline - time.monotonic()))
            if return_code != 0:
                raise SenderProcessError(
                    f"distributed sender exited with status {return_code}"
                )
            self._closed = True
            self._close_streams()
        except Exception:
            self._failed = True
            self.terminate()
            raise

    def close(self, *, request_id: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        self.begin_close(request_id=request_id)
        self.finish_close(timeout=max(0, deadline - time.monotonic()))

    def _close_streams(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def terminate(self) -> None:
        if self._closed:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._closed = True
        self._close_streams()


@dataclass
class DistributedSession:
    engine: object
    sender: DistributedSender
    group_name: str
    _closed: bool = False

    @classmethod
    def open(
        cls,
        engine: object,
        tp_size: int,
        timeout: float,
        *,
        master_address: str = "127.0.0.1",
        master_port: int | None = None,
        group_name: str | None = None,
    ) -> DistributedSession:
        if type(tp_size) is not int or tp_size <= 0:
            raise DistributedSessionError("tp_size must be a positive integer")
        deadline = time.monotonic() + timeout
        port = 0 if master_port is None else master_port
        resolved_group = group_name or f"adapter-eq-{uuid4().hex}"
        spec = SenderSpec(
            master_address=master_address,
            master_port=port,
            rank=0,
            world_size=tp_size + 1,
            group_name=resolved_group,
            device=0,
            group_timeout_seconds=timeout,
        )
        child = DistributedSender.spawn(spec, timeout)
        executor = None
        future = None
        init_started = __import__("threading").Event()

        def initialize_engine():
            init_started.set()
            return engine.init_weights_update_group(
                master_address=master_address,
                master_port=port,
                rank_offset=1,
                world_size=tp_size + 1,
                group_name=resolved_group,
                backend="nccl",
            )

        try:
            port = child.wait_rendezvous(deadline - time.monotonic())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SenderTimeout("sender rendezvous initialization timed out")
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="adapter-group"
            )
            future = executor.submit(initialize_engine)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not init_started.wait(remaining):
                raise SenderTimeout("engine group initialization did not start")
            remaining = deadline - time.monotonic()
            child.wait_ready(remaining)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SenderTimeout("engine group initialization timed out")
            raw_result = future.result(timeout=remaining)
            from .server import normalize_control_result

            result = normalize_control_result(raw_result)
            if not result.success:
                raise DistributedSessionError(result.message)
            return cls(engine=engine, sender=child, group_name=resolved_group)
        except Exception as error:
            if future is not None:
                try:
                    engine.destroy_weights_update_group(resolved_group)
                except Exception:
                    pass
            child.terminate()
            if isinstance(error, DistributedSessionError):
                raise
            if isinstance(error, FutureTimeout):
                error = SenderTimeout("engine group initialization timed out")
            raise DistributedSessionError(str(error)) from error
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def close(self, timeout: float) -> None:
        if self._closed:
            return
        deadline = time.monotonic() + timeout
        failures = []
        try:
            # NCCL destruction can require all ranks to participate. Do not
            # wait for the engine while the peer is still blocked on stdin.
            self.sender.begin_close(request_id=f"close-{uuid4().hex}")
        except Exception as error:
            failures.append(str(error))
        try:
            from .server import normalize_control_result

            result = normalize_control_result(
                self.engine.destroy_weights_update_group(self.group_name)
            )
            if not result.success:
                failures.append(result.message)
        except Exception as error:  # noqa: BLE001 - still close the sender peer
            failures.append(str(error))
        try:
            remaining = max(deadline - time.monotonic(), 0.001)
            self.sender.finish_close(timeout=remaining)
        except Exception as error:  # noqa: BLE001 - report both teardown sides
            failures.append(str(error))
        self._closed = True
        if failures:
            raise DistributedSessionError("; ".join(failures))


def _commands_from_stream(stream: TextIO):
    for line in stream:
        if line.strip():
            yield SenderCommand.from_json(line)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--group-timeout", required=True, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = SenderSpec(
        master_address=args.master_address,
        master_port=args.master_port,
        rank=args.rank,
        world_size=args.world_size,
        group_name=args.group_name,
        device=args.device,
        group_timeout_seconds=args.group_timeout,
    )
    # Native libraries (including NCCL) write directly to fd 1. Keep the JSON
    # pipe on its own descriptor before importing the runtime, then send all
    # ordinary Python and native stdout to the inherited diagnostic stream.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        return run_protocol(
            spec,
            _commands_from_stream(sys.stdin),
            output=output,
            runtime_root=args.runtime_root,
        )


if __name__ == "__main__":
    raise SystemExit(main())

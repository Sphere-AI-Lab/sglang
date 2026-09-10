"""Temporary fault boundaries around real tokenizer communicator replies.

``rank`` selects a collected reply position, not a CUDA rank: tokenizer replies
are already TP-consensus results and arrive without physical-rank metadata.
Actual rank-zero/nonzero worker failures require the registered TP tests.
"""

from __future__ import annotations

import asyncio
import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from .scenarios import ScenarioContractError


@dataclass(frozen=True)
class RankFailure:
    rank: int
    message: str

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 0 or not self.message:
            raise ScenarioContractError(
                "rank failure requires a nonnegative rank and message"
            )


class PhaseGate:
    """One-shot asynchronous barrier with thread-safe external release."""

    def __init__(self, phase):
        if phase not in {"fan-out", "publication", "rollback", "eviction"}:
            raise ScenarioContractError(f"unknown fault phase: {phase}")
        self.phase = phase
        self.entered = threading.Event()
        self.released = threading.Event()
        self.exited = threading.Event()
        self.request = None

    async def block(self, request=None):
        if self.entered.is_set():
            raise ScenarioContractError("phase gate entered more than once")
        self.request = request
        self.entered.set()
        while not self.released.is_set():
            await asyncio.sleep(0.001)
        self.exited.set()

    async def wait_entered(self, timeout):
        async def wait():
            while not self.entered.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait(), timeout)

    def release(self):
        self.released.set()


class FaultController:
    """Inject one result after real IPC, and restore exact attributes in finally."""

    def __init__(self, manager, mode="native_lora"):
        self.manager = manager
        self.mode = mode
        self.injected = []
        self.requests = []

    def attribute(self, operation):
        if operation in {"load", "unload"}:
            kind = "oft" if self.mode == "native_oft" else "lora"
            return f"update_{kind}_adapter_communicator"
        try:
            return {
                "stage": "update_adapter_from_distributed_communicator",
                "activate": "activate_adapter_version_communicator",
                "rollback": "discard_adapter_stage_communicator",
            }[operation]
        except KeyError as error:
            raise ScenarioContractError(
                f"unknown fault operation: {operation}"
            ) from error

    @contextmanager
    def wrap(self, operation, *, failure=None, gate=None, observe=None, when=None):
        attribute = self.attribute(operation)
        original = getattr(self.manager, attribute)
        used = False

        async def wrapped(request):
            nonlocal used
            self.requests.append(request)
            selected = when is None or when(request)
            if observe is not None and selected:
                observe(request)
            results = await original(request)
            if failure is not None and not used and selected:
                if type(results) not in (list, tuple) or failure.rank >= len(results):
                    raise ScenarioContractError("injected rank has no collected reply")
                if any(
                    getattr(result, "success", None) is not True for result in results
                ):
                    raise ScenarioContractError(
                        "real communicator failed before injection"
                    )
                modified = copy.copy(results[failure.rank])
                field = (
                    "error_message" if hasattr(modified, "error_message") else "message"
                )
                if not hasattr(modified, field):
                    raise ScenarioContractError(
                        "rank result has no failure message field"
                    )
                modified.success = False
                setattr(modified, field, f"rank {failure.rank}: {failure.message}")
                results = list(results)
                results[failure.rank] = modified
                self.injected.append(failure)
                used = True
            if gate is not None and selected:
                await gate.block(request)
            return results

        setattr(self.manager, attribute, wrapped)
        try:
            yield self
        finally:
            if gate is not None:
                gate.release()
            setattr(self.manager, attribute, original)

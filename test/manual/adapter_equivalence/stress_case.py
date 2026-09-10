"""Bounded native adapter churn using the same real controls as lifecycle runs."""

from __future__ import annotations

import asyncio
import copy
import math
import time
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable

from .scenarios import ScenarioContractError
from .schema import Observation, canonical_sha256
from .server import AdapterControl, ControlResult


@dataclass(frozen=True)
class StressSpec:
    control: AdapterControl
    a_tensors: dict[str, object]
    a_config: dict[str, object]
    b_tensors: dict[str, object]
    b_config: dict[str, object]
    generate_mixed: Callable[[int, bool], list[Observation]]
    expected_final_state: dict[str, object]
    upsert_distributed: Callable[[str, int, bool], ControlResult]
    operation_timeout: float = 300.0
    job_timeout: float = 3600.0
    emit: Callable[..., None] | None = None

    def __post_init__(self):
        for value in (self.operation_timeout, self.job_timeout):
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ScenarioContractError(
                    "stress deadlines must be positive and finite"
                )


@dataclass(frozen=True)
class StressResult:
    cycles_completed: int
    requests_completed: int
    final_state: dict[str, object]
    completion_hash: str


def run_stress(spec: StressSpec) -> StressResult:
    # Reuse the daemon-worker deadline mechanism; no executor exit join can turn
    # a failed deadline into an unbounded wait. The owning ShardRunner tears down.
    from .run_case import OperationTimeout, _Worker, require_control_success

    worker = _Worker("adapter-stress")
    started = time.monotonic()
    deadline = started + spec.job_timeout
    cycles_completed = requests_completed = 0
    cycle = 0

    def emit(event, **fields):
        if spec.emit is not None:
            now = time.monotonic()
            try:
                spec.emit(
                    event,
                    cycle=cycle,
                    cycles_completed=cycles_completed,
                    requests_completed=requests_completed,
                    elapsed_seconds=now - started,
                    remaining_job_seconds=max(0.0, deadline - now),
                    **fields,
                )
            except Exception:
                pass  # Telemetry must not change the stress verdict.

    def call(operation, function, adapter=None):
        call_started = time.monotonic()
        job_remaining = deadline - call_started
        call_deadline = min(deadline, call_started + spec.operation_timeout)
        remaining = call_deadline - call_started
        scope = "whole-job" if job_remaining <= spec.operation_timeout else "operation"

        def record(event, **fields):
            emit(
                event,
                operation=operation,
                adapter=adapter,
                duration_seconds=time.monotonic() - call_started,
                **fields,
            )

        def timeout(operation_submitted):
            record(
                "stress.operation.timeout",
                timeout_scope=scope,
                operation_submitted=operation_submitted,
                wait_budget_seconds=max(0.0, remaining),
            )
            return OperationTimeout(
                f"stress {scope} deadline exceeded: cycle={cycle} "
                f"operation={operation} adapter={adapter} "
                f"cycles_completed={cycles_completed} requests_completed={requests_completed}"
            )

        if remaining <= 0:
            raise timeout(False)
        record("stress.operation.started", wait_budget_seconds=remaining)
        # Synchronous diagnostic I/O consumes this call's original budget.
        # Do not dispatch new native work after that absolute deadline.
        remaining = call_deadline - time.monotonic()
        if remaining <= 0:
            raise timeout(False)
        future = worker.submit(function)
        remaining = max(0.0, call_deadline - time.monotonic())
        try:
            result = future.result(timeout=remaining)
        except FutureTimeout as error:
            if future.done():
                # Resolve completion racing the wait deadline: a native
                # TimeoutError must retain its identity, while an already
                # available successful result needs no additional wait.
                try:
                    result = future.result()
                except BaseException as native_error:
                    failure(native_error, record)
                    raise
            else:
                raise timeout(True) from error
        except BaseException as error:
            failure(error, record)
            raise
        record("stress.operation.completed", outcome="returned")
        return result

    def failure(error, record):
        # Diagnostic I/O failure must not replace the native failure (including
        # the acknowledged CancelledError used by every tenth cycle).
        if isinstance(error, asyncio.CancelledError):
            record("stress.operation.completed", outcome="cancelled")
        else:
            record(
                "stress.operation.failed",
                error_type=type(error).__name__,
                message=str(error),
            )

    def records(operation, adapter):
        state = call(operation, spec.control.inspect_state, adapter)
        return {record["name"]: record for record in state["registered"]}

    expected = copy.deepcopy(spec.expected_final_state)
    try:
        for cycle in range(100):
            cycle_started = time.monotonic()
            emit("stress.cycle.started")
            for name, tensors, config in (
                ("policy-a", spec.a_tensors, spec.a_config),
                ("policy-b", spec.b_tensors, spec.b_config),
            ):
                require_control_success(
                    call(
                        "load",
                        lambda: spec.control.load_tensors(name, tensors, config),
                        name,
                    ),
                    "stress.load",
                )
            for name in ("policy-a", "policy-b"):
                before = records("inspect.before-upsert", name)
                if set(before) != {"policy-a", "policy-b"}:
                    raise ScenarioContractError(
                        "stress upsert requires both live adapters"
                    )
                cancel = name == "policy-b" and (cycle + 1) % 10 == 0
                try:
                    result = call(
                        "upsert",
                        lambda: spec.upsert_distributed(name, cycle, cancel),
                        name,
                    )
                except asyncio.CancelledError:
                    if not cancel:
                        raise
                else:
                    if cancel:
                        raise ScenarioContractError(
                            "stress cancellation was not acknowledged"
                        )
                    require_control_success(result, "stress.upsert")
                after = records("inspect.after-upsert", name)
                other = "policy-b" if name == "policy-a" else "policy-a"
                if (
                    set(after) != set(before)
                    or after[name]["id"] != before[name]["id"]
                    or after[name]["version"] == before[name]["version"]
                    or after[other] != before[other]
                ):
                    raise ScenarioContractError(
                        "stress upsert did not publish an exact in-place refresh"
                    )
            observations = call(
                "generate", lambda: spec.generate_mixed(10, bool(cycle % 2))
            )
            if not isinstance(observations, (tuple, list)) or len(observations) != 10:
                raise ScenarioContractError("stress request count differs")
            for observation in observations:
                if (
                    not isinstance(observation, Observation)
                    or not observation.output_ids
                    or observation.error is not None
                ):
                    raise ScenarioContractError(
                        "stress missing successful inference evidence"
                    )
                observation.validate()
            requests_completed += len(observations)
            for name in ("policy-b", "policy-a"):
                require_control_success(
                    call("unload", lambda: spec.control.unload(name), name),
                    "stress.unload",
                )
            cycles_completed += 1
            emit(
                "stress.cycle.completed",
                duration_seconds=time.monotonic() - cycle_started,
            )
        final_state = copy.deepcopy(call("inspect.final", spec.control.inspect_state))
        if final_state != expected:
            raise ScenarioContractError("stress final adapter state differs")
        return StressResult(100, 1000, final_state, canonical_sha256(final_state))
    finally:
        worker.close()

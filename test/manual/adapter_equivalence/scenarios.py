"""Frozen lifecycle contracts for adapter-equivalence runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .schema import Observation


class ScenarioContractError(ValueError):
    """Raised when a lifecycle run violates the frozen oracle contract."""


_ADAPTER_LIFECYCLE_TRANSITIONS = (
    "base.initial",
    "startup.adapter",
    "dynamic.load",
    "dynamic.infer",
    "dynamic.unload",
    "dynamic.post-unload-base",
    "switch.a",
    "switch.b",
    "switch.a-again",
    "mixed.base-a-b",
    "concurrent.stream",
    "concurrent.non-stream",
    "prefill.short",
    "prefill.long",
    "decode.short",
    "decode.long",
    "stage.v1",
    "activate.v1",
    "stage.v2",
    "activate.v2",
    "reject.duplicate",
    "reject.stale",
    "reject.invalid-id",
    "reject.invalid-config",
    "rollback.previous",
    "restart.same-manifest",
)

_ADAPTER_MODES = {
    "legacy_oft",
    "canonical_oft",
    "legacy_lora",
    "native_lora",
}


@dataclass(frozen=True)
class LifecycleStep:
    """One observable transition in the adapter lifecycle state machine."""

    name: str
    action: str
    adapter: str | None = None
    version: str | None = None
    prompt_id: str | None = None
    stream: bool | None = None
    max_new_tokens: int | None = None


class LifecycleExecutor(Protocol):
    def execute(self, step: LifecycleStep) -> Observation:
        """Execute one transition and return its immutable observation."""


_ADAPTER_LIFECYCLE_STEPS = (
    LifecycleStep("base.initial", "generate", prompt_id="factual"),
    LifecycleStep(
        "startup.adapter",
        "startup_generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("dynamic.load", "load", adapter="policy-a"),
    LifecycleStep(
        "dynamic.infer", "generate", adapter="policy-a", prompt_id="factual"
    ),
    LifecycleStep("dynamic.unload", "unload", adapter="policy-a"),
    LifecycleStep(
        "dynamic.post-unload-base", "generate", prompt_id="factual"
    ),
    LifecycleStep("switch.a", "generate", adapter="policy-a", prompt_id="factual"),
    LifecycleStep("switch.b", "generate", adapter="policy-b", prompt_id="factual"),
    LifecycleStep(
        "switch.a-again", "generate", adapter="policy-a", prompt_id="factual"
    ),
    LifecycleStep("mixed.base-a-b", "mixed_batch", prompt_id="batch-8"),
    LifecycleStep(
        "concurrent.stream", "concurrent", prompt_id="batch-8", stream=True
    ),
    LifecycleStep(
        "concurrent.non-stream", "concurrent", prompt_id="batch-8", stream=False
    ),
    LifecycleStep("prefill.short", "generate", prompt_id="factual"),
    LifecycleStep("prefill.long", "generate", prompt_id="long-prefix"),
    LifecycleStep(
        "decode.short", "generate", prompt_id="factual", max_new_tokens=1
    ),
    LifecycleStep(
        "decode.long", "generate", prompt_id="factual", max_new_tokens=128
    ),
    LifecycleStep("stage.v1", "stage", adapter="policy-a", version="1"),
    LifecycleStep("activate.v1", "activate", adapter="policy-a", version="1"),
    LifecycleStep("stage.v2", "stage", adapter="policy-a", version="2"),
    LifecycleStep("activate.v2", "activate", adapter="policy-a", version="2"),
    LifecycleStep(
        "reject.duplicate", "reject_duplicate", adapter="policy-a", version="2"
    ),
    LifecycleStep(
        "reject.stale", "reject_stale", adapter="policy-a", version="1"
    ),
    LifecycleStep(
        "reject.invalid-id", "reject_invalid_id", adapter="missing", version="3"
    ),
    LifecycleStep(
        "reject.invalid-config",
        "reject_invalid_config",
        adapter="policy-a",
        version="3",
    ),
    LifecycleStep(
        "rollback.previous", "generate", adapter="policy-a", prompt_id="factual"
    ),
    LifecycleStep("restart.same-manifest", "restart", prompt_id="factual"),
)


def lifecycle_transition_names(mode: str) -> tuple[str, ...]:
    """Return the ordered transition contract for one adapter mode."""

    if mode not in _ADAPTER_MODES:
        raise ScenarioContractError(f"unknown adapter lifecycle mode: {mode}")
    return _ADAPTER_LIFECYCLE_TRANSITIONS


def lifecycle_steps(mode: str) -> tuple[LifecycleStep, ...]:
    """Return the complete ordered state machine for one adapter mode."""

    lifecycle_transition_names(mode)
    return _ADAPTER_LIFECYCLE_STEPS


def execute_lifecycle(
    mode: str,
    executor: LifecycleExecutor,
) -> dict[str, Observation]:
    """Execute every transition exactly once and validate cross-step invariants."""

    observations: dict[str, Observation] = {}
    for step in lifecycle_steps(mode):
        observation = executor.execute(step)
        if not isinstance(observation, Observation):
            raise ScenarioContractError(
                f"executor returned non-Observation for {step.name}"
            )
        if step.name in observations:
            raise ScenarioContractError(f"duplicate lifecycle observation: {step.name}")
        observations[step.name] = observation
    validate_lifecycle_observations(observations)
    return observations


def _output_payload(observation: Observation) -> tuple[object, ...]:
    return (
        observation.output_ids,
        observation.text,
        observation.token_logprobs,
        observation.selected_logits,
        observation.error,
    )


def validate_lifecycle_observations(
    observations: Mapping[str, Observation],
) -> None:
    """Validate cross-transition invariants that require exact identity."""

    required = ("base.initial", "dynamic.post-unload-base")
    missing = [name for name in required if name not in observations]
    if missing:
        raise ScenarioContractError(
            "missing lifecycle observations: " + ", ".join(missing)
        )

    initial = observations["base.initial"]
    restored = observations["dynamic.post-unload-base"]
    if _output_payload(restored) != _output_payload(initial):
        raise ScenarioContractError(
            "post-unload output does not exactly match initial base output"
        )


def classify_startup_failure(
    revision_kind: str,
    mode: str,
    traceback: str,
) -> dict[str, str]:
    """Classify unchanged-source legacy startup failures without hiding defects."""

    if revision_kind != "source" or mode not in {"legacy_oft", "legacy_lora"}:
        raise ScenarioContractError(
            "startup failure is not eligible for legacy-source classification"
        )
    if not traceback:
        raise ScenarioContractError("startup failure traceback must be non-empty")
    return {
        "status": "unsupported_by_legacy",
        "mode": mode,
        "traceback": traceback,
    }

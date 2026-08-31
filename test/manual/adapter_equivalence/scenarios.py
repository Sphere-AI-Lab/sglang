"""Frozen lifecycle contracts for adapter-equivalence runs."""

from __future__ import annotations

from collections.abc import Mapping

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


def lifecycle_transition_names(mode: str) -> tuple[str, ...]:
    """Return the ordered transition contract for one adapter mode."""

    if mode not in _ADAPTER_MODES:
        raise ScenarioContractError(f"unknown adapter lifecycle mode: {mode}")
    return _ADAPTER_LIFECYCLE_TRANSITIONS


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

"""Server argument contracts for frozen source and candidate revisions."""

from __future__ import annotations

from .scenarios import ScenarioContractError


_SOURCE_MODE_ARGS = {
    "base": (),
    "legacy_oft": ("--peft-method", "oft", "--oft-impl", "peft"),
    "canonical_oft": ("--peft-method", "oft", "--oft-impl", "sibling"),
    "legacy_lora": ("--peft-method", "lora"),
    "native_lora": ("--enable-lora", "--enable-lora-staging"),
}

_CANDIDATE_MODE_ARGS = {
    "base": (),
    "canonical_oft": ("--peft-method", "oft"),
    "native_lora": ("--enable-lora", "--enable-lora-staging"),
}


def mode_server_args(revision_kind: str, mode: str) -> tuple[str, ...]:
    """Return the exact adapter-selection arguments for a frozen revision."""

    if revision_kind == "source":
        arguments = _SOURCE_MODE_ARGS
    elif revision_kind == "candidate":
        if mode in {"legacy_oft", "legacy_lora"}:
            raise ScenarioContractError(f"{mode} is a source-only oracle mode")
        arguments = _CANDIDATE_MODE_ARGS
    else:
        raise ScenarioContractError(f"unknown revision kind: {revision_kind}")

    try:
        return arguments[mode]
    except KeyError as error:
        raise ScenarioContractError(
            f"unknown {revision_kind} adapter mode: {mode}"
        ) from error

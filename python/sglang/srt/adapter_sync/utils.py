"""Helpers the shared adapter core needs, kept here so ``srt/adapter_sync``
does not import any single method's package."""

from typing import Set


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:
    """Return the entry of ``target_modules`` that matches ``full_module_name``.

    Copied from ``srt/oft/utils.py`` (WS2-1): the shared core must not depend on
    ``srt/oft``, or a LoRA-only deployment would drag the OFT package in.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )

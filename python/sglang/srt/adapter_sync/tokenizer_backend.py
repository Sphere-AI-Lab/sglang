"""Tokenizer-layer staging interface shared by every staged adapter method.

Each PEFT method that supports staged (RL) weight updates implements this
ABC once, so TokenizerControlMixin.update_adapter_from_distributed /
activate_adapter_version can orchestrate the shared pause/lock/dispatch
sequence without branching on which method is active.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple


async def finish_irreversible_update(operation):
    """Keep an already-started worker mutation alive through caller cancellation."""
    task = asyncio.create_task(operation())
    caller_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            caller_cancelled = True
    result = task.result()
    if caller_cancelled:
        raise asyncio.CancelledError
    return result


class AdapterStagingBackend(ABC):
    """One instance per TokenizerManager, selected by the server's staging
    server_args flag (see ``_STAGING_BACKENDS`` below)."""

    @staticmethod
    def _validate_adapter_id(obj, expected_id: str) -> None:
        """Treat an explicit caller ID as an assertion of the known identity."""
        if obj.adapter_id is not None and obj.adapter_id != expected_id:
            raise ValueError(
                f"Requested adapter_id {obj.adapter_id!r} does not match "
                f"expected adapter_id {expected_id!r}"
            )

    @property
    @abstractmethod
    def lifecycle_lock(self):
        """Lock shared with load, upsert, and unload for this adapter type."""

    @abstractmethod
    async def reserve_stage(self, obj) -> None:
        """Called before dispatch to the scheduler. Reserve/mint the adapter
        identity for this stage request, mutating ``obj.adapter_id`` if the
        method resolves identity from a name. Reject an explicit ID that
        differs from an already-known active or pending ID before mutation.
        Must be safe to call twice with the same (name, version) — the caller
        does not deduplicate."""

    @abstractmethod
    def prepare_activation(self, obj) -> None:
        """Called before dispatch, synchronously, for an activate request.
        Validate the pending stage matches ``obj``'s identity/version and
        raise ``ValueError`` if not. Mutates ``obj.adapter_id`` the same way
        ``reserve_stage`` does, for the activate wire path."""

    @abstractmethod
    async def finish_activation(self, obj, results: List) -> Tuple[bool, str]:
        """Called after every worker's stage-then-activate or activate RPC
        returns. ``results`` is the list of per-worker RPC outputs. Must
        publish the new identity into the method's registry only on
        all-worker success, and return ``(success, message)``."""

    @abstractmethod
    def clear_stage_reservation(self, obj) -> None:
        """Validate the exact identity, then clear only its pending reservation.

        Called with the lifecycle lock held after unanimous worker rollback.
        Validation must finish before changing any tokenizer state.
        """

    @abstractmethod
    def _quarantine(self, name: str, message: str) -> None:
        """Keep an adapter unavailable when stage cleanup cannot be proven."""


# Explicit registry, not decorator/import-side-effect magic: this codebase's
# packages are deliberately lazy-imported (see srt/oft/__init__.py's PEP 562
# pattern) to avoid circular imports and boot-time cost, so a module isn't
# guaranteed to have run its top-level code before dispatch needs it. Adding
# a third adapter method means adding one entry here -- never touching
# tokenizer_control_mixin.py again.
#
# Task 4 registers native LoRA; Task 5 adds canonical OFT without changing
# the shared dispatch loop.
#
# is_enabled is a lambda doing DIRECT attribute access (sa.enable_lora_staging),
# not a string-keyed getattr(sa, flag_name, False) -- these fields are
# guaranteed real ServerArgs fields, so a getattr-with-default there would be
# exactly the no-getattr-defensive anti-pattern this codebase's rules forbid.
# getattr(module, class_name) below is a different, necessary case: this
# module defines AdapterStagingBackend, and srt/lora/staged_manager.py
# imports it to subclass -- an eager top-level import back from here would
# be a genuine circular import (staged_manager.py can't finish defining its
# class until AdapterStagingBackend already exists). The lazy
# importlib.import_module + getattr(module, class_name) is what breaks that
# cycle, not a style choice.
_STAGING_BACKENDS = [
    (
        lambda sa, obj: sa.enable_lora_staging and obj.load_format == "lora_adapter",
        "sglang.srt.lora.staged_manager",
        "LoRAStagingBackend",
    ),
    (
        lambda sa, obj: sa.peft_method == "oft" and obj.load_format == "oft_adapter",
        "sglang.srt.oft.staged_manager",
        "OFTStagingBackend",
    ),
]


def get_staging_backend(tm, obj):
    """Resolve the active staging backend for this server, or ``None`` if
    no staging method is enabled."""
    import importlib

    for is_enabled, module_path, class_name in _STAGING_BACKENDS:
        if is_enabled(tm.server_args, obj):
            module = importlib.import_module(module_path)
            return getattr(module, class_name)(tm)
    return None

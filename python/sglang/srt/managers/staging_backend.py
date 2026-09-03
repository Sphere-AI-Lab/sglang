"""Staged-adapter backend dispatch, shared by every adapter method (LoRA,
OFT) that supports staged (RL) weight updates.

TokenizerControlMixin.update_adapter_from_distributed / activate_adapter_
version resolve the active backend through ``get_staging_backend`` and call
its ``reserve_stage`` / ``prepare_activation`` / ``finish_activation``
methods to orchestrate the shared pause/lock/dispatch sequence without
branching on which method is active. Each backend (``LoRAStagingBackend``,
``OFTStagingBackend``) implements that same three-method shape independently
-- duck-typed, not through a shared base class, since dispatch here never
does an isinstance/issubclass check.
"""

from __future__ import annotations

import importlib

# Explicit registry, not decorator/import-side-effect magic: this codebase's
# packages are deliberately lazy-imported (see srt/oft/__init__.py's docstring)
# to avoid boot-time cost of importing both staged managers when only one (or
# neither) is active. Adding a third adapter method means adding one entry
# here -- never touching tokenizer_control_mixin.py again.
#
# Keyed by the server_args enable-flag, not obj.load_format: native LoRA and
# OFT are already mutually-exclusive server configurations (enable_lora vs.
# enable_oft), so at most one flag below is ever true for a given
# server process -- checking obj.load_format on top of that would be
# re-deriving information the server's own config already determines. Each
# backend's reserve_stage still validates obj.load_format internally as a
# sanity check; that's the right layer for it, not this lookup.
#
# is_enabled is a lambda doing DIRECT attribute access (sa.enable_lora_staging),
# not a string-keyed getattr(sa, flag_name, False) -- these fields are
# guaranteed real ServerArgs fields, so a getattr-with-default there would be
# exactly the no-getattr-defensive anti-pattern this codebase's rules forbid.
_STAGING_BACKENDS = [
    (lambda sa: sa.enable_lora_staging, "sglang.srt.lora.staged_manager", "LoRAStagingBackend"),
    (lambda sa: sa.oft_impl == "staged", "sglang.srt.oft.staged_manager", "OFTStagingBackend"),
]


def get_staging_backend(tm, obj):
    """Resolve the active staging backend for this server, or ``None`` if
    no staging method is enabled."""
    for is_enabled, module_path, class_name in _STAGING_BACKENDS:
        if is_enabled(tm.server_args):
            module = importlib.import_module(module_path)
            return getattr(module, class_name)(tm)
    return None

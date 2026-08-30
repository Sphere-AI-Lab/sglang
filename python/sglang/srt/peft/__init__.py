"""Orbit PEFT package -- FROZEN REFERENCE since the 2026-08-29 cutover.

The default OFT serving implementation is the sibling package ``sglang.srt.oft``
(see ``--oft-impl``; this package remains fully functional as the rollback lever
via ``--oft-impl peft``). Kept intact deliberately -- do not flag as dead code;
its fate is settled by the restructure plan, not by reference sweeps.

Original package docstring follows.

Orbit PEFT package: OFT (and, later, our own LoRA) adapters plus the thin
integration seams into upstream SGLang.

Curated public API. Everything outside this package should import only from here.
The integration façade (``register_peft_args``, ``validate_peft_args``,
``maybe_init_peft_manager``, ``maybe_apply_forward``, ``maybe_prepare_peft_batch``, ...) is
added in Task 5 once ``peft/integration.py`` exists; for now this exposes the OFT
manager/registry that were relocated verbatim from ``srt/oft/``.

Imports are LAZY (PEP 562): eagerly importing ``OFTManager`` at package init pulls
the full OFT stack, whose transitive imports reach ``sglang.srt.distributed`` and
cause a circular import during engine boot. Deferring to first attribute access
keeps ``from sglang.srt.peft import OFTManager`` working without that cycle.
"""

import importlib
from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "OFTManager": "sglang.srt.peft.oft.oft_manager",
    "OFTRef": "sglang.srt.peft.oft.oft_registry",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):  # PEP 562
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(list(globals()) + __all__)


if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from sglang.srt.peft.oft.oft_manager import OFTManager
    from sglang.srt.peft.oft.oft_registry import OFTRef

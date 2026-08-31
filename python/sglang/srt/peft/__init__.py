"""Orbit PEFT package: the thin integration seams into upstream SGLang for OFT
(Orthogonal Finetuning) adapters. The OFT serving implementations themselves
live in the sibling package ``sglang.srt.oft`` (see ``--oft-impl``: 'sibling'
or 'staged'); this package owns the CLI-flag surface (``peft/config.py``) and
the façade ``model_runner.py`` calls through (``peft/integration.py``).

Curated public API. Everything outside this package should import only from here.

Imports are LAZY (PEP 562): eagerly importing ``OFTManager`` at package init pulls
the full OFT stack, whose transitive imports reach ``sglang.srt.distributed`` and
cause a circular import during engine boot. Deferring to first attribute access
keeps ``from sglang.srt.peft import OFTManager`` working without that cycle.
"""

import importlib
from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "OFTManager": "sglang.srt.oft.oft_manager",
    "OFTRef": "sglang.srt.oft.oft_registry",
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
    from sglang.srt.oft.oft_manager import OFTManager
    from sglang.srt.oft.oft_registry import OFTRef

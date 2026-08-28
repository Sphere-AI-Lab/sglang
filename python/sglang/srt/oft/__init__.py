"""Orbit OFT serving -- the sibling package of upstream ``sglang.srt.lora``.

srt/oft mirrors srt/lora file-for-file and name-for-name wherever a
counterpart exists, so upstream protocol changes map 1:1 onto this package.
Mirror map (upstream v0.5.18 -> here):

    lora_manager.py            -> oft_manager.py     (validate/prepare/update naming aligned)
    lora.py                    -> oft.py
    lora_config.py             -> oft_config.py
    lora_registry.py           -> oft_registry.py
    mem_pool.py                -> mem_pool.py
    layers.py                  -> layers.py          (oft_active ~ lora_active; slice API without tp_rank)
    lora_moe_runner_marlin.py  -> oft_moe_runner_marlin.py
    lora_moe_runners.py        -> oft_moe_runners.py (the invoke-replacement seam)
    deepseek_mla_correction.py -> deepseek_mla_correction.py
    backend/base_backend.py    -> backend/base_backend.py
    backend/{triton,torch}_backend.py -> same names
    backend/lora_registry.py   -> backend/oft_registry.py
    torch_ops/, triton_ops/    -> same layout (triton_ops has no upstream dir; OFT kernels)

Deliberate no-mirrors against v0.5.18, each with a reason -- do not "fix"
these without a design decision:

    reset_batch_state / reset_lora_batch      DP-attention idle forwards unsupported in OFT;
                                              a set_oft layer without a prepared batch should
                                              fail loudly (see oft_active docstring).
    init_prefill_cuda_graph_batch_info,       OFT prefill CUDA-graph capture runs through the
    supports/can_use/prefill_cuda_graph_max_bs  peft integration facade + prepare_oft_batch,
                                              not a dedicated prefill batch-info protocol.
    eviction_policy.py, lora_drainer.py,      single-active serving today; multi-tenant OFT
    lora_overlap_loader.py                    (registry/eviction/drainer) is the Phase B backlog.
    backend/lmhead_mixing.py                  OFT embed/lm_head handled in-layer.
    backend/{chunked,ascend}_backend.py,      upstream-specific backends/experiments with no
    marlin_lora_temp/, trtllm_lora_temp/      OFT counterpart planned.

Transitional notes: base/ (AdapterManager and friends) has no upstream
analogue and folds into this package later; OFT server flags still live in
``sglang.srt.peft.config.PEFTArgs`` until the config-surface decision.

Imports are LAZY (PEP 562), matching sglang.srt.peft's front door: eager
OFTManager import at package init pulls the full stack and hits a circular
import during engine boot.
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

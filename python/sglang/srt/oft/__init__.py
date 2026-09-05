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
    eviction_policy.py, lora_drainer.py,      B1 multi-tenancy keeps every boot adapter resident
    lora_overlap_loader.py                    (capacity-capped in init_state), so pool-overflow
                                              eviction/drainer/overlap-load stay out until B2.
    backend/lmhead_mixing.py                  OFT embed/lm_head handled in-layer.
    backend/{chunked,ascend}_backend.py,      upstream-specific backends/experiments with no
    marlin_lora_temp/, trtllm_lora_temp/      OFT counterpart planned.

OFT server flags live in ``sglang.srt.oft.config.OFTArgs``.

Every caller imports submodules directly (``from sglang.srt.oft.oft_manager
import OFTManager``, etc.) rather than through this package's own namespace,
so this file carries no re-exports.
"""

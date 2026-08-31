"""OFT correction for absorbed-MLA ``kv_b_proj``.

The absorbed-MLA path in ``DeepseekV2AttentionMLA`` bypasses
``kv_b_proj.forward()`` and folds the K/V contribution into two BMMs against the
pre-computed ``w_kc`` / ``w_vc`` weights, which are built from the RAW
``kv_b_proj.weight``. A ``ColumnParallelLinearWithOFT`` wrapper on ``kv_b_proj``
would therefore never see the activations and its input rotation ``R`` would be
silently dropped -- the OFT analogue of the LoRA gap handled in
the corresponding native LoRA correction. Megatron trains ``kv_b`` with a
materialized MLA, so it *does* apply ``R``; dropping it on the rollout side is a
K2.5(MLA)-specific train/rollout divergence.

OFT rotates the ``kv_b_proj`` INPUT -- the post-layernorm ``kv_a_normed`` latent
(``kv_lora_rank`` dim): ``[k_nope; v] = W_kvb @ (R @ x)``. In the absorbed form
both ``k_nope`` and ``v`` are recovered from the *same* cached latent ``x`` via
``w_kc`` / ``w_vc``, so rotating ``x`` once by ``R`` before it is cached/attended
is exactly equivalent to the materialized ``kv_b_proj(R @ x)``: substituting
``x -> R x`` into the absorbed ``q·W_kc·x`` (k side) and ``(attn·x)·w_vc`` (v
side) reproduces ``q·W_kc·(R x)`` and ``(attn·(R x))·w_vc``. The ``k_pe`` (rope)
half never passes through ``kv_b_proj`` and is left untouched.

Used from ``deepseek_common/attention_forward_methods/forward_mla.py``. Gate the
call with :func:`is_kv_b_oft_active` so non-OFT forwards take a single
``getattr`` and skip the helper entirely (mirrors ``is_kv_b_lora_active``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


def is_kv_b_oft_active(attn_module: DeepseekV2AttentionMLA) -> bool:
    """Cheap precondition used at the absorbed-MLA call site: True only when a
    ``kv_b_proj`` OFT adapter is wrapped and active on this attention module
    (the uncommon case). Non-OFT forwards pay a single ``getattr``."""
    return getattr(attn_module.kv_b_proj, "set_oft", False)


def apply_kv_b_rotation(
    attn_module: DeepseekV2AttentionMLA,
    kv_a_normed: torch.Tensor,
) -> torch.Tensor:
    """Rotate the compressed KV latent by the ``kv_b_proj`` OFT ``R``.

    ``kv_a_normed`` is the post-layernorm ``kv_lora_rank`` latent that the
    absorbed path caches and attends against; its leading dims are
    ``(num_tokens, 1)`` (single MQA KV head). We flatten to
    ``(num_tokens, kv_lora_rank)`` so the per-token adapter routing inside
    ``run_oft_r_sgemm`` lines up with the batch's token order, apply the SAME
    rotation the materialized ``ColumnParallelLinearWithOFT.forward`` would
    (``kv_b_proj.apply_oft`` == ``R @ x``), then restore the original shape.
    """
    shape = kv_a_normed.shape
    rotated = attn_module.kv_b_proj.apply_oft(kv_a_normed.reshape(-1, shape[-1]))
    return rotated.reshape(shape)

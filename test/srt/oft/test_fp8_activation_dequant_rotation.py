"""Guards the fp8-activation path through OFT's expert rotation.

Upstream's SGLANG_OPT_MOE_QUANT_ONCE optimization can hand invoke_fused_moe_kernel
a pre-quantized fp8 activation (``a1_q``/``a1_scale``, per-token-group-128).
OFT's rotation kernel needs bf16 tl.dot operands, so oft/oft_moe_runners.py's
``invoke()`` wrapper dequantizes with the existing single-pass block-dequant
kernel (``dequant_fp8_block_triton``, built for OFT weight-parity checks --
same per-token-group math applies to activations), rotates in bf16 as usual,
and hands back with ``A_scale=None`` so invoke_fused_moe_kernel's own quant
branch re-quantizes it.

Two failure modes with no other coverage (grep confirms no existing test
touches ``oft_moe_runners.py`` or the fp8 branch):
  1. Wrong dequant math/shape wiring (scale orientation, group indexing) would
     silently rotate garbage -- ``test_dequant_then_rotate_matches_direct_rotate``
     pins the derived property that quantize->dequant->rotate matches a direct
     bf16 rotate, within fp8's own quantization error.
  2. Wrong integration wiring in the invoke() closure (forgetting to clear
     A_scale, passing the still-quantized tensor, reverting to the old raise)
     -- ``test_moe_invoke_dequantizes_fp8_activation_before_rotating`` exercises
     the actual closure built by make_oft_invoke.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.oft.oft_moe_runners import make_oft_invoke
from sglang.srt.oft.triton_ops.block_rotate import apply_oft_rotation_triton
from sglang.srt.oft.triton_ops.parity_dequant_fp8 import dequant_fp8_block_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

M = 8
K = 256
EXPERTS = 2
TOP_K = 1
BLOCK_M = 8
GROUP = 128
# fp8 e4m3's ~3 mantissa bits give per-element relative error up to ~1/16;
# the rotation is a bounded (orthogonal-ish) linear combination so error
# doesn't blow up, but this is genuinely looser than the bf16-only kernel
# tests' TOL=2e-3.
FP8_TOL_ATOL = 0.06
FP8_TOL_RTOL = 0.1


def _random_oft_r(block_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks = K // block_size
    eye = torch.eye(block_size, device="cuda", dtype=torch.float32)
    noise = (
        torch.randn(
            EXPERTS, blocks, block_size, block_size, generator=generator, device="cuda"
        )
        * 0.02
    )
    skew = noise - noise.transpose(-1, -2)
    return (eye + skew).to(torch.bfloat16).contiguous()


def _routing(seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = (
        torch.randn(M, K, generator=generator, device="cuda", dtype=torch.bfloat16)
        * 0.3
    ).contiguous()
    topk_ids = torch.randint(
        0, EXPERTS, (M, TOP_K), generator=generator, device="cuda", dtype=torch.int32
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, EXPERTS
    )
    return hidden_states, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_dequant_then_rotate_matches_direct_rotate(block_size):
    hidden_states, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        _routing(seed=100 + block_size)
    )
    oft_r = _random_oft_r(block_size, seed=200 + block_size)

    a1_q, a1_scale = sglang_per_token_group_quant_fp8(hidden_states, GROUP)
    dequantized = dequant_fp8_block_triton(a1_q, a1_scale, out_dtype=torch.bfloat16)

    rotate_kwargs = dict(
        oft_r=oft_r,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=TOP_K,
        block_m=BLOCK_M,
    )
    rotated_from_fp8 = apply_oft_rotation_triton(dequantized, **rotate_kwargs)
    rotated_reference = apply_oft_rotation_triton(hidden_states, **rotate_kwargs)
    torch.cuda.synchronize()

    max_abs = (rotated_from_fp8.float() - rotated_reference.float()).abs().max().item()
    assert torch.allclose(
        rotated_from_fp8.float(),
        rotated_reference.float(),
        atol=FP8_TOL_ATOL,
        rtol=FP8_TOL_RTOL,
    ), f"BS={block_size} max_abs={max_abs:.3e}"


def test_moe_invoke_dequantizes_fp8_activation_before_rotating():
    hidden_states, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        _routing(seed=42)
    )
    oft_r = _random_oft_r(block_size=16, seed=43)

    a1_q, a1_scale = sglang_per_token_group_quant_fp8(hidden_states, GROUP)

    # w13_weight only needs distinct identity from w2_weight (make_oft_invoke
    # dispatches on `B is layer.w13_weight` / `B is layer.w2_weight`); the
    # down-GEMM path (w2_oft_r) avoids the split-w13 branch's extra machinery.
    w2_weight = torch.empty(EXPERTS, K, K, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        w13_weight=torch.empty(0),
        w2_weight=w2_weight,
        w2_oft_r=oft_r,
    )

    captured = {}

    def fake_real_invoke(
        A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids_,
        sorted_token_ids_, expert_ids_, num_tokens_post_padded_,
        mul_routed_weight, top_k, config, **kw,
    ):
        captured["A"] = A
        captured["A_scale"] = A_scale
        captured["top_k"] = top_k

    invoke = make_oft_invoke(layer, fake_real_invoke)

    EM = sorted_token_ids.shape[0]
    C = torch.zeros(EM, K, device="cuda", dtype=torch.bfloat16)
    topk_weights = torch.ones(M, TOP_K, device="cuda", dtype=torch.float32)

    invoke(
        a1_q,
        w2_weight,
        None,
        C,
        a1_scale,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        TOP_K,
        {"BLOCK_SIZE_M": BLOCK_M},
        use_fp8_w8a8=True,
        block_shape=[128, 128],
        compute_type=torch.bfloat16,
    )

    assert captured["A_scale"] is None, "must clear A_scale so the GEMM re-quantizes"
    assert captured["A"].dtype == torch.bfloat16
    assert captured["top_k"] == 1

    dequantized = dequant_fp8_block_triton(a1_q, a1_scale, out_dtype=torch.bfloat16)
    expected = apply_oft_rotation_triton(
        dequantized,
        oft_r,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        TOP_K,
        BLOCK_M,
    )
    torch.cuda.synchronize()
    max_abs = (captured["A"].float() - expected.float()).abs().max().item()
    assert torch.allclose(
        captured["A"].float(), expected.float(), atol=1e-3, rtol=1e-3
    ), f"max_abs={max_abs:.3e}"

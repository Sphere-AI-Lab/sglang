"""Marlin MoE runner core for expert OFT (pre-GEMM shared-R input rotation).

Mirrors the PROVEN base Marlin MoE path ``fused_marlin_moe``
(``layers/moe/fused_moe_triton/fused_marlin_moe.py``) EXACTLY -- same
``moe_align_block_size`` (with ``global_num_experts``), same dynamically-sized
workspace, same combined ``intermediate_cache13`` buffer, same 4-arg
``get_scalar_type``, same computed ``use_atomic_add``, the two
``moe_wna16_marlin_gemm`` calls with their native shapes, and the
``silu_and_mul`` / ``moe_sum_reduce`` tail. (An earlier version mirrored
``MarlinLoraRunnerCore``, which DIVERGES from ``fused_marlin_moe`` in exactly
those points and OOB'd the gate-up GEMM at ``M=1`` decode.)

The ONLY change vs ``fused_marlin_moe`` is the adapter injection: OFT rotates
each GEMM *input* by a block-diagonal orthogonal matrix ``R`` before the GEMM.

**Shared-R invariant.** The Marlin path is reached ONLY for the legacy/merged OFT
adapter (``w13_oft_r``); the canonical/split path (``w1_oft_r``/``w3_oft_r``)
raises in ``_select_gate_up_oft_r``. The legacy adapter wraps the ENTIRE grouped
FC1/down projection with a SINGLE ``OFTLinear`` -- one rotation ``R`` fanned out
IDENTICALLY across all experts (a grouped projection cannot be split per-expert).
So although ``w13_oft_r`` / ``w2_oft_r`` are stored per-expert
(``(num_local, num_blocks, bs, bs)``), every expert slot holds the SAME ``R``.

Because the rotation is a single shared ``R`` applied to the layer INPUT (outside
the expert GEMM), it composes trivially with the native marlin kernel: rotate the
UN-EXPANDED input by ``R`` (``y = x @ R``, block-diagonal, via the reusable
``apply_block_diag_orth`` OFT primitive), then run the native marlin GEMM in its
proven shapes. Buffers are read LIVE per call; an absent/None buffer (identity
boot / not-yet-loaded) falls through to the un-rotated GEMM for that stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()

if _is_cuda:
    import torch.nn.functional as F

    from sgl_kernel import moe_sum_reduce

    from sglang.kernels.ops.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        get_scalar_type,
    )


def _select_gate_up_oft_r(peft_layer):
    """Select the gate-up rotation buffer, mirroring ``make_oft_invoke``.

    Returns the merged/legacy fused ``w13_oft_r`` rotation, or ``None`` for the
    identity-boot / not-yet-loaded case (the caller then runs the gate-up GEMM
    un-rotated). Buffers are read via ``getattr(..., None)`` because they may be
    genuinely absent before the OFT manager injects them (cuda-graph capture /
    identity boot), exactly as ``make_oft_invoke`` does.

    The split ``w1_oft_r``/``w3_oft_r`` path is unsupported on Marlin: it is a
    genuinely per-expert rotation (each expert its own R), which cannot ride the
    shared-R input-rotation fast path, and the triton core restricts the split
    path to BF16 (``_assert_unquantized``). Fail loud rather than silently drop a
    rotation.
    """
    w13 = getattr(peft_layer, "w13_oft_r", None)
    w1 = getattr(peft_layer, "w1_oft_r", None)
    if w13 is not None and w1 is not None:
        # Invariant (enforced by the OFT manager via oft_type): a layer carries
        # EITHER the legacy fused w13 rotation OR the split w1/w3 rotation.
        raise RuntimeError(
            "Split expert gate/up OFT (w1/w3_oft_r) cannot be active together "
            "with legacy w13_oft_r on the same MoE layer"
        )
    if w1 is not None:
        raise RuntimeError(
            "Split expert gate/up OFT is per-expert / BF16-only; the Marlin "
            "(quantized) MoE path supports only the legacy merged shared-R "
            "w13_oft_r rotation. Per-expert (canonical/split) OFT must use the "
            "triton MoE runner."
        )
    return w13


def _shared_r(oft_r: torch.Tensor, name: str) -> torch.Tensor:
    """Return the single shared rotation ``R`` (``(num_blocks, bs, bs)``) from a
    per-expert OFT buffer ``(num_local, num_blocks, bs, bs)``.

    The legacy/merged adapter fans ONE R across every expert, so all slots are
    identical and slot 0 is representative. Verify that invariant so a genuinely
    per-expert adapter reaching this path fails LOUD instead of silently using
    expert 0's R for every token. The value check syncs (returns a Python bool),
    so it is skipped under CUDA-graph capture -- it runs during eager warmup /
    non-graph batches, which is enough to catch a mis-loaded adapter.
    """
    if oft_r.shape[0] > 1 and not (
        oft_r.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        if not (
            torch.equal(oft_r[0], oft_r[1]) and torch.equal(oft_r[0], oft_r[-1])
        ):
            raise RuntimeError(
                f"{name} differs across experts, but the Marlin OFT path only "
                f"supports the legacy merged SHARED-R adapter (one R fanned out "
                f"identically across experts). A per-expert (canonical/split) "
                f"adapter must use the triton MoE runner, not Marlin."
            )
    return oft_r[0]


class MarlinOFTRunnerCore:
    """Marlin MoE runner using pre-GEMM shared-R input rotation instead of hooks."""

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    def run_from_dispatch(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: MarlinMoeQuantInfo,
        runner_config: MoeRunnerConfig,
        peft_layer,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.oft.torch_ops.oft_ops import apply_block_diag_orth

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        assert runner_config.activation == "silu", "Only SiLU activation is supported."
        inplace = runner_config.inplace
        routed_scaling_factor = runner_config.routed_scaling_factor
        is_gated = runner_config.is_gated

        # ---- scalar setup (mirror fused_marlin_moe) ----
        num_bits = quant_info.weight_bits
        w1_zeros = quant_info.w13_qzeros
        w2_zeros = quant_info.w2_qzeros
        w1_scale = quant_info.w13_scales
        w2_scale = quant_info.w2_scales
        w1_global_scale = quant_info.w13_global_scale
        w2_global_scale = quant_info.w2_global_scale

        M, K = hidden_states.shape
        E = quant_info.w13_qweight.shape[0]
        N = quant_info.w2_qweight.shape[1] * 16
        topk = topk_ids.shape[1]
        gemm1_n = 2 * N if is_gated else N

        is_mxfp4_marlin = (
            num_bits == 4
            and w1_zeros is None
            and w2_zeros is None
            and w1_scale.dtype == torch.float8_e8m0fnu
            and w2_scale.dtype == torch.float8_e8m0fnu
        )

        # M block size selection logic (verbatim from fused_marlin_moe)
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break

        # EP correctness: get_marlin_quant_info leaves expert_map / global_num_experts
        # UNSET (unlike CompressedTensorsWNA16MoE.apply_weights, which computes them
        # inline from the FusedMoE dispatcher). Without them, moe_align_block_size
        # buckets GLOBAL topk_ids (0..num_experts-1) into a LOCAL-sized expert array
        # -> OOB expert index -> illegal memory access at EP>1. Recover them from the
        # layer's dispatcher exactly as apply_weights does.
        expert_map = quant_info.expert_map
        global_num_experts = quant_info.global_num_experts
        if expert_map is None and global_num_experts == -1:
            _disp = getattr(peft_layer, "dispatcher", None)
            _lem = getattr(_disp, "local_expert_mapping", None)
            if _lem is not None:
                expert_map = _lem
                global_num_experts = runner_config.num_experts
        if global_num_experts == -1:
            global_num_experts = E
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts
        )

        # Workspace: match the PROVEN production call site
        # (moe_runner/marlin.py::fused_experts_none_to_marlin), which passes
        # marlin_make_workspace(max_blocks_per_sm=4) == torch.zeros(sms * 4). The
        # kernel launches sms*4 threadblocks and indexes the workspace by
        # threadblock id, so it needs the FULL sms*4 slots. fused_marlin_moe's
        # own `workspace is None` branch caps a dynamic estimate at sms*4 but can
        # fall BELOW it at small M (decode), under-sizing the workspace -> the
        # kernel writes past the end (illegal memory access). Every real caller
        # overrides that default with marlin_make_workspace; so do we.
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        device = hidden_states.device
        workspace = marlin_make_workspace(device, max_blocks_per_sm=4)

        scalar_type1 = get_scalar_type(
            num_bits, w1_zeros is not None, w1_scale, w1_global_scale
        )
        scalar_type2 = get_scalar_type(
            num_bits, w2_zeros is not None, w2_scale, w2_global_scale
        )

        # Combined intermediate buffer (verbatim from fused_marlin_moe): cache1
        # and cache3 alias one cache13 allocation; cache1 is consumed by the
        # activation before cache3 is written, so the aliasing is safe.
        intermediate_cache2 = torch.empty(
            (M * topk, N), device=device, dtype=hidden_states.dtype
        )
        intermediate_cache13 = torch.empty(
            (M * topk * max(gemm1_n, K),), device=device, dtype=hidden_states.dtype
        )
        intermediate_cache1 = intermediate_cache13[: M * topk * gemm1_n].view(
            -1, gemm1_n
        )
        intermediate_cache3 = intermediate_cache13[: M * topk * K].view(-1, K)

        use_atomic_add = (
            hidden_states.dtype == torch.half
            or torch.cuda.get_device_capability(device)[0] >= 9
        ) and (not is_mxfp4_marlin)

        # ---- Gate/Up (Marlin) with shared-R input rotation ----
        # Rotate the UN-EXPANDED hidden_states by the layer's single shared gate-up
        # R, then run the native fused_marlin_moe gate-up GEMM on the rotated input.
        gate_up_oft_r = _select_gate_up_oft_r(peft_layer)
        if gate_up_oft_r is not None:
            gate_up_input = apply_block_diag_orth(
                hidden_states, _shared_r(gate_up_oft_r, "w13_oft_r")
            )
        else:
            gate_up_input = hidden_states

        intermediate_cache1 = moe_wna16_marlin_gemm(
            gate_up_input,
            intermediate_cache1,
            quant_info.w13_qweight,
            quant_info.w13_bias,
            w1_scale,
            w1_global_scale,
            w1_zeros,
            quant_info.w13_g_idx,
            quant_info.w13_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=expert_map is not None,
            b_q_type=scalar_type1,
            size_m=M,
            size_n=gemm1_n,
            size_k=K,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        # ---- Activation ----
        if is_gated:
            silu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)
        else:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))

        if expert_map is not None:
            intermediate_cache3.zero_()

        # ---- Down (Marlin) with shared-R input rotation ----
        w2_oft_r = getattr(peft_layer, "w2_oft_r", None)
        if w2_oft_r is not None:
            down_input = apply_block_diag_orth(
                intermediate_cache2, _shared_r(w2_oft_r, "w2_oft_r")
            )
        else:
            down_input = intermediate_cache2

        intermediate_cache3 = moe_wna16_marlin_gemm(
            down_input,
            intermediate_cache3,
            quant_info.w2_qweight,
            quant_info.w2_bias,
            w2_scale,
            w2_global_scale,
            w2_zeros,
            quant_info.w2_g_idx,
            quant_info.w2_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=expert_map is not None,
            b_q_type=scalar_type2,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        ).view(-1, topk, K)

        # ---- Reduction (verbatim from fused_marlin_moe) ----
        output = hidden_states if inplace else torch.empty_like(hidden_states)
        if is_mxfp4_marlin:
            torch.sum(intermediate_cache3, dim=1, out=output)
        else:
            if routed_scaling_factor is None:
                routed_scaling_factor = 1.0
            moe_sum_reduce(intermediate_cache3, output, routed_scaling_factor)

        return StandardCombineInput(hidden_states=output)

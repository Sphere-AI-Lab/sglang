from typing import Optional

import torch

from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import moe_sum_reduce, silu_and_mul

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        apply_oft_rotation_triton,
    )


def get_scalar_type(num_bits: int, has_zp: bool):
    from sgl_kernel.scalar_type import scalar_types

    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def _fused_marlin_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    # Prepack-split (canonical-OFT path): half-output Marlin weights.
    # When both `w1_gate` and `w1_up` are non-None and OFT split is active,
    # the canonical-split branch uses two half-N GEMMs instead of two full-2N
    # GEMMs with discarded halves. See plan 2026-05-21-marlin-prepack-split-fc1.md.
    w1_gate: Optional[torch.Tensor] = None,
    w1_gate_scale: Optional[torch.Tensor] = None,
    w1_up: Optional[torch.Tensor] = None,
    w1_up_scale: Optional[torch.Tensor] = None,
    w1_oft_r: Optional[torch.Tensor] = None,
    w2_oft_r: Optional[torch.Tensor] = None,
    w3_oft_r: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    # Halves are all-or-nothing. Mixed is a caller bug -- fail loudly.
    _halves = (w1_gate, w1_up, w1_gate_scale, w1_up_scale)
    _have_any = any(t is not None for t in _halves)
    _have_all = all(t is not None for t in _halves)
    assert _have_any == _have_all, (
        "fused_marlin_moe: w1_gate/w1_up/w1_gate_scale/w1_up_scale must all "
        "be None or all be non-None. Got: "
        f"w1_gate={'set' if w1_gate is not None else 'None'}, "
        f"w1_up={'set' if w1_up is not None else 'None'}, "
        f"w1_gate_scale={'set' if w1_gate_scale is not None else 'None'}, "
        f"w1_up_scale={'set' if w1_up_scale is not None else 'None'}."
    )
    if _have_all:
        assert w1 is None and w1_scale is None, (
            "fused_marlin_moe: when split halves are supplied, w1 and "
            "w1_scale must be None (the kernel branch must not read them)."
        )
    else:
        assert w1 is not None and w1_scale is not None, (
            "fused_marlin_moe: when split halves are absent, w1 and "
            "w1_scale must be non-None."
        )

    halves_present = _have_all
    # Reference tensor for w1-shape / dtype invariants. When halves are
    # present, the gate half has shape (E, K/16, 2*N_half) int32 — same
    # K-axis size as the fused 2N form, so the K invariant still holds.
    w1_ref = w1_gate if halves_present else w1
    w1_scale_ref = w1_gate_scale if halves_present else w1_scale

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1_ref.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1_ref.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert (
        hidden_states.dtype == w1_scale_ref.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale_ref.dtype})"
    assert (
        hidden_states.dtype == w2_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1_ref.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )
    has_oft = (
        w1_oft_r is not None or w2_oft_r is not None or w3_oft_r is not None
    )
    if has_oft:
        flat_topk_ids = topk_ids.reshape(-1, 1).contiguous()
        flat_topk_weights = topk_weights.reshape(-1, 1).contiguous()
        (
            flat_sorted_token_ids,
            flat_expert_ids,
            flat_num_tokens_post_padded,
        ) = moe_align_block_size(flat_topk_ids, block_size_m, global_num_experts)
        oft_block_size_m = 1 << (block_size_m - 1).bit_length()
        if oft_block_size_m == block_size_m:
            oft_sorted_token_ids = sorted_token_ids
            oft_expert_ids = expert_ids
            oft_num_tokens_post_padded = num_tokens_post_padded
            flat_oft_sorted_token_ids = flat_sorted_token_ids
            flat_oft_expert_ids = flat_expert_ids
            flat_oft_num_tokens_post_padded = flat_num_tokens_post_padded
        else:
            (
                oft_sorted_token_ids,
                oft_expert_ids,
                oft_num_tokens_post_padded,
            ) = moe_align_block_size(topk_ids, oft_block_size_m, global_num_experts)
            (
                flat_oft_sorted_token_ids,
                flat_oft_expert_ids,
                flat_oft_num_tokens_post_padded,
            ) = moe_align_block_size(
                flat_topk_ids, oft_block_size_m, global_num_experts
            )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            (
                max(sorted_token_ids.size(0), flat_sorted_token_ids.size(0))
                if has_oft
                else sorted_token_ids.size(0)
            )
            // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )

    if has_oft:
        is_canonical_split = w1_oft_r is not None and w3_oft_r is not None
        if is_canonical_split:
            # Canonical OFT split: rotate hidden_states twice (gate / up).
            # If prepack-split halves (w1_gate / w1_up) are present, run two
            # half-N Marlin GEMMs (zero discarded compute). Otherwise fall back
            # to v1 behavior: two full-2N GEMMs with the SAME fused w1 weight
            # against the two rotated inputs, then keep the gate half of the
            # first result and the up half of the second.
            gate_input = apply_oft_rotation_triton(
                hidden_states,
                w1_oft_r,
                topk_ids,
                oft_sorted_token_ids,
                oft_expert_ids,
                oft_num_tokens_post_padded,
                top_k=topk,
                block_m=oft_block_size_m,
            )
            up_input = apply_oft_rotation_triton(
                hidden_states,
                w3_oft_r,
                topk_ids,
                oft_sorted_token_ids,
                oft_expert_ids,
                oft_num_tokens_post_padded,
                top_k=topk,
                block_m=oft_block_size_m,
            )

            has_prepack_split = (
                w1_gate is not None
                and w1_up is not None
                and w1_gate_scale is not None
                and w1_up_scale is not None
            )
            if has_prepack_split:
                # Prepack-split: two half-N Marlin GEMMs, zero discarded compute.
                gate_out = torch.empty(
                    (M * topk, N),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                gate_out = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
                    gate_input,
                    gate_out,
                    w1_gate,
                    None,  # b_bias_or_none
                    w1_gate_scale,
                    None,  # global_scale_or_none
                    w1_zeros,
                    g_idx1,
                    sort_indices1,
                    workspace,
                    flat_sorted_token_ids,
                    flat_expert_ids,
                    flat_num_tokens_post_padded,
                    flat_topk_weights,
                    moe_block_size=block_size_m,
                    top_k=1,
                    mul_topk_weights=False,
                    is_ep=expert_map is not None,
                    b_q_type_id=scalar_type1.id,
                    size_m=M * topk,
                    size_n=N,
                    size_k=K,
                    is_k_full=is_k_full,
                    use_atomic_add=use_atomic_add,
                    use_fp32_reduce=True,
                    is_zp_float=False,
                )
                up_out = torch.empty(
                    (M * topk, N),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                up_out = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
                    up_input,
                    up_out,
                    w1_up,
                    None,  # b_bias_or_none
                    w1_up_scale,
                    None,  # global_scale_or_none
                    w1_zeros,
                    g_idx1,
                    sort_indices1,
                    workspace,
                    flat_sorted_token_ids,
                    flat_expert_ids,
                    flat_num_tokens_post_padded,
                    flat_topk_weights,
                    moe_block_size=block_size_m,
                    top_k=1,
                    mul_topk_weights=False,
                    is_ep=expert_map is not None,
                    b_q_type_id=scalar_type1.id,
                    size_m=M * topk,
                    size_n=N,
                    size_k=K,
                    is_k_full=is_k_full,
                    use_atomic_add=use_atomic_add,
                    use_fp32_reduce=True,
                    is_zp_float=False,
                )
                intermediate_cache1 = torch.cat([gate_out, up_out], dim=-1)
            else:
                gate_full = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
                    gate_input,
                    torch.empty_like(intermediate_cache1),
                    w1,
                    None,  # b_bias_or_none
                    w1_scale,
                    None,  # global_scale_or_none
                    w1_zeros,
                    g_idx1,
                    sort_indices1,
                    workspace,
                    flat_sorted_token_ids,
                    flat_expert_ids,
                    flat_num_tokens_post_padded,
                    flat_topk_weights,
                    moe_block_size=block_size_m,
                    top_k=1,
                    mul_topk_weights=False,
                    is_ep=expert_map is not None,
                    b_q_type_id=scalar_type1.id,
                    size_m=M * topk,
                    size_n=2 * N,
                    size_k=K,
                    is_k_full=is_k_full,
                    use_atomic_add=use_atomic_add,
                    use_fp32_reduce=True,
                    is_zp_float=False,
                )
                up_full = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
                    up_input,
                    torch.empty_like(intermediate_cache1),
                    w1,
                    None,  # b_bias_or_none
                    w1_scale,
                    None,  # global_scale_or_none
                    w1_zeros,
                    g_idx1,
                    sort_indices1,
                    workspace,
                    flat_sorted_token_ids,
                    flat_expert_ids,
                    flat_num_tokens_post_padded,
                    flat_topk_weights,
                    moe_block_size=block_size_m,
                    top_k=1,
                    mul_topk_weights=False,
                    is_ep=expert_map is not None,
                    b_q_type_id=scalar_type1.id,
                    size_m=M * topk,
                    size_n=2 * N,
                    size_k=K,
                    is_k_full=is_k_full,
                    use_atomic_add=use_atomic_add,
                    use_fp32_reduce=True,
                    is_zp_float=False,
                )
                # gate_full[:, :N] -- correctly-rotated gate half;
                # up_full[:, N:]  -- correctly-rotated up half.
                intermediate_cache1 = torch.cat(
                    [gate_full[:, :N], up_full[:, N:]], dim=-1
                )
        else:
            if w1_oft_r is not None:
                first_gemm_input = apply_oft_rotation_triton(
                    hidden_states,
                    w1_oft_r,
                    topk_ids,
                    oft_sorted_token_ids,
                    oft_expert_ids,
                    oft_num_tokens_post_padded,
                    top_k=topk,
                    block_m=oft_block_size_m,
                )
            else:
                first_gemm_input = hidden_states.repeat_interleave(
                    topk, dim=0
                ).contiguous()

            intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
                first_gemm_input,
                intermediate_cache1,
                w1,
                None,  # b_bias_or_none
                w1_scale,
                None,  # global_scale_or_none
                w1_zeros,
                g_idx1,
                sort_indices1,
                workspace,
                flat_sorted_token_ids,
                flat_expert_ids,
                flat_num_tokens_post_padded,
                flat_topk_weights,
                moe_block_size=block_size_m,
                top_k=1,
                mul_topk_weights=False,
                is_ep=expert_map is not None,
                b_q_type_id=scalar_type1.id,
                size_m=M * topk,
                size_n=2 * N,
                size_k=K,
                is_k_full=is_k_full,
                use_atomic_add=use_atomic_add,
                use_fp32_reduce=True,
                is_zp_float=False,
            )
    else:
        intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
            hidden_states,
            intermediate_cache1,
            w1,
            None,  # b_bias_or_none
            w1_scale,
            None,  # global_scale_or_none
            w1_zeros,
            g_idx1,
            sort_indices1,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=expert_map is not None,
            b_q_type_id=scalar_type1.id,
            size_m=M,
            size_n=2 * N,
            size_k=K,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    second_gemm_input = intermediate_cache2
    second_sorted_token_ids = sorted_token_ids
    second_expert_ids = expert_ids
    second_num_tokens_post_padded = num_tokens_post_padded
    second_topk_weights = topk_weights
    if has_oft:
        second_sorted_token_ids = flat_sorted_token_ids
        second_expert_ids = flat_expert_ids
        second_num_tokens_post_padded = flat_num_tokens_post_padded
        second_topk_weights = flat_topk_weights
        if w2_oft_r is not None:
            second_gemm_input = apply_oft_rotation_triton(
                intermediate_cache2,
                w2_oft_r,
                flat_topk_ids,
                flat_oft_sorted_token_ids,
                flat_oft_expert_ids,
                flat_oft_num_tokens_post_padded,
                top_k=1,
                block_m=oft_block_size_m,
            )

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        second_gemm_input,
        intermediate_cache3,
        w2,
        None,  # b_bias_or_none
        w2_scale,
        None,  # global_scale_or_none
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        second_sorted_token_ids,
        second_expert_ids,
        second_num_tokens_post_padded,
        second_topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type2.id,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    moe_sum_reduce(
        intermediate_cache3,
        output,
        routed_scaling_factor,
    )
    return output


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    w1_gate: Optional[torch.Tensor] = None,
    w1_gate_scale: Optional[torch.Tensor] = None,
    w1_up: Optional[torch.Tensor] = None,
    w1_up_scale: Optional[torch.Tensor] = None,
    w1_oft_r: Optional[torch.Tensor] = None,
    w2_oft_r: Optional[torch.Tensor] = None,
    w3_oft_r: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    return _fused_marlin_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        gating_output=gating_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        w1_gate=w1_gate,
        w1_gate_scale=w1_gate_scale,
        w1_up=w1_up,
        w1_up_scale=w1_up_scale,
        w1_oft_r=w1_oft_r,
        w2_oft_r=w2_oft_r,
        w3_oft_r=w3_oft_r,
        num_bits=num_bits,
        is_k_full=is_k_full,
        inplace=inplace,
        routed_scaling_factor=routed_scaling_factor,
    )

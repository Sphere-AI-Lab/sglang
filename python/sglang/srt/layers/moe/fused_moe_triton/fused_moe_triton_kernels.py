from __future__ import annotations

import functools
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.int8_kernel import (
    per_token_group_quant_int8,
    per_token_quant_int8,
    sglang_per_token_group_quant_int8,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_sm90_supported,
)

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    _support_tensor_descriptor = True
except:
    _support_tensor_descriptor = False

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    pass
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    pass

padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


def support_tensor_descriptor():
    return _support_tensor_descriptor


# swap_ab benefits SM90 GPUs (H20, H100, H200, etc.) for certain block shapes.
@functools.lru_cache(maxsize=8)
def should_enable_swap_ab(
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> bool:
    if not _is_cuda or is_batch_invariant_mode_enabled():
        return False

    return is_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
    filter_expert: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if filter_expert and off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not even_Ks:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if c_sorted:
        c_ptrs = (
            c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
        )
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# -----------------------------------------------------------------------------
# TMA allocator: set once per process (avoid per-call triton.set_allocator)
# -----------------------------------------------------------------------------
_TMA_ALLOCATOR_SET = False


def _set_triton_tma_allocator():
    """TMA descriptors require a global allocator; set it once to avoid per-call overhead."""
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        # NOTE: keep this allocation on CUDA device
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _TMA_ALLOCATOR_SET = True


# --- B TensorDescriptor cache (LRU) ---
_B_DESC_CACHE_MAX = 64
_B_DESC_CACHE: "OrderedDict[tuple, TensorDescriptor]" = OrderedDict()


def _get_b_tma_desc_cached(B: torch.Tensor, block_n: int, block_k: int):
    """
    Cache TensorDescriptor for constant weight B.
    Keyed by storage ptr + shape/stride/dtype + tile shape.
    """
    key = (
        int(B.data_ptr()),
        tuple(B.shape),
        tuple(B.stride()),
        str(B.dtype),
        int(block_n),
        int(block_k),
    )

    desc = _B_DESC_CACHE.get(key, None)
    if desc is not None:
        _B_DESC_CACHE.move_to_end(key)
        return desc

    # Create outside lock to reduce lock hold time (ok if duplicated rarely)
    desc = TensorDescriptor(
        B,
        B.shape,
        B.stride(),
        [1, block_n, block_k],
    )

    _B_DESC_CACHE[key] = desc
    _B_DESC_CACHE.move_to_end(key)
    if len(_B_DESC_CACHE) > _B_DESC_CACHE_MAX:
        _B_DESC_CACHE.popitem(last=False)

    return desc


# ---------------------------------------------------------------------------
# Standalone Triton kernel: block-diagonal OFT rotation
# ---------------------------------------------------------------------------


@triton.jit
def _oft_block_rotate_kernel(
    # Input: (M, K)
    A_ptr,
    stride_am,
    stride_ak,
    # Output: (M_expanded, K)  — one row per token-expert pair
    A_rot_ptr,
    stride_arm,
    stride_ark,
    # R matrices: (E, num_blocks, bs, bs)
    R_ptr,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    # Token routing
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    # Dimensions
    top_k: tl.constexpr,
    K: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Apply per-expert block-diagonal OFT rotation to input features.

    Grid: (cdiv(EM, BLOCK_M), num_blocks)
    - pid_m: block of sorted tokens (all same expert via moe_align_block_size)
    - pid_blk: which OFT block (0 .. K // OFT_BLOCK_SIZE - 1)

    For each program: loads BLOCK_M tokens' input at one OFT block position,
    loads the expert's R matrix block, computes rotation via tl.dot, and
    writes the rotated result to A_rot.
    """
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)

    # Bounds check
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    # Load sorted token ids and expert for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(sorted_token_ids_ptr + offs_m)
    sorted_ids = sorted_ids.to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Original token ids (A is indexed by token, not token*top_k)
    orig_ids = sorted_ids // top_k

    # Base K offset for this OFT block
    k_base = pid_blk * OFT_BLOCK_SIZE

    # EP-dispatched non-local experts are encoded as -1. The base MoE GEMM
    # filters those blocks, but the OFT pre-rotation runs before that filter,
    # so skip here to avoid reading outside the R tensor.
    if expert < 0:
        return

    # Tiled rotation: accumulate (BLOCK_M, OFT_BLOCK_SIZE) result
    # by iterating over TILE_K chunks of the inner k dimension.
    #
    # OFT convention: rot_accum[t, c] = sum_k A[t, k_base+k] * R[k, c]
    # i.e. A_rot[:, k_base:k_base+bs] = A[:, k_base:k_base+bs] @ R[expert, pid_blk]
    # (NOT R^T — the dense OFT kernel `sgemm_oft_r.py` and `apply_block_diag_orth`
    # both apply x @ R; matching that here keeps train/inference parity).

    rot_accum = tl.zeros((BLOCK_M, OFT_BLOCK_SIZE), dtype=tl.float32)

    for k_off in range(0, OFT_BLOCK_SIZE, TILE_K):
        # Load A tile: (BLOCK_M, TILE_K) from A[orig_ids, k_base + k_off : ...]
        k_tile_offs = (k_base + k_off + tl.arange(0, TILE_K)).to(tl.int64)
        a_ptrs = A_ptr + orig_ids[:, None] * stride_am + k_tile_offs[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

        # Load R sub-block: R[expert, pid_blk, k_off:k_off+TILE_K, :]
        # Shape: (TILE_K, OFT_BLOCK_SIZE) — rows = k inner axis, cols = c output axis.
        r_row_offs = (k_off + tl.arange(0, TILE_K)).to(tl.int64)
        r_col_offs = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
        r_ptrs = (
            R_ptr
            + expert * stride_re
            + pid_blk * stride_rb
            + r_row_offs[:, None] * stride_ri
            + r_col_offs[None, :] * stride_rj
        )
        r_sub = tl.load(r_ptrs)  # (TILE_K, OFT_BLOCK_SIZE)

        # (BLOCK_M, TILE_K) @ (TILE_K, OFT_BLOCK_SIZE)  →  x @ R per block
        # input_precision="ieee" is a no-op for bf16 operands (Triton 3.5.1
        # only honors it for fp32×fp32) but kept defensive: if R is ever
        # promoted to fp32, this enforces ieee not tf32, matching the Bridge
        # train-side `sgemm_oft_r_single.py:71` annotation.
        rot_accum += tl.dot(a_tile, r_sub, input_precision="ieee")

    # Store rotated output: A_rot[sorted_ids, k_base : k_base + bs]
    out_k_offs = (k_base + tl.arange(0, OFT_BLOCK_SIZE)).to(tl.int64)
    out_ptrs = A_rot_ptr + sorted_ids[:, None] * stride_arm + out_k_offs[None, :] * stride_ark
    tl.store(out_ptrs, rot_accum.to(A_rot_ptr.dtype.element_ty), mask=token_mask[:, None])


def apply_oft_rotation_triton(
    A: torch.Tensor,           # (M, K)
    oft_r: torch.Tensor,       # (E, num_blocks, bs, bs)
    topk_ids: torch.Tensor,    # (M, top_k)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int = 64,
) -> torch.Tensor:
    """Apply per-expert block-diagonal OFT rotation using a Triton kernel.

    Returns A_rot of shape (M * top_k, K) where each row is rotated
    by its assigned expert's block-diagonal R matrix.
    """
    M, K = A.shape
    bs = oft_r.shape[2]
    num_blocks = K // bs

    # Output buffer: one row per token-expert pair
    A_rot = torch.empty(M * top_k, K, device=A.device, dtype=A.dtype)

    # TILE_K: chunk size for the inner dimension of the rotation matmul
    tile_k = min(64, bs)

    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, block_m), num_blocks)

    _oft_block_rotate_kernel[grid](
        A, A.stride(0), A.stride(1),
        A_rot, A_rot.stride(0), A_rot.stride(1),
        oft_r, oft_r.stride(0), oft_r.stride(1), oft_r.stride(2), oft_r.stride(3),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_ids.numel(),
        top_k=top_k,
        K=K,
        OFT_BLOCK_SIZE=bs,
        BLOCK_M=block_m,
        TILE_K=tile_k,
    )

    return A_rot


# ---------------------------------------------------------------------------
# Standalone Triton kernels: LoRA delta  (two-pass: A-proj then B-proj+add)
# ---------------------------------------------------------------------------


@triton.jit
def _lora_a_proj_kernel(
    # Input A: (M, K)
    A_ptr, stride_am, stride_ak,
    # Intermediate H: (M*top_k, num_sub, rank)
    H_ptr, stride_hm, stride_hs, stride_hr,
    # LoRA A weights: (E, [num_sub,] rank, K)
    LA_ptr, stride_la_e, stride_la_s, stride_la_r, stride_la_k,
    # Routing
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    num_valid_tokens,
    top_k: tl.constexpr,
    K: tl.constexpr,
    LORA_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Pass 1: project input down to LoRA rank per expert.

    Grid: (cdiv(EM, BLOCK_M), num_sub_projections)
    Computes H[sorted_ids, sub_proj, :] = A[sorted_ids // top_k] @ lora_a[expert, sub_proj]^T
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)  # sub-projection (0=gate, 1=up for w13; 0 for w2)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert < 0:
        return
    orig_ids = sorted_ids // top_k

    # Tile over K: h = A @ lora_a^T  ->  (BLOCK_M, LORA_RANK)
    h = tl.zeros((BLOCK_M, LORA_RANK), dtype=tl.float32)
    r_offs = tl.arange(0, LORA_RANK).to(tl.int64)

    for k_off in range(0, K, TILE_K):
        k_offs = (k_off + tl.arange(0, TILE_K)).to(tl.int64)

        # Load A tile: (BLOCK_M, TILE_K)
        a_ptrs = A_ptr + orig_ids[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=token_mask[:, None] & (k_offs[None, :] < K), other=0.0)

        # Load lora_a tile: (TILE_K, LORA_RANK)
        # lora_a layout: [expert, sub_proj, r, k]
        la_ptrs = (
            LA_ptr
            + expert * stride_la_e
            + pid_s * stride_la_s
            + r_offs[None, :] * stride_la_r
            + k_offs[:, None] * stride_la_k
        )
        la_tile = tl.load(la_ptrs, mask=k_offs[:, None] < K, other=0.0)  # (TILE_K, LORA_RANK)

        # (BLOCK_M, TILE_K) @ (TILE_K, LORA_RANK)
        h += tl.dot(a_tile, la_tile)

    # Store H[sorted_ids, sub_proj, :]
    h_ptrs = H_ptr + sorted_ids[:, None] * stride_hm + pid_s * stride_hs + r_offs[None, :] * stride_hr
    tl.store(h_ptrs, h.to(H_ptr.dtype.element_ty), mask=token_mask[:, None])


@triton.jit
def _lora_b_proj_add_kernel(
    # Intermediate H: (M*top_k, num_sub, rank)
    H_ptr, stride_hm, stride_hs, stride_hr,
    # LoRA B weights: (E, [num_sub,] N_per_sub, rank)
    LB_ptr, stride_lb_e, stride_lb_s, stride_lb_n, stride_lb_r,
    # Output C to add delta to
    C_ptr, stride_cm, stride_cn,
    lora_scaling,
    topk_weights_ptr,
    # Routing
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    num_valid_tokens,
    N,
    lora_inter_per_tp,
    LORA_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """Pass 2: project H up to output dim and add scaled delta to C.

    Grid: (cdiv(EM, BLOCK_M), cdiv(N, BLOCK_N))
    Computes C[sorted_ids, n_block] += scaling * H[sorted_ids, sub_proj] @ lora_b[expert, sub_proj, n_block]^T
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert < 0:
        return

    # Determine sub-projection from N position
    first_n = pid_n * BLOCK_N
    sub_proj = tl.where(first_n < lora_inter_per_tp, 0, 1).to(tl.int64)
    local_n = first_n - sub_proj * lora_inter_per_tp

    # Load H[sorted_ids, sub_proj, :rank]  ->  (BLOCK_M, LORA_RANK)
    r_offs = tl.arange(0, LORA_RANK).to(tl.int64)
    h_ptrs = H_ptr + sorted_ids[:, None] * stride_hm + sub_proj * stride_hs + r_offs[None, :] * stride_hr
    h = tl.load(h_ptrs, mask=token_mask[:, None], other=0.0)

    # Load lora_b tile: (BLOCK_N, LORA_RANK)
    n_offs = (local_n + tl.arange(0, BLOCK_N)).to(tl.int64)
    lb_ptrs = (
        LB_ptr
        + expert * stride_lb_e
        + sub_proj * stride_lb_s
        + n_offs[:, None] * stride_lb_n
        + r_offs[None, :] * stride_lb_r
    )
    n_valid = n_offs < lora_inter_per_tp
    lb_tile = tl.load(lb_ptrs, mask=n_valid[:, None], other=0.0)  # (BLOCK_N, LORA_RANK)

    # Delta: (BLOCK_M, LORA_RANK) @ (LORA_RANK, BLOCK_N)
    delta = tl.dot(h, tl.trans(lb_tile))
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + sorted_ids, mask=token_mask, other=0)
        delta *= moe_weight[:, None]

    # Add scaled delta to C
    global_n_offs = (first_n + tl.arange(0, BLOCK_N)).to(tl.int64)
    c_ptrs = C_ptr + sorted_ids[:, None] * stride_cm + global_n_offs[None, :] * stride_cn
    n_mask = global_n_offs[None, :] < N
    c = tl.load(c_ptrs, mask=token_mask[:, None] & n_mask, other=0.0).to(tl.float32)
    c += lora_scaling * delta
    tl.store(c_ptrs, c.to(C_ptr.dtype.element_ty), mask=token_mask[:, None] & n_mask)


def apply_lora_delta_triton(
    A: torch.Tensor,           # (M, K)
    C: torch.Tensor,           # (M, top_k, N) — output to add delta to
    lora_a: torch.Tensor,      # (E, 2, rank, K) or (E, rank, K)
    lora_b: torch.Tensor,      # (E, 2, N_sub, rank) or (E, N, rank)
    lora_scaling: float,
    lora_inter_per_tp: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    topk_weights: Optional[torch.Tensor] = None,
    mul_routed_weight: bool = False,
    block_m: int = 64,
) -> None:
    """Apply per-expert LoRA delta to output C using two Triton kernel passes."""
    if mul_routed_weight and topk_weights is None:
        raise ValueError("topk_weights is required when mul_routed_weight is True")

    M, K = A.shape
    N = C.shape[-1]
    rank = lora_a.shape[-2]
    num_sub = 2 if lora_a.dim() == 4 else 1
    num_valid = M * top_k

    # Intermediate buffer: (M*top_k, num_sub, rank)
    H = torch.empty(num_valid, num_sub, rank, device=A.device, dtype=A.dtype)

    EM = sorted_token_ids.shape[0]
    tile_k = min(64, K)

    # Stride helpers for lora_a / lora_b (handle 3D vs 4D)
    la_stride_e = lora_a.stride(0)
    la_stride_s = lora_a.stride(1) if lora_a.dim() == 4 else 0
    la_stride_r = lora_a.stride(-2)
    la_stride_k = lora_a.stride(-1)
    lb_stride_e = lora_b.stride(0)
    lb_stride_s = lora_b.stride(1) if lora_b.dim() == 4 else 0
    lb_stride_n = lora_b.stride(-2)
    lb_stride_r = lora_b.stride(-1)

    # Pass 1: A projection  ->  H
    grid1 = (triton.cdiv(EM, block_m), num_sub)
    _lora_a_proj_kernel[grid1](
        A, A.stride(0), A.stride(1),
        H, H.stride(0), H.stride(1), H.stride(2),
        lora_a, la_stride_e, la_stride_s, la_stride_r, la_stride_k,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        num_valid,
        top_k=top_k, K=K, LORA_RANK=rank,
        BLOCK_M=block_m, TILE_K=tile_k,
    )

    # Pass 2: B projection + add to C
    block_n = 64
    grid2 = (triton.cdiv(EM, block_m), triton.cdiv(N, block_n))
    _lora_b_proj_add_kernel[grid2](
        H, H.stride(0), H.stride(1), H.stride(2),
        lora_b, lb_stride_e, lb_stride_s, lb_stride_n, lb_stride_r,
        C, C.stride(-2), C.stride(-1),
        lora_scaling,
        topk_weights if topk_weights is not None else C,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        num_valid, N, lora_inter_per_tp,
        LORA_RANK=rank, BLOCK_M=block_m, BLOCK_N=block_n,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
    )


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
    lora_a: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    lora_scaling: float = 0.0,
    lora_inter_per_tp: int = 0,
    oft_r: Optional[torch.Tensor] = None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        swap_ab = should_enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    else:
        swap_ab = False

    padded_size = 0

    # OFT and LoRA are mutually exclusive per adapter load.
    assert not (oft_r is not None and lora_a is not None), (
        "OFT and LoRA cannot be active simultaneously on the same layer"
    )

    if oft_r is not None:
        # Apply the expert-specific OFT rotation before activation
        # quantization. FP8 activations cannot be used as tl.dot operands in
        # the rotation kernel, and rotating before quantization matches the
        # dense OFT path mathematically.
        A = apply_oft_rotation_triton(
            A,
            oft_r,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_m=config["BLOCK_SIZE_M"],
        )

        top_k = 1
        C = C.reshape(-1, 1, C.shape[-1])
        topk_weights = topk_weights.reshape(-1, 1)
        topk_ids = topk_ids.reshape(-1, 1)

        from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
            moe_align_block_size,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], B.shape[0]
        )

    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation tensor-wise fp8 quantization, dynamic or static
            padded_size = padding_size
            # activations apply per-token quantization when weights apply per-channel quantization by default
            A, A_scale = scaled_fp8_quant(
                A, A_scale, use_per_token_if_dynamic=per_channel_quant
            )
        else:
            # activation block-wise fp8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (
                per_channel_quant
            ), "int8 quantization only supports channel-wise quantization except for block-wise quantization"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_int8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False

    if (
        (use_int8_w8a16 or use_int4_w4a16)
        and block_shape is not None
        and block_shape[1] > 0
    ):
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        assert bias is None
        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            even_Ks=even_Ks,
            filter_expert=filter_expert,
            **config,
        )

    else:
        if a_use_tma or b_use_tma:
            _set_triton_tma_allocator()

        if a_use_tma:
            a_desc = TensorDescriptor(
                A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]]
            )
        else:
            a_desc = None
        if b_use_tma:
            # B is constant weights -> cache descriptor
            b_desc = _get_b_tma_desc_cached(
                B,
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_K"],
            )
        else:
            b_desc = None

        fused_moe_kernel[grid](
            A,
            a_desc,
            B,
            b_desc,
            bias,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            B.shape[2] - padded_size,
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bias.stride(0) if bias is not None else 0,
            bias.stride(1) if bias is not None else 0,
            C.stride(-2),
            C.stride(-1),
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            even_Ks=even_Ks,
            c_sorted=c_sorted,
            filter_expert=filter_expert,
            swap_ab=swap_ab,
            **config,
        )

        # --- Separate Triton kernel for LoRA delta ---
        # Apply after the base kernel so the delta is added to the output.
        # OFT and LoRA are mutually exclusive, so A/top_k/sorted_token_ids
        # are unchanged when LoRA is active.
        if lora_a is not None:
            if lora_a.dim() == 4:
                assert lora_inter_per_tp % config["BLOCK_SIZE_N"] == 0, (
                    f"lora_inter_per_tp ({lora_inter_per_tp}) must be a multiple of "
                    f"BLOCK_SIZE_N ({config['BLOCK_SIZE_N']})"
                )
            assert lora_a.shape[-2] % 16 == 0, (
                f"LoRA rank must be a multiple of 16, got {lora_a.shape[-2]}"
            )
            apply_lora_delta_triton(
                A, C, lora_a, lora_b, lora_scaling, lora_inter_per_tp,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                top_k,
                topk_weights=topk_weights,
                mul_routed_weight=mul_routed_weight,
                block_m=config["BLOCK_SIZE_M"],
            )


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input
    """
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f"Unsupported activation: {ACTIVATION_TYPE}")


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
):
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.
    """
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)

    if expert_id == -1:
        return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)

        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)
        gate_output_activated = gate_output_activated.to(InDtype)

        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    config: Dict[str, Any],
    topk_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    down_moe_use_tma: bool = False,
    activation: str = "silu",
) -> None:
    """
    Args:
        gateup_output: Input tensor containing gate and up outputs concatenated
        down_input: Output tensor for the result
        config: Configuration dictionary with BLOCK_SIZE_M and BLOCK_SIZE_N
        topk_ids: Expert IDs for unsorted routing (used when down_moe_use_tma=False)
        expert_ids: Expert IDs for sorted routing (used when down_moe_use_tma=True)
        down_moe_use_tma: Whether to use sorted routing layout
        activation: Activation type ("silu" or "gelu")
    """
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    expert_ids_row = topk_ids.view(-1) if not down_moe_use_tma else expert_ids
    expert_step = 1 if not down_moe_use_tma else config["BLOCK_SIZE_M"]
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        expert_ids_row,
        expert_step,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
    )


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


@triton.jit
def _fused_append_shared_experts_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    N_BASE,  # runtime scalar
    scale_factor,  # runtime scalar
    K: tl.constexpr,
    S: tl.constexpr,
):
    """
    for m in range(M):
        for n in range(K):
            fused_ids[m, n] = topk_ids[m, n]
            fused_weights[m, n] = topk_weights[m, n]
        for s in range(S):
            fused_ids[m, K + s] = N + s
            fused_weights[m, K + s] = scale_factor
    """
    pid = tl.program_id(0)

    ids_row_ptr = pid * K
    w_row_ptr = pid * K
    out_ids_row_ptr = pid * (K + S)
    out_w_row_ptr = pid * (K + S)

    offs_k = tl.arange(0, K)
    ids = tl.load(topk_ids_ptr + ids_row_ptr + offs_k)
    ws = tl.load(topk_weights_ptr + w_row_ptr + offs_k)

    tl.store(out_ids_ptr + out_ids_row_ptr + offs_k, ids)
    tl.store(out_weights_ptr + out_w_row_ptr + offs_k, ws)

    offs_s = tl.arange(0, S)

    shared_ids = tl.cast(N_BASE + offs_s, ids.dtype)
    shared_ws = tl.full([S], scale_factor, dtype=ws.dtype)

    tl.store(out_ids_ptr + out_ids_row_ptr + K + offs_s, shared_ids)
    tl.store(out_weights_ptr + out_w_row_ptr + K + offs_s, shared_ws)


def fused_append_shared_experts(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N=None
):
    assert N is not None, "N (shared expert base id) must be provided"
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )

    _fused_append_shared_experts_kernel[(m,)](
        topk_ids,
        topk_weights,
        out_ids,
        out_weights,
        N_BASE=N,
        scale_factor=scale_factor,
        K=k,
        S=s,
        num_warps=1,
    )
    return out_ids, out_weights

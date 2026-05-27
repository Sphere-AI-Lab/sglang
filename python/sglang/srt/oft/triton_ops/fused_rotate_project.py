"""Fused rotate-and-project Triton kernel for SGLang's OFT rollout path.

Replaces ``oft_backend.run_qkv_oft + per-slice F.linear + torch.cat`` with a
single fused kernel: input rotation (block-diagonal R applied to x),
per-slice projection (W slice GEMM), and direct contiguous output assembly.

Per-slice offsets/widths are baked in as ``tl.constexpr`` ints so the launcher
never allocates a side tensor — important because torch.tensor inside a
CUDA-graph-captured region corrupts the stream.

The kernel reads ``slot_idx`` and ``block_size_val`` from 0-d device tensors
inside the kernel body. This mirrors the legacy ``gemm_oft_r_fwd`` pattern so
a single captured CUDA graph correctly handles BOTH (a) the auto-registered
identity slot used during capture and KL ref-model forward (block_size==0:
the kernel skips the rotation matmul and degenerates to a plain fused GEMM),
and (b) post-capture adapter swaps where ``prepare_oft_batch`` mutates the
0-d tensors via ``fill_`` to point at the active slot.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_rotate_project_qkv_kernel(
    x_ptr,                   # (M, K) bf16 input, row-major
    R_ptr,                   # (max_slots, 3 * blocks_per_slice, BS, BS) bf16
    W_ptr,                   # (sum_out, K) bf16 fused weight (slices stacked along rows)
    bias_ptr,                # (sum_out,) bf16; ignored when HAS_BIAS is False
    out_ptr,                 # (M, sum_out) bf16 output, row-major
    slot_idx_ptr,            # 0-d int32: which slot of R to read at runtime
    bsv_ptr,                 # 0-d int32: 0 -> skip rotation (identity); non-zero -> rotate
    M, K, sum_out,
    blocks_per_slice,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,  # per-slice widths
    BS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One program covers (m_tile, slice_idx, n_tile-group-within-slice) for QKV."""
    pid_m = tl.program_id(0)
    pid_slice = tl.program_id(1)
    pid_n = tl.program_id(2)

    # constexpr per-slice offsets (cumulative)
    O0: tl.constexpr = 0
    O1: tl.constexpr = S0
    O2: tl.constexpr = S0 + S1

    if pid_slice == 0:
        slice_offset = O0
        slice_width = S0
    elif pid_slice == 1:
        slice_offset = O1
        slice_width = S1
    else:
        slice_offset = O2
        slice_width = S2

    n_group_start = pid_n * BLOCK_N * GROUP_N
    if n_group_start >= slice_width:
        return

    NUM_SLICES: tl.constexpr = 3
    slot = tl.load(slot_idx_ptr)
    bsv = tl.load(bsv_ptr)
    slot_R_offset = slot * NUM_SLICES * blocks_per_slice * BS * BS

    _fused_rotate_project_inner(
        x_ptr, R_ptr, W_ptr, bias_ptr, out_ptr,
        pid_m, pid_slice, n_group_start, slice_offset, slice_width,
        slot_R_offset, bsv,
        M, K, sum_out, blocks_per_slice,
        BS, BLOCK_M, BLOCK_N, GROUP_N, HAS_BIAS,
    )


@triton.jit
def _fused_rotate_project_gate_up_kernel(
    x_ptr,
    R_ptr,
    W_ptr,
    bias_ptr,
    out_ptr,
    slot_idx_ptr,
    bsv_ptr,
    M, K, sum_out,
    blocks_per_slice,
    S0: tl.constexpr, S1: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """N=2 variant (gate/up)."""
    pid_m = tl.program_id(0)
    pid_slice = tl.program_id(1)
    pid_n = tl.program_id(2)

    O0: tl.constexpr = 0
    O1: tl.constexpr = S0

    if pid_slice == 0:
        slice_offset = O0
        slice_width = S0
    else:
        slice_offset = O1
        slice_width = S1

    n_group_start = pid_n * BLOCK_N * GROUP_N
    if n_group_start >= slice_width:
        return

    NUM_SLICES: tl.constexpr = 2
    slot = tl.load(slot_idx_ptr)
    bsv = tl.load(bsv_ptr)
    slot_R_offset = slot * NUM_SLICES * blocks_per_slice * BS * BS

    _fused_rotate_project_inner(
        x_ptr, R_ptr, W_ptr, bias_ptr, out_ptr,
        pid_m, pid_slice, n_group_start, slice_offset, slice_width,
        slot_R_offset, bsv,
        M, K, sum_out, blocks_per_slice,
        BS, BLOCK_M, BLOCK_N, GROUP_N, HAS_BIAS,
    )


@triton.jit
def _fused_rotate_gate_up_inputs_kernel(
    x_ptr,                   # (M, K) bf16 input, row-major
    R_ptr,                   # (max_slots, 2 * blocks_per_slice, BS, BS) bf16
    out_gate_ptr,            # (M, K) bf16 output
    out_up_ptr,              # (M, K) bf16 output
    slot_idx_ptr,            # 0-d int32: which slot of R to read at runtime
    bsv_ptr,                 # 0-d int32: 0 -> identity passthrough
    M, K,
    blocks_per_slice,
    BS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    block_idx = tl.program_id(1)

    if block_idx >= blocks_per_slice:
        return

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    bs_range = tl.arange(0, BS)
    offs_k = block_idx * BS + bs_range
    x_block = tl.load(
        x_ptr + offs_m[:, None] * K + offs_k[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )

    slot = tl.load(slot_idx_ptr)
    bsv = tl.load(bsv_ptr)

    if bsv == 0:
        tl.store(
            out_gate_ptr + offs_m[:, None] * K + offs_k[None, :],
            x_block,
            mask=m_mask[:, None],
        )
        tl.store(
            out_up_ptr + offs_m[:, None] * K + offs_k[None, :],
            x_block,
            mask=m_mask[:, None],
        )
        return

    slot_R_offset = slot * 2 * blocks_per_slice * BS * BS
    R_inner_offsets = bs_range[:, None] * BS + bs_range[None, :]

    gate_R_base = slot_R_offset + block_idx * BS * BS
    up_R_base = slot_R_offset + (blocks_per_slice + block_idx) * BS * BS

    R_gate = tl.load(R_ptr + gate_R_base + R_inner_offsets)
    R_up = tl.load(R_ptr + up_R_base + R_inner_offsets)

    gate = tl.dot(x_block, R_gate, out_dtype=tl.float32)
    up = tl.dot(x_block, R_up, out_dtype=tl.float32)

    tl.store(
        out_gate_ptr + offs_m[:, None] * K + offs_k[None, :],
        gate.to(tl.bfloat16),
        mask=m_mask[:, None],
    )
    tl.store(
        out_up_ptr + offs_m[:, None] * K + offs_k[None, :],
        up.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


@triton.jit
def _fused_rotate_project_inner(
    x_ptr,
    R_ptr,
    W_ptr,
    bias_ptr,
    out_ptr,
    pid_m, pid_slice, n_group_start, slice_offset, slice_width,
    slot_R_offset, bsv,
    M, K, sum_out,
    blocks_per_slice,
    BS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    offs_n0 = n_group_start + tl.arange(0, BLOCK_N)
    n_mask0 = offs_n0 < slice_width
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if GROUP_N >= 2:
        offs_n1 = n_group_start + BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask1 = offs_n1 < slice_width
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if GROUP_N >= 4:
        offs_n2 = n_group_start + 2 * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n3 = n_group_start + 3 * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask2 = offs_n2 < slice_width
        n_mask3 = offs_n3 < slice_width
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if GROUP_N >= 8:
        offs_n4 = n_group_start + 4 * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n5 = n_group_start + 5 * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n6 = n_group_start + 6 * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n7 = n_group_start + 7 * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask4 = offs_n4 < slice_width
        n_mask5 = offs_n5 < slice_width
        n_mask6 = offs_n6 < slice_width
        n_mask7 = offs_n7 < slice_width
        acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc5 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc6 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc7 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    rotation_block_start = pid_slice * blocks_per_slice

    bs_range = tl.arange(0, BS)
    R_inner_offsets = bs_range[:, None] * BS + bs_range[None, :]

    # bsv == 0 is the legacy identity-passthrough sentinel: the active OFT
    # slot's R buffer is uninitialized / not meaningful; behave as if R were
    # identity (skip the rotation matmul, project x directly).
    do_rotation = bsv != 0

    if do_rotation:
        for block_idx in range(0, blocks_per_slice):
            k_block_start = block_idx * BS
            offs_k = k_block_start + bs_range

            x_block = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            R_block_base = slot_R_offset + (rotation_block_start + block_idx) * BS * BS
            R_block = tl.load(R_ptr + R_block_base + R_inner_offsets)
            x_rot = tl.dot(x_block, R_block, out_dtype=tl.float32)
            x_for_proj = x_rot.to(tl.bfloat16)

            w_rows0 = slice_offset + offs_n0
            W_block0 = tl.load(
                W_ptr + w_rows0[:, None] * K + offs_k[None, :],
                mask=n_mask0[:, None],
                other=0.0,
            )
            acc0 += tl.dot(
                x_for_proj, tl.trans(W_block0), out_dtype=tl.float32, allow_tf32=False
            )

            if GROUP_N >= 2:
                w_rows1 = slice_offset + offs_n1
                W_block1 = tl.load(
                    W_ptr + w_rows1[:, None] * K + offs_k[None, :],
                    mask=n_mask1[:, None],
                    other=0.0,
                )
                acc1 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block1),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )

            if GROUP_N >= 4:
                w_rows2 = slice_offset + offs_n2
                w_rows3 = slice_offset + offs_n3
                W_block2 = tl.load(
                    W_ptr + w_rows2[:, None] * K + offs_k[None, :],
                    mask=n_mask2[:, None],
                    other=0.0,
                )
                W_block3 = tl.load(
                    W_ptr + w_rows3[:, None] * K + offs_k[None, :],
                    mask=n_mask3[:, None],
                    other=0.0,
                )
                acc2 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block2),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )
                acc3 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block3),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )

            if GROUP_N >= 8:
                w_rows4 = slice_offset + offs_n4
                w_rows5 = slice_offset + offs_n5
                w_rows6 = slice_offset + offs_n6
                w_rows7 = slice_offset + offs_n7
                W_block4 = tl.load(
                    W_ptr + w_rows4[:, None] * K + offs_k[None, :],
                    mask=n_mask4[:, None],
                    other=0.0,
                )
                W_block5 = tl.load(
                    W_ptr + w_rows5[:, None] * K + offs_k[None, :],
                    mask=n_mask5[:, None],
                    other=0.0,
                )
                W_block6 = tl.load(
                    W_ptr + w_rows6[:, None] * K + offs_k[None, :],
                    mask=n_mask6[:, None],
                    other=0.0,
                )
                W_block7 = tl.load(
                    W_ptr + w_rows7[:, None] * K + offs_k[None, :],
                    mask=n_mask7[:, None],
                    other=0.0,
                )
                acc4 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block4),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )
                acc5 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block5),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )
                acc6 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block6),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )
                acc7 += tl.dot(
                    x_for_proj,
                    tl.trans(W_block7),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )
    else:
        for block_idx in range(0, blocks_per_slice):
            k_block_start = block_idx * BS
            offs_k = k_block_start + bs_range

            x_block = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )

            w_rows0 = slice_offset + offs_n0
            W_block0 = tl.load(
                W_ptr + w_rows0[:, None] * K + offs_k[None, :],
                mask=n_mask0[:, None],
                other=0.0,
            )
            acc0 += tl.dot(
                x_block, tl.trans(W_block0), out_dtype=tl.float32, allow_tf32=False
            )

            if GROUP_N >= 2:
                w_rows1 = slice_offset + offs_n1
                W_block1 = tl.load(
                    W_ptr + w_rows1[:, None] * K + offs_k[None, :],
                    mask=n_mask1[:, None],
                    other=0.0,
                )
                acc1 += tl.dot(
                    x_block, tl.trans(W_block1), out_dtype=tl.float32, allow_tf32=False
                )

            if GROUP_N >= 4:
                w_rows2 = slice_offset + offs_n2
                w_rows3 = slice_offset + offs_n3
                W_block2 = tl.load(
                    W_ptr + w_rows2[:, None] * K + offs_k[None, :],
                    mask=n_mask2[:, None],
                    other=0.0,
                )
                W_block3 = tl.load(
                    W_ptr + w_rows3[:, None] * K + offs_k[None, :],
                    mask=n_mask3[:, None],
                    other=0.0,
                )
                acc2 += tl.dot(
                    x_block, tl.trans(W_block2), out_dtype=tl.float32, allow_tf32=False
                )
                acc3 += tl.dot(
                    x_block, tl.trans(W_block3), out_dtype=tl.float32, allow_tf32=False
                )

            if GROUP_N >= 8:
                w_rows4 = slice_offset + offs_n4
                w_rows5 = slice_offset + offs_n5
                w_rows6 = slice_offset + offs_n6
                w_rows7 = slice_offset + offs_n7
                W_block4 = tl.load(
                    W_ptr + w_rows4[:, None] * K + offs_k[None, :],
                    mask=n_mask4[:, None],
                    other=0.0,
                )
                W_block5 = tl.load(
                    W_ptr + w_rows5[:, None] * K + offs_k[None, :],
                    mask=n_mask5[:, None],
                    other=0.0,
                )
                W_block6 = tl.load(
                    W_ptr + w_rows6[:, None] * K + offs_k[None, :],
                    mask=n_mask6[:, None],
                    other=0.0,
                )
                W_block7 = tl.load(
                    W_ptr + w_rows7[:, None] * K + offs_k[None, :],
                    mask=n_mask7[:, None],
                    other=0.0,
                )
                acc4 += tl.dot(
                    x_block, tl.trans(W_block4), out_dtype=tl.float32, allow_tf32=False
                )
                acc5 += tl.dot(
                    x_block, tl.trans(W_block5), out_dtype=tl.float32, allow_tf32=False
                )
                acc6 += tl.dot(
                    x_block, tl.trans(W_block6), out_dtype=tl.float32, allow_tf32=False
                )
                acc7 += tl.dot(
                    x_block, tl.trans(W_block7), out_dtype=tl.float32, allow_tf32=False
                )

    if HAS_BIAS:
        bias_vals0 = tl.load(bias_ptr + slice_offset + offs_n0, mask=n_mask0, other=0.0)
        acc0 = acc0 + bias_vals0[None, :].to(tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n0)[None, :],
        acc0.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask0[None, :],
    )

    if GROUP_N >= 2:
        if HAS_BIAS:
            bias_vals1 = tl.load(
                bias_ptr + slice_offset + offs_n1, mask=n_mask1, other=0.0
            )
            acc1 = acc1 + bias_vals1[None, :].to(tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n1)[None, :],
            acc1.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask1[None, :],
        )

    if GROUP_N >= 4:
        if HAS_BIAS:
            bias_vals2 = tl.load(
                bias_ptr + slice_offset + offs_n2, mask=n_mask2, other=0.0
            )
            bias_vals3 = tl.load(
                bias_ptr + slice_offset + offs_n3, mask=n_mask3, other=0.0
            )
            acc2 = acc2 + bias_vals2[None, :].to(tl.float32)
            acc3 = acc3 + bias_vals3[None, :].to(tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n2)[None, :],
            acc2.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask2[None, :],
        )
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n3)[None, :],
            acc3.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask3[None, :],
        )

    if GROUP_N >= 8:
        if HAS_BIAS:
            bias_vals4 = tl.load(
                bias_ptr + slice_offset + offs_n4, mask=n_mask4, other=0.0
            )
            bias_vals5 = tl.load(
                bias_ptr + slice_offset + offs_n5, mask=n_mask5, other=0.0
            )
            bias_vals6 = tl.load(
                bias_ptr + slice_offset + offs_n6, mask=n_mask6, other=0.0
            )
            bias_vals7 = tl.load(
                bias_ptr + slice_offset + offs_n7, mask=n_mask7, other=0.0
            )
            acc4 = acc4 + bias_vals4[None, :].to(tl.float32)
            acc5 = acc5 + bias_vals5[None, :].to(tl.float32)
            acc6 = acc6 + bias_vals6[None, :].to(tl.float32)
            acc7 = acc7 + bias_vals7[None, :].to(tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n4)[None, :],
            acc4.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask4[None, :],
        )
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n5)[None, :],
            acc5.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask5[None, :],
        )
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n6)[None, :],
            acc6.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask6[None, :],
        )
        tl.store(
            out_ptr + offs_m[:, None] * sum_out + (slice_offset + offs_n7)[None, :],
            acc7.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask7[None, :],
        )


def _pick_tiles(M: int, max_slice_width: int):
    # FC1-class shapes: wide outputs benefit most from GROUP_N reuse.
    if max_slice_width >= 1024:
        if M >= 1024:
            return 32, 64, 4
        return 16, 64, 4
    # QKV-class shapes: smaller slices have less rotation recompute.
    if M >= 1024:
        return 128, 128, 2
    return 64, 64, 2


def _pick_qkv_tiles(M: int, max_slice_width: int, block_size: int):
    # GROUP_N=2 with BS=128 exceeds the B200 per-block shared memory limit
    # under CUDA 13/Triton for Qwen2.5-style QKV shapes.
    if block_size >= 128:
        return 64, 64, 1
    return _pick_tiles(M, max_slice_width)


def _validate_inputs(x, R, W, output_sizes, num_slices_expected):
    num_slices = len(output_sizes)
    assert num_slices == num_slices_expected, (
        f"expected {num_slices_expected} slices; got {num_slices}"
    )
    M, K = x.shape
    if R.dim() == 4:
        total_blocks = R.shape[1]
    elif R.dim() == 3:
        total_blocks = R.shape[0]
    else:
        raise AssertionError(f"R must be 3D (legacy) or 4D; got shape {tuple(R.shape)}")
    BS = R.shape[-1]
    blocks_per_slice = total_blocks // num_slices
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert R.dtype == torch.bfloat16, f"R must be bf16, got {R.dtype}"
    assert W.dtype == torch.bfloat16, f"W must be bf16, got {W.dtype}"
    assert x.is_contiguous(), "x must be contiguous"
    assert R.is_contiguous(), "R must be contiguous"
    assert W.is_contiguous(), "W must be contiguous"
    assert blocks_per_slice * BS == K, (
        f"blocks_per_slice={blocks_per_slice} * BS={BS} != K={K}"
    )
    assert BS >= 16, f"Triton tl.dot requires BS >= 16; got BS={BS}"
    return M, K, BS, blocks_per_slice


def _ensure_4d_R_and_runtime_tensors(R, slot_idx_t, bsv_t, BS, device):
    """Normalize R to 4D and provide 0-d slot/bsv tensors for the kernel.

    - If R is 3D, wraps it as 4D with a single slot (slot=0); slot_idx_t /
      bsv_t default to (0, BS) so rotation is enabled.
    - If R is 4D, returns it as-is and requires slot_idx_t / bsv_t to be
      passed by the caller — these MUST be persistent 0-d tensors (not
      freshly allocated per call) when used under CUDA-graph capture.
    """
    if R.dim() == 3:
        R_4d = R.unsqueeze(0).contiguous() if not R.is_contiguous() else R.unsqueeze(0)
        if slot_idx_t is None:
            slot_idx_t = torch.zeros((), dtype=torch.int32, device=device)
        if bsv_t is None:
            bsv_t = torch.tensor(BS, dtype=torch.int32, device=device)
        return R_4d, slot_idx_t, bsv_t
    # R is 4D — caller must supply persistent 0-d tensors for graph safety.
    if slot_idx_t is None or bsv_t is None:
        raise RuntimeError(
            "4D R requires explicit slot_idx_t and bsv_t (persistent 0-d tensors)"
        )
    return R, slot_idx_t, bsv_t


def fused_rotate_gate_up_inputs(
    x: torch.Tensor,
    R: torch.Tensor,
    *,
    slot_idx_t: Optional[torch.Tensor] = None,
    bsv_t: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate FC1/gate-up input once and return ``(x_gate, x_up)`` tensors."""
    if R.dim() == 4:
        total_blocks = R.shape[1]
    elif R.dim() == 3:
        total_blocks = R.shape[0]
    else:
        raise AssertionError(f"R must be 3D (legacy) or 4D; got shape {tuple(R.shape)}")

    M, K = x.shape
    BS = R.shape[-1]
    assert total_blocks % 2 == 0, (
        f"gate/up R blocks must be divisible by 2; got {total_blocks}"
    )
    blocks_per_slice = total_blocks // 2
    assert blocks_per_slice * BS == K, (
        f"blocks_per_slice={blocks_per_slice} * BS={BS} != K={K}"
    )
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert R.dtype == torch.bfloat16, f"R must be bf16, got {R.dtype}"
    assert x.is_contiguous(), "x must be contiguous"
    assert R.is_contiguous(), "R must be contiguous"
    assert BS >= 16, f"Triton tl.dot requires BS >= 16; got BS={BS}"

    R, slot_idx_t, bsv_t = _ensure_4d_R_and_runtime_tensors(
        R, slot_idx_t, bsv_t, BS, x.device
    )

    x_gate = torch.empty_like(x)
    x_up = torch.empty_like(x)
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), blocks_per_slice)

    _fused_rotate_gate_up_inputs_kernel[grid](
        x,
        R,
        x_gate,
        x_up,
        slot_idx_t,
        bsv_t,
        M,
        K,
        blocks_per_slice,
        BS=BS,
        BLOCK_M=BLOCK_M,
    )
    return x_gate, x_up


def fused_rotate_project_qkv(
    x: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    output_sizes: List[int],
    bias: Optional[torch.Tensor] = None,
    *,
    slot_idx_t: Optional[torch.Tensor] = None,
    bsv_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused rotate-and-project for QKV (N=3, GQA Q!=K=V layout).

    R is either 3D ``(total_blocks, BS, BS)`` (legacy single-slot test path)
    or 4D ``(max_slots, total_blocks_per_slot, BS, BS)`` (production). With
    4D R, ``slot_idx_t`` and ``bsv_t`` must be persistent 0-d int32 tensors
    so a captured CUDA graph picks up runtime adapter swaps via ``.fill_()``.
    ``bsv_t == 0`` toggles identity-passthrough (skip rotation matmul).
    """
    M, K, BS, blocks_per_slice = _validate_inputs(x, R, W, output_sizes, 3)
    R, slot_idx_t, bsv_t = _ensure_4d_R_and_runtime_tensors(
        R, slot_idx_t, bsv_t, BS, x.device
    )
    sum_out = int(sum(output_sizes))
    S0, S1, S2 = int(output_sizes[0]), int(output_sizes[1]), int(output_sizes[2])

    # When bias is None we pass x as a dummy pointer; HAS_BIAS=False makes the
    # kernel never read from it. This avoids a torch.empty(0) alloc per call,
    # which is harmless in eager but adds bookkeeping in CUDA graph capture.
    bias_arg = bias if bias is not None else x
    has_bias = bias is not None

    out = torch.empty(M, sum_out, dtype=torch.bfloat16, device=x.device)

    max_slice_width = max(S0, S1, S2)
    BLOCK_M, BLOCK_N, GROUP_N = _pick_qkv_tiles(M, max_slice_width, BS)

    grid = (
        triton.cdiv(M, BLOCK_M),
        3,
        triton.cdiv(max_slice_width, BLOCK_N * GROUP_N),
    )

    _fused_rotate_project_qkv_kernel[grid](
        x, R, W, bias_arg, out,
        slot_idx_t, bsv_t,
        M, K, sum_out,
        blocks_per_slice,
        S0, S1, S2,
        BS=BS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_N=GROUP_N,
        HAS_BIAS=has_bias,
    )
    return out


def fused_rotate_project_gate_up(
    x: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    output_sizes: List[int],
    bias: Optional[torch.Tensor] = None,
    *,
    slot_idx_t: Optional[torch.Tensor] = None,
    bsv_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused rotate-and-project for fused gate/up (N=2, equal output dims)."""
    M, K, BS, blocks_per_slice = _validate_inputs(x, R, W, output_sizes, 2)
    R, slot_idx_t, bsv_t = _ensure_4d_R_and_runtime_tensors(
        R, slot_idx_t, bsv_t, BS, x.device
    )
    assert output_sizes[0] == output_sizes[1], (
        f"gate/up must have equal output sizes; got {output_sizes}"
    )
    sum_out = int(sum(output_sizes))
    S0, S1 = int(output_sizes[0]), int(output_sizes[1])

    # When bias is None we pass x as a dummy pointer; HAS_BIAS=False makes the
    # kernel never read from it. This avoids a torch.empty(0) alloc per call,
    # which is harmless in eager but adds bookkeeping in CUDA graph capture.
    bias_arg = bias if bias is not None else x
    has_bias = bias is not None

    out = torch.empty(M, sum_out, dtype=torch.bfloat16, device=x.device)

    max_slice_width = max(S0, S1)
    BLOCK_M, BLOCK_N, GROUP_N = _pick_tiles(M, max_slice_width)

    grid = (
        triton.cdiv(M, BLOCK_M),
        2,
        triton.cdiv(max_slice_width, BLOCK_N * GROUP_N),
    )

    _fused_rotate_project_gate_up_kernel[grid](
        x, R, W, bias_arg, out,
        slot_idx_t, bsv_t,
        M, K, sum_out,
        blocks_per_slice,
        S0, S1,
        BS=BS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_N=GROUP_N,
        HAS_BIAS=has_bias,
    )
    return out


if __name__ == "__main__":
    """Standalone benchmark vs realistic baselines.

    Three paths timed at each shape:
      * fused      — this file's kernel (rotation + per-slice GEMM + write in one launch).
      * legacy     — production OFT decode path: ``gemm_oft_r_fwd`` (Triton rotation)
                     + per-slice ``F.linear`` (cuBLAS) + ``torch.cat``. This is the actual
                     baseline the fused kernel has to beat in live serving.
      * merged     — ``F.linear(x, W_full, b)`` only. Theoretical lower bound assuming
                     OFT R is pre-merged into W. The fused kernel can't go faster than
                     this — it's the floor.

    Two adapter regimes per shape:
      * identity (bsv=0) — what CUDA-graph capture and KL ref-model forward see.
                          Both fused and legacy take their no-rotation branch; output
                          equals merged. Useful for measuring wrapper overhead alone.
      * rotate (bsv=BS)  — what live rollout decode replay does once an adapter is
                          loaded. Real rotation work; output differs from merged.

    M coverage spans realistic decode batch sizes (1, 8, 32, 64) through capture/
    large-decode (256) and prefill (1024, 4096, 8192) so the per-regime perf gap is
    visible end-to-end.

    Run: python python/sglang/srt/oft/triton_ops/fused_rotate_project.py
    """
    import time

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required to run this benchmark.")

    from sglang.srt.oft.triton_ops import gemm_oft_r_fwd

    def _production_legacy_path(x, R_4d, W, output_sizes, bias, slot_t, bsv_t):
        """Mirror SGLang's current OFT decode path: ``gemm_oft_r_fwd`` (Triton rotation,
        identity-passthrough when bsv=0) + per-slice ``F.linear`` + ``torch.cat``.
        """
        num_slices = len(output_sizes)
        K = x.shape[-1]
        rotated = gemm_oft_r_fwd(x, R_4d, slot_t, bsv_t, num_slices=num_slices)
        input_slices = list(torch.split(rotated, K, dim=-1))
        W_slices = torch.split(W, output_sizes, dim=0)
        if bias is None:
            b_slices: List[Optional[torch.Tensor]] = [None] * num_slices
        else:
            b_slices = list(torch.split(bias, output_sizes, dim=0))
        outs = [
            F.linear(input_slices[i], W_slices[i], b_slices[i])
            for i in range(num_slices)
        ]
        return torch.cat(outs, dim=-1)

    def _gate_up_two_output_input_path(x, R_4d, W, output_sizes, bias, slot_t, bsv_t):
        """Experimental FC1 path: rotate into separate gate/up tensors, then cuBLAS."""
        assert len(output_sizes) == 2
        x_gate, x_up = fused_rotate_gate_up_inputs(
            x,
            R_4d,
            slot_idx_t=slot_t,
            bsv_t=bsv_t,
        )
        W_gate, W_up = torch.split(W, output_sizes, dim=0)
        if bias is None:
            b_gate = b_up = None
        else:
            b_gate, b_up = torch.split(bias, output_sizes, dim=0)
        return torch.cat(
            [
                F.linear(x_gate, W_gate, b_gate),
                F.linear(x_up, W_up, b_up),
            ],
            dim=-1,
        )

    def _merged_weight_path(x, W, bias):
        """Lower bound: OFT pre-merged into W. Single fused F.linear."""
        return F.linear(x, W, bias)

    def _make_inputs(M, K, output_sizes, block_size, dtype=torch.bfloat16, seed=0):
        torch.manual_seed(seed)
        blocks_per_slice = K // block_size
        total_blocks = blocks_per_slice * len(output_sizes)
        x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.01
        R_eye = (
            torch.eye(block_size, device="cuda", dtype=dtype)
            .expand(total_blocks, -1, -1)
            .contiguous()
        )
        R_noise = torch.randn(
            total_blocks, block_size, block_size, device="cuda", dtype=dtype
        ) * 0.01
        # 4D R with 2 slots so the fused kernel reads at slot=1 (matches the
        # typical RL setup: slot 0 = base/identity placeholder, slot 1 = active).
        R_3d = (R_eye + R_noise).contiguous()
        R_4d = torch.stack([R_3d, R_3d], dim=0).contiguous()
        W = torch.randn(sum(output_sizes), K, device="cuda", dtype=dtype) * 0.02
        return x, R_4d, W

    def _time_us(fn, warmup=10, iters=50):
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e6 / iters

    # Qwen2.5-7B per-rank shapes with TP=8: hidden=3584, Q=448, K=V=64,
    # gate=up=2368, OFT block size 32.
    M_LIST = [1, 8, 32, 64, 256, 1024, 4096, 8192]
    WRAPPERS = [
        ("QKV", {"K": 3584, "output_sizes": [448, 64, 64], "block_size": 32},
         fused_rotate_project_qkv),
        ("FC1", {"K": 3584, "output_sizes": [2368, 2368], "block_size": 32},
         fused_rotate_project_gate_up),
    ]
    BSV_REGIMES = [
        ("identity", 0),   # bsv=0 → skip rotation; captures the capture-time / KL forward case
        ("rotate",   32),  # bsv=BS → full rotation; captures the live-decode case
    ]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("Shapes: Qwen2.5-7B per-rank, TP=8, OFT block_size=32, bf16")
    print("Baselines: 'legacy' = gemm_oft_r_fwd + per-slice F.linear + cat (production).")
    print("           'merged' = single F.linear on full fused weight (lower bound).")
    print()

    header = (
        f"{'wrap':<4} {'M':>5} | {'mode':<8} | "
        f"{'fused us':>9} {'legacy us':>9} {'merged us':>9} | "
        f"{'fused/legacy':>13} {'fused/merged':>13} {'legacy/merged':>13} | parity"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    for wrap_name, shape, fused_fn in WRAPPERS:
        for M in M_LIST:
            x, R_4d, W = _make_inputs(M=M, **shape, dtype=torch.bfloat16)
            output_sizes = shape["output_sizes"]
            slot_t = torch.tensor(1, dtype=torch.int32, device="cuda")

            for bsv_label, bsv_val in BSV_REGIMES:
                bsv_t = torch.tensor(bsv_val, dtype=torch.int32, device="cuda")

                fused_out = fused_fn(
                    x, R_4d, W, output_sizes, bias=None,
                    slot_idx_t=slot_t, bsv_t=bsv_t,
                )
                legacy_out = _production_legacy_path(
                    x, R_4d, W, output_sizes, bias=None,
                    slot_t=slot_t, bsv_t=bsv_t,
                )

                max_abs = (fused_out.float() - legacy_out.float()).abs().max().item()
                parity_ok = max_abs <= 2e-3
                all_pass &= parity_ok

                fused_us = _time_us(lambda: fused_fn(
                    x, R_4d, W, output_sizes, bias=None,
                    slot_idx_t=slot_t, bsv_t=bsv_t,
                ))
                legacy_us = _time_us(lambda: _production_legacy_path(
                    x, R_4d, W, output_sizes, bias=None,
                    slot_t=slot_t, bsv_t=bsv_t,
                ))
                merged_us = _time_us(lambda: _merged_weight_path(x, W, bias=None))

                vs_legacy = legacy_us / fused_us
                vs_merged = merged_us / fused_us
                legacy_vs_merged = merged_us / legacy_us
                verdict = "PASS" if parity_ok else f"FAIL({max_abs:.1e})"

                print(
                    f"{wrap_name:<4} {M:>5} | {bsv_label:<8} | "
                    f"{fused_us:>9.1f} {legacy_us:>9.1f} {merged_us:>9.1f} | "
                    f"{vs_legacy:>12.2f}x {vs_merged:>12.2f}x "
                    f"{legacy_vs_merged:>12.2f}x | {verdict}"
                )
            print()

    print()
    print("FC1 two-output rotate-input experiment")
    header2 = (
        f"{'M':>5} | {'mode':<8} | "
        f"{'input2 us':>9} {'legacy us':>9} {'fullfused us':>12} "
        f"{'merged us':>9} | {'input2/legacy':>13} | parity"
    )
    print(header2)
    print("-" * len(header2))

    fc1_shape = {"K": 3584, "output_sizes": [2368, 2368], "block_size": 32}
    for M in M_LIST:
        x, R_4d, W = _make_inputs(M=M, **fc1_shape, dtype=torch.bfloat16)
        output_sizes = fc1_shape["output_sizes"]
        slot_t = torch.tensor(1, dtype=torch.int32, device="cuda")

        for bsv_label, bsv_val in BSV_REGIMES:
            bsv_t = torch.tensor(bsv_val, dtype=torch.int32, device="cuda")

            input2_out = _gate_up_two_output_input_path(
                x, R_4d, W, output_sizes, bias=None,
                slot_t=slot_t, bsv_t=bsv_t,
            )
            legacy_out = _production_legacy_path(
                x, R_4d, W, output_sizes, bias=None,
                slot_t=slot_t, bsv_t=bsv_t,
            )
            input2_max_abs = (
                input2_out.float() - legacy_out.float()
            ).abs().max().item()
            max_abs = input2_max_abs
            parity_ok = max_abs <= 2e-3
            all_pass &= parity_ok

            input2_us = _time_us(lambda: _gate_up_two_output_input_path(
                x, R_4d, W, output_sizes, bias=None,
                slot_t=slot_t, bsv_t=bsv_t,
            ))
            legacy_us = _time_us(lambda: _production_legacy_path(
                x, R_4d, W, output_sizes, bias=None,
                slot_t=slot_t, bsv_t=bsv_t,
            ))
            full_fused_us = _time_us(lambda: fused_rotate_project_gate_up(
                x, R_4d, W, output_sizes, bias=None,
                slot_idx_t=slot_t, bsv_t=bsv_t,
            ))
            merged_us = _time_us(lambda: _merged_weight_path(x, W, bias=None))

            verdict = "PASS" if parity_ok else f"FAIL({max_abs:.1e})"
            print(
                f"{M:>5} | {bsv_label:<8} | "
                f"{input2_us:>9.1f} {legacy_us:>9.1f} "
                f"{full_fused_us:>12.1f} {merged_us:>9.1f} | "
                f"{legacy_us / input2_us:>12.2f}x | {verdict}"
            )
        print()

    print()
    if all_pass:
        print("All shapes within bf16 tolerance (max_abs <= 2e-3) vs production legacy.")
    else:
        raise SystemExit("Parity check failed for at least one shape.")

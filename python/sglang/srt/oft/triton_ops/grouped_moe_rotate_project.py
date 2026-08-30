from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.oft.utils import validate_oft_block_size


@triton.jit
def _split_w13_oft_grouped_moe_kernel(
    hidden_states_ptr,
    w13_ptr,
    w1_oft_r_ptr,
    w3_oft_r_ptr,
    out_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    HALF: tl.constexpr,
    TOP_K: tl.constexpr,
    EM: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    half_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(HALF, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    token_routes = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = token_routes < (M * TOP_K)
    token_idx = token_routes // TOP_K
    route_idx = token_routes - token_idx * TOP_K

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < HALF
    half_offset = half_id * HALF

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    block_range = tl.arange(0, BLOCK_SIZE)
    r_offsets = block_range[:, None] * BLOCK_SIZE + block_range[None, :]

    if BLOCK_SIZE >= 16:
        for block_idx in range(0, BLOCKS):
            k_offsets = block_idx * BLOCK_SIZE + block_range
            x = tl.load(
                hidden_states_ptr + token_idx[:, None] * K + k_offsets[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )

            r_base = expert * BLOCKS * BLOCK_SIZE * BLOCK_SIZE
            if half_id == 0:
                r = tl.load(w1_oft_r_ptr + r_base + block_idx * BLOCK_SIZE * BLOCK_SIZE + r_offsets)
            else:
                r = tl.load(w3_oft_r_ptr + r_base + block_idx * BLOCK_SIZE * BLOCK_SIZE + r_offsets)

            x_rot = tl.dot(x, r, input_precision="ieee", out_dtype=tl.float32).to(tl.bfloat16)
            w = tl.load(
                w13_ptr
                + expert * N * K
                + (half_offset + offs_n[:, None]) * K
                + k_offsets[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(
                x_rot,
                tl.trans(w),
                out_dtype=tl.float32,
                allow_tf32=False,
            )
    else:
        tiny_cols = tl.arange(0, 16)
        blocks_in_tile: tl.constexpr = 16 // BLOCK_SIZE
        block_in_tile = tiny_cols // BLOCK_SIZE
        col_in_block = tiny_cols - block_in_tile * BLOCK_SIZE
        for block_group in range(0, BLOCKS, blocks_in_tile):
            block_ids = block_group + block_in_tile
            valid_blocks = block_ids < BLOCKS
            x_rot = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
            r_base = (
                expert * BLOCKS * BLOCK_SIZE * BLOCK_SIZE
                + block_ids * BLOCK_SIZE * BLOCK_SIZE
            )
            for k in range(BLOCK_SIZE):
                x_k = tl.load(
                    hidden_states_ptr
                    + token_idx[:, None] * K
                    + (block_ids * BLOCK_SIZE + k)[None, :],
                    mask=token_mask[:, None] & valid_blocks[None, :],
                    other=0.0,
                ).to(tl.float32)
                if half_id == 0:
                    r_kj = tl.load(
                        w1_oft_r_ptr + r_base + k * BLOCK_SIZE + col_in_block,
                        mask=valid_blocks,
                        other=0.0,
                    )
                else:
                    r_kj = tl.load(
                        w3_oft_r_ptr + r_base + k * BLOCK_SIZE + col_in_block,
                        mask=valid_blocks,
                        other=0.0,
                    )
                x_rot += x_k * r_kj.to(tl.float32)[None, :]
            k_offsets = block_group * BLOCK_SIZE + tiny_cols
            w = tl.load(
                w13_ptr
                + expert * N * K
                + (half_offset + offs_n[:, None]) * K
                + k_offsets[None, :],
                mask=n_mask[:, None] & valid_blocks[None, :],
                other=0.0,
            )
            acc += tl.dot(
                x_rot.to(tl.bfloat16),
                tl.trans(w),
                out_dtype=tl.float32,
                allow_tf32=False,
            )

    out_offsets = token_idx[:, None] * TOP_K * N + route_idx[:, None] * N
    out_offsets += half_offset + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc.to(tl.bfloat16),
        mask=token_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _pack_split_oft_grouped_bmm_inputs_kernel(
    hidden_states_ptr,
    w1_oft_r_ptr,
    w3_oft_r_ptr,
    packed_inputs_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    expert_offsets_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_PADDED_TOKENS_PER_EXPERT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)
    half_id = tl.program_id(axis=2)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert < 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    token_routes = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = (offs_m < num_tokens_post_padded) & (token_routes < (M * TOP_K))
    token_idx = token_routes // TOP_K
    rank_offsets = offs_m - tl.load(expert_offsets_ptr + expert).to(tl.int64)

    r_base = expert * BLOCKS * BLOCK_SIZE * BLOCK_SIZE + block_idx * BLOCK_SIZE * BLOCK_SIZE
    packed_batch = half_id * EXPERTS + expert
    if BLOCK_SIZE >= 16:
        block_range = tl.arange(0, BLOCK_SIZE)
        k_offsets = block_idx * BLOCK_SIZE + block_range
        x = tl.load(
            hidden_states_ptr + token_idx[:, None] * K + k_offsets[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )

        r_offsets = block_range[:, None] * BLOCK_SIZE + block_range[None, :]
        if half_id == 0:
            r = tl.load(w1_oft_r_ptr + r_base + r_offsets)
        else:
            r = tl.load(w3_oft_r_ptr + r_base + r_offsets)

        x_rot = tl.dot(x, r, input_precision="ieee", out_dtype=tl.float32).to(tl.bfloat16)
        packed_offsets = (
            (packed_batch * MAX_PADDED_TOKENS_PER_EXPERT + rank_offsets[:, None]) * K
            + k_offsets[None, :]
        )
        tl.store(
            packed_inputs_ptr + packed_offsets,
            x_rot,
            mask=token_mask[:, None],
        )
    else:
        for j in range(BLOCK_SIZE):
            x_rot_j = tl.zeros((BLOCK_M,), dtype=tl.float32)
            for k in range(BLOCK_SIZE):
                x_k = tl.load(
                    hidden_states_ptr
                    + token_idx * K
                    + block_idx * BLOCK_SIZE
                    + k,
                    mask=token_mask,
                    other=0.0,
                ).to(tl.float32)
                if half_id == 0:
                    r_kj = tl.load(
                        w1_oft_r_ptr + r_base + k * BLOCK_SIZE + j
                    ).to(tl.float32)
                else:
                    r_kj = tl.load(
                        w3_oft_r_ptr + r_base + k * BLOCK_SIZE + j
                    ).to(tl.float32)
                x_rot_j += x_k * r_kj
            packed_col = block_idx * BLOCK_SIZE + j
            packed_offset = (
                packed_batch * MAX_PADDED_TOKENS_PER_EXPERT + rank_offsets
            ) * K + packed_col
            tl.store(
                packed_inputs_ptr + packed_offset,
                x_rot_j.to(tl.bfloat16),
                mask=token_mask,
            )


@triton.jit
def _unpack_split_oft_grouped_bmm_outputs_kernel(
    projected_ptr,
    out_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    expert_offsets_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    HALF: tl.constexpr,
    TOP_K: tl.constexpr,
    EXPERTS: tl.constexpr,
    MAX_PADDED_TOKENS_PER_EXPERT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    half_id = tl.program_id(axis=2)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert < 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    token_routes = tl.load(sorted_token_ids_ptr + offs_m).to(tl.int64)
    token_mask = (offs_m < num_tokens_post_padded) & (token_routes < (M * TOP_K))
    rank_offsets = offs_m - tl.load(expert_offsets_ptr + expert).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < HALF
    packed_batch = half_id * EXPERTS + expert

    vals = tl.load(
        projected_ptr
        + (packed_batch * MAX_PADDED_TOKENS_PER_EXPERT + rank_offsets[:, None]) * HALF
        + offs_n[None, :],
        mask=token_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    half_offset = half_id * HALF
    tl.store(
        out_ptr + token_routes[:, None] * N + half_offset + offs_n[None, :],
        vals,
        mask=token_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _expert_offsets_from_block_ids_kernel(
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    expert_offsets_ptr,
    BLOCK_M: tl.constexpr,
    MAX_M_BLOCKS: tl.constexpr,
    BLOCKS_PER_TILE: tl.constexpr,
):
    expert = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    sentinel = MAX_M_BLOCKS * BLOCK_M
    best = tl.full((), sentinel, dtype=tl.int64)

    for block_start in range(0, MAX_M_BLOCKS, BLOCKS_PER_TILE):
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_TILE)
        in_range = block_offsets < MAX_M_BLOCKS
        active = block_offsets * BLOCK_M < num_tokens_post_padded
        block_experts = tl.load(
            expert_ids_ptr + block_offsets,
            mask=in_range & active,
            other=-2,
        )
        candidates = tl.where(
            block_experts == expert,
            (block_offsets * BLOCK_M).to(tl.int64),
            sentinel,
        )
        best = tl.minimum(best, tl.min(candidates, axis=0))

    tl.store(expert_offsets_ptr + expert, tl.where(best == sentinel, 0, best))


def _validate_bf16(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be bf16, got {tensor.dtype}")


def _is_cuda_graph_capture_setup() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return True
    except RuntimeError:
        pass
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def fused_split_w13_oft_grouped_moe(
    *,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w1_oft_r: torch.Tensor,
    w3_oft_r: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_m: int,
    block_n: int = 64,
    group_size_m: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    """Return expert FC1 output shaped (M, top_k, 2 * half_intermediate).

    Gate rows use w1_oft_r, up rows use w3_oft_r. This helper is OFT-specific
    and intentionally lives under sglang.srt.oft.triton_ops, even though the
    call site is the MoE runner.
    """
    _validate_bf16("hidden_states", hidden_states)
    _validate_bf16("w13", w13)
    if w1_oft_r.dtype != torch.bfloat16 or w3_oft_r.dtype != torch.bfloat16:
        raise RuntimeError("w1_oft_r and w3_oft_r must be bf16")

    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not w13.is_contiguous():
        w13 = w13.contiguous()
    if not w1_oft_r.is_contiguous():
        w1_oft_r = w1_oft_r.contiguous()
    if not w3_oft_r.is_contiguous():
        w3_oft_r = w3_oft_r.contiguous()

    if hidden_states.dim() != 2:
        raise RuntimeError(f"hidden_states must be 2D, got {tuple(hidden_states.shape)}")
    if w13.dim() != 3:
        raise RuntimeError(f"w13 must be 3D, got {tuple(w13.shape)}")
    if w1_oft_r.dim() != 4 or w3_oft_r.dim() != 4:
        raise RuntimeError(
            "w1_oft_r and w3_oft_r must be 4D (experts, blocks, block_size, block_size)"
        )

    m, hidden = hidden_states.shape
    experts, n, hidden_from_w = w13.shape
    if hidden_from_w != hidden:
        raise RuntimeError(f"w13 hidden dim {hidden_from_w} != hidden_states dim {hidden}")
    if n % 2 != 0:
        raise RuntimeError(f"w13 output dim must be even, got {n}")
    if tuple(w1_oft_r.shape) != tuple(w3_oft_r.shape):
        raise RuntimeError(
            f"w1/w3 OFT shapes differ: {tuple(w1_oft_r.shape)} vs {tuple(w3_oft_r.shape)}"
        )
    if w1_oft_r.shape[0] != experts:
        raise RuntimeError(
            f"OFT expert count {w1_oft_r.shape[0]} != w13 expert count {experts}"
        )
    if w1_oft_r.shape[-2] != w1_oft_r.shape[-1]:
        raise RuntimeError(f"OFT blocks must be square, got {tuple(w1_oft_r.shape[-2:])}")

    block_size = w1_oft_r.shape[-1]
    validate_oft_block_size(block_size)
    blocks = w1_oft_r.shape[1]
    if blocks * block_size != hidden:
        raise RuntimeError(f"OFT blocks {blocks} * block_size {block_size} != hidden {hidden}")
    if block_m <= 0:
        raise RuntimeError(f"block_m must be positive, got {block_m}")
    if block_n <= 0:
        raise RuntimeError(f"block_n must be positive, got {block_n}")
    if group_size_m <= 0:
        raise RuntimeError(f"group_size_m must be positive, got {group_size_m}")

    top_k = topk_ids.shape[1]
    # NOTE: zeros, not empty. The kernel returns without writing when
    # expert_ids[program_m] < 0 (non-local expert under EP), so any uninitialized
    # bytes in those output rows would propagate as NaN/inf. Downstream code
    # filters those positions by topk_ids, but defensive zeroing is cheap and
    # keeps the helper safe for callers that read the full tensor.
    out = torch.zeros((m, top_k, n), device=hidden_states.device, dtype=hidden_states.dtype)

    em = sorted_token_ids.shape[0]
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n // 2, block_n), 2)

    _split_w13_oft_grouped_moe_kernel[grid](
        hidden_states,
        w13,
        w1_oft_r,
        w3_oft_r,
        out,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        M=m,
        K=hidden,
        N=n,
        HALF=n // 2,
        TOP_K=top_k,
        EM=em,
        BLOCKS=blocks,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


def _make_grouped_bmm_expert_offsets(
    *,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    experts: int,
    block_m: int,
) -> Tuple[torch.Tensor, int, int]:
    num_tokens = int(num_tokens_post_padded.item())
    active_m_blocks = triton.cdiv(num_tokens, block_m)
    active_expert_ids = expert_ids[:active_m_blocks]
    active_expert_ids = active_expert_ids[active_expert_ids >= 0].to(torch.int64)
    block_counts = torch.bincount(active_expert_ids, minlength=experts)
    padded_counts = block_counts * block_m
    expert_offsets = _make_static_grouped_bmm_expert_offsets(
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        experts=experts,
        block_m=block_m,
        max_m_blocks=active_m_blocks,
    )
    max_padded_tokens_per_expert = int(padded_counts.max().item())
    if max_padded_tokens_per_expert <= 0:
        raise RuntimeError("Grouped-BMM experiment needs at least one local expert block")
    return expert_offsets, max_padded_tokens_per_expert, active_m_blocks


def _make_static_grouped_bmm_expert_offsets(
    *,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    experts: int,
    block_m: int,
    max_m_blocks: int,
) -> torch.Tensor:
    expert_offsets = torch.empty(
        (experts,), dtype=torch.int64, device=expert_ids.device
    )
    _expert_offsets_from_block_ids_kernel[(experts,)](
        expert_ids,
        num_tokens_post_padded,
        expert_offsets,
        BLOCK_M=block_m,
        MAX_M_BLOCKS=max_m_blocks,
        BLOCKS_PER_TILE=128,
        num_warps=4,
        num_stages=3,
    )
    return expert_offsets


def _make_w13_grouped_bmm_weight(w13: torch.Tensor) -> torch.Tensor:
    half = w13.shape[1] // 2
    return torch.cat(
        [
            w13[:, :half, :].transpose(1, 2),
            w13[:, half:, :].transpose(1, 2),
        ],
        dim=0,
    ).contiguous()


def packed_bmm_split_w13_oft_grouped_moe(
    *,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w1_oft_r: torch.Tensor,
    w3_oft_r: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_m: int,
    expert_offsets: Optional[torch.Tensor] = None,
    max_padded_tokens_per_expert: Optional[int] = None,
    active_m_blocks: Optional[int] = None,
    w13_grouped_bmm_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pre-rotate/pack routed tokens, then use cuBLAS batched GEMM.

    This path favors large-M/prefill-like grouped MoE FC1 shapes: one Triton
    kernel rotates and packs routed token rows by expert, torch.bmm projects
    those packed rows through expert weights, and a second Triton kernel scatters
    the result back to (M, top_k, 2 * half_intermediate).
    """
    _validate_bf16("hidden_states", hidden_states)
    _validate_bf16("w13", w13)
    _validate_bf16("w1_oft_r", w1_oft_r)
    _validate_bf16("w3_oft_r", w3_oft_r)

    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not w13.is_contiguous():
        w13 = w13.contiguous()
    if not w1_oft_r.is_contiguous():
        w1_oft_r = w1_oft_r.contiguous()
    if not w3_oft_r.is_contiguous():
        w3_oft_r = w3_oft_r.contiguous()

    m, hidden = hidden_states.shape
    experts, n, hidden_from_w = w13.shape
    if hidden_from_w != hidden:
        raise RuntimeError(f"w13 hidden dim {hidden_from_w} != hidden_states dim {hidden}")
    if n % 2 != 0:
        raise RuntimeError(f"w13 output dim must be even, got {n}")
    if tuple(w1_oft_r.shape) != tuple(w3_oft_r.shape):
        raise RuntimeError(
            f"w1/w3 OFT shapes differ: {tuple(w1_oft_r.shape)} vs {tuple(w3_oft_r.shape)}"
        )
    if w1_oft_r.shape[0] != experts:
        raise RuntimeError(
            f"OFT expert count {w1_oft_r.shape[0]} != w13 expert count {experts}"
        )
    block_size = w1_oft_r.shape[-1]
    validate_oft_block_size(block_size)
    blocks = w1_oft_r.shape[1]
    if blocks * block_size != hidden:
        raise RuntimeError(f"OFT blocks {blocks} * block_size {block_size} != hidden {hidden}")

    top_k = topk_ids.shape[1]
    half = n // 2
    if (
        expert_offsets is None
        or max_padded_tokens_per_expert is None
        or active_m_blocks is None
    ):
        if _is_cuda_graph_capture_setup():
            active_m_blocks = triton.cdiv(sorted_token_ids.shape[0], block_m)
            max_padded_tokens_per_expert = triton.cdiv(m * top_k, block_m) * block_m
            expert_offsets = _make_static_grouped_bmm_expert_offsets(
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                experts=experts,
                block_m=block_m,
                max_m_blocks=active_m_blocks,
            )
        else:
            expert_offsets, max_padded_tokens_per_expert, active_m_blocks = (
                _make_grouped_bmm_expert_offsets(
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    experts=experts,
                    block_m=block_m,
                )
            )

    if not expert_offsets.is_contiguous():
        expert_offsets = expert_offsets.contiguous()
    if w13_grouped_bmm_weight is not None and not w13_grouped_bmm_weight.is_contiguous():
        w13_grouped_bmm_weight = w13_grouped_bmm_weight.contiguous()

    packed_inputs = torch.empty(
        (experts * 2, max_padded_tokens_per_expert, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    pack_grid = (active_m_blocks, blocks, 2)
    _pack_split_oft_grouped_bmm_inputs_kernel[pack_grid](
        hidden_states,
        w1_oft_r,
        w3_oft_r,
        packed_inputs,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        expert_offsets,
        M=m,
        K=hidden,
        TOP_K=top_k,
        EXPERTS=experts,
        BLOCKS=blocks,
        BLOCK_SIZE=block_size,
        MAX_PADDED_TOKENS_PER_EXPERT=max_padded_tokens_per_expert,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=3,
    )

    if w13_grouped_bmm_weight is None:
        projected = torch.empty(
            (experts * 2, max_padded_tokens_per_expert, half),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        torch.bmm(
            packed_inputs[:experts],
            w13[:, :half, :].transpose(1, 2),
            out=projected[:experts],
        )
        torch.bmm(
            packed_inputs[experts:],
            w13[:, half:, :].transpose(1, 2),
            out=projected[experts:],
        )
    else:
        projected = torch.bmm(packed_inputs, w13_grouped_bmm_weight)
    out = torch.zeros((m, top_k, n), device=hidden_states.device, dtype=hidden_states.dtype)

    block_n = 64
    unpack_grid = (active_m_blocks, triton.cdiv(half, block_n), 2)
    _unpack_split_oft_grouped_bmm_outputs_kernel[unpack_grid](
        projected,
        out,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        expert_offsets,
        M=m,
        N=n,
        HALF=half,
        TOP_K=top_k,
        EXPERTS=experts,
        MAX_PADDED_TOKENS_PER_EXPERT=max_padded_tokens_per_expert,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )
    return out


def _benchmark_config(m: int, experts: int) -> Dict[str, int]:
    if m <= experts:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


def _make_benchmark_inputs(
    *,
    m: int,
    hidden: int,
    half: int,
    experts: int,
    top_k: int,
    block_size: int,
) -> List[torch.Tensor]:
    hidden_states = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(experts, half * 2, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    blocks = hidden // block_size
    eye = torch.eye(block_size, device="cuda", dtype=torch.bfloat16)
    w1_oft_r = eye.expand(experts, blocks, block_size, block_size).clone()
    w3_oft_r = eye.expand(experts, blocks, block_size, block_size).clone()
    w1_oft_r.add_(torch.randn_like(w1_oft_r) * 0.005)
    w3_oft_r.add_(torch.randn_like(w3_oft_r) * 0.005)
    topk_ids = torch.randint(0, experts, (m, top_k), device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(m, top_k, device="cuda", dtype=torch.bfloat16)
    return [hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids, topk_weights]


def _bench_cuda(fn, *, warmup: int = 10, rep: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / rep


def _measure_or_error(fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        return out, _bench_cuda(fn), None
    except Exception as exc:  # benchmark must retain unsupported baseline rows
        return None, None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def _run_benchmark(json_path: Optional[str] = None) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for grouped_moe_rotate_project benchmark")

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(7)
    hidden = 2048
    half = 384
    experts = 128
    top_k = 8
    block_sizes = [4, 8, 16]
    ms = [1, 8, 32, 64, 256, 1024]
    rows: List[Dict[str, object]] = []
    failed = False

    header = (
        f"{'BS':>4} {'M':>6} {'variant':>12} {'us':>11} {'max_abs':>10}  note"
    )
    print("Grouped MoE OFT FC1 benchmark (lower us is better)")
    print(header)
    print("-" * len(header))
    for block_size in block_sizes:
        for m in ms:
            hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids, topk_weights = (
                _make_benchmark_inputs(
                    m=m,
                    hidden=hidden,
                    half=half,
                    experts=experts,
                    top_k=top_k,
                    block_size=block_size,
                )
            )
            config = _benchmark_config(m, experts)
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, config["BLOCK_SIZE_M"], experts
            )
            expert_offsets, max_padded_tokens_per_expert, active_m_blocks = (
                _make_grouped_bmm_expert_offsets(
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    experts=experts,
                    block_m=config["BLOCK_SIZE_M"],
                )
            )
            w13_grouped_bmm_weight = _make_w13_grouped_bmm_weight(w13)

            def run_direct() -> torch.Tensor:
                return fused_split_w13_oft_grouped_moe(
                    hidden_states=hidden_states,
                    w13=w13,
                    w1_oft_r=w1_oft_r,
                    w3_oft_r=w3_oft_r,
                    topk_ids=topk_ids,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    block_m=config["BLOCK_SIZE_M"],
                )

            def run_legacy() -> torch.Tensor:
                legacy = torch.empty(
                    m, top_k, half * 2, device="cuda", dtype=torch.bfloat16
                )
                for half_slice, oft_r in (
                    (slice(None, half), w1_oft_r),
                    (slice(half, None), w3_oft_r),
                ):
                    half_cache = torch.empty(
                        m, top_k, half, device="cuda", dtype=torch.bfloat16
                    )
                    invoke_fused_moe_kernel(
                        hidden_states,
                        w13[:, half_slice, :].contiguous(),
                        None,
                        half_cache,
                        None,
                        None,
                        None,
                        topk_weights,
                        topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        False,
                        top_k,
                        config,
                        compute_type=tl.bfloat16,
                        use_fp8_w8a8=False,
                        use_int8_w8a8=False,
                        use_int8_w8a16=False,
                        use_int4_w4a16=False,
                        per_channel_quant=False,
                        block_shape=None,
                        oft_r=oft_r,
                    )
                    legacy[..., half_slice].copy_(half_cache)
                return legacy

            def run_packed_bmm() -> torch.Tensor:
                return packed_bmm_split_w13_oft_grouped_moe(
                    hidden_states=hidden_states,
                    w13=w13,
                    w1_oft_r=w1_oft_r,
                    w3_oft_r=w3_oft_r,
                    topk_ids=topk_ids,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    block_m=config["BLOCK_SIZE_M"],
                    expert_offsets=expert_offsets,
                    max_padded_tokens_per_expert=max_padded_tokens_per_expert,
                    active_m_blocks=active_m_blocks,
                    w13_grouped_bmm_weight=w13_grouped_bmm_weight,
                )

            measured = {
                "legacy": _measure_or_error(run_legacy),
                "direct": _measure_or_error(run_direct),
                "packed_bmm": _measure_or_error(run_packed_bmm),
            }
            legacy_out = measured["legacy"][0]
            for variant, (out, us, error) in measured.items():
                max_abs = None
                if out is not None and legacy_out is not None:
                    max_abs = (out.float() - legacy_out.float()).abs().max().item()
                row = {
                    "BS": block_size,
                    "M": m,
                    "variant": variant,
                    "us": us,
                    "max_abs": max_abs,
                    "error": error,
                }
                rows.append(row)
                us_text = f"{us:11.2f}" if us is not None else f"{'--':>11}"
                err_text = (
                    f"{max_abs:10.5f}" if max_abs is not None else f"{'--':>10}"
                )
                print(
                    f"{block_size:>4} {m:>6} {variant:>12} "
                    f"{us_text} {err_text}  {error or ''}"
                )
                if error is None and max_abs is not None and max_abs > 3e-2:
                    failed = True

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {json_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark grouped-MoE OFT paths")
    parser.add_argument("--json", default=None, help="write complete rows to this file")
    args = parser.parse_args()
    raise SystemExit(_run_benchmark(args.json))

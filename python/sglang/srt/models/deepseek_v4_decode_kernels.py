from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    value = 1
    while value < x:
        value <<= 1
    return value


@triton.jit
def _indexer_score_decode_kernel(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    req_indices_ptr,
    group_count_ptr,
    active_ptr,
    out_ptr,
    MAX_GROUPS: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    block_t = tl.program_id(1)
    offs_t = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < MAX_GROUPS
    offs_d = tl.arange(0, BLOCK_D)

    req = tl.load(req_indices_ptr + b)
    group_count = tl.load(group_count_ptr + b)
    active = tl.load(active_ptr + b)
    score = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for h in tl.static_range(0, N_HEADS):
        dot = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for d0 in tl.static_range(0, HEAD_DIM, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr + (b * N_HEADS + h) * HEAD_DIM + d,
                mask=d < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            kv = tl.load(
                kv_cache_ptr + (req * MAX_GROUPS + offs_t[:, None]) * HEAD_DIM + d[None, :],
                mask=mask_t[:, None] & (d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            dot += tl.sum(kv * q[None, :], axis=1)
        dot_bf16 = dot.to(tl.bfloat16)
        relu_bf16 = tl.where(dot_bf16 > 0.0, dot_bf16, 0.0).to(tl.bfloat16)
        weight_bf16 = tl.load(weights_ptr + b * N_HEADS + h).to(tl.bfloat16)
        score += (relu_bf16 * weight_bf16).to(tl.bfloat16).to(tl.float32)

    valid = mask_t & (offs_t < group_count) & active
    score = tl.where(valid, score, -float("inf"))
    tl.store(out_ptr + b * MAX_GROUPS + offs_t, score, mask=mask_t)


def deepseek_v4_indexer_score_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    req_indices: torch.Tensor,
    group_count: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if q.dim() != 4 or q.shape[1] != 1:
        raise RuntimeError(
            f"deepseek_v4_indexer_score_decode expects q [batch, 1, heads, dim], got {tuple(q.shape)}"
        )
    if weights.shape[:2] != q.shape[:2] or weights.shape[-1] != q.shape[2]:
        raise RuntimeError(
            "deepseek_v4_indexer_score_decode weights shape mismatch: "
            f"q={tuple(q.shape)} weights={tuple(weights.shape)}"
        )
    q = q.contiguous()
    weights = weights.contiguous()
    batch, _, n_heads, head_dim = q.shape
    max_groups = kv_cache.shape[1]
    out = torch.empty((batch, 1, max_groups), device=q.device, dtype=q.dtype)
    block_t = 16
    block_d = 64
    grid = (batch, triton.cdiv(max_groups, block_t))
    _indexer_score_decode_kernel[grid](
        q,
        kv_cache,
        weights,
        req_indices.reshape(-1).contiguous(),
        group_count.reshape(-1).contiguous(),
        active.reshape(-1).contiguous(),
        out,
        MAX_GROUPS=max_groups,
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _window_topk_decode_kernel(
    out_ptr,
    start_pos_ptr,
    active_ptr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_W)
    mask = offs < WINDOW_SIZE
    start = tl.load(start_pos_ptr + b)
    active = tl.load(active_ptr + b)
    wrap = start % WINDOW_SIZE
    rotated = (offs + wrap + 1) % WINDOW_SIZE
    early = tl.where(offs <= start, offs, -1)
    topk = tl.where(start >= WINDOW_SIZE - 1, rotated, early)
    topk = tl.where(active, topk, -1)
    tl.store(out_ptr + b * WINDOW_SIZE + offs, topk, mask=mask)


def deepseek_v4_window_topk_decode(
    window_size: int,
    start_pos: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    batch = start_pos.numel()
    out = torch.empty((batch, 1, window_size), device=start_pos.device, dtype=torch.int64)
    block_w = _next_power_of_2(window_size)
    _window_topk_decode_kernel[(batch,)](
        out,
        start_pos.reshape(-1).contiguous(),
        active.reshape(-1).contiguous(),
        WINDOW_SIZE=window_size,
        BLOCK_W=block_w,
        num_warps=4,
    )
    return out


@triton.jit
def _compress_topk_decode_kernel(
    out_ptr,
    start_pos_ptr,
    active_ptr,
    RATIO: tl.constexpr,
    OFFSET: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    b = tl.program_id(0)
    block_g = tl.program_id(1)
    groups = block_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask = groups < MAX_GROUPS
    start = tl.load(start_pos_ptr + b)
    active = tl.load(active_ptr + b)
    group_count = (start + 1) // RATIO
    topk = tl.where((groups < group_count) & active, groups + OFFSET, -1)
    tl.store(out_ptr + b * MAX_GROUPS + groups, topk, mask=mask)


def deepseek_v4_compress_topk_decode(
    ratio: int,
    start_pos: torch.Tensor,
    offset: int,
    max_groups: int,
    active: torch.Tensor,
) -> torch.Tensor:
    batch = start_pos.numel()
    out = torch.empty((batch, 1, max_groups), device=start_pos.device, dtype=torch.int64)
    block_g = min(1024, _next_power_of_2(max_groups))
    _compress_topk_decode_kernel[(batch, triton.cdiv(max_groups, block_g))](
        out,
        start_pos.reshape(-1).contiguous(),
        active.reshape(-1).contiguous(),
        RATIO=ratio,
        OFFSET=int(offset),
        MAX_GROUPS=max_groups,
        BLOCK_G=block_g,
        num_warps=8 if block_g >= 512 else 4,
    )
    return out


@triton.jit
def _gather_active_cache_decode_kernel(
    cache_ptr,
    out_ptr,
    req_indices_ptr,
    active_ptr,
    TIME: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    b = tl.program_id(0)
    block_e = tl.program_id(1)
    offs = block_e * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < (TIME * DIM)
    req = tl.load(req_indices_ptr + b)
    active = tl.load(active_ptr + b)
    values = tl.load(
        cache_ptr + req * TIME * DIM + offs,
        mask=mask & active,
        other=0.0,
    )
    tl.store(out_ptr + b * TIME * DIM + offs, values, mask=mask)


def deepseek_v4_gather_active_cache_decode(
    cache: torch.Tensor,
    req_indices: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if cache.dim() != 3:
        raise RuntimeError(
            f"deepseek_v4_gather_active_cache_decode expects cache [slots, time, dim], got {tuple(cache.shape)}"
        )
    req_indices = req_indices.reshape(-1).contiguous()
    active = active.reshape(-1).contiguous()
    batch = req_indices.numel()
    time, dim = cache.shape[1], cache.shape[2]
    out = torch.empty((batch, time, dim), device=cache.device, dtype=cache.dtype)
    block_e = 1024
    _gather_active_cache_decode_kernel[(batch, triton.cdiv(time * dim, block_e))](
        cache,
        out,
        req_indices,
        active,
        TIME=time,
        DIM=dim,
        BLOCK_E=block_e,
        num_warps=8,
    )
    return out


dsv4_indexer_score_decode = deepseek_v4_indexer_score_decode
dsv4_window_topk_decode = deepseek_v4_window_topk_decode
dsv4_compress_topk_decode = deepseek_v4_compress_topk_decode
dsv4_gather_active_cache_decode = deepseek_v4_gather_active_cache_decode

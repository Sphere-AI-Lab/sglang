from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv4_update_cache_slots_kernel(
    cache_ptr,
    req_indices_ptr,
    write_idx_ptr,
    values_ptr,
    active_ptr,
    dim: tl.constexpr,
    cache_stride_b: tl.constexpr,
    cache_stride_t: tl.constexpr,
    cache_stride_d: tl.constexpr,
    values_stride_b: tl.constexpr,
    values_stride_d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < dim
    active = tl.load(active_ptr + b)

    req = tl.load(req_indices_ptr + b)
    write_idx = tl.load(write_idx_ptr + b)
    values = tl.load(
        values_ptr + b * values_stride_b + offs_d * values_stride_d,
        mask=mask_d & active,
        other=0.0,
    )
    tl.store(
        cache_ptr
        + req * cache_stride_b
        + write_idx * cache_stride_t
        + offs_d * cache_stride_d,
        values,
        mask=mask_d & active,
    )


def _next_power_of_2(x: int) -> int:
    value = 1
    while value < x:
        value <<= 1
    return value


def deepseek_v4_update_cache_slots_(
    cache: torch.Tensor,
    req_indices: torch.Tensor,
    write_idx: torch.Tensor,
    values: torch.Tensor,
    active: torch.Tensor,
) -> None:
    if values.dim() == 3 and values.shape[1] == 1:
        values = values.squeeze(1)
    if values.dim() != 2:
        raise RuntimeError(
            f"deepseek_v4_update_cache_slots_ expects values [batch, dim], got {tuple(values.shape)}"
        )
    if cache.dim() != 3:
        raise RuntimeError(
            f"deepseek_v4_update_cache_slots_ expects cache [slots, time, dim], got {tuple(cache.shape)}"
        )

    bsz, dim = values.shape
    if req_indices.device != cache.device or req_indices.dtype != torch.int64:
        raise RuntimeError(
            "deepseek_v4_update_cache_slots_ requires req_indices to be int64 on "
            f"{cache.device}, got device={req_indices.device} dtype={req_indices.dtype}"
        )
    if write_idx.device != cache.device or write_idx.dtype != torch.int64:
        raise RuntimeError(
            "deepseek_v4_update_cache_slots_ requires write_idx to be int64 on "
            f"{cache.device}, got device={write_idx.device} dtype={write_idx.dtype}"
        )
    if active.device != cache.device or active.dtype != torch.bool:
        raise RuntimeError(
            "deepseek_v4_update_cache_slots_ requires active to be bool on "
            f"{cache.device}, got device={active.device} dtype={active.dtype}"
        )
    req_indices = req_indices.reshape(-1)
    write_idx = write_idx.reshape(-1)
    active = active.reshape(-1)
    if req_indices.numel() != bsz or active.numel() != bsz:
        raise RuntimeError(
            "deepseek_v4_update_cache_slots_ batch mismatch: "
            f"values={tuple(values.shape)} req_indices={tuple(req_indices.shape)} "
            f"active={tuple(active.shape)}"
        )
    if write_idx.numel() == 1 and bsz != 1:
        write_idx = write_idx.expand(bsz)
    if write_idx.numel() != bsz:
        raise RuntimeError(
            "deepseek_v4_update_cache_slots_ write_idx mismatch: "
            f"values={tuple(values.shape)} write_idx={tuple(write_idx.shape)}"
        )

    req_indices = req_indices.contiguous()
    write_idx = write_idx.contiguous()
    active = active.contiguous()
    values = values.contiguous()

    block_d = min(1024, _next_power_of_2(dim))
    num_warps = 8 if block_d >= 512 else 4
    grid = (bsz, triton.cdiv(dim, block_d))
    _dsv4_update_cache_slots_kernel[grid](
        cache,
        req_indices,
        write_idx,
        values,
        active,
        dim,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        values.stride(0),
        values.stride(1),
        BLOCK_D=block_d,
        num_warps=num_warps,
    )


dsv4_update_cache_slots_ = deepseek_v4_update_cache_slots_

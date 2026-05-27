"""Single-adapter, un-segmented OFT block-diagonal rotation (forward).

The kernel in `sgemm_oft_r.py` supports per-request adapter selection via
per-segment metadata (`seg_indptr`, `weight_indices`, `oft_block_sizes`) —
useful when different requests in the batch use different OFT adapters,
but pure overhead when every token shares one adapter (the common case in
serving and RL rollout, including DUMMY_OFT identity benchmarking).

This file holds the un-segmented variant. Differences from the general
kernel:
  - Grid is 2-D `(num_blocks * cdiv(total_tokens, BLOCK_S), num_slices)`
    instead of 3-D `(num_blocks * num_token_tiles, num_slices, num_segments)`.
  - `BLOCK_S=64` (vs 16) since tokens are now contiguous and aligned —
    higher utilization on decode batches.
  - `adapter_idx` and `block_size_val` are 0-d int tensor pointers
    (cuda-graph-rebindable), read once per program rather than via
    per-segment indirection.
  - `total_tokens` is `tl.constexpr`, fixed at launch — under cuda graphs
    the captured grid covers the captured maximum batch size.

Naming: `gemm_oft_r_fwd`. The leading `s` of the existing
`sgemm_oft_r_fwd` referred to "segmented" (the kernel's own docstring
calls itself a "segmented block-diagonal OFT rotation"); since this
variant is explicitly un-segmented, the prefix is dropped.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_oft_r_kernel(
    x_ptr,
    output_ptr,
    weights_ptr,
    adapter_idx_ptr,
    block_size_val_ptr,
    input_dim,
    weights_stride_0,
    weights_stride_1,
    weights_stride_2,
    weights_stride_3,
    x_stride_0,
    output_stride_0,
    total_tokens: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-adapter OFT block-diagonal rotation.

    Grid: (num_blocks * cdiv(total_tokens, BLOCK_S), num_slices)
      axis 0: block_idx * num_token_tiles + token_tile_idx
      axis 1: slice_id (0 standard; 0/1 gate_up; 0/1/2 qkv)

    For each slice s in [0, num_slices):
      output[:, s*input_dim:(s+1)*input_dim] = x @ R_s  (block-diagonal)
    where R_s = weights[adapter_idx, s*num_blocks:(s+1)*num_blocks, :, :].

    When block_size_val == 0, the kernel performs an identity passthrough
    (copy `x` into the corresponding output slice). Same contract as the
    general kernel for the "no adapter" case.
    """
    pid_0 = tl.program_id(0)
    slice_id = tl.program_id(1)

    num_token_tiles = tl.cdiv(total_tokens, BLOCK_S)
    block_idx = pid_0 // num_token_tiles
    token_tile_idx = pid_0 % num_token_tiles

    if block_idx >= num_blocks:
        return

    token_offset = token_tile_idx * BLOCK_S
    if token_offset >= total_tokens:
        return

    adapter_idx = tl.load(adapter_idx_ptr)
    block_size_val = tl.load(block_size_val_ptr)

    s_offsets = tl.arange(0, BLOCK_S)
    actual_s = token_offset + s_offsets
    s_mask = actual_s < total_tokens

    c_offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_SIZE)

    x_base = x_ptr + block_idx * BLOCK_SIZE
    out_base = output_ptr + slice_id * input_dim + block_idx * BLOCK_SIZE

    if block_size_val == 0:
        x_vals = tl.load(
            x_base + actual_s[:, None] * x_stride_0 + c_offsets[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )
        tl.store(
            out_base + actual_s[:, None] * output_stride_0 + c_offsets[None, :],
            x_vals,
            mask=s_mask[:, None],
        )
        return

    weight_block_idx = slice_id * num_blocks + block_idx
    R_base = (
        weights_ptr
        + adapter_idx * weights_stride_0
        + weight_block_idx * weights_stride_1
    )

    if BLOCK_SIZE >= 16:
        x_tile = tl.load(
            x_base + actual_s[:, None] * x_stride_0 + k_offsets[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )
        R_block = tl.load(
            R_base
            + k_offsets[:, None] * weights_stride_2
            + c_offsets[None, :] * weights_stride_3,
        )
        out = tl.dot(x_tile, R_block, input_precision="ieee")
    else:
        # Element-wise loop for tiny block sizes (block_size < 16).
        # Matches the general kernel's small-block fallback bit-for-bit.
        acc = tl.zeros((BLOCK_S, BLOCK_SIZE), dtype=tl.float32)
        for k in range(BLOCK_SIZE):
            x_col = tl.load(
                x_base + actual_s * x_stride_0 + k,
                mask=s_mask,
                other=0.0,
            ).to(tl.float32)
            R_row = tl.load(
                R_base + k * weights_stride_2 + c_offsets * weights_stride_3,
            ).to(tl.float32)
            acc += x_col[:, None] * R_row[None, :]
        out = acc

    tl.store(
        out_base + actual_s[:, None] * output_stride_0 + c_offsets[None, :],
        out.to(x_ptr.dtype.element_ty),
        mask=s_mask[:, None],
    )


def gemm_oft_r_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    adapter_idx_t: torch.Tensor,
    block_size_val_t: torch.Tensor,
    num_slices: int = 1,
    BLOCK_S: int = 64,
) -> torch.Tensor:
    """Single-adapter OFT rotation launcher (forward).

    Args:
        x: (total_tokens, input_dim). Under cuda graphs this is the
            captured maximum batch size; trailing padding rows are
            processed and harmlessly written through.
        weights: (num_ofts, num_slices * num_blocks, block_size, block_size)
            precomputed R buffers (same layout as the general kernel).
        adapter_idx_t: 0-d int tensor with the active adapter slot.
            Tensor (not int) so the captured graph stays valid across
            replays when the active adapter changes between batches.
        block_size_val_t: 0-d int tensor. 0 means identity passthrough.
            Typically `oft_block_sizes[adapter_idx_t]`.
        num_slices: 1 (standard linear), 2 (gate_up fused), 3 (qkv fused).
        BLOCK_S: token-tile size. 64 is a good default for H100 decode;
            smaller values reduce utilization, larger values cost occupancy.

    Returns:
        (total_tokens, num_slices * input_dim) rotated output.
    """
    total_tokens, input_dim = x.shape
    if weights.numel() == 0:
        return x.repeat(1, num_slices)

    num_ofts, total_blocks_buf, block_size, _ = weights.shape
    if block_size == 0 or total_blocks_buf == 0:
        return x.repeat(1, num_slices)

    if input_dim % block_size != 0:
        raise ValueError(
            f"OFT input_dim ({input_dim}) must be divisible by block_size ({block_size})"
        )

    num_blocks = input_dim // block_size

    output = torch.empty(
        (total_tokens, num_slices * input_dim), device=x.device, dtype=x.dtype
    )

    BLOCK_SIZE = block_size
    num_token_tiles = triton.cdiv(total_tokens, BLOCK_S)
    grid = (num_blocks * num_token_tiles, num_slices)

    _gemm_oft_r_kernel[grid](
        x,
        output,
        weights,
        adapter_idx_t,
        block_size_val_t,
        input_dim,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        x.stride(0),
        output.stride(0),
        total_tokens=total_tokens,
        num_blocks=num_blocks,
        BLOCK_S=BLOCK_S,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

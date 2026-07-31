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

# Shared memory available to one Triton program on sm_90 (H100), in bytes.
# Measured, not documented: Triton reports it verbatim in the OutOfResources it
# raises ("Hardware limit: 232448").
OFT_SMEM_BUDGET = 232448
# At or below this the kernel is left exactly as it shipped -- one K iteration,
# one N iteration -- because these block sizes always fit and the untiled code
# is what every existing deployment has been running.
OFT_UNTILED_MAX_BS = 128
# Tile widths for the contraction (K) and the output columns (N) above that.
OFT_TILE_K = 128
OFT_TILE_N = 128
# Triton software-pipelines the inner loop, so the staged operands are live for
# several iterations at once. Three is the default `num_stages` and the
# conservative assumption: overestimating shrinks the tile, which costs a little
# speed, while underestimating brings back the launch failure this file exists
# to remove.
_PIPELINE_STAGES = 3


def _tiled_smem_bytes(block_s, tile_k, tile_n, itemsize=2, stages=_PIPELINE_STAGES):
    """Bytes staged per program: the x tile plus the R tile, times the pipeline.

    Note what is absent: `block_size`. That is the whole point -- the footprint
    depends only on the tile widths, so a bigger OFT block costs more loop
    iterations rather than more shared memory.
    """
    return stages * itemsize * (block_s * tile_k + tile_k * tile_n)


def _pick_tiles(block_size, block_s):
    """(tile_k, tile_n) for this block size: the largest that fit the budget.

    Returns `(block_size, block_size)` at or below OFT_UNTILED_MAX_BS, which
    makes both loops single-iteration and the emitted code equivalent to the
    untiled original -- that equivalence is the no-regression contract.

    Halving only tile_k, never tile_n: tile_n also sets the accumulator width,
    and the accumulator lives in registers rather than shared memory, so
    shrinking it would trade a shared-memory saving for nothing.
    """
    if block_size <= OFT_UNTILED_MAX_BS:
        return block_size, block_size
    tile_n = min(OFT_TILE_N, block_size)
    tile_k = min(OFT_TILE_K, block_size)
    while tile_k > 16 and _tiled_smem_bytes(block_s, tile_k, tile_n) > OFT_SMEM_BUDGET:
        tile_k //= 2
    return tile_k, tile_n


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
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Single-adapter OFT block-diagonal rotation.

    Grid: (num_blocks * cdiv(total_tokens, BLOCK_S) * (BLOCK_SIZE // TILE_N),
           num_slices)
      axis 0: (block_idx * num_token_tiles + token_tile_idx) * num_col_tiles
              + col_tile_idx
      axis 1: slice_id (0 standard; 0/1 gate_up; 0/1/2 qkv)

    Each program owns one BLOCK_S x TILE_N patch of the output and walks the
    contraction in TILE_K steps. At BLOCK_SIZE <= OFT_UNTILED_MAX_BS the
    launcher sets TILE_K = TILE_N = BLOCK_SIZE, both loops run once, and this is
    the original one-shot `tl.dot`.

    For each slice s in [0, num_slices):
      output[:, s*input_dim:(s+1)*input_dim] = x @ R_s  (block-diagonal)
    where R_s = weights[adapter_idx, s*num_blocks:(s+1)*num_blocks, :, :].

    When block_size_val == 0, the kernel performs an identity passthrough
    (copy `x` into the corresponding output slice). Same contract as the
    general kernel for the "no adapter" case.
    """
    pid_0 = tl.program_id(0)
    slice_id = tl.program_id(1)

    num_col_tiles: tl.constexpr = BLOCK_SIZE // TILE_N
    num_token_tiles = tl.cdiv(total_tokens, BLOCK_S)
    tiles_per_block = num_token_tiles * num_col_tiles

    block_idx = pid_0 // tiles_per_block
    rem = pid_0 % tiles_per_block
    token_tile_idx = rem // num_col_tiles
    col_tile_idx = rem % num_col_tiles

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

    # Columns of the rotation block this program is responsible for. At
    # TILE_N == BLOCK_SIZE there is one tile and this is `arange(0, BLOCK_SIZE)`,
    # exactly as before.
    col_offset = col_tile_idx * TILE_N
    c_offsets = col_offset + tl.arange(0, TILE_N)

    x_base = x_ptr + block_idx * BLOCK_SIZE
    out_base = output_ptr + slice_id * input_dim + block_idx * BLOCK_SIZE

    if block_size_val == 0:
        # `block_size_val` is a runtime load, so this branch is compiled
        # whether or not it ever executes and its staged tile counts against
        # the same budget as the dot branch. Reading only this program's TILE_N
        # columns is what keeps that true at large BLOCK_SIZE.
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
        acc = tl.zeros((BLOCK_S, TILE_N), dtype=tl.float32)
        for k0 in range(0, BLOCK_SIZE, TILE_K):
            k_offsets = k0 + tl.arange(0, TILE_K)
            x_tile = tl.load(
                x_base + actual_s[:, None] * x_stride_0 + k_offsets[None, :],
                mask=s_mask[:, None],
                other=0.0,
            )
            R_tile = tl.load(
                R_base
                + k_offsets[:, None] * weights_stride_2
                + c_offsets[None, :] * weights_stride_3,
            )
            acc += tl.dot(x_tile, R_tile, input_precision="ieee")
        out = acc
    else:
        # Element-wise loop for tiny block sizes (block_size < 16).
        # Matches the general kernel's small-block fallback bit-for-bit.
        # TILE_N == BLOCK_SIZE here, so `c_offsets` spans the whole block.
        acc = tl.zeros((BLOCK_S, TILE_N), dtype=tl.float32)
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
    # At BLOCK_SIZE <= OFT_UNTILED_MAX_BS this returns (BLOCK_SIZE, BLOCK_SIZE):
    # one column tile, one K step, and the grid below reduces to what it was.
    TILE_K, TILE_N = _pick_tiles(BLOCK_SIZE, BLOCK_S)
    num_token_tiles = triton.cdiv(total_tokens, BLOCK_S)
    num_col_tiles = BLOCK_SIZE // TILE_N
    grid = (num_blocks * num_token_tiles * num_col_tiles, num_slices)

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
        TILE_K=TILE_K,
        TILE_N=TILE_N,
    )

    return output

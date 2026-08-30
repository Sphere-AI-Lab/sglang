"""Single-adapter OFT block-diagonal rotation (backward).

Forward: y = x @ R  (block-diagonal; see `gemm_oft_r.py` for the layout).
Backward identities:
  grad_x_block[t, k]    = sum_s grad_y[t, s*D + b*BS + c] * R[a, s*nB+b, k, c]
                        = sum_s grad_y_slice[t, b, c] @ R_slice[b, c, k]^T
  grad_R[a, s*nB+b, k, c] = sum_t x[t, b*BS+k] * grad_y[t, s*D + b*BS + c]
                        = (x_block[:, b, k]).T @ grad_y_slice[:, b, c]

For identity adapters (block_size_val == 0):
  grad_x_block[t, k] = sum_s grad_y[t, s*D + b*BS + k]   (just sum slices)
  grad_R              = N/A (no R for identity)

Two kernels, both un-segmented (single-adapter):
  - `_grad_x_kernel`  computes grad_x: grid (num_blocks * cdiv(T, BLOCK_S),)
    Loops over slices internally so the slice contributions accumulate.
  - `_grad_R_kernel`  computes grad_R: grid (num_blocks, num_slices).
    Each program reduces over the token dimension via TILE_T tiles.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grad_x_kernel(
    grad_y_ptr,
    weights_ptr,
    grad_x_ptr,
    adapter_idx_ptr,
    block_size_val_ptr,
    input_dim,
    weights_stride_0,
    weights_stride_1,
    weights_stride_2,
    weights_stride_3,
    grad_y_stride_0,
    grad_x_stride_0,
    total_tokens: tl.constexpr,
    num_slices: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-adapter grad_x for block-diagonal OFT rotation.

    Grid: (num_blocks * cdiv(total_tokens, BLOCK_S),)
      pid_0 = block_idx * num_token_tiles + token_tile_idx

    Computes:
      grad_x[:, b*BS:(b+1)*BS] = sum_s grad_y[:, s*D + b*BS:s*D + (b+1)*BS]
                                       @ R[a, s*num_blocks + b, :, :]^T
    Identity (block_size_val == 0): the rotation is a per-slice copy, so
    grad_x reduces to summing the slice contributions of grad_y.
    """
    pid_0 = tl.program_id(0)

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

    gx_base = grad_x_ptr + block_idx * BLOCK_SIZE

    if block_size_val == 0:
        acc = tl.zeros((BLOCK_S, BLOCK_SIZE), dtype=tl.float32)
        for s in range(num_slices):
            gy_addr = grad_y_ptr + s * input_dim + block_idx * BLOCK_SIZE
            gy_tile = tl.load(
                gy_addr + actual_s[:, None] * grad_y_stride_0 + c_offsets[None, :],
                mask=s_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += gy_tile
        tl.store(
            gx_base + actual_s[:, None] * grad_x_stride_0 + c_offsets[None, :],
            acc.to(grad_y_ptr.dtype.element_ty),
            mask=s_mask[:, None],
        )
        return

    # General case: accumulate sum_s grad_y_slice @ R_slice^T over slices.
    acc = tl.zeros((BLOCK_S, BLOCK_SIZE), dtype=tl.float32)
    for s in range(num_slices):
        gy_addr = grad_y_ptr + s * input_dim + block_idx * BLOCK_SIZE
        gy_tile = tl.load(
            gy_addr + actual_s[:, None] * grad_y_stride_0 + c_offsets[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )
        weight_block_idx = s * num_blocks + block_idx
        R_base = (
            weights_ptr
            + adapter_idx * weights_stride_0
            + weight_block_idx * weights_stride_1
        )
        # R_block: (BS_k, BS_c) — load it transposed-readable so that
        # tl.dot(grad_y_slice, R^T) becomes tl.dot(grad_y_slice, R_block.T).
        R_block = tl.load(
            R_base
            + k_offsets[:, None] * weights_stride_2
            + c_offsets[None, :] * weights_stride_3,
        )
        # gy_tile: (BLOCK_S, BS_c), R_block.T: (BS_c, BS_k) -> acc += (BLOCK_S, BS_k)
        if BLOCK_SIZE >= 16:
            acc += tl.dot(gy_tile, tl.trans(R_block), input_precision="ieee")
        else:
            for kk in range(BLOCK_SIZE):
                gy_col = tl.load(
                    gy_addr + actual_s * grad_y_stride_0 + kk,
                    mask=s_mask,
                    other=0.0,
                ).to(tl.float32)
                R_row = tl.load(
                    R_base + k_offsets * weights_stride_2 + kk * weights_stride_3,
                ).to(tl.float32)
                acc += gy_col[:, None] * R_row[None, :]

    tl.store(
        gx_base + actual_s[:, None] * grad_x_stride_0 + k_offsets[None, :],
        acc.to(grad_y_ptr.dtype.element_ty),
        mask=s_mask[:, None],
    )


@triton.jit
def _grad_R_kernel(
    x_ptr,
    grad_y_ptr,
    grad_R_ptr,
    adapter_idx_ptr,
    grad_R_stride_0,
    grad_R_stride_1,
    grad_R_stride_2,
    grad_R_stride_3,
    x_stride_0,
    grad_y_stride_0,
    input_dim,
    total_tokens: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_T: tl.constexpr,
):
    """Single-adapter grad_R for block-diagonal OFT rotation.

    Grid: (num_blocks, num_slices)

    grad_R[a, s*num_blocks + b, k, c] = sum_t x[t, b*BS+k] * grad_y[t, s*D + b*BS + c]
    """
    block_idx = tl.program_id(0)
    slice_id = tl.program_id(1)

    adapter_idx = tl.load(adapter_idx_ptr)

    col_base_x = block_idx * BLOCK_SIZE
    col_base_gy = slice_id * input_dim + block_idx * BLOCK_SIZE

    k_offsets = tl.arange(0, BLOCK_SIZE)
    c_offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for t_start in range(0, total_tokens, TILE_T):
        t_offsets = t_start + tl.arange(0, TILE_T)
        t_mask = t_offsets < total_tokens

        x_tile = tl.load(
            x_ptr + t_offsets[:, None] * x_stride_0 + col_base_x + k_offsets[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )
        gy_tile = tl.load(
            grad_y_ptr + t_offsets[:, None] * grad_y_stride_0 + col_base_gy + c_offsets[None, :],
            mask=t_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(tl.trans(x_tile), gy_tile, input_precision="ieee")

    weight_block_idx = slice_id * num_blocks + block_idx
    out_base = (
        grad_R_ptr
        + adapter_idx * grad_R_stride_0
        + weight_block_idx * grad_R_stride_1
    )
    tl.store(
        out_base + k_offsets[:, None] * grad_R_stride_2 + c_offsets[None, :] * grad_R_stride_3,
        acc.to(grad_R_ptr.dtype.element_ty),
    )


def gemm_oft_r_bwd_grad_x(
    grad_y: torch.Tensor,
    weights: torch.Tensor,
    adapter_idx_t: torch.Tensor,
    block_size_val_t: torch.Tensor,
    num_slices: int = 1,
    BLOCK_S: int = 64,
) -> torch.Tensor:
    """Single-adapter grad_x launcher.

    Args:
        grad_y: (total_tokens, num_slices * input_dim) upstream gradient.
        weights: (num_ofts, num_slices * num_blocks, block_size, block_size).
        adapter_idx_t: 0-d int tensor — active adapter slot.
        block_size_val_t: 0-d int tensor — 0 means identity (slice-sum).
        num_slices: 1, 2, or 3.
        BLOCK_S: token-tile size.

    Returns:
        grad_x: (total_tokens, input_dim).
    """
    total_tokens, total_out_dim = grad_y.shape
    if total_out_dim % num_slices != 0:
        raise ValueError(
            f"grad_y dim {total_out_dim} not divisible by num_slices {num_slices}"
        )
    input_dim = total_out_dim // num_slices

    if weights.numel() == 0:
        # No weights — fall back to identity case (sum slices).
        return grad_y.view(total_tokens, num_slices, input_dim).sum(dim=1)

    _, _, block_size, _ = weights.shape
    if input_dim % block_size != 0:
        raise ValueError(
            f"input_dim {input_dim} not divisible by block_size {block_size}"
        )
    num_blocks = input_dim // block_size

    grad_x = torch.empty(
        (total_tokens, input_dim), device=grad_y.device, dtype=grad_y.dtype
    )

    BLOCK_SIZE = block_size
    num_token_tiles = triton.cdiv(total_tokens, BLOCK_S)
    grid = (num_blocks * num_token_tiles,)

    _grad_x_kernel[grid](
        grad_y,
        weights,
        grad_x,
        adapter_idx_t,
        block_size_val_t,
        input_dim,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        weights.stride(3),
        grad_y.stride(0),
        grad_x.stride(0),
        total_tokens=total_tokens,
        num_slices=num_slices,
        num_blocks=num_blocks,
        BLOCK_S=BLOCK_S,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_x


def gemm_oft_r_bwd_grad_R(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    adapter_idx_t: torch.Tensor,
    weights_shape: tuple,
    num_slices: int = 1,
    TILE_T: int = 32,
) -> torch.Tensor:
    """Single-adapter grad_R launcher.

    Caller must NOT call this when the active adapter has block_size_val == 0
    (identity has no R to gradient). It's the caller's responsibility to
    branch on `block_size_val` and skip this call for identity adapters.

    Args:
        x: (total_tokens, input_dim) — saved input from forward.
        grad_y: (total_tokens, num_slices * input_dim) upstream gradient.
        adapter_idx_t: 0-d int tensor — active adapter slot.
        weights_shape: (num_ofts, num_slices * num_blocks, block_size, block_size).
            We allocate grad_R in this shape; only the active adapter's slot
            is written.
        num_slices: 1, 2, or 3.
        TILE_T: token-axis tile size for the reduction. Powers of 2 in
            [16, 128] work well; 32 is a sane default for decode-style T.

    Returns:
        grad_R: same shape as `weights_shape`. All slots except the active
        adapter are uninitialized (caller should mask them out or zero the
        buffer beforehand).
    """
    total_tokens, input_dim = x.shape
    expected_gy_dim = num_slices * input_dim
    if grad_y.shape != (total_tokens, expected_gy_dim):
        raise ValueError(
            f"grad_y shape {tuple(grad_y.shape)} doesn't match expected "
            f"({total_tokens}, {expected_gy_dim})"
        )

    num_ofts, total_blocks_buf, block_size, block_size_2 = weights_shape
    if block_size_2 != block_size:
        raise ValueError(f"weights_shape block dims must be square, got {weights_shape}")
    if total_blocks_buf != num_slices * (input_dim // block_size):
        raise ValueError(
            f"weights_shape total blocks {total_blocks_buf} doesn't match "
            f"num_slices * num_blocks = {num_slices} * {input_dim // block_size}"
        )
    num_blocks = input_dim // block_size

    grad_R = torch.zeros(weights_shape, device=x.device, dtype=x.dtype)

    grid = (num_blocks, num_slices)

    _grad_R_kernel[grid](
        x,
        grad_y,
        grad_R,
        adapter_idx_t,
        grad_R.stride(0),
        grad_R.stride(1),
        grad_R.stride(2),
        grad_R.stride(3),
        x.stride(0),
        grad_y.stride(0),
        input_dim,
        total_tokens=total_tokens,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
        TILE_T=TILE_T,
    )

    return grad_R


def gemm_oft_r_bwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    grad_y: torch.Tensor,
    adapter_idx_t: torch.Tensor,
    block_size_val_t: torch.Tensor,
    num_slices: int = 1,
    BLOCK_S: int = 64,
    TILE_T: int = 32,
) -> tuple:
    """Convenience: compute grad_x and grad_R together for the active adapter.

    Returns (grad_x, grad_R). When the active adapter is identity
    (`block_size_val_t == 0`), grad_R is a zero tensor of weights' shape
    (no learnable rotation to gradient).
    """
    grad_x = gemm_oft_r_bwd_grad_x(
        grad_y=grad_y,
        weights=weights,
        adapter_idx_t=adapter_idx_t,
        block_size_val_t=block_size_val_t,
        num_slices=num_slices,
        BLOCK_S=BLOCK_S,
    )

    block_size_val = int(block_size_val_t.item()) if block_size_val_t.numel() else 0
    if block_size_val == 0:
        grad_R = torch.zeros_like(weights)
    else:
        grad_R = gemm_oft_r_bwd_grad_R(
            x=x,
            grad_y=grad_y,
            adapter_idx_t=adapter_idx_t,
            weights_shape=tuple(weights.shape),
            num_slices=num_slices,
            TILE_T=TILE_T,
        )

    return grad_x, grad_R

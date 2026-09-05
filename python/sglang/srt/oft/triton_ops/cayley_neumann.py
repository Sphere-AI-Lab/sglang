"""Triton-accelerated Cayley-Neumann transform for OFT rotation matrices.

R = I + 2Q + 2Q² + 2Q³ + Q⁴  from skew-symmetric Q.

Single-program-per-block approach: each triton program handles one
(block_size x block_size) block entirely in registers, fusing all
matrix multiplications and accumulations.

For small blocks (fp16: <= 128, fp32: <= 32), uses a triton kernel (3-4x faster).
For larger blocks, falls back to torch (register pressure makes triton slower).

Hardcoded for NUM_TERMS=5 (the standard OFT configuration).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.oft.torch_ops.oft_ops import cayley_neumann as _torch_cayley_neumann

NUM_TERMS = 5  # R = I + 2Q + 2Q² + 2Q³ + Q⁴


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _cayley_fwd_kernel(
    Q_ptr, R_ptr,
    stride_b, stride_r, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute R = I + 2Q + 2Q² + 2Q³ + Q⁴ for one block. Grid: (num_blocks,)"""
    bid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    Q_base = Q_ptr + bid * stride_b
    Q = tl.load(Q_base + rows[:, None] * stride_r + cols[None, :] * stride_c)

    # Accumulate R
    I = (rows[:, None] == cols[None, :]).to(Q.dtype)
    R = I + 2.0 * Q

    # Q² = Q @ Q  (cast dot output back to input dtype for next dot)
    Q2 = tl.dot(Q, Q, input_precision="ieee").to(Q.dtype)
    R = R + 2.0 * Q2

    # Q³ = Q² @ Q
    Q3 = tl.dot(Q2, Q, input_precision="ieee").to(Q.dtype)
    R = R + 2.0 * Q3

    # Q⁴ = Q³ @ Q
    Q4 = tl.dot(Q3, Q, input_precision="ieee").to(Q.dtype)
    R = R + Q4

    R_base = R_ptr + bid * stride_b
    tl.store(
        R_base + rows[:, None] * stride_r + cols[None, :] * stride_c,
        R.to(Q.dtype),
    )


def cayley_neumann_fwd(Q_skew: torch.Tensor, num_terms: int = 5) -> torch.Tensor:
    """Triton Cayley-Neumann forward (block_size must be <= 128 for fp16, <= 32 for fp32)."""
    assert num_terms == NUM_TERMS, f"Only num_terms={NUM_TERMS} supported, got {num_terms}"
    num_blocks, block_size, _ = Q_skew.shape
    if block_size < 16:
        return _torch_cayley_neumann(Q_skew, num_terms)
    R = torch.empty_like(Q_skew)
    _cayley_fwd_kernel[(num_blocks,)](
        Q_skew, R,
        Q_skew.stride(0), Q_skew.stride(1), Q_skew.stride(2),
        BLOCK_SIZE=block_size,
    )
    return R

"""Fused FP8 block-wise dequant triton kernel for OFT parity mode.

Semantics of the math this replaces
-----------------------------------
    out[..., m, n] = fp32(w_fp8[..., m, n]) * scale[..., m//BH, n//BW]
    cast to out_dtype (bf16 / fp16)

    - ``BH`` = block-height = M / scale.shape[-2]
    - ``BW`` = block-width  = N / scale.shape[-1]

Why this exists
---------------
The PyTorch reference implementation (``peft/fp8_utils.py::dequant_fp8`` in
Megatron-Bridge, and ``_bridge_dequant_fp8`` in sglang) does:

    w = w_fp8.float().reshape(E, sr, BH, sc, BW)   # full fp32 copy
    w = w * scale.float().view(E, sr, 1, sc, 1)
    return w.reshape(E, M, N).to(out_dtype)

For a 200B-parameter MoE tensor that's 800GB of transient fp32, three full
memory passes (fp8→fp32, multiply, cast-to-bf16). This kernel does it in
one pass with no fp32 materialization — FP8 loaded to fp32 in-register,
multiplied by one scale per tile, stored as bf16.

Used as a drop-in fast path inside ``_bridge_dequant_fp8`` when the
tensors are on CUDA and triton is available; falls back to the PyTorch
reshape-multiply otherwise.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _dequant_fp8_block_kernel(
        w_ptr,
        scale_ptr,
        out_ptr,
        # Shape (batched): E may be 1 for dense, >1 for MoE
        E,
        M,
        N,
        SR,
        SC,
        # Strides — batch strides are 0 for dense (broadcast ignored by grid=1).
        w_stride_e,
        w_stride_m,
        w_stride_n,
        s_stride_e,
        s_stride_m,
        s_stride_n,
        o_stride_e,
        o_stride_m,
        o_stride_n,
        BH: tl.constexpr,
        BW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Grid: (E, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))."""
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N
        tile_mask = m_mask[:, None] & n_mask[None, :]

        w_fp8 = tl.load(
            w_ptr
            + pid_e * w_stride_e
            + m_offs[:, None] * w_stride_m
            + n_offs[None, :] * w_stride_n,
            mask=tile_mask,
            other=0.0,
        )
        w_f32 = w_fp8.to(tl.float32)

        # Per-element scale indices (cheap integer div; when BLOCK_M ≤ BH,
        # the whole tile shares one row index).
        sm = m_offs // BH
        sn = n_offs // BW
        s_mask_m = sm < SR
        s_mask_n = sn < SC
        scale_mask = s_mask_m[:, None] & s_mask_n[None, :]
        scale = tl.load(
            scale_ptr
            + pid_e * s_stride_e
            + sm[:, None] * s_stride_m
            + sn[None, :] * s_stride_n,
            mask=scale_mask,
            other=1.0,
        )

        out = (w_f32 * scale).to(out_ptr.dtype.element_ty)

        tl.store(
            out_ptr
            + pid_e * o_stride_e
            + m_offs[:, None] * o_stride_m
            + n_offs[None, :] * o_stride_n,
            out,
            mask=tile_mask,
        )


def dequant_fp8_block_triton(
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused FP8 block-wise dequant. CUDA-only; caller must gate on device.

    Accepts 2-D ``[M, N]`` or 3-D ``[E, M, N]`` inputs; returns same rank.
    ``scale`` must have a matching ``[sr, sc]`` or ``[E, sr, sc]`` shape
    such that ``M % sr == 0`` and ``N % sc == 0``.
    """
    assert _HAS_TRITON, "triton not available"
    assert w_fp8.is_cuda, "dequant_fp8_block_triton requires CUDA"

    squeeze_out = False
    if w_fp8.dim() == 2:
        w_fp8 = w_fp8.unsqueeze(0)
        scale = scale.unsqueeze(0)
        squeeze_out = True
    assert w_fp8.dim() == 3 and scale.dim() == 3, (
        f"expected [E,M,N] and [E,sr,sc], got {w_fp8.shape} and {scale.shape}"
    )
    E, M, N = w_fp8.shape
    E_s, SR, SC = scale.shape
    assert E_s == E, (E, E_s)
    assert M % SR == 0 and N % SC == 0, (M, SR, N, SC)
    BH = M // SR
    BW = N // SC

    out = torch.empty((E, M, N), dtype=out_dtype, device=w_fp8.device)

    # Tile sizes: cap at 128 so shared memory stays reasonable; when BH or
    # BW is smaller, match it so every tile maps to a small integer number
    # of scale entries.
    BLOCK_M = min(128, max(16, BH))
    BLOCK_N = min(128, max(16, BW))

    grid = (E, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _dequant_fp8_block_kernel[grid](
        w_fp8, scale, out,
        E, M, N, SR, SC,
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BH=BH, BW=BW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out.squeeze(0) if squeeze_out else out

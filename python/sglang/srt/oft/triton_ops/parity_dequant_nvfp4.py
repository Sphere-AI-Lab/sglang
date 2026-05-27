"""Fused NVFP4 dequant triton kernel for OFT parity mode (fallback path).

Semantics
---------
    w_bf16[n, k] = e2m1_decode(nibble(w_u8[n, k//2], k%2))
                   * scale[n, k//16]
                   * scale_2

Where:
    w_u8:    uint8,  [out, in/2]  — two FP4 (e2m1) packed per byte
    scale:   fp32,   [out, in/16] — per-16-element block scale (converted
                                    from e4m3 by the caller)
    scale_2: fp32,   scalar       — global amax scale

Used only when ``modelopt`` isn't importable. When modelopt is present,
``parity_dequant._dequant_nvfp4`` delegates to ``NVFP4QTensor.dequantize``
and this kernel is never invoked. Kept as a safety net and for
environments without modelopt.

The E2M1 → fp32 decode is a 16-entry gather from a small on-device
lookup table cached per (device, dtype).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


# E2M1 decoded values, indexed by nibble (low 4 bits).
#  sign/exp/mantissa = 1/2/1 with bias=1 and subnormal support.
_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

_DEVICE_LOOKUP: dict = {}


def _get_lookup_table(device: torch.device) -> torch.Tensor:
    t = _DEVICE_LOOKUP.get(device)
    if t is None:
        t = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
        _DEVICE_LOOKUP[device] = t
    return t


if _HAS_TRITON:

    @triton.jit
    def _dequant_nvfp4_kernel(
        w_ptr,          # uint8, [M, N/2]
        scale_ptr,      # fp32,  [M, N/16]
        scale_2_ptr,    # fp32,  scalar
        table_ptr,      # fp32,  [16]  e2m1 lookup
        out_ptr,        # out_dtype, [M, N]
        M,
        N,
        w_stride_m,
        w_stride_n,
        s_stride_m,
        s_stride_n,
        o_stride_m,
        o_stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,  # columns of output (even)
    ):
        """Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N
        tile_mask = m_mask[:, None] & n_mask[None, :]

        # Each output column j reads byte j//2; low nibble if j even, high if odd.
        byte_idx = n_offs // 2
        use_high = (n_offs & 1) == 1

        packed = tl.load(
            w_ptr
            + m_offs[:, None] * w_stride_m
            + byte_idx[None, :] * w_stride_n,
            mask=tile_mask,
            other=0,
        )
        low_nib = (packed & 0xF).to(tl.int32)
        high_nib = ((packed >> 4) & 0xF).to(tl.int32)
        nibble = tl.where(use_high[None, :], high_nib, low_nib)

        # Gather FP4 value from the 16-entry lookup.
        val = tl.load(table_ptr + nibble)

        # Block scale: n//16 picks the scale column.
        s_col = n_offs // 16
        s_col_mask = s_col < (N // 16)
        scale_tile_mask = m_mask[:, None] & s_col_mask[None, :]
        scale = tl.load(
            scale_ptr
            + m_offs[:, None] * s_stride_m
            + s_col[None, :] * s_stride_n,
            mask=scale_tile_mask,
            other=1.0,
        )
        scale_2 = tl.load(scale_2_ptr)

        out = (val * scale * scale_2).to(out_ptr.dtype.element_ty)

        tl.store(
            out_ptr
            + m_offs[:, None] * o_stride_m
            + n_offs[None, :] * o_stride_n,
            out,
            mask=tile_mask,
        )


def dequant_nvfp4_triton(
    w_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused NVFP4 dequant. CUDA-only.

    ``weight_scale`` may arrive as e4m3 or fp32; converted to fp32 here.
    """
    assert _HAS_TRITON, "triton not available"
    assert w_u8.is_cuda, "dequant_nvfp4_triton requires CUDA"
    assert w_u8.dtype == torch.uint8, w_u8.dtype

    M, half_N = w_u8.shape
    N = half_N * 2
    assert weight_scale.dim() == 2, weight_scale.shape
    scale_fp32 = weight_scale.to(torch.float32).contiguous()
    scale_2_fp32 = weight_scale_2.to(torch.float32).reshape(1).contiguous()

    table = _get_lookup_table(w_u8.device)

    out = torch.empty((M, N), dtype=out_dtype, device=w_u8.device)

    BLOCK_M = 64
    BLOCK_N = 128  # must be even

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _dequant_nvfp4_kernel[grid](
        w_u8, scale_fp32, scale_2_fp32, table, out,
        M, N,
        w_u8.stride(0), w_u8.stride(1),
        scale_fp32.stride(0), scale_fp32.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out

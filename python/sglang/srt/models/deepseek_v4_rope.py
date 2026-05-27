"""Fused fp32-precision RoPE kernel for DeepSeek V4 MLA in SGLang.

Mirrors the DeepSeek V4 RoPE kernel (same math, same bit
output) but exposes in-place wrappers matching the eager ``_apply_rotary_emb``
and ``_apply_rotary_emb_decode`` semantics in ``deepseek_v4.py``. The eager
implementations materialise four intermediate tensors per call (fp32 upcast,
complex64 product, bf16 cast, full-size cat); this kernel keeps fp32
arithmetic in registers and writes a single bf16 output (or directly
overwrites the input slice when called in-place).

Batch-invariant by construction: element-wise per (token, head); fixed
constexpr tile sizes; no atomics; no cross-batch reductions.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _splice_rotary_kernel(
    x_ptr,                 # [N, HEAD_DIM]
    out_ptr,               # [N, HEAD_DIM] (may alias x_ptr; loads precede stores)
    freqs_real_ptr,        # [seqlen, ROPE_DIM/2] fp32
    freqs_imag_ptr,        # [seqlen, ROPE_DIM/2] fp32
    N,
    seqlen,
    inner_stride,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    STATIC_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_STATIC: tl.constexpr,
    BLOCK_ROPE_HALF: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    if STATIC_DIM > 0:
        for s_start in tl.static_range(0, STATIC_DIM, BLOCK_STATIC):
            s_offs = s_start + tl.arange(0, BLOCK_STATIC)
            s_mask = s_offs < STATIC_DIM
            ptrs = x_ptr + n_offs[:, None] * HEAD_DIM + s_offs[None, :]
            x_st = tl.load(
                ptrs, mask=n_mask[:, None] & s_mask[None, :], other=0.0
            )
            tl.store(
                out_ptr + n_offs[:, None] * HEAD_DIM + s_offs[None, :],
                x_st,
                mask=n_mask[:, None] & s_mask[None, :],
            )

    rope_half = ROPE_DIM // 2
    pair_offs = tl.arange(0, BLOCK_ROPE_HALF)
    pair_mask = pair_offs < rope_half
    real_col = STATIC_DIM + 2 * pair_offs
    imag_col = real_col + 1

    seq_idx = (n_offs // inner_stride) % seqlen

    x_real = tl.load(
        x_ptr + n_offs[:, None] * HEAD_DIM + real_col[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_imag = tl.load(
        x_ptr + n_offs[:, None] * HEAD_DIM + imag_col[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    f_real = tl.load(
        freqs_real_ptr + seq_idx[:, None] * rope_half + pair_offs[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    )
    f_imag = tl.load(
        freqs_imag_ptr + seq_idx[:, None] * rope_half + pair_offs[None, :],
        mask=n_mask[:, None] & pair_mask[None, :],
        other=0.0,
    )

    if INVERSE:
        f_imag = -f_imag

    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    tl.store(
        out_ptr + n_offs[:, None] * HEAD_DIM + real_col[None, :],
        out_real,
        mask=n_mask[:, None] & pair_mask[None, :],
    )
    tl.store(
        out_ptr + n_offs[:, None] * HEAD_DIM + imag_col[None, :],
        out_imag,
        mask=n_mask[:, None] & pair_mask[None, :],
    )


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _launch_inplace_rope(
    x: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
    seqlen: int,
    inner_stride: int,
    inverse: bool,
):
    rope_dim = x.shape[-1]
    assert rope_dim % 2 == 0
    rope_half = rope_dim // 2
    head_dim = rope_dim  # rope-only slice, no static prefix
    static_dim = 0

    N = x.numel() // head_dim
    BLOCK_N = 64
    BLOCK_STATIC = 64  # unused when STATIC_DIM == 0
    BLOCK_ROPE_HALF = _next_pow2(rope_half)

    grid = (triton.cdiv(N, BLOCK_N),)
    _splice_rotary_kernel[grid](
        x, x,  # in-place: out aliases input
        freqs_real, freqs_imag,
        N, seqlen, inner_stride,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        STATIC_DIM=static_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_STATIC=BLOCK_STATIC,
        BLOCK_ROPE_HALF=BLOCK_ROPE_HALF,
        INVERSE=inverse,
    )


def _apply_inplace(x: torch.Tensor, fr, fi, seqlen, inner_stride, inverse):
    """Run the kernel in-place on ``x``. If ``x`` is non-contiguous (common
    when the caller passes a slice like ``kv[..., -rope_dim:]``), operate
    on a contiguous copy and write the result back via ``copy_`` so the
    original strided storage gets the rotated values.
    """
    if x.is_contiguous():
        _launch_inplace_rope(x, fr, fi, seqlen, inner_stride, inverse)
    else:
        x_c = x.contiguous()
        _launch_inplace_rope(x_c, fr, fi, seqlen, inner_stride, inverse)
        x.copy_(x_c)


def _freqs_real_imag(freqs_cis: torch.Tensor, rope_half: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert freqs_cis.is_complex(), "freqs_cis must be complex"
    fr = freqs_cis.real.contiguous()
    fi = freqs_cis.imag.contiguous()
    if fr.ndim != 2:
        fr = fr.reshape(-1, rope_half)
        fi = fi.reshape(-1, rope_half)
    assert fr.shape[-1] == rope_half
    return fr, fi


def apply_rotary_emb_triton(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place replacement for sglang ``_apply_rotary_emb``.

    ``x`` is the rope-only slice (e.g., ``kv[..., -rope_dim:]``). Layout is
    either 3D ``[batch, seq, rope_dim]`` (inner_stride=1, freqs indexed by seq)
    or 4D ``[batch, seq, heads, rope_dim]`` (inner_stride=heads).
    """
    assert x.is_cuda, "Triton RoPE requires CUDA tensors"
    rope_half = x.shape[-1] // 2
    fr, fi = _freqs_real_imag(freqs_cis, rope_half)
    if x.ndim == 4:
        seqlen, inner_stride = x.shape[1], x.shape[2]
    elif x.ndim == 3:
        seqlen, inner_stride = x.shape[1], 1
    elif x.ndim == 2:
        seqlen, inner_stride = x.shape[0], 1
    else:
        raise ValueError(f"unsupported x.ndim={x.ndim}")
    _apply_inplace(x, fr, fi, seqlen, inner_stride, inverse)
    return x


def apply_rotary_emb_decode_triton(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place replacement for sglang ``_apply_rotary_emb_decode``.

    ``x`` is the rope-only slice of shape ``[batch, 1, ..., rope_dim]``;
    ``freqs_cis`` is ``[batch, rope_dim/2]`` (one freq per batch element,
    not per seq position). Maps to the same kernel with ``seqlen=batch`` and
    ``inner_stride=heads`` (4D) or 1 (3D).
    """
    assert x.is_cuda, "Triton RoPE requires CUDA tensors"
    rope_half = x.shape[-1] // 2
    fr, fi = _freqs_real_imag(freqs_cis, rope_half)
    # x is [B, 1, (H,) rope_dim]; flatten to [N, rope_dim] with seq pos 0
    # decided by batch index. n = b*H + h (4D) or b (3D).
    if x.ndim == 4:
        # [B, 1, H, rope_dim] -> inner_stride = H, seqlen = B (interpret as batch dim)
        seqlen, inner_stride = x.shape[0], x.shape[2]
    elif x.ndim == 3:
        seqlen, inner_stride = x.shape[0], 1
    else:
        raise ValueError(f"unsupported decode x.ndim={x.ndim}")
    _apply_inplace(x, fr, fi, seqlen, inner_stride, inverse)
    return x

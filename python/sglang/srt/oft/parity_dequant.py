"""Dequantize quantized base weights for OFT parity-mode forward.

When `ServerArgs.oft_parity_mode` is on, an OFT-wrapped linear dequantizes its
base weight to the activation dtype before running `F.linear`. This matches
Megatron-Bridge's `_forward_fp8` / `_forward_int4*` / NVFP4 training graph,
which dequantizes on every forward and runs a bf16 GEMM on the OFT-rotated
input. Slower than a native quant GEMM but bit-closer to training.

Public API:
- ``parity_linear(base_layer, x, apply_bias=True)`` — drop-in replacement for
  ``base_layer.quant_method.apply(base_layer, x, bias)`` used by OFT forwards.
- ``dequant_base_weight_to(base_layer, dtype)`` — low-level dequant used by
  ``parity_linear`` and callable directly by tests / MoE hooks.

Supported formats:
- Unquantized (bf16 / fp16 / fp32).
- FP8 block-wise (``float8_e4m3fn`` + ``weight_scale_inv``).
- FP8 per-tensor (``float8_e4m3fn`` + ``weight_scale``).
- INT4 AWQ (``qweight``/``qzeros``/``scales`` via ``awq_dequantize``).
- INT4 WNA16 MoE experts (``w{13,2}_qweight``/``w{13,2}_scales``).
- NVFP4 (``weight``/``weight_scale``/``weight_scale_2``).

Marlin / Machete / other pre-shuffled dense INT4 layouts cannot be
dequantized by the generic AWQ/GPTQ helpers after
``process_weights_after_loading``. Dense OFT falls back to the layer's native
quantized kernel for those prepacked formats.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "is_parity_mode",
    "parity_linear",
    "dequant_base_weight_to",
    "dequant_moe_expert_weight_to",
    "dequant_moe_experts_bf16",
]


# ─────────────────────────────────────────────────────────────────────────────
# Bridge FP8 dequant — ported verbatim from
# megatron/bridge/peft/fp8_utils.py::dequant_fp8 (qoft branch) so sglang
# runs the same math Bridge trains with, independent of whether
# Megatron-Bridge is installed. Keep in sync when Bridge updates
# (e.g. ue8m0 scale support).
# ─────────────────────────────────────────────────────────────────────────────


def _bridge_dequant_fp8(
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weight: ``w = w_fp8 * scale_inv`` (block-wise or per-tensor).

    On CUDA with block-wise scales, dispatches to a fused triton kernel
    (``parity_dequant_fp8.dequant_fp8_block_triton``) that avoids the full
    fp32 intermediate. Per-tensor (scalar scale) stays on the PyTorch path
    — it's already a single-pass op.

    Args:
        w_fp8: ``[out, in]`` in ``float8_e4m3fn``.
        scale_inv: ``[1]`` (per-tensor) or ``[out//B, in//B]`` (block-wise).
        out_dtype: Target dtype.
    """
    if scale_inv.numel() == 1:
        return (w_fp8.float() * scale_inv.float().item()).to(out_dtype)

    # Fused triton fast path (2-D block-wise FP8 on CUDA).
    if w_fp8.is_cuda and w_fp8.dim() == 2 and scale_inv.dim() == 2:
        try:
            from sglang.srt.oft.triton_ops.parity_dequant_fp8 import (
                dequant_fp8_block_triton,
            )

            return dequant_fp8_block_triton(
                w_fp8, scale_inv.to(torch.float32), out_dtype
            )
        except Exception:
            # Fall through to PyTorch reference on any kernel failure.
            pass

    out_feat, in_feat = w_fp8.shape
    sr, sc = scale_inv.shape
    bh, bw = out_feat // sr, in_feat // sc
    w = w_fp8.float().reshape(sr, bh, sc, bw)
    w = w * scale_inv.float().unsqueeze(1).unsqueeze(3)
    return w.reshape(out_feat, in_feat).to(out_dtype)


def is_parity_mode() -> bool:
    """Single source of truth for the parity-mode flag.

    Reads ``ServerArgs.oft_parity_mode`` from the global server args. All
    OFT/MoE call sites check this at forward time instead of caching state
    on every wrapped layer. The value is effectively immutable after server
    startup, so the per-forward lookup is free.
    """
    try:
        from sglang.srt.server_args import get_global_server_args

        return bool(getattr(get_global_server_args(), "oft_parity_mode", False))
    except Exception:
        return False


def parity_linear(
    base_layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    apply_bias: bool = True,
) -> torch.Tensor:
    """Dequant ``base_layer.weight`` to ``x.dtype`` and run ``F.linear``.

    Matches Megatron-Bridge's ``_forward_fp8`` / ``_forward_int4*`` / NVFP4
    training forward. The rotated input ``x`` is consumed directly; this
    function replaces only the inner GEMM, so TP collectives and OFT R
    handling in the calling layer are preserved.
    """
    bias: Optional[torch.Tensor] = None
    if apply_bias and not getattr(base_layer, "skip_bias_add", False):
        bias = getattr(base_layer, "bias", None)

    try:
        W = dequant_base_weight_to(base_layer, x.dtype)
    except NotImplementedError:
        quant_method = getattr(base_layer, "quant_method", None)
        if (
            quant_method is not None
            and hasattr(quant_method, "apply")
            and hasattr(base_layer, "weight_packed")
            and hasattr(base_layer, "weight_scale")
        ):
            return quant_method.apply(base_layer, x, bias)
        raise

    return F.linear(x, W, bias)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────


def dequant_base_weight_to(
    base_layer: torch.nn.Module, dtype: torch.dtype
) -> torch.Tensor:
    """Return a 2-D ``[out, in]`` dequantized copy of the base weight.

    Raises ``NotImplementedError`` for quant formats not yet wired up. The
    caller decides whether an unsupported prepacked format can safely fall
    back to the native quantized kernel on already-rotated OFT input.
    """
    # INT4 AWQ: qweight is stored under a different attribute from 'weight'.
    if hasattr(base_layer, "qweight") and hasattr(base_layer, "scales"):
        return _dequant_awq(base_layer, dtype)

    w = getattr(base_layer, "weight", None)
    if w is None:
        raise NotImplementedError(
            f"Parity mode: base layer {type(base_layer).__name__} has no "
            "`weight`, `qweight`, or supported `weight_packed` attribute."
        )

    if _is_unquantized(w):
        return w.to(dtype)

    if w.dtype == torch.float8_e4m3fn or (
        hasattr(torch, "float8_e5m2") and w.dtype == torch.float8_e5m2
    ):
        return _dequant_fp8(base_layer, dtype)

    # NVFP4 stores packed FP4 as uint8 with e4m3 per-16-block scale + fp32 scale2.
    if w.dtype == torch.uint8 and hasattr(base_layer, "weight_scale_2"):
        return _dequant_nvfp4(base_layer, dtype)

    raise NotImplementedError(
        f"Parity mode does not yet support base weight dtype {w.dtype} "
        f"(layer {type(base_layer).__name__}). Implement the dequant in "
        "`sglang/srt/oft/parity_dequant.py` or turn off --oft-parity-mode."
    )


def _is_unquantized(weight: torch.Tensor) -> bool:
    return weight.dtype in (torch.float32, torch.float16, torch.bfloat16)


def _dequant_int4_packed_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize signed symmetric INT4 packed along the last dimension."""
    if weight_packed.dtype == torch.uint8:
        pack_factor = 2
        shifts = torch.tensor([0, 4], device=weight_packed.device, dtype=torch.uint8)
        unpacked = ((weight_packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.float32)
    elif weight_packed.dtype == torch.int32:
        pack_factor = 8
        shifts = torch.arange(8, device=weight_packed.device, dtype=torch.int32) * 4
        unpacked = (
            (weight_packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF
        ).to(torch.float32)
    else:
        raise NotImplementedError(
            "INT4 parity dequant supports uint8 or int32 packed weights, "
            f"got {weight_packed.dtype}."
        )

    unpacked = unpacked.reshape(
        *weight_packed.shape[:-1], weight_packed.shape[-1] * pack_factor
    )
    unpacked = unpacked - 8

    scale = weight_scale.to(torch.float32)
    target_in = unpacked.shape[-1]
    groups = max(1, scale.shape[-1])
    elements_per_group = max(1, (target_in + groups - 1) // groups)
    scale = scale.repeat_interleave(elements_per_group, dim=-1)[..., :target_in]
    return (unpacked * scale).to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# FP8
# ─────────────────────────────────────────────────────────────────────────────


def _dequant_fp8(base_layer: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize an FP8 base weight to ``dtype`` via Bridge's ``dequant_fp8``.

    Handles both per-tensor (``weight_scale`` scalar) and block-wise
    (``weight_scale_inv`` 2-D) layouts with one call.
    """
    w = base_layer.weight
    scale_inv = getattr(base_layer, "weight_scale_inv", None)
    if scale_inv is None:
        scale_inv = getattr(base_layer, "weight_scale", None)
    if scale_inv is None:
        raise NotImplementedError(
            "FP8 dequant for parity mode found neither weight_scale_inv (block) "
            "nor weight_scale (per-tensor) on the base layer."
        )
    return _bridge_dequant_fp8(w, scale_inv, out_dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# INT4 AWQ
# ─────────────────────────────────────────────────────────────────────────────


def _dequant_awq(base_layer: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize AWQ INT4 weight to ``dtype`` via sglang's ``awq_dequantize``.

    sglang's AWQ ``apply`` already does dequant + bf16 matmul, so the fast
    path and parity path are algorithmically equivalent for AWQ; parity mode
    goes through this helper to keep the code path uniform.

    Returns ``[out, in]``. ``awq_dequantize`` produces ``[in, out]``.
    """
    try:
        from sgl_kernel import awq_dequantize  # noqa: F401
        from sglang.srt.layers.quantization.awq import awq_dequantize as _awq_dq
    except ImportError:
        from sglang.srt.layers.quantization.awq_triton import (
            awq_dequantize_triton as _awq_dq,
        )

    # Marlin / Machete variants pre-shuffle qweight in process_weights_after_loading;
    # awq_dequantize would read the permuted layout and produce garbage.
    qcfg = getattr(getattr(base_layer, "quant_method", None), "quant_config", None)
    qcfg_name = type(qcfg).__name__ if qcfg is not None else ""
    if "Marlin" in qcfg_name or "Machete" in qcfg_name:
        raise NotImplementedError(
            f"Parity mode on {qcfg_name}: the INT4 qweight is pre-shuffled "
            "for Marlin/Machete kernels and cannot be dequantized by the "
            "generic AWQ path. Load the checkpoint without the Marlin "
            "backend, or extend parity_dequant with a Marlin-aware dequant."
        )

    W_in_out = _awq_dq(base_layer.qweight, base_layer.scales, base_layer.qzeros)
    return W_in_out.t().contiguous().to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# NVFP4 (ModelOpt)
# ─────────────────────────────────────────────────────────────────────────────

# E2M1 (FP4) decode table: sign/exp/mantissa = 1/2/1 → 16 values.
# Values in standard order for nibble [e2 e1 e1 m1] packed layout.
_E2M1_DECODE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _dequant_nvfp4(base_layer: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize NVFP4 base weight to ``dtype``.

    Storage layout (sglang ``ModelOptFp4LinearMethod.create_weights``):
        weight:         uint8, ``[out, in/2]``  (two e2m1 per byte)
        weight_scale:   e4m3,  ``[out, in/16]`` (per-16-element block scale)
        weight_scale_2: fp32 scalar              (global scale)

    Prefers modelopt's own dequant when available — modelopt is the source
    of truth for NVFP4 encoding/decoding and what Bridge uses at training
    time via its TE ``TensorQuantizer`` wrapper. Falls back to a local
    e2m1 decode (matching the standard 16-entry table + low-nibble-first
    packing) when modelopt isn't importable.
    """
    # Prefer modelopt: it owns the NVFP4 encoding/decoding convention.
    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    except ImportError:
        NVFP4QTensor = None

    w_u8 = base_layer.weight
    ws = base_layer.weight_scale          # e4m3, [out, in/16]
    ws2 = base_layer.weight_scale_2       # fp32 scalar

    out_features = int(getattr(base_layer, "output_size_per_partition", w_u8.shape[0]))
    in_features = int(getattr(base_layer, "input_size_per_partition", ws.shape[-1] * 16))
    half_in_features = (in_features + 1) // 2
    scale_cols = (in_features + 15) // 16

    # SGLang may pad the packed weight for FP4 serving kernels after loading.
    # Parity mode should dequantize the logical checkpoint weight, matching
    # Megatron-Bridge's ``dequantize_nvfp4(..., weight_shape=...)`` path.
    w_u8 = w_u8[:out_features, :half_in_features]
    ws = ws[:out_features, :scale_cols]

    if NVFP4QTensor is not None:
        try:
            try:
                qtensor = NVFP4QTensor(torch.Size((out_features, in_features)), dtype, w_u8)
            except TypeError:
                qtensor = NVFP4QTensor(
                    w_u8,
                    metadata={"shape": (out_features, in_features), "dtype": dtype},
                )
            return qtensor.dequantize(
                dtype=dtype,
                scale=ws,
                double_scale=ws2.reshape(()).to(device=ws.device, dtype=torch.float32),
                block_sizes={-1: 16},
            )
        except Exception:
            # Fall through to the local decode path if the modelopt
            # API shape differs from what we pass (version drift).
            pass

    # CUDA fast path: fused triton dequant (no full-weight fp32 staging).
    if w_u8.is_cuda:
        try:
            from sglang.srt.oft.triton_ops.parity_dequant_nvfp4 import (
                dequant_nvfp4_triton,
            )

            return dequant_nvfp4_triton(w_u8, ws, ws2, dtype)
        except Exception:
            pass

    ws2 = ws2.to(torch.float32)
    out_dim, half_in = w_u8.shape
    in_dim = half_in * 2

    # Unpack two e2m1 values per byte → int64 nibble indices.
    # Convention (matches flashinfer / sglang FP4 encoding): low nibble = even k,
    # high nibble = odd k.
    low = (w_u8 & 0x0F).to(torch.int64)
    high = ((w_u8 >> 4) & 0x0F).to(torch.int64)
    lookup = torch.tensor(_E2M1_DECODE, dtype=torch.float32, device=w_u8.device)
    vals_low = lookup[low]   # [out, in/2]
    vals_high = lookup[high] # [out, in/2]
    # Interleave: out[:, 2k] = low, out[:, 2k+1] = high
    vals = torch.empty(out_dim, in_dim, dtype=torch.float32, device=w_u8.device)
    vals[:, 0::2] = vals_low
    vals[:, 1::2] = vals_high

    # Broadcast the per-16-block scale across the block dim.
    ws_f32 = ws.to(torch.float32)                                  # [out, in/16]
    ws_exp = ws_f32.repeat_interleave(16, dim=-1)                  # [out, in]
    ws_exp = ws_exp[:, :in_dim]

    W = vals * ws_exp * ws2
    return W.to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# MoE expert parity hook (scaffolded — FusedMoE side wiring is a follow-up).
# ─────────────────────────────────────────────────────────────────────────────


def _batched_dequant_fp8_all_experts(
    w: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequant all E experts of ``w`` at once via one fp32 broadcast multiply.

    ``w``: ``[E, out, in]`` FP8. ``scale`` is one of:
      - per-tensor-per-expert: ``[E]`` or ``[E, 1, 1]``
      - block-wise: ``[E, out//bh, in//bw]``
      - ``None`` (unquantized): just cast
    Returns ``[E, out, in]`` in ``dtype``.
    """
    if scale is None or _is_unquantized(w):
        return w.to(dtype)

    E, out_feat, in_feat = w.shape
    s = scale
    # Normalize per-tensor-per-expert shape to [E, 1, 1].
    if s.dim() == 1 and s.shape[0] == E:
        s = s.view(E, 1, 1)
    if s.dim() == 3 and s.shape[1:] == (1, 1):
        return (w.float() * s.float()).to(dtype)
    if s.dim() == 3 and s.shape[0] == E:
        # Fused triton batched fast path.
        if w.is_cuda:
            try:
                from sglang.srt.oft.triton_ops.parity_dequant_fp8 import (
                    dequant_fp8_block_triton,
                )

                return dequant_fp8_block_triton(w, s.to(torch.float32), dtype)
            except Exception:
                pass
        _, sr, sc = s.shape
        bh, bw = out_feat // sr, in_feat // sc
        wf = w.float().reshape(E, sr, bh, sc, bw)
        wf = wf * s.float().reshape(E, sr, 1, sc, 1)
        return wf.reshape(E, out_feat, in_feat).to(dtype)
    raise NotImplementedError(
        f"Unsupported FP8 MoE scale shape {tuple(s.shape)} for w {tuple(w.shape)}."
    )


def dequant_moe_experts_bf16(
    layer: torch.nn.Module, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched FP8 → ``dtype`` dequant of every expert's ``w13`` and ``w2``.

    Pure function: reads ``layer.{w13_weight, w2_weight,
    w{13,2}_weight_scale_inv, w{13,2}_weight_scale}`` and returns
    ``(W13_bf16, W2_bf16)`` with shapes ``[E, 2*inter, hidden]`` and
    ``[E, hidden, inter]``. One fp32 broadcast multiply per linear — no
    Python per-expert loop.

    Math matches Bridge's ``_forward_fp8`` + ``dequant_fp8`` applied per
    expert. Caller is responsible for whatever comes after (running the
    native bf16 MoE runner on these weights).
    """
    def _pick_scale(prefix: str) -> Optional[torch.Tensor]:
        s = getattr(layer, f"{prefix}_weight_scale_inv", None)
        if s is None:
            s = getattr(layer, f"{prefix}_weight_scale", None)
        return s

    w13_packed = getattr(layer, "w13_weight_packed", None)
    w2_packed = getattr(layer, "w2_weight_packed", None)
    w13_weight_scale = getattr(layer, "w13_weight_scale", None)
    w2_weight_scale = getattr(layer, "w2_weight_scale", None)
    if all(
        isinstance(t, torch.Tensor)
        for t in (w13_packed, w2_packed, w13_weight_scale, w2_weight_scale)
    ):
        return (
            _dequant_wna16_moe_packed_weight(w13_packed, w13_weight_scale, dtype),
            _dequant_wna16_moe_packed_weight(w2_packed, w2_weight_scale, dtype),
        )

    w13_qweight = getattr(layer, "w13_qweight", None)
    w2_qweight = getattr(layer, "w2_qweight", None)
    w13_scales = getattr(layer, "w13_scales", None)
    w2_scales = getattr(layer, "w2_scales", None)
    if all(
        isinstance(t, torch.Tensor)
        for t in (w13_qweight, w2_qweight, w13_scales, w2_scales)
    ):
        w13_qzeros = getattr(layer, "w13_qzeros", None)
        w2_qzeros = getattr(layer, "w2_qzeros", None)
        has_w13_zp = isinstance(w13_qzeros, torch.Tensor) and w13_qzeros.numel() > 0
        has_w2_zp = isinstance(w2_qzeros, torch.Tensor) and w2_qzeros.numel() > 0
        if has_w13_zp or has_w2_zp:
            raise NotImplementedError(
                "Parity MoE dequant does not yet support INT4 expert zero "
                "points. Symmetric W4A16 expert weights are supported."
            )
        return (
            _dequant_int4_packed_weight(w13_qweight, w13_scales, dtype),
            _dequant_int4_packed_weight(w2_qweight, w2_scales, dtype),
        )

    W13 = _batched_dequant_fp8_all_experts(
        layer.w13_weight, _pick_scale("w13"), dtype
    )
    W2 = _batched_dequant_fp8_all_experts(
        layer.w2_weight, _pick_scale("w2"), dtype
    )
    return W13, W2


def _dequant_wna16_moe_packed_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize compressed-tensors WNA16 MoE weights to ``[E, N, K]``.

    ``CompressedTensorsWNA16TritonMoE.process_weights_after_loading`` converts
    checkpoint tensors from ``[E, K/8, N]`` int32 into ``[E, N, K/2]`` uint8
    and transposes scales to ``[E, N, K/group]``. Parity mode normally sees the
    post-processed form, but accepting the pre-processed orientation makes this
    helper safe for focused unit tests.
    """
    if weight_packed.dim() != 3 or weight_scale.dim() != 3:
        raise NotImplementedError(
            "WNA16 MoE parity dequant expects 3-D packed weights and scales, "
            f"got weight={tuple(weight_packed.shape)} scale={tuple(weight_scale.shape)}."
        )

    if (
        weight_packed.dtype == torch.int32
        and weight_packed.shape[0] == weight_scale.shape[0]
        and weight_packed.shape[2] == weight_scale.shape[2]
    ):
        packed = weight_packed.transpose(1, 2).contiguous().view(torch.uint8)
        scale = weight_scale.transpose(1, 2).contiguous()
        return _dequant_int4_packed_weight(packed, scale, dtype)

    if weight_packed.shape[:-1] == weight_scale.shape[:-1]:
        return _dequant_int4_packed_weight(weight_packed, weight_scale, dtype)

    raise NotImplementedError(
        "Unsupported WNA16 MoE packed/scale layout for parity dequant: "
        f"weight={tuple(weight_packed.shape)} {weight_packed.dtype}, "
        f"scale={tuple(weight_scale.shape)} {weight_scale.dtype}."
    )


def dequant_moe_expert_weight_to(
    moe_layer: torch.nn.Module,
    expert_id: int,
    which: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize one expert's ``w13`` or ``w2`` weight to ``dtype``.

    ``which`` is one of ``"w13"`` / ``"w2"``. Returns the per-expert 2-D
    ``[out, in]`` weight. Supports FP8 (per-tensor or block) and raises for
    other formats — extend as needed.

    This is the building block for a parity-mode grouped GEMM on MoE
    experts. The FusedMoE forward wiring (replacing the native grouped
    quant-GEMM with per-expert dequant + bf16 matmul) lives outside this
    file; see ``python/sglang/srt/oft/oft_manager.py::apply_oft_R_for_moe_experts``
    for where OFT writes ``moe.w13_oft_r`` / ``moe.w2_oft_r``.
    """
    assert which in ("w13", "w2"), which

    packed = getattr(moe_layer, f"{which}_weight_packed", None)
    scale = getattr(moe_layer, f"{which}_weight_scale", None)
    if isinstance(packed, torch.Tensor) and isinstance(scale, torch.Tensor):
        return _dequant_wna16_moe_packed_weight(
            packed[expert_id : expert_id + 1],
            scale[expert_id : expert_id + 1],
            dtype,
        )[0]

    qweight = getattr(moe_layer, f"{which}_qweight", None)
    qscales = getattr(moe_layer, f"{which}_scales", None)
    if isinstance(qweight, torch.Tensor) and isinstance(qscales, torch.Tensor):
        qzeros = getattr(moe_layer, f"{which}_qzeros", None)
        if isinstance(qzeros, torch.Tensor) and qzeros.numel() > 0:
            raise NotImplementedError(
                "Parity MoE dequant does not yet support INT4 expert zero "
                "points. Symmetric W4A16 expert weights are supported."
            )
        return _dequant_int4_packed_weight(
            qweight[expert_id], qscales[expert_id], dtype
        )

    w = getattr(moe_layer, f"{which}_weight")[expert_id]
    # FP8 per-tensor (weight_scale) or block (weight_scale_inv) via Bridge's
    # ported dequant — same math for both layouts.
    scale_inv = getattr(moe_layer, f"{which}_weight_scale_inv", None)
    if scale_inv is not None:
        return _bridge_dequant_fp8(w, scale_inv[expert_id], out_dtype=dtype)

    scale_per_tensor = getattr(moe_layer, f"{which}_weight_scale", None)
    if scale_per_tensor is not None:
        return _bridge_dequant_fp8(
            w, scale_per_tensor[expert_id], out_dtype=dtype
        )

    if _is_unquantized(w):
        return w.to(dtype)

    raise NotImplementedError(
        f"Parity MoE dequant: unsupported weight dtype {w.dtype} on "
        f"{type(moe_layer).__name__} for {which}. Extend parity_dequant."
    )

"""OFT MoE kernel-invoker wrapper for expert pre-rotation.

Upstream sglang ships only POST-GEMM void hooks (``LoRAHooks.after_gate_up`` /
``after_down``) that mutate the GEMM *output* in place. OFT is different: it is a
per-expert MULTIPLICATIVE rotation applied to the GEMM *input* before the GEMM.
The rotation expands rows (``num_tokens -> num_tokens * top_k``), returns a NEW
tensor, collapses ``top_k -> 1`` and re-aligns the block metadata.

Rather than branch on that inside upstream's ``_fused_moe_kernel_sequence`` (the
WS-A/WS-B "hooks struct" design, which cost ~155 forked lines there), the fork
substitutes the KERNEL INVOKER itself: ``_fused_moe_kernel_sequence`` takes an
``invoke`` callable defaulting to ``invoke_fused_moe_kernel``, and this module
builds the OFT-rotating replacement. Everything the rotation needs is already an
argument of ``invoke_fused_moe_kernel`` (``A``, ``C``, ``sorted_token_ids``,
``expert_ids``, ``num_tokens_post_padded``, ``top_k``, ``config``), so upstream
keeps its own call sites verbatim.

Because the wrapper transforms only its OWN arguments, the sequence's locals are
never rebound -- so the LoRA ``after_down`` hook and the combine path keep seeing
the ORIGINAL ``topk_weights``/``topk_ids``/``topk`` by construction, which the
previous design had to preserve by hand via renamed variables.

``_oft_prerotate`` and the split gate-up body are verbatim moves of the
in-kernel originals, so numerics stay bit-identical. The OFT MoE golden tests
guard this contract across three configurations with zero permitted drift.

The wrapper closes over the LAYER and re-reads ``layer.w*_oft_r`` on EVERY call,
so a streamed in-place weight sync stays visible under CUDA-graph replay, and an
identity boot (buffers injected after the wrapper is built) is picked up. It is
self-gating: with no OFT buffer present it delegates straight to the real kernel.
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.oft.utils import MoEOFTBatchInfo

_QUANT_KWARGS = (
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "use_int4_w4a16",
    "per_channel_quant",
)


@dataclass
class OFTInfo:
    w13_oft_r: torch.Tensor | None
    w1_oft_r: torch.Tensor | None
    w3_oft_r: torch.Tensor | None
    w2_oft_r: torch.Tensor | None
    batch_info: MoEOFTBatchInfo
    num_experts: int
    max_ofts: int
    has_active_oft: bool
    cg_buffers: dict | None = None


def _compute_oft_alignment(
    topk_ids: torch.Tensor,
    oft_info: OFTInfo,
    row_group_factor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group routed rows by adapter slot and expert for OFT rotation."""
    from sglang.kernels.ops.moe.moe_lora_align import moe_lora_align_block_size

    if row_group_factor < 1:
        raise ValueError(f"row_group_factor must be positive, got {row_group_factor}")

    block_size_m = 64
    max_ofts = oft_info.max_ofts
    max_num_tokens_padded = topk_ids.numel() + oft_info.num_experts * (
        block_size_m - 1
    )
    max_num_tokens_padded = (
        (max_num_tokens_padded + block_size_m - 1) // block_size_m
    ) * block_size_m
    max_num_m_blocks = max_num_tokens_padded // block_size_m
    device = topk_ids.device
    cg = oft_info.cg_buffers if get_is_capture_mode() else None

    if cg is None:
        sorted_token_ids = torch.empty(
            max_ofts * max_num_tokens_padded,
            dtype=torch.int32,
            device=device,
        )
        expert_ids = torch.empty(
            max_ofts * max_num_m_blocks,
            dtype=torch.int32,
            device=device,
        )
        num_tokens_post_padded = torch.zeros(
            max_ofts,
            dtype=torch.int32,
            device=device,
        )
        oft_ids = torch.arange(max_ofts, dtype=torch.int32, device=device)
    else:
        sorted_token_ids = cg["sorted_token_ids_oft"][
            : max_ofts * max_num_tokens_padded
        ]
        expert_ids = cg["expert_ids_oft"][: max_ofts * max_num_m_blocks]
        num_tokens_post_padded = cg["num_tokens_post_padded_oft"][:max_ofts]
        num_tokens_post_padded.zero_()
        oft_ids = cg["oft_ids"][:max_ofts]

    batch = oft_info.batch_info
    seg_indptr = batch.seg_indptr
    if row_group_factor != 1:
        seg_indptr = seg_indptr * row_group_factor

    moe_lora_align_block_size(
        topk_ids,
        seg_indptr,
        batch.req_to_oft,
        int(oft_info.num_experts),
        block_size_m,
        int(max_ofts),
        int(max_num_tokens_padded),
        int(max_num_m_blocks),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        batch.adapter_enabled,
        oft_ids,
        cumsum_buffer=cg.get("cumsum_buffer") if cg is not None else None,
        token_mask=cg.get("token_mask") if cg is not None else None,
    )

    return (
        sorted_token_ids.view(max_ofts, max_num_tokens_padded),
        expert_ids.view(max_ofts, max_num_m_blocks),
        num_tokens_post_padded,
        oft_ids,
    )


def _oft_prerotate(
    A,
    oft_r,
    oft_info,
    C,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    top_k,
    num_experts,
    block_size_m,
):
    """Expert-specific OFT rotation + top_k->1 collapse + re-align for a single
    fused-MoE GEMM. Verbatim body (was ``fused_moe.py::_apply_expert_oft_prerotate``,
    then ``moe_hooks.py::_oft_prerotate``). Rotating before activation
    quantization matches the dense OFT path mathematically (FP8 activations
    cannot be tl.dot operands in the rotation kernel). Returns the rotated +
    re-aligned tensors to feed ONE kernel call with ``top_k=1`` and no ``oft_r``.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.oft.triton_ops import apply_oft_rotation_triton

    if oft_info is None:
        raise RuntimeError("MoE OFT rotation requires request-aware OFTInfo")
    num_batch_tokens = oft_info.batch_info.num_tokens
    if num_batch_tokens <= 0 or A.shape[0] % num_batch_tokens != 0:
        raise ValueError(
            f"OFT MoE rows {A.shape[0]} are not grouped by {num_batch_tokens} batch tokens"
        )
    row_group_factor = A.shape[0] // num_batch_tokens
    (
        oft_sorted_token_ids,
        oft_expert_ids,
        oft_num_tokens_post_padded,
        oft_ids,
    ) = _compute_oft_alignment(
        topk_ids,
        oft_info,
        row_group_factor=row_group_factor,
    )

    A = apply_oft_rotation_triton(
        A,
        oft_r,
        topk_ids,
        oft_sorted_token_ids,
        oft_expert_ids,
        oft_num_tokens_post_padded,
        oft_ids,
        oft_info.batch_info.adapter_enabled,
        top_k=top_k,
        block_m=block_size_m,
    )
    C = C.reshape(-1, 1, C.shape[-1])
    topk_weights = topk_weights.reshape(-1, 1)
    topk_ids = topk_ids.reshape(-1, 1)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, num_experts
    )
    return (
        A,
        C,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _assert_unquantized(kw):
    """The split/canonical gate-up path is BF16-only. Was the caller-side guard
    in ``_fused_moe_kernel_sequence``; lives here now so the upstream file keeps
    no OFT knowledge."""
    if any(kw.get(name) for name in _QUANT_KWARGS) or kw.get("block_shape") is not None:
        raise RuntimeError(
            "Split expert gate/up OFT is currently implemented for BF16/unquantized "
            "FusedMoE only. Quantized split expert OFT needs a dedicated quantized "
            "first-GEMM path."
        )


def _run_gate_up_split(
    layer,
    real_invoke,
    oft_info,
    A,
    B,
    bias,
    C,
    A_scale,
    B_scale,
    B_zp,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    config,
    kw,
):
    """Own the ENTIRE gate-up GEMM for the split/canonical OFT path.

    Verbatim body (was ``moe_hooks.py::_oft_run_gate_up_split``, itself the
    ``if split_w13_oft:`` block from ``_fused_moe_kernel_sequence``). Rotates
    gate/up inputs independently (``layer.w1_oft_r`` / ``layer.w3_oft_r``, read
    live) and writes the two ``N // 2`` halves of ``C`` (= intermediate_cache1).
    Each half GEMM writes a contiguous scratch buffer (the kernel needs a
    contiguous C), then copies into the non-contiguous half-column view.

    ``N``/``total_tokens`` are derived from ``C`` -- upstream allocates it as
    ``torch.empty((total_tokens, N))`` -- instead of being threaded down.
    """
    w1_oft_r = layer.w1_oft_r
    w3_oft_r = layer.w3_oft_r
    if w1_oft_r is None or w3_oft_r is None:
        raise RuntimeError(
            "Split expert gate/up OFT requires both w1_oft_r and w3_oft_r"
        )
    total_tokens, N = C.shape[0], C.shape[1]
    c_sorted = kw.get("c_sorted", False)
    filter_expert = kw.get("filter_expert", True)
    compute_type = kw["compute_type"]

    for half_slice, oft_r in (
        (slice(None, N // 2), w1_oft_r),
        (slice(N // 2, None), w3_oft_r),
    ):
        half_cache = torch.empty(
            (total_tokens, N // 2),
            device=A.device,
            dtype=A.dtype,
        )
        b_half = B[:, half_slice, :].contiguous()
        (
            a_in,
            c_in,
            tw,
            ti,
            sti,
            ei,
            ntpp,
        ) = _oft_prerotate(
            A,
            oft_r,
            oft_info,
            half_cache,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            # the gate-up GEMM's own top_k (rotation collapses it to 1)
            top_k,
            b_half.shape[0],
            config["BLOCK_SIZE_M"],
        )
        real_invoke(
            a_in,
            b_half,
            None if bias is None else bias[:, half_slice].contiguous(),
            c_in,
            A_scale,
            None if B_scale is None else B_scale[:, half_slice].contiguous(),
            B_zp,
            tw,
            ti,
            sti,
            ei,
            ntpp,
            mul_routed_weight,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
            c_sorted=c_sorted,
            filter_expert=filter_expert,
        )
        C[:, half_slice].copy_(half_cache)


def make_oft_invoke(
    layer: Any,
    real_invoke: Callable,
    oft_info: OFTInfo | None = None,
) -> Callable:
    """Build the OFT-rotating replacement for ``invoke_fused_moe_kernel``.

    Dispatches on the WEIGHT TENSOR IDENTITY: ``triton.py`` passes
    ``w1=quant_info.w13_weight`` / ``w2=quant_info.w2_weight``, and both
    ``unquant.py`` and ``fp8.py`` build those as ``layer.w13_weight`` /
    ``layer.w2_weight`` -- the very objects this closure holds. The split path's
    inner half-GEMMs call ``real_invoke`` directly (with a freshly sliced weight),
    so there is no recursion.

    Self-gating: a GEMM whose OFT buffer is ``None`` falls through to
    ``real_invoke`` unchanged, so there is no separate "is any hook set?" gate to
    get wrong (the WS-A Finding-#1 bug class becomes unrepresentable).
    """

    def invoke(
        A,
        B,
        bias,
        C,
        A_scale,
        B_scale,
        B_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        **kw,
    ):
        is_gate_up = B is layer.w13_weight
        is_down = B is layer.w2_weight
        if not (is_gate_up or is_down):
            # Not one of this layer's two expert GEMMs (e.g. the split path's own
            # sliced half-weights, or a future extra GEMM): no rotation to apply.
            return real_invoke(
                A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                mul_routed_weight, top_k, config, **kw,
            )

        if is_gate_up:
            w13 = getattr(layer, "w13_oft_r", None)
            w1 = getattr(layer, "w1_oft_r", None)
            if w13 is not None and w1 is not None:
                # Invariant (enforced by the OFT manager via oft_type): a layer
                # carries EITHER the legacy fused w13 rotation OR the split w1/w3
                # rotation, never both. Fail loud rather than silently pick one.
                raise RuntimeError(
                    "Split expert gate/up OFT (w1/w3_oft_r) cannot be active "
                    "together with legacy w13_oft_r on the same MoE layer"
                )
            if w1 is not None:
                _assert_unquantized(kw)
                return _run_gate_up_split(
                    layer,
                    real_invoke,
                    oft_info,
                    A,
                    B,
                    bias,
                    C,
                    A_scale,
                    B_scale,
                    B_zp,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    mul_routed_weight,
                    top_k,
                    config,
                    kw,
                )
            oft_r = w13
        else:
            oft_r = getattr(layer, "w2_oft_r", None)

        if oft_r is None:
            return real_invoke(
                A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                mul_routed_weight, top_k, config, **kw,
            )

        # FP8 activations cannot be tl.dot operands in the rotation kernel (see
        # _oft_prerotate's docstring). use_fp8_w8a8 alone is NOT the right check
        # here -- that's the WEIGHT quantization scheme, and fp8-weight OFT-MoE
        # with rotation on raw bf16 activations is validated and correct. The
        # actual risk is A itself already being pre-quantized (e.g. the opt-in
        # SGLANG_OPT_MOE_QUANT_ONCE a1_q path), which would otherwise silently
        # rotate the wrong tensor or crash inside Triton.
        #
        # Fix: undo the pre-quantization for this one call. Dequant with the
        # existing single-pass block-dequant kernel (built for OFT weight-parity
        # checks in Megatron-Bridge import; same per-token-group math applies to
        # activations), rotate in bf16 as usual, then hand back with
        # A_scale=None so invoke_fused_moe_kernel's own quant branch
        # (fused_moe_triton_kernels.py: `elif _is_cuda: ... =
        # sglang_per_token_group_quant_fp8(A, block_k)`) re-quantizes it -- the
        # exact path already used when SGLANG_OPT_MOE_QUANT_ONCE is off. One
        # dequant pass, no new rotation-kernel numerics, no new golden-drift
        # risk on the already-validated bf16 rotation math.
        if A.dtype == torch.float8_e4m3fn:
            from sglang.srt.oft.triton_ops.parity_dequant_fp8 import (
                dequant_fp8_block_triton,
            )

            A = dequant_fp8_block_triton(A, A_scale, out_dtype=C.dtype)
            A_scale = None

        a, c, tw, ti, sti, ei, ntpp = _oft_prerotate(
            A,
            oft_r,
            oft_info,
            C,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            B.shape[0],
            config["BLOCK_SIZE_M"],
        )
        # The rotation collapses top_k -> 1 for this GEMM. The down GEMM already
        # ran at top_k=1; kw's own `router_topk` (the ORIGINAL topk) passes
        # through untouched, as the combine path requires.
        return real_invoke(
            a, B, bias, c, A_scale, B_scale, B_zp, tw, ti, sti, ei, ntpp,
            mul_routed_weight, 1, config, **kw,
        )

    return invoke

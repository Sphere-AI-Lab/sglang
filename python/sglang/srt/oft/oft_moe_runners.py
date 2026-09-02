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

``_oft_prerotate`` and the split gate-up body are VERBATIM moves (originally from
the now-deleted ``peft/oft/moe_hooks.py``, itself a verbatim move of the in-kernel
originals) so numerics stay bit-identical to the in-kernel originals; this was
checked against an A/B golden (drift MUST be 0, 3 configs) before the legacy
package was deleted.

The wrapper closes over the LAYER and re-reads ``layer.w*_oft_r`` on EVERY call,
so a streamed in-place weight sync stays visible under CUDA-graph replay, and an
identity boot (buffers injected after the wrapper is built) is picked up. It is
self-gating: with no OFT buffer present it delegates straight to the real kernel.
"""

from typing import Any, Callable

import torch

_QUANT_KWARGS = (
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "use_int4_w4a16",
    "per_channel_quant",
)


def _oft_prerotate(
    A,
    oft_r,
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

    A = apply_oft_rotation_triton(
        A,
        oft_r,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
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


def _oft_prerotate_multi_tenant(
    A,
    oft_r_all_slots,
    slot_ids,
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
    """Multi-tenant sibling of _oft_prerotate: selects each token's own
    adapter's R-block (via slot_ids) instead of applying one shared R to
    the whole batch. Exercised whenever ANY real resident adapter is
    referenced in the current forward batch -- not just ">=2 distinct"
    adapters: in the plain native-RPC ("sibling") pool, buffer slot 0
    (memory_pool.active_idx) is permanently reserved for the boot-time
    base/identity registration, so a real adapter can never land there,
    meaning OFTManager._compute_moe_multi_tenant_slot_ids already takes
    this general per-token path for any batch with a single real adapter
    too (see that function's docstring for the full history). Also forced
    under CUDA-graph capture of the "oft_multi" dual-capture variant
    (decode_cuda_graph_runner.py's _resolve_record_oft_variant_graph /
    _resolve_oft_variant), even for the zero-real-adapter capture-time
    dummy batch, so the captured graph learns to read per-token slot ids
    at all -- see the 2026-09-01-oft-moe-cuda-graph-dual-capture plan. The
    single-adapter fast path (_oft_prerotate) is completely unaffected by
    this function's existence.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.oft.triton_ops import apply_oft_rotation_triton_multi_slot

    A = apply_oft_rotation_triton_multi_slot(
        A,
        oft_r_all_slots,
        slot_ids,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
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
    slot_ids = getattr(layer, "_oft_moe_multi_tenant_slot_ids", None)

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
        if slot_ids is not None:
            oft_r_all_slots = (
                layer._oft_w1_oft_r_all_slots
                if oft_r is w1_oft_r
                else layer._oft_w3_oft_r_all_slots
            )
            (
                a_in,
                c_in,
                tw,
                ti,
                sti,
                ei,
                ntpp,
            ) = _oft_prerotate_multi_tenant(
                A,
                oft_r_all_slots,
                slot_ids,
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
        else:
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


def make_oft_invoke(layer: Any, real_invoke: Callable) -> Callable:
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
                    layer, real_invoke, A, B, bias, C, A_scale, B_scale, B_zp,
                    topk_weights, topk_ids, sorted_token_ids, expert_ids,
                    num_tokens_post_padded, mul_routed_weight, top_k, config, kw,
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

        # This branch IS the hot path under a replayed decode CUDA graph for
        # a correctly-configured server, and that is now SAFE: the
        # 2026-09-01-oft-moe-cuda-graph-dual-capture plan's dual-capture
        # mechanism captures a dedicated "oft_multi" graph variant whenever
        # any real resident adapter is possible
        # (decode_cuda_graph_runner.py's _resolve_record_oft_variant_graph /
        # _resolve_oft_variant), and OFTManager._compute_moe_multi_tenant_
        # slot_ids writes this batch's routing decision IN PLACE into a
        # persistent buffer (allocated once by init_cuda_graph_batch_info)
        # rather than a fresh per-call tensor -- so the captured kernel's
        # pointer stays valid across every replay instead of going stale.
        # Capture-time forcing (get_capture_oft_variant() == "oft_multi")
        # makes sure the graph that gets captured is the one that actually
        # reads slot_ids, even though the capture-time dummy batch itself
        # carries zero real adapters. peft/config.py's validate_peft_args
        # only disables decode CUDA graphs for the residual configurations
        # dual-capture does NOT cover (DP-attention, insufficient effective
        # adapter capacity -- see _resolve_record_oft_variant_graph's
        # docstring); everywhere dual-capture does engage, this branch runs
        # under replay by design, not by accident. The slot_ids-is-None path
        # below (reading the persistent, in-place-updated pool buffers) is
        # the OTHER captured variant ("oft_single"), read when the batch has
        # zero real resident adapters at all -- in the plain native-RPC pool
        # a real adapter can never land at active_idx, so slot_ids is None
        # only for that zero-real-adapter case; the moment one real
        # MoE-target adapter loads, this branch (not the slot_ids-is-None
        # one) is what runs, exactly as intended.
        slot_ids = getattr(layer, "_oft_moe_multi_tenant_slot_ids", None)
        if slot_ids is not None:
            all_slots_attr = (
                "_oft_w13_oft_r_all_slots" if is_gate_up else "_oft_w2_oft_r_all_slots"
            )
            oft_r_all_slots = getattr(layer, all_slots_attr, None)
            if oft_r_all_slots is None:
                # A legacy-fused static adapter loaded at boot short-circuits
                # _init_identity_expert_oft_for_cuda_graph before it binds this
                # attribute (see oft_manager.py), so it can be legitimately
                # absent even with multi-tenancy active elsewhere. Fail loud
                # rather than let a bare AttributeError hit the forward-pass
                # hot path.
                raise RuntimeError(
                    f"Multi-tenant expert OFT is active for this layer but "
                    f"{all_slots_attr} is not bound (unsupported layer "
                    "configuration for multi-tenancy)"
                )
            # slot_ids arrives with one entry per ORIGINAL token, which is what
            # the gate-up GEMM wants: its A is (num_tokens, K) and it is invoked
            # with the real top_k, so the rotation kernel does the top_k
            # expansion itself.
            #
            # The DOWN GEMM is the other shape. Its A is the gate-up output,
            # already expanded to num_tokens * router_topk rows in TOKEN-MAJOR
            # order (row = token * router_topk + k -- see fused_moe.py's
            # `intermediate_cache1[: num_tokens * topk].view(num_tokens, topk, N)`
            # and `intermediate_cache3 = torch.empty((num_tokens, topk, ...))`),
            # and it is invoked with top_k=1, so the kernel expands nothing.
            # slot_ids must therefore arrive already per-row. `router_topk` is
            # threaded into kw at that call site for exactly this purpose.
            #
            # Not handled: `down_moe_use_tma` lays A out expert-SORTED and
            # block-padded (`c_sorted=down_moe_use_tma` on the gate-up GEMM), so
            # row != token * router_topk + k and A gains padding rows. That
            # layout already breaks the pre-existing single-slot down rotation
            # (which indexes A[sorted_ids] the same token-major way), so it is
            # out of scope here rather than newly broken; it is opt-in via
            # USE_TMA in a tuned down config. apply_oft_rotation_triton_multi_slot's
            # own length check rejects it loudly instead of mis-rotating,
            # because the padded A has more rows than the expansion produces.
            router_topk = kw.get("router_topk", 1)
            if is_down and router_topk != 1:
                slot_ids = slot_ids.repeat_interleave(router_topk)
            a, c, tw, ti, sti, ei, ntpp = _oft_prerotate_multi_tenant(
                A,
                oft_r_all_slots,
                slot_ids,
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
        else:
            a, c, tw, ti, sti, ei, ntpp = _oft_prerotate(
                A,
                oft_r,
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

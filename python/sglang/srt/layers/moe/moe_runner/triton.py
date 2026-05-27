from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_hip, is_xpu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = bool(int(os.getenv("SGLANG_USE_AITER", "0")))
_is_xpu = is_xpu()
_MOE_PADDING_SIZE = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


def _enable_fused_grouped_moe_oft_fc1() -> bool:
    value = os.getenv("SGLANG_OFT_ENABLE_FUSED_GROUPED_MOE_FC1", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _select_fused_grouped_moe_oft_fc1_tiles(
    num_tokens: int, default_block_m: int
) -> dict:
    if 2 <= num_tokens <= 16:
        return {
            "block_m": 8,
            "block_n": 64,
            "group_size_m": 1,
            "num_warps": 2,
        }
    return {"block_m": default_block_m}


def _make_config_expert_lora_compatible(config: dict, lora_inter_per_tp: int) -> dict:
    block_n = config.get("BLOCK_SIZE_N")
    if not block_n or lora_inter_per_tp % block_n == 0:
        return config

    for candidate in (64, 32, 16):
        if candidate <= block_n and lora_inter_per_tp % candidate == 0:
            compatible_config = dict(config)
            compatible_config["BLOCK_SIZE_N"] = candidate
            return compatible_config

    return config


@functools.lru_cache(maxsize=1)
def _load_fused_split_w13_oft_grouped_moe():
    # Hoisting this import out of the hot path: sglang.srt.oft.triton_ops/__init__
    # pulls in other OFT kernels.
    from sglang.srt.oft.triton_ops import fused_split_w13_oft_grouped_moe

    return fused_split_w13_oft_grouped_moe


@functools.lru_cache(maxsize=1)
def _load_packed_bmm_split_w13_oft_grouped_moe():
    from sglang.srt.oft.triton_ops import packed_bmm_split_w13_oft_grouped_moe

    return packed_bmm_split_w13_oft_grouped_moe


if _is_cuda or _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul

    if _is_hip:
        _has_vllm = False
        if _use_aiter:
            try:
                from aiter import moe_sum
            except ImportError:
                raise ImportError(
                    "aiter is required when SGLANG_USE_AITER is set to True"
                )
        else:
            try:
                from vllm import _custom_ops as vllm_ops  # moe_sum

                _has_vllm = True
            except ImportError:
                # Fallback: vllm not available, will use triton moe_sum
                _has_vllm = False
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_xpu:
    from sgl_kernel import moe_sum_reduce, silu_and_mul


if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel import (  # noqa: F401
        moe_align_block_size as sgl_moe_align_block_size,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    # Expert LoRA
    w13_lora_a: Optional[torch.Tensor] = None
    w13_lora_b: Optional[torch.Tensor] = None
    w2_lora_a: Optional[torch.Tensor] = None
    w2_lora_b: Optional[torch.Tensor] = None
    lora_scaling: float = 0.0
    # Expert OFT
    w13_oft_r: Optional[torch.Tensor] = None
    w1_oft_r: Optional[torch.Tensor] = None
    w3_oft_r: Optional[torch.Tensor] = None
    w2_oft_r: Optional[torch.Tensor] = None


# NOTE: zero-copy as_strided view into intermediate_cache1 was tried but does
# not land — invoke_fused_moe_kernel's ``C = C.reshape(-1, 1, C.shape[-1])``
# collapses the size-1 mid-dim stride to ``half``, so the two halves overlap.
# Needs kernel-side explicit output row-stride to revisit.


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
    ) -> TritonRunnerOutput:

        # TODO: move these functions to the triton runner
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            _megatron_compiled_weighted_swiglu,
            _swiglu_gpt_oss_sigmoid_alpha,
            _swiglu_silu_clamp_mul,
            invoke_fused_moe_kernel,
            moe_sum_reduce_torch_compile,
            moe_sum_reduce_triton,
        )

        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input
        apply_router_weight_after_activation = self.config.apply_router_weight_after_activation

        assert self.config.is_gated, "Only gated MoEs are supported for Triton runner"
        assert not (apply_router_weight_on_input and apply_router_weight_after_activation), (
            "apply_router_weight_on_input and apply_router_weight_after_activation "
            "cannot both be True — router weight would be applied twice"
        )

        M = hidden_states.shape[0]
        E, N, _ = w13.shape
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        intermediate_cache1 = torch.empty(
            (M, topk_ids.shape[1], N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        split_w13_oft = quant_info.w1_oft_r is not None or quant_info.w3_oft_r is not None
        if split_w13_oft and (
            use_fp8_w8a8
            or use_int8_w8a8
            or use_int8_w8a16
            or use_int4_w4a16
            or per_channel_quant
            or block_shape is not None
        ):
            raise RuntimeError(
                "Split expert gate/up OFT is currently implemented for BF16/unquantized "
                "FusedMoE only. Quantized split expert OFT needs a dedicated quantized "
                "first-GEMM path."
            )

        if split_w13_oft:
            if quant_info.w1_oft_r is None or quant_info.w3_oft_r is None:
                raise RuntimeError(
                    "Split expert gate/up OFT requires both w1_oft_r and w3_oft_r"
                )
            if quant_info.w13_oft_r is not None:
                raise RuntimeError(
                    "Split expert gate/up OFT cannot be active together with legacy w13_oft_r"
                )

            fast_split_w13_oft = (
                _enable_fused_grouped_moe_oft_fc1()
                and _is_cuda
                and hidden_states.is_cuda
                and b13 is None
                and not apply_router_weight_on_input
                and not use_fp8_w8a8
                and not use_int8_w8a8
                and not use_int8_w8a16
                and not use_int4_w4a16
                and not per_channel_quant
                and block_shape is None
                and hidden_states.dtype == torch.bfloat16
                and w13.dtype == torch.bfloat16
                and quant_info.w1_oft_r.dtype == torch.bfloat16
                and quant_info.w3_oft_r.dtype == torch.bfloat16
            )

            if fast_split_w13_oft:
                oft_fc1_tile_kwargs = running_state.get(
                    "oft_fc1_tile_kwargs",
                    {"block_m": running_state["config"]["BLOCK_SIZE_M"]},
                )
                intermediate_cache1 = _load_fused_split_w13_oft_grouped_moe()(
                    hidden_states=hidden_states,
                    w13=w13,
                    w1_oft_r=quant_info.w1_oft_r,
                    w3_oft_r=quant_info.w3_oft_r,
                    topk_ids=topk_ids,
                    sorted_token_ids=sorted_token_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    **oft_fc1_tile_kwargs,
                )
            else:
                half_cache_shape = (intermediate_cache1.shape[0], intermediate_cache1.shape[1], N // 2)
                for half_slice, oft_r in (
                    (slice(None, N // 2), quant_info.w1_oft_r),
                    (slice(N // 2, None), quant_info.w3_oft_r),
                ):
                    half_cache = torch.empty(
                        half_cache_shape,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    invoke_fused_moe_kernel(
                        hidden_states,
                        w13[:, half_slice, :].contiguous(),
                        None if b13 is None else b13[:, half_slice].contiguous(),
                        half_cache,
                        a13_scale,
                        None if w13_scale is None else w13_scale[:, half_slice].contiguous(),
                        w13_zp,
                        topk_weights,
                        topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        apply_router_weight_on_input,
                        topk_ids.shape[1],
                        running_state["config"],
                        compute_type=compute_type,
                        use_fp8_w8a8=False,
                        use_int8_w8a8=False,
                        use_int8_w8a16=False,
                        use_int4_w4a16=False,
                        per_channel_quant=False,
                        block_shape=None,
                        lora_a=None,
                        lora_b=None,
                        lora_scaling=0.0,
                        lora_inter_per_tp=N // 2,
                        oft_r=oft_r,
                    )
                    intermediate_cache1[..., half_slice].copy_(half_cache)
        else:
            invoke_fused_moe_kernel(
                hidden_states,
                w13,
                b13,
                intermediate_cache1,
                a13_scale,
                w13_scale,
                w13_zp,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                apply_router_weight_on_input,
                topk_ids.shape[1],
                running_state["config"],
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
                lora_a=quant_info.w13_lora_a,
                lora_b=quant_info.w13_lora_b,
                lora_scaling=quant_info.lora_scaling,
                lora_inter_per_tp=N // 2,  # gate/up boundary
                oft_r=quant_info.w13_oft_r,
            )

        intermediate_cache2 = torch.empty(
            (M * topk_ids.shape[1], N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        routed_weight_applied_after_activation = False
        if activation == "silu":
            if (
                apply_router_weight_after_activation
                and gemm1_alpha is None
                and gemm1_limit is None
            ):
                # Match Megatron Core's compiled weighted SwiGLU expression.
                # The separate silu_and_mul + multiply route rounds before
                # applying the routing weight and leaves a measurable parity gap.
                intermediate_cache2 = _megatron_compiled_weighted_swiglu(
                    intermediate_cache1.view(-1, N),
                    topk_weights.reshape(-1).unsqueeze(-1),
                )
                routed_weight_applied_after_activation = True
            elif gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(
                    intermediate_cache1.view(-1, N), gemm1_limit
                )
            elif _is_cuda or _is_hip or _is_xpu:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.silu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        elif activation == "gelu":
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        # Apply routing weights after activation (matching Megatron Core's
        # placement in mlp.py:325-328 and experts.py:320-322).
        # This ensures bit-wise identical outputs between training (Megatron)
        # and rollout (sglang) for RL infrastructure.
        if (
            apply_router_weight_after_activation
            and not routed_weight_applied_after_activation
        ):
            original_dtype = intermediate_cache2.dtype
            flat_weights = topk_weights.reshape(-1)
            intermediate_cache2 = intermediate_cache2 * flat_weights.unsqueeze(-1)
            intermediate_cache2 = intermediate_cache2.to(original_dtype)

        intermediate_cache3 = torch.empty(
            (M, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if no_combine:
            assert not inplace
            out_hidden_states = torch.empty(
                (M, topk_ids.shape[1], w2.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states.unsqueeze(0)
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input and not apply_router_weight_after_activation,
            1,
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            lora_a=quant_info.w2_lora_a,
            lora_b=quant_info.w2_lora_b,
            lora_scaling=quant_info.lora_scaling,
            lora_inter_per_tp=w2.shape[1],  # full output dim, no split
            oft_r=quant_info.w2_oft_r,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states,
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if M <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                )
            elif _has_vllm:
                vllm_ops.moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                )
            else:
                # Fallback: use triton moe_sum when vllm is not available
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
        elif _is_xpu:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )
        else:
            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )

        return TritonRunnerOutput(
            hidden_states=out_hidden_states,
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    output = fused_experts(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        b1=quant_info.b13,
        b2=quant_info.b2,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=quant_info.w13_zp,
        w2_zp=quant_info.w2_zp,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=quant_info.block_shape,
    )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # Standard no-A2A usually takes the fused shortcut. Expert adapters bypass
    # that shortcut because the legacy fused path does not accept adapter tensors.

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        get_config_dtype_str,
        moe_align_block_size,
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    num_tokens = hidden_states.shape[0]
    num_local_experts = runner_config.num_local_experts

    if (
        not (quant_info.use_fp8_w8a8 or quant_info.use_int8_w8a8)
        or quant_info.block_shape is not None
        or _use_aiter
    ):
        padding_size = 0
    else:
        padding_size = _MOE_PADDING_SIZE

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        quant_info.w13_weight.shape,
        (
            num_local_experts,
            quant_info.w2_weight.shape[1],
            quant_info.w2_weight.shape[2] - padding_size,
        ),
        topk_output.topk_ids.shape[1],
        config_dtype,
        block_shape=quant_info.block_shape,
        per_channel_quant=quant_info.per_channel_quant,
    )

    config = get_config_func(num_tokens)
    running_state.pop("oft_fc1_tile_kwargs", None)
    split_w13_oft = quant_info.w1_oft_r is not None or quant_info.w3_oft_r is not None
    if (
        split_w13_oft
        and _enable_fused_grouped_moe_oft_fc1()
        and _is_cuda
        and hidden_states.is_cuda
        and quant_info.b13 is None
        and not runner_config.apply_router_weight_on_input
        and not quant_info.use_fp8_w8a8
        and not quant_info.use_int8_w8a8
        and not quant_info.use_int8_w8a16
        and not quant_info.use_int4_w4a16
        and not quant_info.per_channel_quant
        and quant_info.block_shape is None
        and hidden_states.dtype == torch.bfloat16
        and quant_info.w13_weight.dtype == torch.bfloat16
        and quant_info.w1_oft_r is not None
        and quant_info.w3_oft_r is not None
        and quant_info.w1_oft_r.dtype == torch.bfloat16
        and quant_info.w3_oft_r.dtype == torch.bfloat16
    ):
        oft_fc1_tile_kwargs = _select_fused_grouped_moe_oft_fc1_tiles(
            num_tokens, config["BLOCK_SIZE_M"]
        )
        if oft_fc1_tile_kwargs["block_m"] != config["BLOCK_SIZE_M"]:
            config = dict(config)
            config["BLOCK_SIZE_M"] = oft_fc1_tile_kwargs["block_m"]
        running_state["oft_fc1_tile_kwargs"] = oft_fc1_tile_kwargs

    if quant_info.w13_lora_a is not None and quant_info.w13_lora_a.dim() == 4:
        config = _make_config_expert_lora_compatible(
            config, quant_info.w13_weight.shape[1] // 2
        )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_output.topk_ids, config["BLOCK_SIZE_M"], num_local_experts
    )

    running_state["config"] = config

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # Standard no-A2A usually takes the fused shortcut. Expert adapters bypass
    # that shortcut because the legacy fused path does not accept adapter tensors.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )

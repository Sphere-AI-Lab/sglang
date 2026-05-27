from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatcherBaseHooks,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_deepep_config,
    get_moe_runner_backend,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_hip,
    is_npu,
    load_json_config,
)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    from deep_ep import ElasticBuffer

    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True
except ImportError:
    use_deepep = False

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


class DeepEPPDispatchHooks(DispatcherBaseHooks):

    def __call__(self, dispatcher: BaseDispatcher):
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class DeepEPNormalDispatchOutput(NamedTuple):
    """DeepEP normal dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class DeepEPLLDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)


class DeepEPNormalCombineInput(NamedTuple):
    """DeepEP normal combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


class DeepEPLLCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalCombineInput, CombineInput)
assert isinstance(DeepEPLLCombineInput, CombineInput)


class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    _buffer = None
    _cuda_graph_buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None
    _num_topk: Optional[int] = None
    _use_fp8_dispatch: Optional[bool] = None

    @classmethod
    def mark_current_buffer_as_cuda_graph_owned(cls):
        cls._cuda_graph_buffer = cls._buffer

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        num_topk: int = -1,
        use_fp8_dispatch: bool = False,
    ):
        _ = param_bytes, deepep_mode
        assert num_max_dispatch_tokens_per_rank != -1
        assert num_experts != -1
        assert num_topk != -1

        allow_hybrid_mode = envs.SGLANG_DEEPEP_ALLOW_HYBRID_MODE.get()
        required_bytes = ElasticBuffer.get_buffer_size_hint(
            group,
            num_max_dispatch_tokens_per_rank,
            hidden_size,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=allow_hybrid_mode,
        )
        is_cuda_graph_capture = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if (
            cls._buffer is not None
            and cls._buffer.group == group
            and cls._buffer.num_bytes >= required_bytes
            and cls._hidden_size == hidden_size
            and cls._num_experts == num_experts
            and cls._num_topk == num_topk
            and cls._use_fp8_dispatch == use_fp8_dispatch
        ):
            cls._num_max_dispatch_tokens_per_rank = max(
                cls._num_max_dispatch_tokens_per_rank or 0,
                num_max_dispatch_tokens_per_rank,
            )
            return cls._buffer

        old_buffer = cls._buffer
        if old_buffer is not None and old_buffer is not cls._cuda_graph_buffer:
            cls._destroy_buffer(old_buffer)

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts
        cls._num_topk = num_topk
        cls._use_fp8_dispatch = use_fp8_dispatch
        cls._buffer = ElasticBuffer(
            group,
            num_bytes=required_bytes,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=allow_hybrid_mode,
            num_allocated_qps=0 if allow_hybrid_mode else 17,
            explicitly_destroy=True,
        )
        if is_cuda_graph_capture:
            cls._cuda_graph_buffer = cls._buffer
        return cls._buffer

    @classmethod
    def clean_buffer(cls):
        buffers = []
        for buffer in (cls._buffer, cls._cuda_graph_buffer):
            if buffer is not None and not any(buffer is item for item in buffers):
                buffers.append(buffer)

        cls._buffer = None
        cls._cuda_graph_buffer = None
        cls._dispatch_mode = None
        cls._hidden_size = None
        cls._num_max_dispatch_tokens_per_rank = None
        cls._num_experts = None
        cls._num_topk = None
        cls._use_fp8_dispatch = None

        if not buffers:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for buffer in buffers:
            cls._destroy_buffer(buffer)

    @staticmethod
    def _destroy_buffer(buffer):
        for method_name in ("destroy", "close"):
            method = getattr(buffer, method_name, None)
            if callable(method):
                method()
                break

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY

    @classmethod
    def set_dispatch_mode(cls, mode: DeepEPMode):
        if mode.is_low_latency():
            cls.set_dispatch_mode_as_low_latency()
        elif mode.is_normal():
            cls.set_dispatch_mode_as_normal()
        else:
            raise Exception("unsupported mode")


class DeepEPConfig(BaseDispatcherConfig):
    _instance = None

    def __init__(self):
        config_str = get_deepep_config()
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            self.num_sms = int(
                config_parsed.get(
                    "num_sms",
                    config_parsed.get("normal_dispatch", {}).get("num_sms", 0),
                )
            )
        else:
            self.num_sms = 0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
        force_bf16_dispatch: bool = False,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.force_bf16_dispatch = force_bf16_dispatch

        self.params_bytes = 2
        # Minimum DeepEP buffer capacity. DeepEP v2 also requires the runtime
        # dispatch capacity to cover the actual max local token count across
        # EP ranks, so _dispatch_core raises this value dynamically.
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

        self.handle = None

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(
        self,
        num_max_dispatch_tokens_per_rank: Optional[int] = None,
        use_fp8_dispatch: bool = False,
    ):
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None

    def _get_num_max_tokens_per_rank(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> int:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return max(self.num_max_dispatch_tokens_per_rank, num_tokens)
        local = torch.tensor([num_tokens], device=device, dtype=torch.int32)
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=self.group)
        return max(self.num_max_dispatch_tokens_per_rank, int(local.item()))


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and not get_moe_runner_backend().is_cutlass()
            and not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            and not self.force_bf16_dispatch
        ):
            # TODO hard code 128 block quant,use fp8 communication
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        previous_event = ElasticBuffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        (
            hidden_states,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return DeepEPNormalDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        use_fp8_dispatch = isinstance(x, tuple)
        local_x = x[0] if use_fp8_dispatch else x
        num_max_dispatch_tokens_per_rank = self._get_num_max_tokens_per_rank(
            local_x.shape[0],
            local_x.device,
        )
        is_cuda_graph_capture = torch.cuda.is_current_stream_capturing()
        buffer = self._get_buffer(
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch=use_fp8_dispatch,
        )
        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.

        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights.float(),
            num_experts=self.num_experts,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            previous_event=previous_event,
            async_with_compute_stream=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            num_sms=DeepEPConfig.get_instance().num_sms,
            do_cpu_sync=not is_cuda_graph_capture,
        )
        if is_cuda_graph_capture:
            num_valid_recv_tokens = self.handle.psum_num_recv_tokens_per_scaleup_rank[
                -1
            ]
            row_ids = torch.arange(
                recv_topk_ids.shape[0],
                device=recv_topk_ids.device,
                dtype=torch.int32,
            )
            valid_recv_mask = row_ids < num_valid_recv_tokens.to(torch.int32)
            recv_topk_ids = recv_topk_ids.masked_fill(
                ~valid_recv_mask[:, None], -1
            )
            recv_topk_weights = recv_topk_weights.masked_fill(
                ~valid_recv_mask[:, None], 0.0
            )
        num_recv_tokens_per_expert = self.handle.num_recv_tokens_per_expert_list
        if not is_cuda_graph_capture:
            get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
                num_recv_tokens_per_expert,
                num_tokens_per_rank=None,
                num_tokens_per_rdma_rank=None,
                num_tokens_per_expert=None,
            )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):

        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()  # triton runner was supported but it's temporarily disabled

        previous_event = ElasticBuffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer(
            num_max_dispatch_tokens_per_rank=self.handle.num_max_tokens_per_rank,
        )
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_with_compute_stream=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            num_sms=DeepEPConfig.get_instance().num_sms,
        )
        return combined_x, event

    def _get_buffer(
        self,
        num_max_dispatch_tokens_per_rank: Optional[int] = None,
        use_fp8_dispatch: bool = False,
    ):
        DeepEPBuffer.set_dispatch_mode_as_normal()
        if num_max_dispatch_tokens_per_rank is None:
            num_max_dispatch_tokens_per_rank = self.num_max_dispatch_tokens_per_rank

        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.router_topk,
            use_fp8_dispatch=use_fp8_dispatch,
        )


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        force_bf16_dispatch: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
            force_bf16_dispatch=force_bf16_dispatch,
        )

        self._normal_dispatcher = _DeepEPDispatcherImplNormal(
            async_finish=async_finish,
            **common_kwargs,
        )
        self._low_latency_dispatcher = None

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode in (DeepEPMode.NORMAL, DeepEPMode.LOW_LATENCY):
            # DeepEP v2 exposes one ElasticBuffer dispatch/combine API. Keep the
            # scheduler mode decision, but use the v2 normal-format path for both.
            return self._normal_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        self._normal_dispatcher.set_overlap_args(
            combine_overlap_args, meta_overlap_args
        )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)

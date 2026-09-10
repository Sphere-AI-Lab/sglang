"""CPU regressions for OFT logits and row-parallel serving contracts."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[4]


class _Linear(nn.Module):
    pass


class _ColumnParallelLinear(_Linear):
    pass


class _MergedColumnParallelLinear(_ColumnParallelLinear):
    pass


class _QKVParallelLinear(_ColumnParallelLinear):
    pass


class _ReplicatedLinear(_Linear):
    pass


class _RowParallelLinear(_Linear):
    pass


class _VocabParallelEmbedding(nn.Module):
    pass


class _ParallelLMHead(_VocabParallelEmbedding):
    pass


class _ForwardBatch:
    pass


def _module(**attrs):
    module = ModuleType("test_dependency")
    module.__dict__.update(attrs)
    return module


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_production_modules():
    """Load the production files while replacing unavailable GPU imports."""

    def noop(*args, **kwargs):
        return None

    original_modules = dict(sys.modules)
    try:
        for package in (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.activation",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.layers.moe",
            "sglang.srt.distributed",
            "sglang.srt.model_executor",
            "sglang.srt.oft",
            "sglang.srt.oft.backend",
            "sglang.srt.utils",
        ):
            sys.modules[package] = _module(__path__=[])

        sys.modules["sglang.srt.distributed"] = _module(
            get_tensor_model_parallel_rank=lambda: 0,
            split_tensor_along_last_dim=lambda x, num_partitions: torch.chunk(
                x, num_partitions, dim=-1
            ),
            tensor_model_parallel_all_gather=noop,
            tensor_model_parallel_all_reduce=lambda x: x,
            get_tp_group=noop,
        )
        sys.modules["sglang.srt.layers.linear"] = _module(
            ColumnParallelLinear=_ColumnParallelLinear,
            MergedColumnParallelLinear=_MergedColumnParallelLinear,
            QKVParallelLinear=_QKVParallelLinear,
            ReplicatedLinear=_ReplicatedLinear,
            RowParallelLinear=_RowParallelLinear,
        )
        sys.modules["sglang.srt.layers.vocab_parallel_embedding"] = _module(
            ParallelLMHead=_ParallelLMHead,
            VocabParallelEmbedding=_VocabParallelEmbedding,
        )
        sys.modules["sglang.srt.oft.backend.base_backend"] = _module(
            BaseOFTBackend=object
        )
        sys.modules["sglang.srt.oft.utils"] = _module(OFTBatchInfo=object)
        sys.modules["sglang.srt.layers.moe.utils"] = _module(
            should_skip_mlp_all_reduce=lambda: False
        )
        sys.modules["sglang.srt.runtime_context"] = _module(get_parallel=lambda: None)

        oft_layers = _load_module(
            "oft_layers_under_test", "python/sglang/srt/oft/layers.py"
        )

        class _CaptureHiddenMode:
            NULL = object()

        class _DpPaddingMode:
            SUM_LEN = object()

        class _LogprobStage:
            PREFILL = object()

        sys.modules["sglang.kernels.ops.activation.softcap"] = _module(
            softcap_inplace_logits=noop
        )
        sys.modules["sglang.srt.distributed.device_communicators"] = _module(
            triton_symm_mem_ag=SimpleNamespace()
        )
        sys.modules["sglang.srt.layers.aux_hidden_states"] = _module(
            AuxHiddenStates=object,
            pack_aux_hidden_states=noop,
        )
        sys.modules["sglang.srt.layers.dp_attention"] = _module(
            DpPaddingMode=_DpPaddingMode,
            attn_tp_all_gather=noop,
            attn_tp_all_gather_into_tensor=noop,
            dp_gather_replicate=noop,
            dp_scatter=noop,
            get_dp_device=noop,
            get_dp_dtype=noop,
            get_dp_hidden_size=noop,
        )
        sys.modules["sglang.srt.layers.logprob_processor"] = _module(
            InputLogprobProcessor=object,
            LogprobStage=_LogprobStage,
            get_token_ids_logprobs_raw=noop,
            get_top_logprobs_raw=noop,
        )
        sys.modules["sglang.srt.model_executor.forward_batch_info"] = _module(
            CaptureHiddenMode=_CaptureHiddenMode,
            ForwardBatch=_ForwardBatch,
            ForwardMode=object,
        )
        sys.modules["sglang.srt.runtime_context"] = _module(
            get_exec=noop,
            get_parallel=noop,
        )
        sys.modules["sglang.srt.true_on_policy"] = _module(
            should_force_bfloat16_lm_head=lambda **kwargs: False
        )
        sys.modules["sglang.srt.utils.common"] = _module(
            is_cpu=lambda: True,
            is_npu=lambda: False,
            is_pin_memory_available=lambda: False,
            use_intel_amx_backend=lambda layer: False,
        )
        logits_processor = _load_module(
            "logits_processor_under_test",
            "python/sglang/srt/layers/logits_processor.py",
        )
        return oft_layers, logits_processor
    finally:
        loaded = {
            name: module
            for name, module in sys.modules.items()
            if name in {"oft_layers_under_test", "logits_processor_under_test"}
        }
        sys.modules.clear()
        sys.modules.update(original_modules)
        sys.modules.update(loaded)


OFT_LAYERS, LOGITS_PROCESSOR = _load_production_modules()
CI_REGISTER = _load_module(
    "ci_register_under_test", "python/sglang/test/ci/ci_register.py"
)
register_cpu_ci = CI_REGISTER.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TorchRotationBackend:
    def run_oft_r_sgemm(self, x, weights):
        return x @ weights.to(x.dtype)


class _SegmentedTorchRotationBackend:
    def __init__(self, segment_lengths):
        self.segment_lengths = segment_lengths

    def run_oft_r_sgemm(self, x, weights):
        if x.shape[0] != sum(self.segment_lengths):
            raise RuntimeError("OFT segment metadata does not match hidden-state rows")
        segments = torch.split(x, self.segment_lengths)
        return torch.cat(
            [segment @ rotation for segment, rotation in zip(segments, weights)]
        )


class _QuantizedLMHeadMethod:
    def apply(self, layer, hidden_states, bias=None):
        if getattr(layer, "quantization_marker", None) != "base-lm-head":
            raise AssertionError("quantization method received the OFT wrapper")
        return hidden_states.float() @ layer.dequantized_weight.T + layer.runtime_bias


def _lm_head(*, quantized=False, backend=None, rotations=None, weight=None):
    base = _ParallelLMHead()
    base.embedding_dim = 2
    base.org_vocab_size = 2
    base.tp_size = 1
    base.bias = None
    if quantized:
        base.weight = torch.tensor([[11, 12], [13, 14]], dtype=torch.int8)
        base.dequantized_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        base.runtime_bias = torch.tensor([5.0, 7.0])
        base.quantization_marker = "base-lm-head"
        base.quant_method = _QuantizedLMHeadMethod()
    else:
        base.weight = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]) if weight is None else weight
        )
        base.quant_method = SimpleNamespace()
    wrapper = OFT_LAYERS.ParallelLMHeadWithOFT(
        base, backend if backend is not None else _TorchRotationBackend()
    )
    wrapper.set_oft_info(
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]]) if rotations is None else rotations
    )
    return wrapper


def _processor(*, fp32=False, return_full_logits=False):
    processor = LOGITS_PROCESSOR.LogitsProcessor.__new__(
        LOGITS_PROCESSOR.LogitsProcessor
    )
    nn.Module.__init__(processor)
    processor.use_fp32_lm_head = fp32
    processor.vocab_size = 2
    processor.logit_scale = None
    processor.final_logit_softcapping = None
    processor.do_tensor_parallel_all_gather = False
    processor.do_tensor_parallel_all_gather_dp_attn = False
    processor.use_attn_tp_group = False
    processor.return_full_logits = return_full_logits
    return processor


class _ForwardMode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode_or_idle(self):
        return self.kind == "decode"

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False

    def is_extend(self):
        return self.kind == "extend"

    def is_dllm_extend(self):
        return self.kind == "dllm"


class _CaptureMode:
    def __init__(self, kind):
        self.kind = kind

    def need_capture(self):
        return self.kind != "none"

    def is_full(self):
        return self.kind == "full"

    def is_last(self):
        return self.kind == "last"


def _metadata(*, mode="extend", capture="none", is_prefill_only=False):
    return LOGITS_PROCESSOR.LogitsMetadata(
        forward_mode=_ForwardMode(mode),
        capture_hidden_mode=_CaptureMode(capture),
        extend_return_logprob=False,
        extend_seq_lens=torch.tensor([3, 2]),
        extend_seq_lens_cpu=[3, 2],
        extend_logprob_start_lens_cpu=[0, 0],
        is_prefill_only=is_prefill_only,
    )


def _segmented_case(*, quantized=False, weight=None):
    hidden_states = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 5.0]]
    )
    rotations = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]], [[0.0, -1.0], [1.0, 0.0]]])
    head = _lm_head(
        quantized=quantized,
        backend=_SegmentedTorchRotationBackend([3, 2]),
        rotations=rotations,
        weight=weight,
    )
    return hidden_states, head


@pytest.mark.parametrize(
    "capture, expected_capture",
    [
        (
            "full",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 4.0],
                    [0.0, 5.0],
                ]
            ),
        ),
        ("last", torch.tensor([[3.0, 0.0], [0.0, 5.0]])),
    ],
)
def test_forward_rotates_full_segmented_batch_before_pruning_without_rotating_capture(
    capture, expected_capture
):
    # Break caught: pruning [3, 2] to two rows before OFT leaves the backend's
    # segment metadata at [0, 3, 5], while capture must remain unrotated.
    hidden_states, lm_head = _segmented_case(quantized=True)

    output = _processor().forward(
        torch.arange(5), hidden_states, lm_head, _metadata(capture=capture)
    )

    torch.testing.assert_close(
        output.next_token_logits, torch.tensor([[11.0, 19.0], [10.0, 22.0]])
    )
    torch.testing.assert_close(output.hidden_states, expected_capture)


def test_logits_dispatch_keeps_fp32_lm_head_precision_after_oft_rotation():
    # Break caught: routing the wrapper's forward skips LogitsProcessor's
    # explicit FP32 LM-head contract.
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float16)

    output = _processor(fp32=True).forward(
        torch.tensor([0]), hidden_states, _lm_head(), _metadata(mode="decode")
    )

    assert output.next_token_logits.dtype == torch.float32
    torch.testing.assert_close(output.next_token_logits, torch.tensor([[0.0, -2.0]]))


def test_multi_item_scoring_rotates_before_delimiter_slicing(monkeypatch):
    # Break caught: MIS delimiter slicing must not shrink the tensor before
    # segment-aware OFT rotation.
    hidden_states, lm_head = _segmented_case(weight=torch.eye(2))
    metadata = _metadata(is_prefill_only=True)
    metadata.extend_return_top_logprob = True
    metadata.top_logprobs_nums = [1, 1]
    batch = _ForwardBatch()
    batch.multi_item_delimiter_indices = [torch.tensor([1]), torch.tensor([1])]
    batch.metadata = metadata
    monkeypatch.setattr(
        LOGITS_PROCESSOR.LogitsMetadata,
        "from_forward_batch",
        classmethod(lambda cls, value: value.metadata),
    )
    monkeypatch.setattr(
        LOGITS_PROCESSOR,
        "get_top_logprobs_raw",
        lambda logprobs, *args, **kwargs: (logprobs, None),
    )

    output = _processor().forward(torch.arange(5), hidden_states, lm_head, batch)

    torch.testing.assert_close(
        output.input_top_logprobs_val,
        torch.tensor([[-1.3132616, -0.31326166], [-0.01814996, -4.01815]]),
    )


def test_dllm_rotates_full_hidden_states_before_full_logits():
    # Break caught: moving rotation into only the common pruning path would
    # leave diffusion full logits unadapted.
    hidden_states, lm_head = _segmented_case(weight=torch.eye(2))

    output = _processor(return_full_logits=True).forward(
        torch.arange(5), hidden_states, lm_head, _metadata(mode="dllm")
    )

    torch.testing.assert_close(
        output.full_logits,
        torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [4.0, 0.0], [5.0, 0.0]]),
    )


class _DenseMethod:
    def apply(self, layer, hidden_states, bias=None):
        return F.linear(hidden_states, layer.weight, bias)


def _row_wrapper(*, rank, reduce_results, use_dp_attention_reduce=False):
    base = _RowParallelLinear()
    base.input_is_parallel = True
    base.input_size = 2
    base.input_size_per_partition = 2
    base.output_size = 1
    base.tp_size = 2
    base.tp_rank = rank
    base.reduce_results = reduce_results
    base.use_decode_attn_tp = False
    base.use_dp_attention_reduce = use_dp_attention_reduce
    base.skip_bias_add = False
    base.weight = torch.tensor([[1.0, 1.0]])
    base.bias = torch.tensor([7.0])
    base.quant_method = _DenseMethod()
    return OFT_LAYERS.RowParallelLinearWithOFT(base, _TorchRotationBackend())


def test_row_parallel_oft_defers_decoder_owned_reduction(monkeypatch):
    # Break caught: eager OFT reduction followed by decoder fusion/scatter
    # reduces the same local contribution twice.
    wrapper = _row_wrapper(rank=0, reduce_results=True)
    wrapper.set_oft_info(torch.eye(2))
    monkeypatch.setattr(OFT_LAYERS, "should_skip_mlp_all_reduce", lambda: True)
    monkeypatch.setattr(
        OFT_LAYERS, "tensor_model_parallel_all_reduce", lambda output: output * 10
    )

    output, _ = wrapper(torch.tensor([[2.0, 3.0]]))

    torch.testing.assert_close(output, torch.tensor([[12.0]]))


@pytest.mark.parametrize(
    "rank, expected",
    [(0, torch.tensor([[12.0]])), (1, torch.tensor([[5.0]]))],
)
def test_row_parallel_oft_adds_deferred_bias_on_rank_zero_only(rank, expected):
    # Break caught: adding replicated bias on every rank over-counts it when a
    # later attention communicator combines local outputs.
    wrapper = _row_wrapper(rank=rank, reduce_results=False)

    output, _ = wrapper(torch.tensor([[2.0, 3.0]]))

    torch.testing.assert_close(output, expected)


def test_row_parallel_oft_uses_attention_tp_reduction_group(monkeypatch):
    # Break caught: DP-attention row shards reduced over global TP mix tokens
    # from different attention groups.
    wrapper = _row_wrapper(rank=0, reduce_results=True, use_dp_attention_reduce=True)
    monkeypatch.setattr(OFT_LAYERS, "should_skip_mlp_all_reduce", lambda: False)
    monkeypatch.setattr(
        OFT_LAYERS, "tensor_model_parallel_all_reduce", lambda output: output + 1000
    )
    monkeypatch.setattr(
        OFT_LAYERS,
        "get_parallel",
        lambda: SimpleNamespace(
            attn_tp_group=SimpleNamespace(all_reduce=lambda output: output + 100)
        ),
    )

    output, _ = wrapper(torch.tensor([[2.0, 3.0]]))

    torch.testing.assert_close(output, torch.tensor([[112.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

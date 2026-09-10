"""Canonical split rotations must never use a quantized shared-input GEMM."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.oft.backend.torch_backend import TorchNativeOFTBackend
from sglang.srt.oft.layers import get_oft_layer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _QuantizedMethod:
    """CPU GEMM boundary for testing wrapper dispatch without quantized kernels."""

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


def _layer(layer_type, *, has_weight=True, skip_bias_add=False):
    # Base constructors allocate TP/device resources. Keep their real types
    # for factory dispatch and supply the metadata for a local dense shard.
    layer = layer_type.__new__(layer_type)
    torch.nn.Module.__init__(layer)
    layer.quant_method = _QuantizedMethod()
    layer.input_size = 12
    layer.output_sizes = [4, 2, 2] if layer_type is QKVParallelLinear else [4, 4]
    layer.output_size = 8
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.skip_bias_add = skip_bias_add
    layer.gather_output = False
    layer.bias = torch.arange(8).float()
    if has_weight:
        layer.weight = torch.arange(96).reshape(8, 12).float() / 100
    return layer


@pytest.mark.parametrize("layer_type", [QKVParallelLinear, MergedColumnParallelLinear])
@pytest.mark.parametrize("has_weight", [False, True])
def test_canonical_quantized_fused_projection_rejected_before_adapter_load(
    layer_type, has_weight
):
    layer = _layer(layer_type, has_weight=has_weight)

    # Rejection must happen during model wrapping, before an adapter can be
    # admitted whose K/V or up rotation would silently be replaced by Q/gate.
    with pytest.raises(NotImplementedError, match="[Cc]anonical.*[Oo][Ff][Tt].*quant"):
        get_oft_layer(layer, None, "canonical_oft")


@pytest.mark.parametrize("layer_type", [QKVParallelLinear, MergedColumnParallelLinear])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_quantized_shared_oft_still_rotates_the_entire_input(layer_type, skip_bias_add):
    layer = _layer(layer_type, skip_bias_add=skip_bias_add)
    backend = TorchNativeOFTBackend(1, torch.device("cpu"))
    backend.batch_info = SimpleNamespace(
        num_segments=1,
        weight_indices=torch.tensor([0]),
        seg_lens=torch.tensor([2]),
        oft_block_sizes=torch.tensor([2]),
    )
    wrapper = get_oft_layer(layer, backend, "oft")
    # Six blocks are divisible by both 2 and 3. A prefix-based fallback must
    # not reinterpret this shared rotation as independent stacked branches.
    block = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotations = block.repeat(6, 1, 1)
    wrapper.set_oft_info(rotations.unsqueeze(0))
    x = torch.arange(24).reshape(2, 12).float() / 10

    actual, bias = wrapper(x)

    rotated = x @ torch.block_diag(*rotations.unbind())
    expected = F.linear(rotated, layer.weight, None if skip_bias_add else layer.bias)
    torch.testing.assert_close(actual, expected)
    if skip_bias_add:
        torch.testing.assert_close(bias, layer.bias)
    else:
        assert bias is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

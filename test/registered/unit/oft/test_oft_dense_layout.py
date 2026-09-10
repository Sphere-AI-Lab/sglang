"""Dense OFT must choose shared versus split rotations from the adapter type."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from sglang.srt.oft.backend.torch_backend import TorchNativeOFTBackend
from sglang.srt.oft.layers import get_oft_layer
from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


TARGETS = [
    ("qkv_proj", QKVParallelLinear, (4, 2, 2)),
    ("gate_up_proj", MergedColumnParallelLinear, (4, 4)),
    ("fused_qkv_a_proj_with_mqa", ReplicatedLinear, (3, 5)),
]


class _FakeUnquantizedMethod:
    """CPU GEMM boundary; avoids device-specific base-linear initialization."""

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


def _layer(layer_type, output_sizes, hidden, tp_size=1, skip_bias_add=False):
    # Constructors allocate distributed/device resources. The wrapper only
    # needs this local shard and the real base type for factory dispatch.
    layer = layer_type.__new__(layer_type)
    torch.nn.Module.__init__(layer)
    output_dim = sum(output_sizes)
    layer.weight = (
        torch.arange(output_dim * hidden).reshape(output_dim, hidden).float() / 100
    )
    layer.bias = torch.arange(output_dim).float() / 10
    layer.input_size = hidden
    layer.output_size = output_dim * tp_size
    layer.skip_bias_add = skip_bias_add
    layer.quant_method = _FakeUnquantizedMethod()
    if layer_type is not ReplicatedLinear:
        layer.output_sizes = [size * tp_size for size in output_sizes]
        layer.tp_size = tp_size
        layer.tp_rank = 0
        layer.gather_output = False
    return layer


def _pool(oft_type):
    pool = object.__new__(OFTMemoryPool)
    pool.oft_type = oft_type
    pool.max_ofts_per_batch = 3
    pool.max_oft_block_size = 2
    pool.tp_size = 1
    pool.tp_rank = 0
    return pool


@pytest.mark.parametrize("target,layer_type,output_sizes", TARGETS)
@pytest.mark.parametrize("blocks", [2, 3, 6])
@pytest.mark.parametrize("oft_type", ["oft", "canonical_oft"])
def test_dense_buffer_shape_respects_shared_or_split_type(
    target, layer_type, output_sizes, blocks, oft_type
):
    pool = _pool(oft_type)
    module = SimpleNamespace(get_oft_input_dim=lambda: blocks * 2)
    shape = pool.get_oft_R_shape(target, None, 2, 0, module)
    slices = 1 if oft_type == "oft" else len(output_sizes)
    assert shape == (3, blocks * slices, 2, 2)


@pytest.mark.parametrize("target,layer_type,output_sizes", TARGETS)
@pytest.mark.parametrize("blocks", [2, 3, 6])
@pytest.mark.parametrize("oft_type", ["oft", "canonical_oft"])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_factory_forward_uses_all_shared_blocks_or_independent_slices(
    target, layer_type, output_sizes, blocks, oft_type, skip_bias_add
):
    hidden = blocks * 2
    tp_size = 1 if layer_type is ReplicatedLinear else 2
    layer = _layer(layer_type, output_sizes, hidden, tp_size, skip_bias_add)
    backend = TorchNativeOFTBackend(1, torch.device("cpu"))
    backend.batch_info = SimpleNamespace(
        num_segments=1,
        weight_indices=torch.tensor([0]),
        seg_lens=torch.tensor([2]),
        oft_block_sizes=torch.tensor([2]),
    )
    wrapper = get_oft_layer(layer, backend, oft_type)
    if layer_type is ReplicatedLinear and oft_type == "canonical_oft":
        wrapper.first_output_dim = output_sizes[0]
    # Six blocks are divisible by BOTH 2 and 3: shape modulo cannot select
    # the layout. Distinct rotations also catch silently using just a prefix.
    choices = torch.tensor([[[0.0, -1.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, -1.0]]])
    rotations = choices[torch.arange(blocks) % 2]
    slices = 1 if oft_type == "oft" else len(output_sizes)
    per_slice = [rotations if i % 2 == 0 else -rotations for i in range(slices)]
    wrapper.set_oft_info(torch.cat(per_slice).unsqueeze(0))
    x = torch.arange(2 * hidden).reshape(2, hidden).float() / 10

    actual, bias = wrapper(x)

    weights = layer.weight.split(output_sizes)
    expected_parts = []
    for i, weight in enumerate(weights):
        rotation = per_slice[0 if oft_type == "oft" else i]
        rotated = x @ torch.block_diag(*rotation.unbind())
        expected_parts.append(F.linear(rotated, weight))
    expected = torch.cat(expected_parts, dim=-1)
    if not skip_bias_add:
        expected += layer.bias
    torch.testing.assert_close(actual, expected)
    if skip_bias_add:
        torch.testing.assert_close(bias, layer.bias)
    else:
        assert bias is None


@pytest.mark.parametrize("boundary", [None, -1, 0, 8, 9, 2.5])
def test_canonical_replicated_requires_a_valid_output_boundary(boundary):
    layer = _layer(ReplicatedLinear, (3, 5), 12)
    wrapper = get_oft_layer(layer, None, "canonical_oft")
    if boundary is not None:
        wrapper.first_output_dim = boundary
    wrapper.set_oft_info(torch.eye(2).repeat(1, 12, 1, 1))

    with pytest.raises(ValueError, match="first_output_dim.*output_size"):
        wrapper(torch.zeros(2, 12))


@pytest.mark.parametrize("oft_type", ["oft", "canonical_oft"])
@pytest.mark.parametrize("nested_config", [False, True])
def test_manager_sets_replicated_boundary_from_q_lora_rank(oft_type, nested_config):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    projection = _layer(ReplicatedLinear, (3, 5), 12)
    model.layers[0].fused_qkv_a_proj_with_mqa = projection
    manager = object.__new__(OFTManager)
    manager.base_model = model
    config = SimpleNamespace(num_hidden_layers=1, q_lora_rank=3)
    manager.base_hf_config = (
        SimpleNamespace(text_config=config) if nested_config else config
    )
    manager.target_modules = {"fused_qkv_a_proj_with_mqa"}
    manager.oft_backend = None
    manager.oft_type = oft_type

    manager.init_oft_modules()

    wrapper = model.layers[0].fused_qkv_a_proj_with_mqa
    assert wrapper.first_output_dim == 3
    assert wrapper.base_layer is projection
    assert manager.adapter_modules[0]["layers.0.fused_qkv_a_proj_with_mqa"] is wrapper


@pytest.mark.parametrize("target,layer_type,output_sizes", TARGETS)
@pytest.mark.parametrize("oft_type", ["oft", "canonical_oft"])
def test_immediate_direct_and_staged_full_compacts_write_identically(
    target, layer_type, output_sizes, oft_type
):
    pool = _pool(oft_type)
    pool.target_modules = {target}
    module = SimpleNamespace(get_oft_input_dim=lambda: 12)
    shape = pool.get_oft_R_shape(target, None, 2, 0, module)
    storage = torch.full(shape, float("nan"))
    pool._groups = {f"R:{target}": {0: storage}}
    slices = 1 if oft_type == "oft" else len(output_sizes)
    compact = torch.arange(1, 6 * slices + 1).reshape(-1, 1).float() / 100
    r = precompute_oft_r(compact, 2)

    pool._precompute_and_store_R(storage[0], compact, 2)
    pool.load_oft_weight_direct(1, f"model.layers.0.{target}.oft_R", compact, 2, [], 0)
    pool._fill_slot(2, {(target, 0): (r, 2, None, 1)})

    assert torch.equal(storage[0], storage[1])
    assert torch.equal(storage[0], storage[2])
    assert torch.equal(storage[0], r)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

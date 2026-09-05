"""Dense single-adapter Triton batches do not need MoE batch metadata."""

from types import SimpleNamespace

import torch

from sglang.srt.oft.backend.triton_backend import TritonOFTBackend
from sglang.srt.oft.oft_manager import OFTManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_dense_uniform_triton_batch_without_segmented_batch_info():
    backend = TritonOFTBackend(max_ofts_per_batch=2, device=torch.device("cpu"))
    backend.prepare_oft_batch(
        forward_batch=SimpleNamespace(batch_size=1),
        weight_indices=[1],
        oft_block_sizes=[0, 4],
        use_cuda_graph=False,
    )
    assert backend._use_single_adapter_fast_path
    assert not hasattr(backend, "batch_info")

    manager = OFTManager.__new__(OFTManager)
    manager.oft_backend = backend
    manager._moe_modules = {}  # Dense Qwen has no MoE modules to receive metadata.
    manager._push_moe_multi_tenant_batch_info()


def test_moe_modules_still_receive_prepared_batch_info():
    batch_info = object()
    module = SimpleNamespace()
    manager = OFTManager.__new__(OFTManager)
    manager.oft_backend = SimpleNamespace(batch_info=batch_info)
    manager._moe_modules = {0: module}
    manager.max_ofts_per_batch = 4

    manager._push_moe_multi_tenant_batch_info()

    assert module._oft_moe_multi_tenant_batch_info is batch_info
    assert module._oft_max_ofts_per_batch == 4

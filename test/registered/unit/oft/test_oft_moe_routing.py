from types import SimpleNamespace

import pytest
import torch

from sglang.srt.oft.backend.base_backend import _compute_moe_oft_info
from sglang.srt.oft.backend.torch_backend import TorchNativeOFTBackend
from sglang.srt.oft.backend.triton_backend import TritonOFTBackend


def test_builds_moe_mapping_for_base_and_two_adapters():
    info = _compute_moe_oft_info(
        num_tokens=5,
        seg_indptr=torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        weight_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_ofts=3,
    )

    assert info.adapter_enabled.tolist() == [0, 1, 1]
    assert info.token_oft_mapping.tolist() == [0, 0, 1, 2, 2]
    assert info.num_tokens == 5


def test_base_only_batch_has_no_active_oft():
    info = _compute_moe_oft_info(
        num_tokens=3,
        seg_indptr=torch.tensor([0, 3], dtype=torch.int32),
        weight_indices=torch.tensor([0], dtype=torch.int32),
        max_ofts=3,
    )

    assert info.adapter_enabled.tolist() == [0, 0, 0]
    assert info.token_oft_mapping.tolist() == [0, 0, 0]


@pytest.mark.parametrize("backend_cls", [TritonOFTBackend, TorchNativeOFTBackend])
def test_prepare_oft_batch_attaches_request_metadata(backend_cls, monkeypatch):
    backend = backend_cls(max_ofts_per_batch=3, device=torch.device("cpu"))
    backend.is_moe_oft = True
    forward_batch = SimpleNamespace(
        batch_size=3,
        extend_seq_lens_cpu=[2, 1, 2],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    torch_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_zeros(*args, **kwargs)

    torch_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    backend_module = __import__(backend_cls.__module__, fromlist=["unused"])
    monkeypatch.setattr(
        backend_module,
        "generate_sequence_lengths",
        lambda _forward_batch, device: torch.tensor([2, 1, 2], dtype=torch.int32),
    )

    backend.prepare_oft_batch(
        forward_batch,
        weight_indices=[0, 1, 2],
        oft_block_sizes=[0, 4, 4],
        use_cuda_graph=False,
    )

    info = backend.batch_info.moe_oft_info
    assert backend.batch_info.has_active_oft is True
    assert info.seg_indptr.tolist() == [0, 2, 3, 5]
    assert info.req_to_oft.tolist() == [0, 1, 2]
    assert info.adapter_enabled.tolist() == [0, 1, 1]
    assert info.token_oft_mapping.tolist() == [0, 0, 1, 2, 2]

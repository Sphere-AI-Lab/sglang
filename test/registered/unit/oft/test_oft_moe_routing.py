from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.oft.backend.base_backend import _compute_moe_oft_info
from sglang.srt.oft.backend.torch_backend import TorchNativeOFTBackend
from sglang.srt.oft.backend.triton_backend import TritonOFTBackend
from sglang.srt.oft.mem_pool import EMPTY_SLOT, OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager


def _expert_pool(groups):
    pool = object.__new__(OFTMemoryPool)
    pool._groups = groups
    pool.max_oft_block_size = 4
    pool.num_layer = 0
    pool.embedding_R_buffer = {}
    pool.lm_head_R_buffer = {}
    return pool


def _block_identity_like(slot):
    return torch.eye(slot.shape[-1], dtype=slot.dtype).expand_as(slot)


def test_expert_group_exposes_all_adapter_slots():
    pool = _expert_pool({"w1_oft_r": {0: torch.empty(3, 2, 1, 4, 4)}})

    group = pool.get_expert_tensor("w1_oft_r", layer_id=0)

    assert group.shape[0] == 3
    assert group.data_ptr() == pool._groups["w1_oft_r"][0].data_ptr()


def test_disk_expert_load_writes_only_selected_slot():
    pool = _expert_pool(
        {
            name: {0: torch.zeros(3, 2, 1, 4, 4)}
            for name in ("w1_oft_r", "w3_oft_r")
        }
    )
    manager = object.__new__(OFTManager)
    manager.memory_pool = pool
    manager.oft_r_dtype = torch.float32
    moe = SimpleNamespace(
        num_local_experts=2,
        moe_tp_rank=0,
        moe_tp_size=1,
        w13_weight=torch.empty(0),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
    )
    expert_weights = {
        0: {
            "gate_proj.oft_R": torch.zeros(1, 6),
            "up_proj.oft_R": torch.zeros(1, 6),
        }
    }
    before = pool.slot("w1_oft_r", 0, 1).clone()

    with patch(
        "sglang.srt.oft.torch_ops.oft_ops.precompute_oft_r",
        return_value=torch.eye(4).view(1, 4, 4),
    ):
        manager._apply_expert_oft_to_module(
            moe, expert_weights, block_size=4, layer_id=0, slot_idx=2
        )

    torch.testing.assert_close(pool.slot("w1_oft_r", 0, 1), before)
    assert not torch.equal(pool.slot("w1_oft_r", 0, 2), before)


def test_expert_load_failure_does_not_publish_slot():
    pool = _expert_pool({})
    pool.eviction_policy = MagicMock()
    pool.uid_to_buffer_id = {}
    pool.buffer_id_to_uid = {2: EMPTY_SLOT}
    pool._acquire_buffer_slot = MagicMock(return_value=2)
    pool.load_oft_weight_to_buffer = MagicMock()
    pool.reset_buffer_slot_to_identity = MagicMock()
    adapter = SimpleNamespace()

    def fail_expert_load(_adapter, _buffer_id):
        raise RuntimeError("Cayley failure")

    with pytest.raises(RuntimeError, match="Cayley failure"):
        pool.prepare_oft_batch(
            cur_uids={"broken"},
            oft_adapters={"broken": adapter},
            oft_modules=[],
            oft_refs={},
            oft_embed_tokens_module=None,
            oft_lm_head_module=None,
            expert_loader=fail_expert_load,
        )

    assert "broken" not in pool.uid_to_buffer_id
    assert "broken" not in pool.buffer_id_to_uid.values()
    pool.reset_buffer_slot_to_identity.assert_called_once_with(2)


def test_existing_slot_reset_also_clears_expert_groups():
    pool = _expert_pool(
        {
            name: {0: torch.zeros(3, 2, 1, 4, 4)}
            for name in ("w1_oft_r", "w3_oft_r", "w13_oft_r", "w2_oft_r")
        }
    )

    pool.reset_buffer_slot_to_identity(2)

    for group in pool._groups.values():
        torch.testing.assert_close(group[0][2], _block_identity_like(group[0][2]))


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

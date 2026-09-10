"""Disk unload releases resident OFT state before the next native admission."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.oft.base.mem_pool import EMPTY_SLOT
from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.oft_registry import OFTRef
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CONFIG = {
    "peft_type": "OFT",
    "target_modules": ["q_proj"],
    "oft_block_size": 4,
}


def _resident_disk_manager(*, pinned=False, resident=True):
    """Represent a disk adapter after its first serving batch admitted it."""
    ref = OFTRef(
        adapter_id="disk-uuid",
        adapter_name="disk-policy",
        adapter_path="/fixtures/disk-policy",
        adapter_version=7,
        pinned=pinned,
        reloadable=True,
    )
    manager = object.__new__(OFTManager)
    manager.refs = {ref.adapter_id: ref}
    manager.configs = {ref.adapter_id: object()}
    manager.adapters = {ref.adapter_id: object()}
    manager.num_pinned = int(pinned)
    manager.device = torch.device("cpu")
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
    manager.load_config = manager.oft_backend = None
    manager.pending_oft_load_events = {}
    pool = object.__new__(OFTMemoryPool)
    pool.max_ofts_per_batch = 2
    pool.max_oft_block_size = 4
    pool.uid_to_buffer_id = {None: 0}
    pool.buffer_id_to_uid = [None, EMPTY_SLOT]
    pool.eviction_policy = get_eviction_policy("lru")
    pool._active_versions = {ref.adapter_id: 7}
    pool._groups = {"w13_oft_r": {0: torch.full((2, 2, 1, 4, 4), -3.0)}}
    pool.embedding_R_buffer = {"embed": torch.full((2, 1, 4, 4), -4.0)}
    pool.lm_head_R_buffer = {"head": torch.full((2, 1, 4, 4), -5.0)}
    if resident:
        pool.uid_to_buffer_id[ref.adapter_id] = 1
        pool.buffer_id_to_uid[1] = ref.adapter_id
        pool.eviction_policy.mark_used(ref.adapter_id)
    manager.memory_pool = pool
    return manager, ref


def _buffers(pool):
    return [
        pool._groups["w13_oft_r"][0],
        pool.embedding_R_buffer["embed"],
        pool.lm_head_R_buffer["head"],
    ]


def _assert_unloaded(manager, ref):
    pool = manager.memory_pool
    assert ref.adapter_id not in manager.refs
    assert ref.adapter_id not in manager.configs
    assert ref.adapter_id not in manager.adapters
    assert ref.adapter_id not in pool.uid_to_buffer_id
    assert ref.adapter_id not in pool.buffer_id_to_uid
    assert ref.adapter_id not in pool.eviction_policy.access_order
    assert ref.adapter_id not in pool._active_versions
    assert pool.uid_to_buffer_id == {None: 0}
    assert pool.buffer_id_to_uid == [None, EMPTY_SLOT]
    assert manager.num_pinned == 0


def test_disk_unload_then_tensor_admission_at_capacity_two():
    manager, disk_ref = _resident_disk_manager()
    unloaded = manager.unload_oft_adapter(disk_ref)
    assert unloaded.success, unloaded.error_message
    wire_ref = OFTRef(
        adapter_id="wire-uuid",
        adapter_name="wire-policy",
        adapter_path="wire-policy",
        adapter_version=1,
        pinned=False,
        reloadable=False,
    )

    # Tensor parsing/commit are independent of admission. Keep the real
    # unload, slot allocator, eviction policy and registration in this cycle.
    with patch(
        "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
        return_value=(({}, {}, {}, []), ""),
    ), patch(
        "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
        return_value=(True, "Success"),
    ):
        loaded = manager.load_adapter_from_tensors(wire_ref, [], CONFIG)

    assert loaded.success, loaded.error_message
    assert manager.refs == {wire_ref.adapter_id: wire_ref}
    assert wire_ref.adapter_id in manager.adapters
    assert manager.memory_pool.uid_to_buffer_id == {None: 0}
    assert manager.memory_pool.buffer_id_to_uid == [None, EMPTY_SLOT]
    assert not manager.memory_pool.eviction_policy.access_order
    assert disk_ref.adapter_id not in manager.memory_pool._active_versions


@pytest.mark.parametrize("pinned", [False, True])
def test_disk_unload_resets_and_releases_slot_once(pinned):
    manager, ref = _resident_disk_manager(pinned=pinned)
    buffers = _buffers(manager.memory_pool)
    original = [buffer.clone() for buffer in buffers]
    pointers = [buffer.data_ptr() for buffer in buffers]
    # The stored registration determines pin accounting, even on a retry
    # request reconstructed with a different default for the pinned field.
    request_ref = replace(ref, pinned=not pinned)

    for _ in range(2):
        result = manager.unload_oft_adapter(request_ref)
        assert result.success, result.error_message
        _assert_unloaded(manager, ref)
        for buffer, before, pointer in zip(buffers, original, pointers):
            assert buffer.data_ptr() == pointer
            torch.testing.assert_close(buffer[0], before[0])
            torch.testing.assert_close(buffer[1], torch.eye(4).expand_as(buffer[1]))


def test_disk_unload_reset_failure_keeps_registration_for_retry():
    manager, ref = _resident_disk_manager(pinned=True)
    pool = manager.memory_pool
    before = [buffer.clone() for buffer in _buffers(pool)]

    with patch.object(
        pool, "reset_buffer_slot_to_identity", side_effect=RuntimeError("reset failed")
    ):
        result = manager.unload_oft_adapter(ref)

    assert not result.success
    assert "reset failed" in result.error_message
    assert manager.refs[ref.adapter_id] is ref
    assert ref.adapter_id in manager.configs
    assert ref.adapter_id in manager.adapters
    assert manager.num_pinned == 1
    assert pool.uid_to_buffer_id[ref.adapter_id] == 1
    assert pool.buffer_id_to_uid[1] == ref.adapter_id
    assert ref.adapter_id in pool.eviction_policy.access_order
    assert pool._active_versions[ref.adapter_id] == 7
    for buffer, original in zip(_buffers(pool), before):
        torch.testing.assert_close(buffer, original)

    retry = manager.unload_oft_adapter(ref)
    assert retry.success, retry.error_message
    _assert_unloaded(manager, ref)


def test_disk_unload_without_residency_preserves_other_slots():
    manager, ref = _resident_disk_manager(pinned=True, resident=False)
    before = [buffer.clone() for buffer in _buffers(manager.memory_pool)]

    result = manager.unload_oft_adapter(ref)

    assert result.success, result.error_message
    _assert_unloaded(manager, ref)
    for buffer, original in zip(_buffers(manager.memory_pool), before):
        torch.testing.assert_close(buffer, original)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

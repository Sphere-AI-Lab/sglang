import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.oft.base.mem_pool import EMPTY_SLOT
from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.oft_registry import OFTRef
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CONFIG = {
    "peft_type": "OFT",
    "target_modules": ["q_proj"],
    "oft_block_size": 4,
}


def _ref(name, *, adapter_id=None, pinned=False, reloadable=False):
    return OFTRef(
        adapter_id=adapter_id or name,
        adapter_name=name,
        adapter_path=name,
        pinned=pinned,
        reloadable=reloadable,
    )


def _manager(capacity=2):
    manager = OFTManager.__new__(OFTManager)
    manager.refs = {}
    manager.configs = {}
    manager.adapters = {}
    manager.num_pinned = 0
    manager._clear_expert_oft = MagicMock()
    pool = SimpleNamespace(
        max_oft_block_size=4,
        max_ofts_per_batch=capacity,
        uid_to_buffer_id={},
        buffer_id_to_uid=[EMPTY_SLOT] * capacity,
        eviction_policy=get_eviction_policy("lru"),
        reset_buffer_slot_to_identity=MagicMock(),
    )
    pool.allocate_buffer_slot_with_eviction = MethodType(
        OFTMemoryPool.allocate_buffer_slot_with_eviction, pool
    )
    manager.memory_pool = pool
    return manager


class TestNativeAdmission(unittest.TestCase):
    def test_dict_payload_is_normalized_before_resolve_and_commit(self):
        manager = _manager()
        ref = _ref("new")
        payload = {"model.layers.0.self_attn.q_proj.oft_R": "tensor"}

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(({}, {}, {}, []), ""),
        ) as resolve, patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ) as commit:
            result = manager.load_adapter_from_tensors(ref, payload, CONFIG)

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(resolve.call_args.args[1], list(payload.items()))
        self.assertEqual(commit.call_args.args[1], list(payload.items()))
        self.assertEqual(manager.memory_pool.uid_to_buffer_id[ref.adapter_id], 0)

    def test_invalid_payload_does_not_evict_resident(self):
        manager = _manager(capacity=1)
        resident = _ref("resident", reloadable=True)
        manager.refs[resident.adapter_id] = resident
        manager.configs[resident.adapter_id] = "config"
        manager.adapters[resident.adapter_id] = "disk-backed"
        manager.memory_pool.uid_to_buffer_id[resident.adapter_id] = 0
        manager.memory_pool.buffer_id_to_uid[0] = resident.adapter_id

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(None, "Unresolved OFT tensor names: bogus"),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups"
        ) as commit:
            result = manager.load_adapter_from_tensors(_ref("new"), [], CONFIG)

        self.assertFalse(result.success)
        commit.assert_not_called()
        self.assertIn(resident.adapter_id, manager.refs)
        self.assertEqual(
            manager.memory_pool.uid_to_buffer_id[resident.adapter_id], 0
        )

    def test_commit_failure_removes_phantom_registration(self):
        manager = _manager()
        ref = _ref("new")

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(({}, {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(False, "write failed"),
        ):
            result = manager.load_adapter_from_tensors(ref, [], CONFIG)

        self.assertFalse(result.success)
        self.assertIn("write failed", result.error_message)
        self.assertNotIn(ref.adapter_id, manager.refs)
        self.assertNotIn(ref.adapter_id, manager.configs)
        self.assertNotIn(ref.adapter_id, manager.memory_pool.uid_to_buffer_id)
        self.assertNotIn(ref.adapter_id, manager.memory_pool.eviction_policy.access_order)

    def test_wire_upsert_cannot_replace_disk_backed_adapter(self):
        manager = _manager()
        disk_ref = _ref("same", adapter_id="disk-id", reloadable=True)
        manager.refs[disk_ref.adapter_id] = disk_ref
        manager.configs[disk_ref.adapter_id] = "config"
        manager.adapters[disk_ref.adapter_id] = "disk-backed"

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(({}, {}, {}, []), ""),
        ):
            result = manager.load_adapter_from_tensors(
                _ref("same", adapter_id="wire-id"), [], CONFIG, upsert=True
            )

        self.assertFalse(result.success)
        self.assertIn("loaded from disk", result.error_message)
        self.assertIn(disk_ref.adapter_id, manager.refs)
        self.assertIn(disk_ref.adapter_id, manager.adapters)

    def test_pinned_count_tracks_register_and_idempotent_unload(self):
        manager = _manager()
        ref = _ref("pinned", pinned=True)

        result = manager.register_streamed_adapter(ref, 0, CONFIG)
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(manager.num_pinned, 1)

        self.assertTrue(manager.unload_streamed_adapter(ref).success)
        self.assertTrue(manager.unload_streamed_adapter(ref).success)
        self.assertEqual(manager.num_pinned, 0)

    def test_distributed_receive_failure_is_returned(self):
        manager = _manager()
        updater = MagicMock()
        updater.receive_weights_from_distributed.side_effect = RuntimeError("boom")

        result = manager.load_adapter_from_distributed(
            _ref("new"), [], [], [], CONFIG, "group", updater
        )

        self.assertFalse(result.success)
        self.assertIn("Failed to receive OFT adapter weights", result.error_message)
        self.assertIn("boom", result.error_message)

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.oft.base.mem_pool import EMPTY_SLOT
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.oft_registry import OFTRef
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CONFIG = {
    "peft_type": "OFT",
    "target_modules": ["q_proj"],
    "oft_block_size": 4,
}


def _ref(
    name,
    *,
    adapter_id=None,
    adapter_version=1,
    pinned=False,
    reloadable=False,
):
    return OFTRef(
        adapter_id=adapter_id or name,
        adapter_name=name,
        adapter_path=name,
        adapter_version=adapter_version,
        pinned=pinned,
        reloadable=reloadable,
    )


def _manager(capacity=2):
    manager = OFTManager.__new__(OFTManager)
    manager.refs = {}
    manager.configs = {}
    manager.adapters = {}
    manager.num_pinned = 0
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
    manager.load_config = manager.oft_backend = None
    manager.device = SimpleNamespace(type="cpu")
    manager._clear_expert_oft = MagicMock()
    pool = SimpleNamespace(
        max_oft_block_size=4,
        max_ofts_per_batch=capacity,
        uid_to_buffer_id={},
        buffer_id_to_uid=[EMPTY_SLOT] * capacity,
        eviction_policy=get_eviction_policy("lru"),
        reset_buffer_slot_to_identity=MagicMock(),
    )
    manager.memory_pool = pool
    return manager


class TestNativeAdmission(unittest.TestCase):
    def test_worker_normalizes_flattened_native_payload(self):
        worker = TpModelWorker.__new__(TpModelWorker)
        worker.ps = SimpleNamespace(tp_rank=0)
        runner = SimpleNamespace(
            device="cuda:0",
            load_oft_adapter_from_tensors=MagicMock(return_value="loaded"),
        )
        worker._model_runner = runner
        ref = _ref("adapter")
        request = SimpleNamespace(
            serialized_named_tensors=[b"rank-0"],
            load_format="oft_adapter",
            config_dict=CONFIG,
            upsert=True,
            to_ref=lambda: ref,
        )
        raw_payload = ("flattened_oft_payload", b"tensor", [], [])
        normalized_payload = [("layer.oft_R", "tensor")]

        with patch.object(
            worker,
            "_deserialize_own_rank",
            return_value=raw_payload,
        ), patch(
            "sglang.srt.managers.tp_worker.normalize_oft_weight_payload",
            return_value=normalized_payload,
            create=True,
        ) as normalize:
            result = worker.load_oft_adapter_from_tensors(request)

        self.assertEqual(result, "loaded")
        normalize.assert_called_once_with(raw_payload, device="cuda:0")
        runner.load_oft_adapter_from_tensors.assert_called_once_with(
            ref,
            normalized_payload,
            CONFIG,
            upsert=True,
        )

    def test_dict_payload_is_snapshotted_before_lazy_publication(self):
        manager = _manager()
        ref = _ref("new")
        payload = {"model.layers.0.self_attn.q_proj.oft_R": torch.ones(1, 6)}

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(({}, {}, {}, []), ""),
        ) as resolve, patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ) as commit:
            result = manager.load_adapter_from_tensors(ref, payload, CONFIG)

        self.assertTrue(result.success, result.error_message)
        snapshot = resolve.call_args.args[1]
        self.assertEqual(snapshot[0][0], next(iter(payload)))
        torch.testing.assert_close(snapshot[0][1], next(iter(payload.values())))
        self.assertNotEqual(
            snapshot[0][1].data_ptr(), next(iter(payload.values())).data_ptr()
        )
        commit.assert_not_called()
        self.assertIn(ref.adapter_id, manager.adapters)
        self.assertNotIn(ref.adapter_id, manager.memory_pool.uid_to_buffer_id)

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
        self.assertEqual(manager.memory_pool.uid_to_buffer_id[resident.adapter_id], 0)

    def test_cpu_construction_failure_does_not_publish_registration(self):
        manager = _manager()
        ref = _ref("new")
        with patch(
            "sglang.srt.oft.oft_manager.OFTAdapter",
            side_effect=RuntimeError("CPU construction failed"),
        ):
            result = manager.load_adapter_from_tensors(ref, [], CONFIG)
        self.assertFalse(result.success)
        self.assertIn("CPU construction failed", result.error_message)
        self.assertNotIn(ref.adapter_id, manager.refs)
        self.assertNotIn(ref.adapter_id, manager.configs)
        self.assertNotIn(ref.adapter_id, manager.adapters)
        self.assertNotIn(ref.adapter_id, manager.memory_pool.uid_to_buffer_id)

    def test_failed_wire_upsert_restores_previous_adapter(self):
        manager = _manager(capacity=1)
        old_ref = _ref("same", adapter_id="wire-id", adapter_version=4)
        new_ref = _ref("same", adapter_id="wire-id", adapter_version=5)
        old_config = object()
        manager.refs[old_ref.adapter_id] = old_ref
        manager.configs[old_ref.adapter_id] = old_config
        manager.memory_pool.uid_to_buffer_id[old_ref.adapter_id] = 0
        manager.memory_pool.buffer_id_to_uid[0] = old_ref.adapter_id
        manager.memory_pool.eviction_policy.mark_used(old_ref.adapter_id)

        manager.memory_pool.staging_idx = 1
        manager.memory_pool.slot_payloads = {0: "old", 1: "identity"}
        manager.memory_pool.staged_identity = lambda: None

        def copy_slot(source, destination):
            manager.memory_pool.slot_payloads[destination] = (
                manager.memory_pool.slot_payloads[source]
            )

        manager.memory_pool.copy_supported_buffer_slot = copy_slot

        def commit(*args):
            buffer_id = args[3]
            manager.memory_pool.slot_payloads[buffer_id] = "partial-new"
            return False, "write failed"

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(({}, {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            side_effect=commit,
        ):
            result = manager.load_adapter_from_tensors(new_ref, [], CONFIG, upsert=True)

        self.assertFalse(result.success)
        self.assertIn("write failed", result.error_message)
        self.assertTrue(result.previous_adapter_preserved)
        self.assertIs(manager.refs[old_ref.adapter_id], old_ref)
        self.assertIs(manager.configs[old_ref.adapter_id], old_config)
        self.assertEqual(manager.memory_pool.uid_to_buffer_id[old_ref.adapter_id], 0)
        self.assertEqual(manager.memory_pool.buffer_id_to_uid[0], old_ref.adapter_id)
        self.assertEqual(manager.memory_pool.slot_payloads[0], "old")
        self.assertEqual(
            result.loaded_adapters[old_ref.adapter_name], old_ref.adapter_path
        )

    def test_failed_wire_upsert_does_not_claim_preservation_when_restore_fails(self):
        manager = _manager(capacity=1)
        old_ref = _ref("same", adapter_id="wire-id", adapter_version=4)
        new_ref = _ref("same", adapter_id="wire-id", adapter_version=5)
        manager.refs[old_ref.adapter_id] = old_ref
        manager.configs[old_ref.adapter_id] = object()
        manager.memory_pool.uid_to_buffer_id[old_ref.adapter_id] = 0
        manager.memory_pool.buffer_id_to_uid[0] = old_ref.adapter_id
        manager.memory_pool.eviction_policy.mark_used(old_ref.adapter_id)
        manager.memory_pool.staging_idx = 1
        manager.memory_pool.staged_identity = lambda: None
        copy_count = 0

        def copy_slot(source, destination):
            nonlocal copy_count
            copy_count += 1
            if copy_count == 2:
                raise RuntimeError("restore exploded")

        manager.memory_pool.copy_supported_buffer_slot = copy_slot

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(({}, {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(False, "write failed"),
        ):
            result = manager.load_adapter_from_tensors(new_ref, [], CONFIG, upsert=True)

        self.assertFalse(result.success)
        self.assertIn("restore exploded", result.error_message)
        self.assertFalse(result.previous_adapter_preserved)

    def test_rejected_wire_upsert_confirms_untouched_previous_adapter(self):
        manager = _manager(capacity=1)
        old_ref = _ref("same", adapter_id="wire-id", adapter_version=4)
        manager.refs[old_ref.adapter_id] = old_ref

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(None, "bad payload"),
        ):
            result = manager.load_adapter_from_tensors(
                _ref("same", adapter_id="wire-id", adapter_version=5),
                [],
                CONFIG,
                upsert=True,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "bad payload")
        self.assertTrue(result.previous_adapter_preserved)
        self.assertIs(manager.refs[old_ref.adapter_id], old_ref)

    def test_rejected_wire_upsert_does_not_preserve_divergent_active_version(self):
        manager = _manager(capacity=1)
        old_ref = _ref("same", adapter_id="wire-id", adapter_version=4)
        manager.refs[old_ref.adapter_id] = old_ref
        manager.memory_pool._active_versions = {old_ref.adapter_id: 3}

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(None, "bad payload"),
        ):
            result = manager.load_adapter_from_tensors(
                _ref("same", adapter_id="wire-id", adapter_version=5),
                [],
                CONFIG,
                upsert=True,
            )

        self.assertFalse(result.success)
        self.assertFalse(result.previous_adapter_preserved)

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
        self.assertIn("disk-loaded", result.error_message)
        self.assertIn(disk_ref.adapter_id, manager.refs)
        self.assertIn(disk_ref.adapter_id, manager.adapters)

    def test_worker_unload_is_idempotent_after_another_rank_succeeded(self):
        result = _manager().unload_adapter(_ref("missing"))

        self.assertTrue(result.success, result.error_message)

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


if __name__ == "__main__":
    unittest.main()

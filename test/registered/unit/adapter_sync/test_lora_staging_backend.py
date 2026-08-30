"""Unit coverage for native hidden-slot LoRA staging and activation."""

import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.adapter_sync.backends.lora import (
    PendingLoRAStage,
    StagedLoRAManager,
    StagedLoRAMemoryPool,
)
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import LoRAMemoryPool


CONFIG_DICT = {"target_modules": ["q_proj"], "r": 4, "lora_alpha": 8}


def _pool(n_slots=2, n_layers=2, rank=4, hidden=8):
    """Build a pool with N advertised slots and one hidden physical slot."""
    pool = object.__new__(StagedLoRAMemoryPool)
    pool.max_loras_per_batch = n_slots
    pool.staging_idx = n_slots
    pool._staged_uid = None
    pool._staged_version = None
    wide = n_slots + 1
    pool.A_buffer = {
        "q_proj": [torch.zeros(wide, rank, hidden) for _ in range(n_layers)]
    }
    pool.B_buffer = {
        "q_proj": [torch.zeros(wide, hidden, rank) for _ in range(n_layers)]
    }
    pool.embedding_A_buffer = {
        "embed_tokens": torch.zeros(wide, rank, hidden)
    }
    pool.embedding_B_buffer = {
        "embed_tokens": torch.zeros(wide, hidden, rank)
    }
    pool.lm_head_A_buffer = {"lm_head": torch.zeros(wide, rank, hidden)}
    pool.lm_head_B_buffer = {"lm_head": torch.zeros(wide, hidden, rank)}
    pool.new_embeddings_buffer = {
        "input_embeddings": torch.zeros(wide, 2, hidden)
    }
    serving_uids = [f"id-{chr(ord('a') + i)}" for i in range(n_slots)]
    pool.uid_to_buffer_id = {uid: i for i, uid in enumerate(serving_uids)}
    pool.buffer_id_to_uid = list(serving_uids)
    pool.can_support = Mock(return_value=True)
    return pool


def _manager(existing_ref=None):
    """Build a manager with only the native tensor-staging state it reads."""
    manager = object.__new__(StagedLoRAManager)
    manager.configs = {}
    manager.loras = {}
    manager.lora_refs = {}
    manager.num_pinned_loras = 0
    manager.max_loras_per_batch = 4
    manager.base_hf_config = MagicMock(vocab_size=32000)
    manager.lora_modules = [{}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    manager.device = torch.device("cpu")
    manager.memory_pool = _pool()
    manager._pending_lora_stage = None
    manager._create_lora_adapter_from_tensors = Mock(return_value=MagicMock())
    manager.memory_pool.load_lora_weight_to_buffer = Mock()

    if existing_ref is not None:
        manager.configs[existing_ref.lora_id] = MagicMock(name="old_config")
        manager.loras[existing_ref.lora_id] = MagicMock(name="old_adapter")
        manager.lora_refs[existing_ref.lora_id] = existing_ref
        manager.num_pinned_loras = int(bool(existing_ref.pinned))
    return manager


class TestSlotReservation(CustomTestCase):
    def test_constructor_preserves_hidden_slot_set_during_native_init(self):
        def fake_native_init(pool, *args, **kwargs):
            pool.staging_idx = 2

        with patch.object(LoRAMemoryPool, "__init__", new=fake_native_init):
            pool = StagedLoRAMemoryPool()

        self.assertEqual(pool.staging_idx, 2)
        self.assertIsNone(pool.staged_identity())

    def test_staging_slot_sits_outside_advertised_capacity(self):
        pool = _pool(n_slots=2)
        self.assertEqual(pool.staging_idx, 2)
        self.assertGreaterEqual(pool.staging_idx, pool.max_loras_per_batch)
        self.assertEqual(pool.available_serving_slots(), 2)

    def test_buffers_are_one_slot_wider_than_advertised(self):
        pool = _pool(n_slots=2)
        self.assertEqual(pool.A_buffer["q_proj"][0].shape[0], 3)
        self.assertEqual(pool.embedding_A_buffer["embed_tokens"].shape[0], 3)


class TestNativePoolStaging(CustomTestCase):
    def test_stage_calls_native_loader_for_hidden_slot(self):
        pool = _pool(n_slots=2)
        adapter = MagicMock()
        pool.load_lora_weight_to_buffer = MagicMock()

        pool.stage(
            uid="id-a",
            version=4,
            adapter=adapter,
            lora_modules=[{}],
            embed_module=None,
            lm_head_module=None,
        )

        pool.load_lora_weight_to_buffer.assert_called_once_with(
            "id-a", 2, adapter, [{}], None, None
        )
        self.assertEqual(pool.staged_identity(), ("id-a", 4))

    def test_same_identity_retry_is_idempotent(self):
        pool = _pool()
        pool.load_lora_weight_to_buffer = MagicMock()
        args = dict(
            uid="id-a",
            version=4,
            adapter=MagicMock(),
            lora_modules=[{}],
            embed_module=None,
            lm_head_module=None,
        )

        pool.stage(**args)
        pool.stage(**args)

        pool.load_lora_weight_to_buffer.assert_called_once()

    def test_conflicting_stage_is_rejected(self):
        pool = _pool()
        pool.load_lora_weight_to_buffer = MagicMock()
        pool.stage("id-a", 4, MagicMock(), [{}], None, None)

        with self.assertRaisesRegex(ValueError, "id-a.*4"):
            pool.stage("id-b", 5, MagicMock(), [{}], None, None)

    def test_failed_native_load_does_not_publish_staged_identity(self):
        pool = _pool()
        pool.load_lora_weight_to_buffer = Mock(side_effect=ValueError("bad shape"))

        with self.assertRaisesRegex(ValueError, "bad shape"):
            pool.stage("id-a", 4, MagicMock(), [{}], None, None)

        self.assertIsNone(pool.staged_identity())

    def test_activate_copies_every_buffer_family_without_changing_maps(self):
        pool = _pool()
        pool.load_lora_weight_to_buffer = MagicMock()
        pool.stage("id-a", 4, MagicMock(), [{}], None, None)
        for tensor in pool._slot_buffers():
            tensor[pool.staging_idx].fill_(7)
        uid_map = dict(pool.uid_to_buffer_id)
        slot_map = list(pool.buffer_id_to_uid)

        pool.activate("id-a", 4, destination=0)

        for tensor in pool._slot_buffers():
            self.assertTrue(torch.equal(tensor[0], tensor[pool.staging_idx]))
        self.assertEqual(pool.uid_to_buffer_id, uid_map)
        self.assertEqual(pool.buffer_id_to_uid, slot_map)
        self.assertEqual(pool.staged_identity(), ("id-a", 4))

    def test_activate_requires_exact_identity_and_serving_destination(self):
        pool = _pool()
        pool.load_lora_weight_to_buffer = MagicMock()
        pool.stage("id-a", 4, MagicMock(), [{}], None, None)

        with self.assertRaisesRegex(ValueError, "id-a.*4"):
            pool.activate("id-a", 5, destination=0)
        with self.assertRaisesRegex(ValueError, "serving slot"):
            pool.activate("id-a", 4, destination=pool.staging_idx)
        with self.assertRaisesRegex(ValueError, "serving slot"):
            pool.activate("id-a", 4, destination=-1)

    def test_discard_requires_exact_identity(self):
        pool = _pool()
        pool.load_lora_weight_to_buffer = MagicMock()
        pool.stage("id-a", 4, MagicMock(), [{}], None, None)

        with self.assertRaisesRegex(ValueError, "id-a.*4"):
            pool.discard_stage("id-b", 4)
        pool.discard_stage("id-a", 4)

        self.assertIsNone(pool.staged_identity())


class TestNativeManagerStaging(CustomTestCase):
    def test_stage_preserves_active_state(self):
        old_ref = LoRARef(
            lora_id="id-a",
            lora_name="policy",
            lora_path="__tensor__",
            pinned=True,
            version=3,
        )
        manager = _manager(old_ref)
        configs = dict(manager.configs)
        loras = dict(manager.loras)
        refs = dict(manager.lora_refs)
        pinned = manager.num_pinned_loras
        uid_map = dict(manager.memory_pool.uid_to_buffer_id)
        slot_map = list(manager.memory_pool.buffer_id_to_uid)

        result = manager.stage_adapter(
            [("q_proj.lora_A.weight", torch.ones(1))],
            CONFIG_DICT,
            name="policy",
            version=4,
            adapter_id="id-a",
        )

        self.assertTrue(result.success)
        self.assertEqual(manager.configs, configs)
        self.assertEqual(manager.loras, loras)
        self.assertEqual(manager.lora_refs, refs)
        self.assertEqual(manager.num_pinned_loras, pinned)
        self.assertEqual(manager.memory_pool.uid_to_buffer_id, uid_map)
        self.assertEqual(manager.memory_pool.buffer_id_to_uid, slot_map)
        self.assertIsInstance(manager._pending_lora_stage, PendingLoRAStage)
        self.assertTrue(manager._pending_lora_stage.ref.pinned)
        self.assertEqual(manager._pending_lora_stage.ref.version, 4)

    def test_same_pending_identity_retry_is_idempotent(self):
        manager = _manager()
        args = (
            [("q_proj.lora_A.weight", torch.ones(1))],
            CONFIG_DICT,
            "policy",
            4,
            "id-a",
        )

        first = manager.stage_adapter(*args)
        second = manager.stage_adapter(*args)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        manager._create_lora_adapter_from_tensors.assert_called_once()

    def test_conflicting_pending_identity_returns_failure(self):
        manager = _manager()
        self.assertTrue(
            manager.stage_adapter([], CONFIG_DICT, "policy", 4, "id-a").success
        )

        result = manager.stage_adapter([], CONFIG_DICT, "other", 5, "id-b")

        self.assertFalse(result.success)
        self.assertIn("id-a", result.error_message)
        self.assertIn("4", result.error_message)


class TestNativeManagerActivation(CustomTestCase):
    def _stage_existing(self, pinned=True):
        old_ref = LoRARef(
            lora_id="id-a",
            lora_name="policy",
            lora_path="__tensor__",
            pinned=pinned,
            version=3,
        )
        manager = _manager(old_ref)
        old_state = (
            manager.configs["id-a"],
            manager.loras["id-a"],
            manager.lora_refs["id-a"],
        )
        result = manager.stage_adapter([], CONFIG_DICT, "policy", 4, "id-a")
        self.assertTrue(result.success)
        return manager, old_state

    def test_existing_pinned_adapter_stays_pinned_after_activation(self):
        manager, _ = self._stage_existing(pinned=True)

        result = manager.activate_adapter("policy", 4, "id-a")

        self.assertTrue(result.success)
        self.assertTrue(manager.lora_refs["id-a"].pinned)
        self.assertEqual(manager.lora_refs["id-a"].version, 4)
        self.assertEqual(manager.num_pinned_loras, 1)
        self.assertIsNone(manager._pending_lora_stage)
        self.assertIsNone(manager.memory_pool.staged_identity())

    def test_nonresident_activation_commits_without_evicting_a_slot(self):
        manager = _manager()
        uid_map = dict(manager.memory_pool.uid_to_buffer_id)
        slot_map = list(manager.memory_pool.buffer_id_to_uid)
        self.assertTrue(
            manager.stage_adapter([], CONFIG_DICT, "new", 1, "id-new").success
        )
        manager.memory_pool.activate = Mock(
            side_effect=AssertionError("nonresident activation must not copy")
        )

        result = manager.activate_adapter("new", 1, "id-new")

        self.assertTrue(result.success)
        self.assertIn("id-new", manager.loras)
        self.assertEqual(manager.memory_pool.uid_to_buffer_id, uid_map)
        self.assertEqual(manager.memory_pool.buffer_id_to_uid, slot_map)
        manager.memory_pool.activate.assert_not_called()

    def test_resident_copy_failure_restores_old_adapter_and_state(self):
        manager, old_state = self._stage_existing()
        manager.memory_pool.activate = Mock(side_effect=RuntimeError("copy failed"))
        manager.memory_pool.load_lora_weight_to_buffer.reset_mock()

        result = manager.activate_adapter("policy", 4, "id-a")

        self.assertFalse(result.success)
        self.assertIn("copy failed", result.error_message)
        self.assertIs(manager.configs["id-a"], old_state[0])
        self.assertIs(manager.loras["id-a"], old_state[1])
        self.assertIs(manager.lora_refs["id-a"], old_state[2])
        manager.memory_pool.load_lora_weight_to_buffer.assert_called_once_with(
            "id-a",
            0,
            old_state[1],
            manager.lora_modules,
            manager.embed_tokens_module,
            manager.lm_head_module,
        )

    def test_failed_restore_requires_worker_restart(self):
        manager, old_state = self._stage_existing()
        manager.memory_pool.activate = Mock(side_effect=RuntimeError("copy failed"))
        manager.memory_pool.load_lora_weight_to_buffer.reset_mock()
        manager.memory_pool.load_lora_weight_to_buffer.side_effect = RuntimeError(
            "restore failed"
        )

        result = manager.activate_adapter("policy", 4, "id-a")

        self.assertFalse(result.success)
        self.assertIn("worker restart required", result.error_message)
        self.assertIs(manager.configs["id-a"], old_state[0])
        self.assertIs(manager.loras["id-a"], old_state[1])
        self.assertIs(manager.lora_refs["id-a"], old_state[2])


if __name__ == "__main__":
    unittest.main()

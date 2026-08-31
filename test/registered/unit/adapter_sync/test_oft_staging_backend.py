import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


TARGET_MODULE = "q_proj"
BLOCK_SIZE = 4
CONFIG_DICT = {
    "peft_type": "oft",
    "target_modules": [TARGET_MODULE],
    "oft_block_size": BLOCK_SIZE,
}


def _make_pool(max_ofts_per_batch=4):
    from sglang.srt.oft.staged_manager import StagedOFTMemoryPool

    base_hf_config = MagicMock()
    base_hf_config.num_hidden_layers = 1
    base_hf_config.hidden_size = 8
    base_model = MagicMock()
    # OFTMemoryPool.init_buffers reads the buffer device off the base
    # model's first parameter, and get_oft_R_shape (with oft_modules=None,
    # as here) resolves every module's dims via
    # base_model.get_hidden_dim(module_name, layer_idx). A bare MagicMock
    # auto-vivifies both `.parameters()` and `.get_hidden_dim` as mocks
    # returning further mocks, which torch.empty(..., device=...) and
    # tuple-unpacking (`input_dim, _ = ...`) both reject. Pin them to
    # concrete values instead.
    base_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    base_model.get_hidden_dim = MagicMock(return_value=(8, 8))
    pool = StagedOFTMemoryPool(
        base_hf_config=base_hf_config,
        max_ofts_per_batch=max_ofts_per_batch,
        dtype=torch.float32,
        tp_size=1,
        tp_rank=0,
        max_oft_block_size=BLOCK_SIZE,
        target_modules={TARGET_MODULE},
        base_model=base_model,
        eviction_policy="lru",
        oft_added_tokens_size=0,
        oft_type="canonical_oft",
    )
    return pool


def _named_tensors_for_layer_0(fill_value: float):
    """Real _fill_slot payload (python/sglang/srt/oft/mem_pool.py:647-677):
    maps (target_module, layer_id) -> (r, block_size, slice_index, split_count).
    r is the compact per-block rotation-generator tensor _write_oft_r_block
    expects; shape/content correctness for the OFT math itself is covered by
    existing tests in test/registered/unit/oft/ -- these tests only need a
    tensor _write_oft_r_block will accept without raising, distinguishable by
    fill_value so slot-isolation assertions can tell slots apart."""
    r = torch.full((BLOCK_SIZE, BLOCK_SIZE), fill_value, dtype=torch.float32)
    return {(TARGET_MODULE, 0): (r, BLOCK_SIZE, None, 1)}


class TestOFTStagingSlotReservation(unittest.TestCase):
    def test_staging_slot_sits_outside_the_advertised_capacity(self):
        pool = _make_pool(max_ofts_per_batch=4)
        self.assertEqual(pool.max_ofts_per_batch, 4)
        self.assertEqual(pool.staging_idx, 4)

    def test_available_serving_slots_excludes_the_hidden_slot(self):
        pool = _make_pool(max_ofts_per_batch=4)
        self.assertEqual(pool.available_serving_slots(), 4)


class TestOFTStagingTransaction(unittest.TestCase):
    def test_stage_then_activate_writes_only_the_destination_slot(self):
        pool = _make_pool(max_ofts_per_batch=4)
        slot_0_before = pool.slot(f"R:{TARGET_MODULE}", 0, 0).clone()

        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=9.0))
        pool.activate("adapter-a", 1, destination=2)

        self.assertTrue(
            (pool.slot(f"R:{TARGET_MODULE}", 0, 0) == slot_0_before).all(),
            "activating one uid must not touch slot 0",
        )
        self.assertTrue(
            (pool.slot(f"R:{TARGET_MODULE}", 0, 2) == 9.0).all(),
            "activate must copy the staged value into the destination slot",
        )
        self.assertEqual(pool.active_version_for("adapter-a"), 1)
        self.assertIsNone(pool.staged_identity())

    def test_activate_rejects_a_different_adapter_than_was_staged(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=9.0))
        with self.assertRaises(ValueError):
            pool.activate("adapter-b", 1, destination=2)

    def test_activate_rejects_the_staging_slot_as_a_destination(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=9.0))
        with self.assertRaises(ValueError):
            pool.activate("adapter-a", 1, destination=pool.staging_idx)


def _manager_for_pool_construction(max_ofts_per_batch=4):
    """Attributes OFTManager.__init__/init_state would have set before
    calling init_memory_pool -- built directly (object.__new__, no real
    constructor call) since the real constructor also builds an OFT backend,
    installs MoE wrappers, etc., none of which init_memory_pool itself reads.
    Mirrors _make_pool's base_model/base_hf_config mocking."""
    from sglang.srt.oft.staged_manager import StagedOFTManager

    base_hf_config = MagicMock()
    base_hf_config.num_hidden_layers = 1
    base_hf_config.hidden_size = 8
    base_model = MagicMock()
    base_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    base_model.get_hidden_dim = MagicMock(return_value=(8, 8))

    manager = object.__new__(StagedOFTManager)
    manager.base_model = base_model
    manager.base_hf_config = base_hf_config
    manager.max_ofts_per_batch = max_ofts_per_batch
    manager.max_adapters_per_batch = max_ofts_per_batch
    manager.oft_r_dtype = torch.float32
    manager.dtype = torch.float32
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.max_oft_block_size = BLOCK_SIZE
    manager.target_modules = {TARGET_MODULE}
    manager.oft_type = "canonical_oft"
    manager.adapter_modules = [{}]
    manager.eviction_policy = "lru"
    manager.oft_added_tokens_size = 0
    manager.memory_saver_adapter = None
    manager.memory_saver_cpu_backup = False
    manager.peft_double_buffer = False
    manager.adapters = {}
    manager.refs = {}
    manager.configs = {}
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    return manager


def _manager(max_ofts_per_batch=4):
    """Manager fixture with only the state stage_adapter/activate_adapter
    actually read -- mirrors test_lora_staged_manager.py's _manager()
    helper. Builds the memory pool via the already-covered _make_pool
    fixture rather than the heavier init_memory_pool path above."""
    from sglang.srt.oft.staged_manager import StagedOFTManager

    manager = object.__new__(StagedOFTManager)
    manager.memory_pool = _make_pool(max_ofts_per_batch=max_ofts_per_batch)
    manager.base_hf_config = manager.memory_pool.base_hf_config
    manager.load_config = MagicMock()
    manager.oft_backend = MagicMock()
    manager.configs = {}
    manager.adapters = {}
    manager.refs = {}
    manager._pending_oft_stage = None
    return manager


class TestStagedOFTManagerConstruction(unittest.TestCase):
    def test_init_memory_pool_builds_a_staged_pool(self):
        from sglang.srt.oft.staged_manager import StagedOFTMemoryPool

        manager = _manager_for_pool_construction(max_ofts_per_batch=4)

        manager.init_memory_pool()

        self.assertIsInstance(manager.memory_pool, StagedOFTMemoryPool)
        self.assertEqual(manager.memory_pool.staging_idx, 4)
        self.assertEqual(manager.memory_pool.available_serving_slots(), 4)


class TestStagedOFTManagerStaging(unittest.TestCase):
    def test_stage_writes_the_hidden_slot_and_tracks_pending(self):
        manager = _manager()

        result = manager.stage_adapter(
            _named_tensors_for_layer_0(fill_value=9.0),
            CONFIG_DICT,
            name="adapter-a",
            version=1,
            adapter_id="adapter-a",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            manager.memory_pool.staged_identity(), ("adapter-a", 1)
        )
        self.assertTrue(
            (
                manager.memory_pool.slot(
                    f"R:{TARGET_MODULE}", 0, manager.memory_pool.staging_idx
                )
                == 9.0
            ).all()
        )
        self.assertIsNotNone(manager._pending_oft_stage)
        self.assertEqual(manager._pending_oft_stage.uid, "adapter-a")

    def test_same_pending_identity_retry_is_idempotent(self):
        manager = _manager()
        args = (
            _named_tensors_for_layer_0(fill_value=9.0),
            CONFIG_DICT,
            "adapter-a",
            1,
            "adapter-a",
        )

        first = manager.stage_adapter(*args)
        second = manager.stage_adapter(*args)

        self.assertTrue(first.success)
        self.assertTrue(second.success)

    def test_conflicting_pending_stage_is_rejected(self):
        manager = _manager()
        self.assertTrue(
            manager.stage_adapter(
                _named_tensors_for_layer_0(fill_value=9.0),
                CONFIG_DICT,
                "adapter-a",
                1,
                "adapter-a",
            ).success
        )

        result = manager.stage_adapter(
            _named_tensors_for_layer_0(fill_value=1.0),
            CONFIG_DICT,
            "adapter-b",
            2,
            "adapter-b",
        )

        self.assertFalse(result.success)
        self.assertIn("adapter-a", result.error_message)


class TestStagedOFTManagerActivation(unittest.TestCase):
    def _staged(self, manager, uid="adapter-a", version=1, fill_value=9.0):
        result = manager.stage_adapter(
            _named_tensors_for_layer_0(fill_value=fill_value),
            CONFIG_DICT,
            uid,
            version,
            uid,
        )
        self.assertTrue(result.success)

    def test_activate_requires_a_pending_stage_matching_identity(self):
        manager = _manager()

        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertFalse(result.success)
        self.assertIn("no OFT stage is pending", result.error_message)

    def test_activate_requires_a_reserved_serving_slot(self):
        manager = _manager()
        self._staged(manager)

        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertFalse(result.success)
        self.assertIn("No serving slot is reserved", result.error_message)

    def test_activate_copies_staged_data_and_clears_pending(self):
        manager = _manager()
        self._staged(manager, fill_value=9.0)
        manager.memory_pool.uid_to_buffer_id["adapter-a"] = 2

        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertTrue(result.success)
        self.assertTrue(
            (manager.memory_pool.slot(f"R:{TARGET_MODULE}", 0, 2) == 9.0).all()
        )
        self.assertIsNone(manager._pending_oft_stage)
        self.assertIsNone(manager.memory_pool.staged_identity())


class TestActivateUpdatesManagerBookkeeping(unittest.TestCase):
    """Guards the exact silent-failure mode described in staged_manager.py's
    activate_adapter comment: prepare_oft_batch (oft_manager.py) looks up
    self.adapters[uid].block_size / self.configs[uid].block_size for every
    resident uid on every forward batch. Before the bookkeeping lines in
    activate_adapter, a newly activated uid is physically live in its GPU
    slot but absent from both dicts -- the next prepare_oft_batch call for
    this uid raises KeyError. This test fails on any regression that drops
    that bookkeeping."""

    def test_activate_populates_configs_and_adapters_for_the_new_uid(self):
        manager = _manager()
        result = manager.stage_adapter(
            _named_tensors_for_layer_0(fill_value=9.0),
            CONFIG_DICT,
            "adapter-a",
            1,
            "adapter-a",
        )
        self.assertTrue(result.success)
        manager.memory_pool.uid_to_buffer_id["adapter-a"] = 2
        self.assertNotIn("adapter-a", manager.configs)
        self.assertNotIn("adapter-a", manager.adapters)

        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertTrue(result.success)
        self.assertIn("adapter-a", manager.configs)
        self.assertIn("adapter-a", manager.adapters)
        self.assertEqual(manager.configs["adapter-a"].block_size, BLOCK_SIZE)
        self.assertEqual(manager.adapters["adapter-a"].block_size, BLOCK_SIZE)


if __name__ == "__main__":
    unittest.main()

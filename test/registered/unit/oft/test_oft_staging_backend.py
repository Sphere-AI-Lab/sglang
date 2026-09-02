import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    fill_value so slot-isolation assertions can tell slots apart. This is the
    memory pool's OWN internal format -- exercised directly against
    StagedOFTMemoryPool.stage()/activate() in TestOFTStaging{SlotReservation,
    Transaction} above. StagedOFTManager.stage_adapter does NOT take this
    format (see _raw_named_tensors_for_layer_0 below)."""
    r = torch.full((BLOCK_SIZE, BLOCK_SIZE), fill_value, dtype=torch.float32)
    return {(TARGET_MODULE, 0): (r, BLOCK_SIZE, None, 1)}


def _raw_named_tensors_for_layer_0(fill_value: float):
    """Raw checkpoint-name compact OFT weight for layer 0's target module --
    the actual shape StagedOFTManager.stage_adapter's named_tensors argument
    takes in production (weight_updater.py -> peft/integration.py ->
    oft_manager.stage_adapter(tensors, ...), and OFTManager._stage_fill's own
    docstring: "raw checkpoint-name tensors"). A single compact block
    (num_blocks=1) so precompute_oft_r's result broadcasts to every block
    position in the runtime buffer regardless of that buffer's own block
    count -- the same "block_share" case _write_oft_r_block already handles
    for a real single-block adapter -- so this test doesn't need to know the
    pool's inferred per-module block count."""
    n_elements = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    compact = torch.full((1, n_elements), fill_value, dtype=torch.float32)
    return [(f"model.layers.0.self_attn.{TARGET_MODULE}.oft_R", compact)]


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


class TestStagingCoexistsWithMultiTenancy(unittest.TestCase):
    """Guards the exact gap found while designing this: AdapterMemPool.activate()
    is pool-wide (one _active_version for the whole pool); StagedOFTMemoryPool
    must NOT have that property, or admitting a second adapter while a first
    is being staged would corrupt the first's serving slot."""

    def test_two_resident_adapters_keep_independent_versions(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.uid_to_buffer_id["adapter-a"] = 0
        pool.uid_to_buffer_id["adapter-b"] = 1

        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=1.0))
        pool.activate("adapter-a", 1, destination=0)

        pool.stage("adapter-b", 5, _named_tensors_for_layer_0(fill_value=2.0))
        pool.activate("adapter-b", 5, destination=1)

        self.assertEqual(pool.active_version_for("adapter-a"), 1)
        self.assertEqual(pool.active_version_for("adapter-b"), 5)

    def test_activating_one_adapter_does_not_touch_a_second_resident_slot(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.uid_to_buffer_id["adapter-a"] = 0
        pool.uid_to_buffer_id["adapter-b"] = 1
        slot_1_before = pool.slot(f"R:{TARGET_MODULE}", 0, 1).clone()

        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=1.0))
        pool.activate("adapter-a", 1, destination=0)

        self.assertTrue((pool.slot(f"R:{TARGET_MODULE}", 0, 1) == slot_1_before).all())


class TestStreamedOFTUnload(unittest.TestCase):
    def test_unload_restores_identity_and_forgets_the_active_version(self):
        from sglang.srt.oft.oft_registry import OFTRef
        from sglang.srt.oft.mem_pool import EMPTY_SLOT

        manager = _manager()
        pool = manager.memory_pool
        uid = "adapter-a"
        slot_id = 2
        ref = OFTRef(
            adapter_id=uid,
            adapter_name="policy",
            adapter_path="__tensor__",
            adapter_version=7,
            pinned=False,
        )
        manager.configs[uid] = MagicMock()
        manager.refs[uid] = ref
        pool.uid_to_buffer_id[uid] = slot_id
        pool.buffer_id_to_uid[slot_id] = uid
        pool._active_versions[uid] = 7
        pool.R_buffer[TARGET_MODULE][0][slot_id].zero_()
        expert_module = SimpleNamespace(
            w1_oft_r=torch.ones(1),
            w3_oft_r=torch.ones(1),
            w13_oft_r=torch.ones(1),
            w2_oft_r=torch.ones(1),
        )
        manager._moe_modules = {0: expert_module}

        result = manager.unload_streamed_adapter(ref)

        self.assertTrue(result.success, result.error_message)
        self.assertNotIn(uid, manager.configs)
        self.assertNotIn(uid, manager.refs)
        self.assertNotIn(uid, pool.uid_to_buffer_id)
        self.assertIs(pool.buffer_id_to_uid[slot_id], EMPTY_SLOT)
        self.assertIsNone(pool.active_version_for(uid))
        identity = torch.eye(BLOCK_SIZE).expand_as(
            pool.R_buffer[TARGET_MODULE][0][slot_id]
        )
        self.assertTrue(
            torch.equal(pool.R_buffer[TARGET_MODULE][0][slot_id], identity)
        )
        self.assertIsNone(expert_module.w13_oft_r)
        self.assertIsNone(expert_module.w2_oft_r)
        self.assertIsNone(expert_module.w1_oft_r)
        self.assertIsNone(expert_module.w3_oft_r)


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
    manager.max_oft_block_size = BLOCK_SIZE
    manager.adapter_modules = [{}]
    manager.configs = {}
    manager.adapters = {}
    manager.refs = {}
    manager.num_pinned = 0
    manager._pending_oft_stage = None
    manager.memory_pool.uid_to_buffer_id[None] = 0
    manager.memory_pool.buffer_id_to_uid[0] = None
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
            _raw_named_tensors_for_layer_0(fill_value=9.0),
            CONFIG_DICT,
            name="adapter-a",
            version=1,
            adapter_id="adapter-a",
        )

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(
            manager.memory_pool.staged_identity(), ("adapter-a", 1)
        )
        self.assertIsNotNone(manager._pending_oft_stage)
        self.assertEqual(manager._pending_oft_stage.uid, "adapter-a")

    def test_same_pending_identity_retry_is_idempotent(self):
        manager = _manager()
        args = (
            _raw_named_tensors_for_layer_0(fill_value=9.0),
            CONFIG_DICT,
            "adapter-a",
            1,
            "adapter-a",
        )

        first = manager.stage_adapter(*args)
        second = manager.stage_adapter(*args)

        self.assertTrue(first.success, first.error_message)
        self.assertTrue(second.success, second.error_message)

    def test_conflicting_pending_stage_is_rejected(self):
        manager = _manager()
        self.assertTrue(
            manager.stage_adapter(
                _raw_named_tensors_for_layer_0(fill_value=9.0),
                CONFIG_DICT,
                "adapter-a",
                1,
                "adapter-a",
            ).success
        )

        result = manager.stage_adapter(
            _raw_named_tensors_for_layer_0(fill_value=1.0),
            CONFIG_DICT,
            "adapter-b",
            2,
            "adapter-b",
        )

        self.assertFalse(result.success)
        self.assertIn("adapter-a", result.error_message)

    def test_stage_rejects_a_block_size_mismatch(self):
        """Guards OFTManager._stage_fill's block_size check (oft_manager.py:
        1440-1449), reused unchanged by _partition_and_precompute -- a
        streamed update whose PEFT config disagrees with the server's
        --max-oft-block-size must be rejected, not silently misapplied."""
        manager = _manager()
        bad_config = dict(CONFIG_DICT, oft_block_size=BLOCK_SIZE * 2)

        result = manager.stage_adapter(
            _raw_named_tensors_for_layer_0(fill_value=9.0),
            bad_config,
            "adapter-a",
            1,
            "adapter-a",
        )

        self.assertFalse(result.success)
        self.assertIn("block_size", result.error_message)

    def test_a_construction_failure_never_touches_the_pool_or_jams_the_slot(self):
        """Regression for round-2 review: OFTConfig.from_dict/OFTAdapter
        construction must run, and fail, BEFORE memory_pool.stage() is ever
        called -- there is no rollback for a pool mutation once it has run.
        Getting the order backwards leaves the hidden staging slot
        permanently occupied by the failed uid (StagedOFTMemoryPool.stage's
        own _require_staged_identity then rejects every subsequent
        stage_adapter call, for ANY uid, until someone retries the exact
        failed (uid, version) with a corrected config -- an unexposed,
        untested recovery path). Mirrors StagedLoRAManager.stage_adapter's
        order: LoRAConfig.from_dict -> validate -> adapter construction, all
        strictly before memory_pool.stage(...)."""
        manager = _manager()
        bad_config = {"target_modules": [TARGET_MODULE], "oft_block_size": BLOCK_SIZE}
        self.assertNotIn("peft_type", bad_config)

        result = manager.stage_adapter(
            _raw_named_tensors_for_layer_0(fill_value=9.0),
            bad_config,
            "adapter-a",
            1,
            "adapter-a",
        )

        self.assertFalse(result.success)
        self.assertIn("peft_type", result.error_message)
        self.assertIsNone(
            manager.memory_pool.staged_identity(),
            "a construction failure must never touch the pool",
        )
        self.assertIsNone(manager._pending_oft_stage)
        self.assertNotIn("adapter-a", manager.memory_pool.uid_to_buffer_id)

        # The slot must not be jammed: a different uid can still stage.
        result = manager.stage_adapter(
            _raw_named_tensors_for_layer_0(fill_value=1.0),
            CONFIG_DICT,
            "adapter-b",
            2,
            "adapter-b",
        )

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(manager.memory_pool.staged_identity(), ("adapter-b", 2))

    def test_an_expert_apply_failure_never_touches_the_pool_or_jams_the_slot(self):
        """Regression for round-3 review: the SAME jamming failure class as
        test_a_construction_failure_never_touches_the_pool_or_jams_the_slot,
        but triggered from apply_streamed_expert_oft (oft_manager.py:1171+,
        unedited) instead of adapter construction. That method has real raise
        paths for a mismatched expert/MoE OFT chunk (a shape/dtype/device
        mismatch via _raise_streamed_expert_oft_buffer_mismatch, or a
        tp_size-divisibility assert) and runs AFTER memory_pool.stage(...)
        has already mutated the dense group and set _staged_uid/
        _staged_version. Without the discard_stage() cleanup in
        stage_adapter's mutation try/except, this leaves the pool believing
        the failed (uid, version) is staged while self._pending_oft_stage is
        never set -- jamming the one hidden slot for every future
        stage_adapter call (any uid).

        Building a real mismatched-expert-buffer fixture would need a
        FusedMoE module wired into base_model (this file's fixtures use a
        bare MagicMock, matching Task 2's own fixtures, which also don't
        cover the expert path) -- mocking apply_streamed_expert_oft's raise
        directly is the maintainable way to trigger this specific method's
        failure without that heavier fixture, while still exercising the
        real stage_adapter code path end to end (real _partition_and_
        precompute, real memory_pool.stage(), real OFTAdapter construction)."""
        manager = _manager()
        raw_expert_tensor = [
            ("model.layers.0.mlp.experts.0.gate_proj.oft_R", torch.randn(2, 6))
        ]

        with patch.object(
            manager,
            "apply_streamed_expert_oft",
            side_effect=RuntimeError("expert buffer mismatch"),
        ):
            result = manager.stage_adapter(
                raw_expert_tensor, CONFIG_DICT, "adapter-a", 1, "adapter-a"
            )

        self.assertFalse(result.success)
        self.assertIn("expert buffer mismatch", result.error_message)
        self.assertIsNone(
            manager.memory_pool.staged_identity(),
            "an expert-apply failure must leave the pool clean, not jammed",
        )
        self.assertIsNone(manager._pending_oft_stage)
        self.assertNotIn("adapter-a", manager.memory_pool.uid_to_buffer_id)

        # The slot must not be jammed: a different uid can still stage.
        result = manager.stage_adapter(
            _raw_named_tensors_for_layer_0(fill_value=1.0),
            CONFIG_DICT,
            "adapter-b",
            2,
            "adapter-b",
        )

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(manager.memory_pool.staged_identity(), ("adapter-b", 2))


class TestStagedOFTManagerActivation(unittest.TestCase):
    def _staged(self, manager, uid="adapter-a", version=1, fill_value=9.0):
        result = manager.stage_adapter(
            _raw_named_tensors_for_layer_0(fill_value=fill_value),
            CONFIG_DICT,
            uid,
            version,
            uid,
        )
        self.assertTrue(result.success, result.error_message)

    def test_activate_requires_a_pending_stage_matching_identity(self):
        manager = _manager()

        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertFalse(result.success)
        self.assertIn("no OFT stage is pending", result.error_message)

    def test_first_stage_reserves_a_serving_slot_for_the_minted_id(self):
        manager = _manager()
        self._staged(manager)

        destination = manager.memory_pool.uid_to_buffer_id["adapter-a"]
        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertTrue(result.success, result.error_message)
        self.assertNotEqual(destination, manager.memory_pool.staging_idx)
        self.assertEqual(
            manager.memory_pool.buffer_id_to_uid[destination], "adapter-a"
        )

    def test_stage_then_activate_from_raw_tensors_lands_the_transformed_value(self):
        """End-to-end through the real transformation (mirrors Task 2's
        TestOFTStagingTransaction.test_stage_then_activate_writes_only_the_
        destination_slot, but driven through StagedOFTManager.stage_adapter/
        activate_adapter with a RAW checkpoint-name tensor -- the shape a
        real weight_updater.py -> peft/integration.py ->
        oft_manager.stage_adapter(...) call actually supplies -- rather than
        the pool's internal format directly)."""
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        manager = _manager()
        compact = _raw_named_tensors_for_layer_0(fill_value=9.0)[0][1]
        slot_0_before = manager.memory_pool.slot(f"R:{TARGET_MODULE}", 0, 0).clone()

        self._staged(manager, fill_value=9.0)
        destination = manager.memory_pool.uid_to_buffer_id["adapter-a"]
        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertTrue(result.success, result.error_message)
        expected_r = precompute_oft_r(compact, BLOCK_SIZE)[0]
        actual = manager.memory_pool.slot(f"R:{TARGET_MODULE}", 0, destination)
        self.assertTrue(
            torch.allclose(actual, expected_r.expand_as(actual)),
            "activate must land the Cayley-transformed raw tensor, not the "
            "raw compact weight, in the destination slot",
        )
        self.assertTrue(
            (manager.memory_pool.slot(f"R:{TARGET_MODULE}", 0, 0) == slot_0_before).all(),
            "activating one uid must not touch slot 0",
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
            _raw_named_tensors_for_layer_0(fill_value=9.0),
            CONFIG_DICT,
            "adapter-a",
            1,
            "adapter-a",
        )
        self.assertTrue(result.success, result.error_message)
        self.assertNotIn("adapter-a", manager.configs)
        self.assertNotIn("adapter-a", manager.adapters)

        result = manager.activate_adapter("adapter-a", 1, "adapter-a")

        self.assertTrue(result.success, result.error_message)
        self.assertIn("adapter-a", manager.configs)
        self.assertIn("adapter-a", manager.adapters)
        self.assertEqual(manager.configs["adapter-a"].block_size, BLOCK_SIZE)
        self.assertEqual(manager.adapters["adapter-a"].block_size, BLOCK_SIZE)


class TestOFTStagingBackendIsSymmetric(unittest.TestCase):
    def test_implements_the_shared_interface(self):
        from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        self.assertTrue(issubclass(OFTStagingBackend, AdapterStagingBackend))


class TestOFTStagingBackendPrepareActivation(unittest.TestCase):
    """Regression for round-1 review: a real client's obj.adapter_id defaults
    to None, so prepare_activation must resolve it from tm.peft_ref_cache
    (the same lookup register_peft_ref itself uses) -- otherwise
    StagedOFTManager.activate_adapter's uid falls back to obj.adapter_name
    and never matches the UUID stage_adapter recorded."""

    def test_resolves_adapter_id_from_the_ref_cache(self):
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        from sglang.srt.oft.oft_registry import OFTRef

        ref = OFTRef(
            adapter_id="uuid-123",
            adapter_name="adapter-a",
            adapter_path="__distributed__",
            adapter_version=7,
            reloadable=False,
        )
        tm = MagicMock()
        tm.pending_oft_stage = ref
        obj = MagicMock()
        obj.adapter_name = "adapter-a"
        obj.adapter_version = "7"
        obj.adapter_id = None

        OFTStagingBackend(tm).prepare_activation(obj)

        self.assertEqual(obj.adapter_id, "uuid-123")

    def test_raises_when_the_adapter_name_is_not_registered(self):
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = MagicMock()
        tm.pending_oft_stage = None
        obj = MagicMock()
        obj.adapter_name = "adapter-a"
        obj.adapter_version = "7"

        with self.assertRaises(ValueError):
            OFTStagingBackend(tm).prepare_activation(obj)


class TestOFTStagingBackendVersioning(unittest.TestCase):
    @staticmethod
    def _tm():
        from sglang.srt.oft.oft_registry import OFTRegistry

        return SimpleNamespace(
            peft_registry=OFTRegistry(),
            peft_ref_cache={},
            peft_update_lock=asyncio.Lock(),
            pending_oft_stage=None,
            failed_oft_activations={},
        )

    def test_rejects_equal_or_stale_versions_before_staging(self):
        from sglang.srt.oft.oft_registry import OFTRef
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = self._tm()
        active = OFTRef(
            adapter_id="id-a",
            adapter_name="policy",
            adapter_path="__distributed__",
            adapter_version=4,
            reloadable=False,
        )
        asyncio.run(tm.peft_registry.register(active))
        tm.peft_ref_cache[active.adapter_name] = active

        for version in ("4", "3"):
            obj = SimpleNamespace(
                load_format="oft_adapter",
                adapter_name="policy",
                adapter_version=version,
                adapter_id=None,
            )
            with self.assertRaisesRegex(ValueError, "newer than active version 4"):
                asyncio.run(OFTStagingBackend(tm).reserve_stage(obj))

    def test_publishes_the_exact_requested_version_only_after_worker_agreement(self):
        from sglang.srt.managers.io_struct import ActivateAdapterVersionReqOutput
        from sglang.srt.oft.oft_registry import OFTRef
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = self._tm()
        active = OFTRef(
            adapter_id="id-a",
            adapter_name="policy",
            adapter_path="__distributed__",
            adapter_version=2,
            reloadable=False,
        )
        asyncio.run(tm.peft_registry.register(active))
        tm.peft_ref_cache[active.adapter_name] = active
        backend = OFTStagingBackend(tm)
        obj = SimpleNamespace(
            load_format="oft_adapter",
            adapter_name="policy",
            adapter_version="7",
            adapter_id=None,
        )

        asyncio.run(backend.reserve_stage(obj))
        self.assertEqual(
            tm.peft_registry.get_all_adapters()["policy"].adapter_version, 2
        )
        success, _ = asyncio.run(
            backend.finish_activation(
                obj,
                [
                    ActivateAdapterVersionReqOutput(
                        success=True,
                        message="activated",
                        active_adapter_version="7",
                    )
                ],
            )
        )

        self.assertTrue(success)
        published = tm.peft_registry.get_all_adapters()["policy"]
        self.assertEqual((published.adapter_id, published.adapter_version), ("id-a", 7))
        self.assertIsNone(tm.pending_oft_stage)

    def test_worker_version_disagreement_does_not_publish(self):
        from sglang.srt.managers.io_struct import ActivateAdapterVersionReqOutput
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = self._tm()
        backend = OFTStagingBackend(tm)
        obj = SimpleNamespace(
            load_format="oft_adapter",
            adapter_name="policy",
            adapter_version="7",
            adapter_id=None,
        )
        asyncio.run(backend.reserve_stage(obj))

        success, message = asyncio.run(
            backend.finish_activation(
                obj,
                [
                    ActivateAdapterVersionReqOutput(
                        success=True,
                        message="wrong version",
                        active_adapter_version="6",
                    )
                ],
            )
        )

        self.assertFalse(success)
        self.assertIn("worker active versions", message)
        self.assertEqual(tm.peft_registry.get_all_adapters(), {})
        self.assertIn("policy", tm.failed_oft_activations)


class TestExpertOFTRejection(unittest.TestCase):
    def test_init_rejects_expert_targets_until_routing_is_request_aware(self):
        from sglang.srt.oft.oft_manager import OFTManager

        manager = object.__new__(OFTManager)
        manager.refs = {}
        manager.init_oft_adapters = MagicMock()
        manager.init_oft_shapes = MagicMock()
        manager.init_oft_modules = MagicMock()
        manager._install_moe_oft_wrappers = MagicMock(return_value=1)

        with self.assertRaisesRegex(ValueError, "request-aware"):
            manager.init_state(max_oft_block_size=4, target_modules=["w1"])


if __name__ == "__main__":
    unittest.main()

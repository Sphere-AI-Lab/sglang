"""Regression tests for the native multi-tenant OFT adapter admission path
added to OFTManager/OFTMemoryPool (`srt/oft/oft_manager.py`,
`srt/oft/mem_pool.py`, `srt/oft/streamed_weight_loader.py`), covering three
bugs found by Task 8's GPU integration suite and fixed afterward:

1. `load_adapter_from_tensors` passed a raw `Dict[str, Tensor]` (the actual
   production shape from `TpModelWorker._deserialize_own_rank`) straight
   through to code that iterates `named_tensors` as `List[Tuple[str, Tensor]]`
   -- crashed every native-RPC load with "too many values to unpack".
2. `allocate_buffer_slot()` (used by the legacy single-active streamed path)
   has no eviction fallback and hard-fails when the pool is full; the new
   multi-tenant admission path needs real LRU eviction instead, without
   changing `allocate_buffer_slot()`'s own behavior.
3. `LRUEvictionPolicy.select_victim` (`srt/lora/eviction_policy.py`, shared
   with LoRA) asserts if none of the eviction candidates were ever
   `mark_used(...)`'d and `None` isn't offered -- exactly the scenario of an
   adapter loaded via the native RPC path but never yet `generate()`'d on
   (which is the normal case for `prepare_oft_batch`'s usual `mark_used`
   calls, which only fire per served forward batch).

Also covers the fix for a related safety gap: eviction (destroying a
resident, working adapter) must not happen before payload validation, or a
malformed new-adapter payload could destroy an unrelated resident adapter
for nothing. `OFTManager.load_adapter_from_tensors` now calls
`_resolve_streamed_oft_tensor_groups` (validate, no buffer/mutation) before
`allocate_buffer_slot_with_eviction`, and only commits
(`_commit_streamed_oft_tensor_groups`) after eviction+registration.
"""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.oft.base.mem_pool import EMPTY_SLOT, AdapterMemPool
from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.streamed_weight_loader import (
    _commit_streamed_oft_tensor_groups,
    _resolve_streamed_oft_tensor_groups,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TARGET_MODULE = "q_proj"
BLOCK_SIZE = 4
CONFIG_DICT = {
    "peft_type": "oft",
    "target_modules": [TARGET_MODULE],
    "oft_block_size": BLOCK_SIZE,
}


def _make_pool(max_ofts_per_batch, max_block_size=BLOCK_SIZE):
    """A lightweight stand-in carrying exactly the attributes
    OFTMemoryPool.allocate_buffer_slot[_with_eviction] touch (bookkeeping
    dicts/lists + the shared eviction_policy) -- no R_buffer/torch tensors,
    since none of this logic reads or writes them. The REAL (unbound)
    OFTMemoryPool methods are bound onto it via MethodType, so what's under
    test is the actual production code, not a reimplementation."""
    pool = SimpleNamespace(
        max_oft_block_size=max_block_size,
        max_ofts_per_batch=max_ofts_per_batch,
        uid_to_buffer_id={},
        buffer_id_to_uid=[EMPTY_SLOT] * max_ofts_per_batch,
        eviction_policy=get_eviction_policy("lru"),
        reset_buffer_slot_to_identity=MagicMock(),
    )
    pool.allocate_buffer_slot_with_eviction = MethodType(
        OFTMemoryPool.allocate_buffer_slot_with_eviction, pool
    )
    pool.allocate_buffer_slot = MethodType(OFTMemoryPool.allocate_buffer_slot, pool)
    return pool


def _make_manager(max_ofts_per_batch=4, max_block_size=BLOCK_SIZE):
    """OFTManager.__new__ bypasses __init__ (which requires a real wired-up
    base_model/backend/memory pool) -- the methods under test only touch
    self.refs/self.configs/self.adapters/self.memory_pool, all set here."""
    mgr = OFTManager.__new__(OFTManager)
    mgr.refs = {}
    mgr.configs = {}
    mgr.adapters = {}
    mgr.num_pinned = 0
    mgr.memory_pool = _make_pool(max_ofts_per_batch, max_block_size)
    return mgr


def _make_base_pool(max_adapters_per_batch):
    """A lightweight stand-in carrying exactly the attributes
    AdapterMemPool._acquire_buffer_slot touches (bookkeeping dict/list + the
    shared eviction_policy) -- no R_buffer/torch tensors, since none of this
    logic reads or writes them. The REAL (unbound) AdapterMemPool method is
    bound onto it via MethodType, so what's under test is the actual
    production code, not a reimplementation. This is the generic base-class
    admission path shared by OFTMemoryPool AND StagedOFTMemoryPool, used by
    prepare_oft_batch's lazy per-batch admission (oft/mem_pool.py:698)."""
    pool = SimpleNamespace(
        max_adapters_per_batch=max_adapters_per_batch,
        uid_to_buffer_id={},
        buffer_id_to_uid=[EMPTY_SLOT] * max_adapters_per_batch,
        eviction_policy=get_eviction_policy("lru"),
    )
    pool._acquire_buffer_slot = MethodType(AdapterMemPool._acquire_buffer_slot, pool)
    return pool


def _make_ref(name, adapter_id=None, pinned=False, reloadable=True):
    return OFTRef(
        adapter_id=adapter_id or name,
        adapter_name=name,
        adapter_path=name,
        pinned=pinned,
        reloadable=reloadable,
    )


class TestLoadAdapterFromTensorsDictInput(unittest.TestCase):
    """Bug 1: named_tensors is a raw dict in production
    (TpModelWorker._deserialize_own_rank preserves whatever the client
    serialized); the resolve/commit helpers require List[Tuple]."""

    def test_dict_named_tensors_reach_resolve_and_commit_as_list_of_tuples(self):
        mgr = _make_manager()
        ref = _make_ref("a1")
        named_tensors_dict = {
            "model.layers.0.self_attn.q_proj.oft_R": "T0",
            "model.layers.1.self_attn.q_proj.oft_R": "T1",
        }
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ) as mock_resolve, patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ) as mock_commit:
            result = mgr.load_adapter_from_tensors(ref, named_tensors_dict, CONFIG_DICT)
        self.assertTrue(result.success, result.error_message)

        resolved_tensors = mock_resolve.call_args.args[1]
        self.assertIsInstance(resolved_tensors, list)
        self.assertEqual(set(resolved_tensors), set(named_tensors_dict.items()))

        committed_tensors = mock_commit.call_args.args[1]
        self.assertIsInstance(committed_tensors, list)
        self.assertEqual(set(committed_tensors), set(named_tensors_dict.items()))

    def test_list_named_tensors_passed_through_unchanged(self):
        mgr = _make_manager()
        ref = _make_ref("a1")
        named_tensors_list = [("model.layers.0.self_attn.q_proj.oft_R", "T0")]
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ) as mock_commit:
            mgr.load_adapter_from_tensors(ref, named_tensors_list, CONFIG_DICT)
        self.assertIs(mock_commit.call_args.args[1], named_tensors_list)


class TestAllocateBufferSlotWithEviction(unittest.TestCase):
    """Bug 3: allocate_buffer_slot_with_eviction (new) evicts an unpinned
    resident real adapter when full; allocate_buffer_slot() (legacy
    primitive, unchanged) still hard-fails with no eviction."""

    def test_empty_slot_used_first_no_eviction(self):
        pool = _make_pool(max_ofts_per_batch=2)
        buffer_id, evicted = pool.allocate_buffer_slot_with_eviction({})
        self.assertEqual(buffer_id, 0)
        self.assertIsNone(evicted)

    def test_full_pool_evicts_lru_unpinned_resident(self):
        pool = _make_pool(max_ofts_per_batch=2)
        refs = {"old": _make_ref("old"), "new_resident": _make_ref("new_resident")}
        pool.uid_to_buffer_id = {"old": 0, "new_resident": 1}
        pool.buffer_id_to_uid = ["old", "new_resident"]
        pool.eviction_policy.mark_used("old")
        pool.eviction_policy.mark_used("new_resident")
        buffer_id, evicted = pool.allocate_buffer_slot_with_eviction(refs)
        self.assertEqual(evicted, "old")  # least-recently marked used
        self.assertEqual(buffer_id, 0)
        self.assertNotIn("old", pool.uid_to_buffer_id)
        self.assertEqual(pool.buffer_id_to_uid[0], EMPTY_SLOT)

    def test_pinned_resident_never_evicted(self):
        pool = _make_pool(max_ofts_per_batch=1)
        refs = {"pinned_one": _make_ref("pinned_one", pinned=True)}
        pool.uid_to_buffer_id = {"pinned_one": 0}
        pool.buffer_id_to_uid = ["pinned_one"]
        with self.assertRaises(ValueError):
            pool.allocate_buffer_slot_with_eviction(refs)
        self.assertIn("pinned_one", pool.uid_to_buffer_id)  # untouched on failure

    def test_unpinned_preferred_over_pinned_when_both_present(self):
        pool = _make_pool(max_ofts_per_batch=2)
        refs = {
            "pinned_one": _make_ref("pinned_one", pinned=True),
            "evict_me": _make_ref("evict_me", pinned=False),
        }
        pool.uid_to_buffer_id = {"pinned_one": 0, "evict_me": 1}
        pool.buffer_id_to_uid = ["pinned_one", "evict_me"]
        pool.eviction_policy.mark_used("pinned_one")
        pool.eviction_policy.mark_used("evict_me")
        buffer_id, evicted = pool.allocate_buffer_slot_with_eviction(refs)
        self.assertEqual(evicted, "evict_me")
        self.assertEqual(buffer_id, 1)
        self.assertIn("pinned_one", pool.uid_to_buffer_id)

    def test_identity_slot_never_a_candidate(self):
        # Pool full of ONLY the base/identity slot (uid=None) -- must raise,
        # never evict the base model placeholder.
        pool = _make_pool(max_ofts_per_batch=1)
        pool.uid_to_buffer_id = {None: 0}
        pool.buffer_id_to_uid = [None]
        with self.assertRaises(ValueError):
            pool.allocate_buffer_slot_with_eviction({})

    def test_non_reloadable_resident_never_evicted(self):
        # C1 fix: an adapter loaded over the wire (reloadable=False) has no
        # CPU-side artifact to re-page from, so evicting it is unrecoverable
        # -- it must never be picked as a victim, mirroring the pinned check.
        pool = _make_pool(max_ofts_per_batch=1)
        refs = {"wire_loaded": _make_ref("wire_loaded", reloadable=False)}
        pool.uid_to_buffer_id = {"wire_loaded": 0}
        pool.buffer_id_to_uid = ["wire_loaded"]
        with self.assertRaises(ValueError):
            pool.allocate_buffer_slot_with_eviction(refs)
        self.assertIn("wire_loaded", pool.uid_to_buffer_id)  # untouched on failure

    def test_reloadable_preferred_over_non_reloadable_when_both_present(self):
        pool = _make_pool(max_ofts_per_batch=2)
        refs = {
            "wire_loaded": _make_ref("wire_loaded", reloadable=False),
            "disk_backed": _make_ref("disk_backed", reloadable=True),
        }
        pool.uid_to_buffer_id = {"wire_loaded": 0, "disk_backed": 1}
        pool.buffer_id_to_uid = ["wire_loaded", "disk_backed"]
        pool.eviction_policy.mark_used("wire_loaded")
        pool.eviction_policy.mark_used("disk_backed")
        buffer_id, evicted = pool.allocate_buffer_slot_with_eviction(refs)
        self.assertEqual(evicted, "disk_backed")
        self.assertEqual(buffer_id, 1)
        self.assertIn("wire_loaded", pool.uid_to_buffer_id)

    def test_pool_full_of_only_pinned_and_non_reloadable_raises(self):
        # No legal victim anywhere in the pool -- must fail cleanly rather
        # than fall back to evicting one of them unsafely.
        pool = _make_pool(max_ofts_per_batch=2)
        refs = {
            "pinned_one": _make_ref("pinned_one", pinned=True),
            "wire_loaded": _make_ref("wire_loaded", reloadable=False),
        }
        pool.uid_to_buffer_id = {"pinned_one": 0, "wire_loaded": 1}
        pool.buffer_id_to_uid = ["pinned_one", "wire_loaded"]
        with self.assertRaises(ValueError):
            pool.allocate_buffer_slot_with_eviction(refs)

    def test_allocate_buffer_slot_itself_still_hard_fails_unchanged(self):
        pool = _make_pool(max_ofts_per_batch=1)
        pool.uid_to_buffer_id = {"x": 0}
        pool.buffer_id_to_uid = ["x"]
        with self.assertRaises(ValueError):
            pool.allocate_buffer_slot()


class TestSelectVictimNeverUsedEdgeCase(unittest.TestCase):
    """The third bug: LRUEvictionPolicy.select_victim asserts if nothing in
    the candidate set was ever mark_used()'d and None isn't offered --
    exactly what happens to an adapter loaded via the native RPC path but
    never yet generate()'d on. Covers both the normal (tracked) case and
    the edge case that used to crash."""

    def test_normal_case_tracked_candidate_is_evicted(self):
        pool = _make_pool(max_ofts_per_batch=1)
        refs = {"tracked": _make_ref("tracked")}
        pool.uid_to_buffer_id = {"tracked": 0}
        pool.buffer_id_to_uid = ["tracked"]
        pool.eviction_policy.mark_used("tracked")
        buffer_id, evicted = pool.allocate_buffer_slot_with_eviction(refs)
        self.assertEqual(evicted, "tracked")
        self.assertEqual(buffer_id, 0)

    def test_never_used_candidate_does_not_crash(self):
        pool = _make_pool(max_ofts_per_batch=1)
        refs = {"never_served": _make_ref("never_served")}
        pool.uid_to_buffer_id = {"never_served": 0}
        pool.buffer_id_to_uid = ["never_served"]
        # No mark_used call -- reproduces Task 8's exact scenario (loaded
        # but never generate()'d on). Must not raise AssertionError.
        buffer_id, evicted = pool.allocate_buffer_slot_with_eviction(refs)
        self.assertEqual(evicted, "never_served")
        self.assertEqual(buffer_id, 0)

    def test_load_adapter_from_tensors_marks_used_at_admission_so_later_eviction_finds_it(self):
        """End-to-end: an adapter admitted via load_adapter_from_tensors and
        never generate()'d on (so prepare_oft_batch's mark_used never ran
        for it) must still be a valid eviction candidate for a later load,
        rather than crashing select_victim."""
        mgr = _make_manager(max_ofts_per_batch=2)
        mgr.memory_pool.uid_to_buffer_id[None] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = None  # identity placeholder

        ref_a = _make_ref("adapter_a")
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ):
            result_a = mgr.load_adapter_from_tensors(ref_a, [], CONFIG_DICT)
        self.assertTrue(result_a.success, result_a.error_message)

        ref_b = _make_ref("adapter_b")
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ):
            result_b = mgr.load_adapter_from_tensors(ref_b, [], CONFIG_DICT)
        self.assertTrue(result_b.success, result_b.error_message)
        self.assertNotIn("adapter_a", mgr.refs)  # evicted
        self.assertIn("adapter_b", mgr.refs)
        self.assertEqual(mgr.memory_pool.uid_to_buffer_id[None], 0)  # identity untouched


class TestValidateBeforeEvict(unittest.TestCase):
    """Failure-safety gap fix: a malformed payload must not destroy a
    resident adapter it never needed to evict in the first place."""

    def test_resolve_failure_never_triggers_eviction(self):
        mgr = _make_manager(max_ofts_per_batch=1)
        resident = _make_ref("resident")
        mgr.refs[resident.adapter_id] = resident
        mgr.configs[resident.adapter_id] = "cfg"
        mgr.memory_pool.uid_to_buffer_id[resident.adapter_id] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = resident.adapter_id

        new_ref = _make_ref("new_adapter")
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(None, "Unresolved OFT tensor names: bogus"),
        ) as mock_resolve, patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups"
        ) as mock_commit:
            result = mgr.load_adapter_from_tensors(new_ref, [], CONFIG_DICT)

        self.assertFalse(result.success)
        self.assertIn("Unresolved OFT tensor names", result.error_message)
        mock_resolve.assert_called_once()
        mock_commit.assert_not_called()
        # The resident adapter must be completely untouched: never evicted
        # for a payload that was invalid anyway.
        self.assertIn("resident", mgr.refs)
        self.assertEqual(mgr.memory_pool.uid_to_buffer_id["resident"], 0)
        self.assertEqual(mgr.memory_pool.buffer_id_to_uid[0], "resident")

    def test_commit_failure_after_real_eviction_names_the_evicted_adapter(self):
        """Residual risk (validation passed, the later GPU-side commit still
        failed) is not silently swallowed: the evicted adapter's name is
        called out explicitly in the error."""
        mgr = _make_manager(max_ofts_per_batch=1)
        resident = _make_ref("resident")
        mgr.refs[resident.adapter_id] = resident
        mgr.configs[resident.adapter_id] = "cfg"
        mgr.memory_pool.uid_to_buffer_id[resident.adapter_id] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = resident.adapter_id
        mgr.memory_pool.eviction_policy.mark_used("resident")

        new_ref = _make_ref("new_adapter")
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(False, "OOM during Cayley precompute"),
        ):
            result = mgr.load_adapter_from_tensors(new_ref, [], CONFIG_DICT)

        self.assertFalse(result.success)
        self.assertIn("resident", result.error_message)
        self.assertIn("evicted", result.error_message)
        self.assertIn("OOM during Cayley precompute", result.error_message)
        # The eviction genuinely happened (validation had already passed).
        self.assertNotIn("resident", mgr.refs)
        # And the NEW adapter must not be left as a phantom resident ref
        # either: register_streamed_adapter + mark_used already ran before
        # commit failed, so without cleanup it would look valid/resident
        # while its buffer's contents are undefined.
        self.assertNotIn(new_ref.adapter_id, mgr.refs)
        self.assertNotIn(new_ref.adapter_id, mgr.configs)
        self.assertNotIn(new_ref.adapter_id, mgr.memory_pool.uid_to_buffer_id)
        # LRU tracking must be cleaned up too, not just the residency maps --
        # otherwise it's a slow leak across repeated failed loads.
        self.assertNotIn(
            new_ref.adapter_id, mgr.memory_pool.eviction_policy.access_order
        )

    def test_commit_failure_does_not_leave_phantom_resident_ref(self):
        """Even with no eviction involved (pool has room), a commit failure
        must not leave the new adapter's ref/config/buffer-mapping
        registered as if it had succeeded."""
        mgr = _make_manager(max_ofts_per_batch=2)
        new_ref = _make_ref("new_adapter")
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(False, "OOM during Cayley precompute"),
        ):
            result = mgr.load_adapter_from_tensors(new_ref, [], CONFIG_DICT)

        self.assertFalse(result.success)
        self.assertIn("OOM during Cayley precompute", result.error_message)
        self.assertNotIn("evicted", result.error_message)  # no eviction happened
        self.assertNotIn(new_ref.adapter_id, mgr.refs)
        self.assertNotIn(new_ref.adapter_id, mgr.configs)
        self.assertNotIn(new_ref.adapter_id, mgr.memory_pool.uid_to_buffer_id)
        # The slot it briefly occupied is back to empty, not phantom-owned.
        self.assertEqual(mgr.memory_pool.buffer_id_to_uid[0], EMPTY_SLOT)
        self.assertNotIn(
            new_ref.adapter_id, mgr.memory_pool.eviction_policy.access_order
        )


class TestEvictionPreservesDiskBackedAdapter(unittest.TestCase):
    """C1 fix #2: once fix #1 excludes wire-loaded (non-reloadable)
    residents from eviction candidacy, the only adapters
    allocate_buffer_slot_with_eviction can ever select as a victim are
    disk-backed (--peft-paths) ones -- self.refs/self.configs are shared by
    both kinds, but only disk-backed adapters also have a self.adapters
    entry. Calling unload_streamed_adapter on such a victim would delete
    its configs/refs while leaving self.adapters behind and num_pinned
    un-decremented -- a half-unloaded state. The eviction (and upsert-of-
    existing-name) call sites must instead only free its buffer slot,
    leaving the rest of its CPU-side bookkeeping intact so it can be
    lazily re-admitted into a fresh slot later, exactly like any adapter
    that simply isn't in the current batch."""

    def test_disk_backed_eviction_victim_keeps_configs_refs_and_adapters(self):
        # max_ofts_per_batch=1: the disk-backed adapter occupies the only
        # slot, so admitting the new one MUST evict it (no empty slot to
        # fall back to first).
        mgr = _make_manager(max_ofts_per_batch=1)
        disk_ref = _make_ref("disk_backed", reloadable=True)
        mgr.refs[disk_ref.adapter_id] = disk_ref
        mgr.configs[disk_ref.adapter_id] = "disk_cfg"
        mgr.adapters[disk_ref.adapter_id] = "disk_adapter_object"
        mgr.memory_pool.uid_to_buffer_id[disk_ref.adapter_id] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = disk_ref.adapter_id
        mgr.memory_pool.eviction_policy.mark_used(disk_ref.adapter_id)

        new_ref = _make_ref("wire_new", reloadable=False)
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ):
            result = mgr.load_adapter_from_tensors(new_ref, [], CONFIG_DICT)

        self.assertTrue(result.success, result.error_message)
        # Lost its GPU buffer slot to the new wire-loaded adapter...
        self.assertNotIn(disk_ref.adapter_id, mgr.memory_pool.uid_to_buffer_id)
        # ...but keeps its CPU-side bookkeeping fully intact -- unlike a
        # wire-loaded victim, which would be fully unloaded (configs/refs
        # deleted too, since it has nothing else to fall back on).
        self.assertIn(disk_ref.adapter_id, mgr.refs)
        self.assertIn(disk_ref.adapter_id, mgr.configs)
        self.assertIn(disk_ref.adapter_id, mgr.adapters)

    def test_upsert_of_disk_backed_name_is_rejected_even_with_mismatched_new_id(self):
        """Same bug shape at the OTHER call site that can unload a resident
        ref by name: an upsert naming an already-loaded disk-backed adapter
        is rejected outright (see TestUpsertRejectsDiskBackedToWireLoadedTransition),
        regardless of what adapter_id the incoming (mismatched-id) ref
        carries -- existing_id is found by NAME match against self.refs, so
        the rejection check does not depend on the new ref's own id. Keeps
        the disk-backed entry completely untouched, same as the realistic
        same-id case."""
        mgr = _make_manager(max_ofts_per_batch=2)
        disk_ref = _make_ref("shared_name", reloadable=True)
        mgr.refs[disk_ref.adapter_id] = disk_ref
        mgr.configs[disk_ref.adapter_id] = "disk_cfg"
        mgr.adapters[disk_ref.adapter_id] = "disk_adapter_object"
        mgr.memory_pool.uid_to_buffer_id[disk_ref.adapter_id] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = disk_ref.adapter_id

        new_ref = _make_ref("shared_name", adapter_id="new_wire_id", reloadable=False)
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ) as mock_commit:
            result = mgr.load_adapter_from_tensors(
                new_ref, [], CONFIG_DICT, upsert=True
            )

        self.assertFalse(result.success)
        self.assertIn(disk_ref.adapter_name, result.error_message)
        mock_commit.assert_not_called()
        self.assertIn(disk_ref.adapter_id, mgr.refs)
        self.assertIn(disk_ref.adapter_id, mgr.configs)
        self.assertIn(disk_ref.adapter_id, mgr.adapters)
        self.assertNotIn("new_wire_id", mgr.refs)


class TestUpsertRejectsDiskBackedToWireLoadedTransition(unittest.TestCase):
    """Second-round fix: upserting a NEW wire-loaded adapter over an
    EXISTING disk-backed (--peft-paths) adapter's name -- reusing the SAME
    adapter_id, the realistic case (resolve_or_reuse reuses ids for a
    matching adapter_name upstream) -- must be rejected outright, not
    silently corrupt self.adapters/num_pinned state.
    _unload_streamed_adapter_if_not_disk_backed no-ops for a disk-backed
    existing entry, but register_streamed_adapter would still overwrite
    self.refs/self.configs for the SAME id afterward, leaving
    self.adapters[existing_id] (the old CPU-side OFTAdapter) stale --
    silently corrupting later num_pinned accounting and disk-vs-streamed
    unload dispatch. Migrating the identity in place was considered and
    rejected as too risky; reject the transition explicitly instead."""

    def test_upsert_with_colliding_id_over_disk_backed_name_is_rejected(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        disk_ref = _make_ref("shared_name", reloadable=True, pinned=True)
        mgr.refs[disk_ref.adapter_id] = disk_ref
        mgr.configs[disk_ref.adapter_id] = "disk_cfg"
        mgr.adapters[disk_ref.adapter_id] = "disk_adapter_object"
        mgr.memory_pool.uid_to_buffer_id[disk_ref.adapter_id] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = disk_ref.adapter_id
        mgr.num_pinned = 1  # disk_ref was loaded pinned, per load_adapter's accounting

        # Reuses the SAME adapter_id as the existing disk-backed entry (the
        # realistic same-id case), not a mismatched one.
        new_ref = _make_ref("shared_name", reloadable=False)
        self.assertEqual(new_ref.adapter_id, disk_ref.adapter_id)

        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups"
        ) as mock_commit:
            result = mgr.load_adapter_from_tensors(
                new_ref, [], CONFIG_DICT, upsert=True
            )

        self.assertFalse(result.success)
        self.assertIn("disk", result.error_message.lower())
        self.assertIn(disk_ref.adapter_name, result.error_message)
        # Rejected before any mutation: the original disk-backed entry, and
        # num_pinned, are completely untouched.
        self.assertIs(mgr.refs[disk_ref.adapter_id], disk_ref)
        self.assertEqual(mgr.configs[disk_ref.adapter_id], "disk_cfg")
        self.assertEqual(mgr.adapters[disk_ref.adapter_id], "disk_adapter_object")
        self.assertEqual(mgr.num_pinned, 1)
        self.assertEqual(mgr.memory_pool.uid_to_buffer_id[disk_ref.adapter_id], 0)
        # The commit path was never reached -- rejected before both
        # _unload_streamed_adapter_if_not_disk_backed and
        # register_streamed_adapter.
        mock_commit.assert_not_called()

    def test_upsert_over_wire_loaded_name_with_same_id_still_works(self):
        """Sanity: the fix must not be over-broad -- upserting a wire-loaded
        adapter over an EXISTING wire-loaded adapter of the same name (the
        normal, previously-working case, no self.adapters entry involved)
        must still succeed."""
        mgr = _make_manager(max_ofts_per_batch=2)
        old_wire_ref = _make_ref("shared_name", reloadable=False, pinned=False)
        mgr.refs[old_wire_ref.adapter_id] = old_wire_ref
        mgr.configs[old_wire_ref.adapter_id] = "old_wire_cfg"
        mgr.memory_pool.uid_to_buffer_id[old_wire_ref.adapter_id] = 0
        mgr.memory_pool.buffer_id_to_uid[0] = old_wire_ref.adapter_id

        new_ref = _make_ref("shared_name", reloadable=False)
        with patch(
            "sglang.srt.oft.streamed_weight_loader._resolve_streamed_oft_tensor_groups",
            return_value=(("fused", {}, {}, []), ""),
        ), patch(
            "sglang.srt.oft.streamed_weight_loader._commit_streamed_oft_tensor_groups",
            return_value=(True, "Success"),
        ):
            result = mgr.load_adapter_from_tensors(
                new_ref, [], CONFIG_DICT, upsert=True
            )
        self.assertTrue(result.success, result.error_message)


class TestAcquireBufferSlotExcludesNonReloadable(unittest.TestCase):
    """Second-round fix: AdapterMemPool._acquire_buffer_slot (the regular
    per-forward-step admission path, called from prepare_oft_batch via
    oft/mem_pool.py:698 -- used by both OFTMemoryPool and the legacy staged
    path's StagedOFTMemoryPool) previously excluded only `ref.pinned` from
    eviction candidacy, missing the `not ref.reloadable` exclusion its
    sibling `allocate_buffer_slot_with_eviction` already had. A wire-loaded
    (non-reloadable, no CPU-side copy) resident adapter could therefore be
    silently evicted here to make room for a different adapter -- and,
    unlike the native-RPC admission path, this call path has no try/except
    wrapper: an uncaught ValueError here reaches run_scheduler_process's
    outermost handler, which SIGQUITs the whole engine."""

    def test_non_reloadable_unpinned_resident_never_evicted(self):
        pool = _make_base_pool(max_adapters_per_batch=1)
        refs = {"wire_loaded": _make_ref("wire_loaded", pinned=False, reloadable=False)}
        pool.uid_to_buffer_id = {"wire_loaded": 0}
        pool.buffer_id_to_uid = ["wire_loaded"]
        with self.assertRaises(ValueError):
            pool._acquire_buffer_slot(cur_uids=set(), refs=refs)
        # untouched on failure
        self.assertIn("wire_loaded", pool.uid_to_buffer_id)
        self.assertEqual(pool.buffer_id_to_uid[0], "wire_loaded")

    def test_mixed_pool_of_wire_loaded_residents_raises_instead_of_evicting(self):
        """The scenario that reproduced the original crash: a memory pool
        entirely full of wire-loaded (non-reloadable, UNPINNED) resident
        adapters, then a request that needs to admit a different adapter
        (e.g. disk-backed) not yet resident. Before this fix, the missing
        `not ref.reloadable` check let the OLD (pinned-only) exclusion treat
        both wire-loaded residents as legal eviction candidates -- one of
        them would be silently evicted. Must now raise a clean ValueError
        instead, mentioning both pinned and non-reloadable adapters."""
        pool = _make_base_pool(max_adapters_per_batch=2)
        refs = {
            "wire_a": _make_ref("wire_a", pinned=False, reloadable=False),
            "wire_b": _make_ref("wire_b", pinned=False, reloadable=False),
        }
        pool.uid_to_buffer_id = {"wire_a": 0, "wire_b": 1}
        pool.buffer_id_to_uid = ["wire_a", "wire_b"]
        pool.eviction_policy.mark_used("wire_a")
        pool.eviction_policy.mark_used("wire_b")

        # cur_uids models prepare_oft_batch's current-batch uid set: it
        # contains the new (not-yet-resident, disk-backed) adapter's uid,
        # but neither wire-loaded resident is needed by this batch.
        with self.assertRaises(ValueError) as ctx:
            pool._acquire_buffer_slot(cur_uids={"disk_backed_new"}, refs=refs)
        message = str(ctx.exception)
        self.assertIn("pinned", message)
        self.assertIn("wire", message)
        # Neither wire-loaded resident was evicted.
        self.assertIn("wire_a", pool.uid_to_buffer_id)
        self.assertIn("wire_b", pool.uid_to_buffer_id)
        self.assertEqual(pool.buffer_id_to_uid, ["wire_a", "wire_b"])

    def test_pinned_still_excluded_unchanged(self):
        # Regression guard: the pre-existing `pinned` exclusion (unrelated to
        # this fix) must keep working.
        pool = _make_base_pool(max_adapters_per_batch=1)
        refs = {"pinned_one": _make_ref("pinned_one", pinned=True, reloadable=True)}
        pool.uid_to_buffer_id = {"pinned_one": 0}
        pool.buffer_id_to_uid = ["pinned_one"]
        with self.assertRaises(ValueError):
            pool._acquire_buffer_slot(cur_uids=set(), refs=refs)
        self.assertIn("pinned_one", pool.uid_to_buffer_id)

    def test_reloadable_unpinned_resident_is_still_evicted(self):
        # Sanity: a genuinely evictable (disk-backed, unpinned) resident is
        # still evicted normally -- the fix must not be over-broad.
        pool = _make_base_pool(max_adapters_per_batch=1)
        refs = {"disk_backed": _make_ref("disk_backed", pinned=False, reloadable=True)}
        pool.uid_to_buffer_id = {"disk_backed": 0}
        pool.buffer_id_to_uid = ["disk_backed"]
        pool.eviction_policy.mark_used("disk_backed")
        buffer_id = pool._acquire_buffer_slot(cur_uids=set(), refs=refs)
        self.assertEqual(buffer_id, 0)
        self.assertNotIn("disk_backed", pool.uid_to_buffer_id)
        self.assertEqual(pool.buffer_id_to_uid[0], EMPTY_SLOT)

    def test_empty_slot_used_first_no_eviction_needed(self):
        pool = _make_base_pool(max_adapters_per_batch=2)
        pool.uid_to_buffer_id = {"resident": 0}
        pool.buffer_id_to_uid = ["resident", EMPTY_SLOT]
        buffer_id = pool._acquire_buffer_slot(cur_uids=set(), refs={})
        self.assertEqual(buffer_id, 1)


class TestAcquireBufferSlotNeverEvictsBasePlaceholder(unittest.TestCase):
    """Critical fix (Task 4b review): AdapterMemPool._acquire_buffer_slot's
    candidate-building loop only DEPRIORITIZED uid=None (preferring to evict
    a real adapter over the base placeholder when both were candidates), it
    never EXCLUDED None outright -- unlike its sibling
    allocate_buffer_slot_with_eviction, which already has `if uid is None:
    continue` (oft/mem_pool.py). Whenever every OTHER resident real adapter
    was pinned, non-reloadable, or already referenced by the current batch
    (cur_uids), None became the only remaining candidate and got evicted --
    reachable at ANY capacity (not just a pool full of only the base slot),
    an ordinary multi-tenant admission scenario, not an edge case. Once a
    real adapter lands at buffer slot 0 (self.active_idx),
    OFTManager._compute_moe_multi_tenant_slot_ids's slot-index-based check
    (idx != 0) treats it identically to "no real adapter", so its per-token
    CUDA-graph persistent buffer is never refreshed for that batch -- while
    the host's own uid-based replay selector (_resolve_oft_variant) still
    selects the general "oft_multi" graph, replaying stale routing data:
    silently applying the wrong adapter's rotation. Exactly the failure mode
    the whole 2026-09-01-oft-moe-cuda-graph-dual-capture plan exists to
    prevent, made reachable in production by Task 4b's own relaxation of
    peft/config.py's decode-CUDA-graph guard for capacity >= 1 sibling
    MoE-target deployments."""

    def test_base_placeholder_never_evicted_when_all_other_residents_are_referenced(
        self,
    ):
        # Reviewer's repro: capacity 3 (max_adapters_per_batch=3), None@0,
        # A@1, D1@2 all resident; a batch referencing {A, D1, D2} needs a
        # slot for the new adapter D2. A and D1 are excluded as candidates
        # because they're in cur_uids -- before the fix, None was the only
        # remaining candidate and got evicted (D2 would silently land at
        # slot 0/active_idx). Must now raise instead.
        pool = _make_base_pool(max_adapters_per_batch=3)
        refs = {
            "A": _make_ref("A", pinned=False, reloadable=True),
            "D1": _make_ref("D1", pinned=False, reloadable=True),
        }
        pool.uid_to_buffer_id = {None: 0, "A": 1, "D1": 2}
        pool.buffer_id_to_uid = [None, "A", "D1"]
        with self.assertRaises(ValueError):
            pool._acquire_buffer_slot(cur_uids={"A", "D1", "D2"}, refs=refs)
        # The base/identity placeholder must still be resident at slot 0.
        self.assertIn(None, pool.uid_to_buffer_id)
        self.assertEqual(pool.buffer_id_to_uid[0], None)

    def test_real_adapter_still_evicted_normally_ahead_of_base_placeholder(self):
        # Sanity: the fix must not be over-broad -- a genuinely evictable
        # (unpinned, reloadable, not-in-cur_uids) real adapter is still
        # evicted normally even though None is nominally also present.
        pool = _make_base_pool(max_adapters_per_batch=2)
        refs = {"evict_me": _make_ref("evict_me", pinned=False, reloadable=True)}
        pool.uid_to_buffer_id = {None: 0, "evict_me": 1}
        pool.buffer_id_to_uid = [None, "evict_me"]
        pool.eviction_policy.mark_used("evict_me")
        buffer_id = pool._acquire_buffer_slot(cur_uids={"new_adapter"}, refs=refs)
        self.assertEqual(buffer_id, 1)
        self.assertNotIn("evict_me", pool.uid_to_buffer_id)
        self.assertIn(None, pool.uid_to_buffer_id)


class TestNumPinnedBookkeeping(unittest.TestCase):
    """I6 fix: register_streamed_adapter/unload_streamed_adapter must keep
    num_pinned in sync for pinned wire-loaded adapters -- previously
    register_streamed_adapter never touched it, silently under-counting
    num_pinned and making validate_new_adapter's anti-starvation guard and
    validate_batch's mem_pool_vacancy arithmetic (base/manager.py) wrong once
    pinned OFT adapters became reachable over the wire."""

    def test_register_increments_num_pinned_for_pinned_ref(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("pinned_wire", pinned=True, reloadable=False)
        result = mgr.register_streamed_adapter(ref, 0, CONFIG_DICT)
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(mgr.num_pinned, 1)

    def test_register_does_not_increment_for_unpinned_ref(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("unpinned_wire", pinned=False, reloadable=False)
        result = mgr.register_streamed_adapter(ref, 0, CONFIG_DICT)
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(mgr.num_pinned, 0)

    def test_unload_decrements_num_pinned_for_pinned_ref(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("pinned_wire", pinned=True, reloadable=False)
        mgr.register_streamed_adapter(ref, 0, CONFIG_DICT)
        self.assertEqual(mgr.num_pinned, 1)
        result = mgr.unload_streamed_adapter(ref)
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(mgr.num_pinned, 0)

    def test_double_unload_does_not_double_decrement(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("pinned_wire", pinned=True, reloadable=False)
        mgr.register_streamed_adapter(ref, 0, CONFIG_DICT)
        mgr.unload_streamed_adapter(ref)
        mgr.unload_streamed_adapter(ref)  # idempotent re-unload
        self.assertEqual(mgr.num_pinned, 0)

    def test_double_register_does_not_double_increment(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("pinned_wire", pinned=True, reloadable=False)
        mgr.register_streamed_adapter(ref, 0, CONFIG_DICT)
        mgr.register_streamed_adapter(ref, 0, CONFIG_DICT)
        self.assertEqual(mgr.num_pinned, 1)


class TestGracefulFailureInsteadOfAssert(unittest.TestCase):
    """C1 fix #3 (defense in depth): a registry/GPU-pool divergence (e.g. a
    dispatch for an adapter the GPU pool has already evicted) must produce a
    graceful failure, not a bare AssertionError that can escape uncaught and
    crash the engine."""

    def test_unload_adapter_missing_config_returns_graceful_failure(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("never_loaded")
        result = mgr.unload_adapter(ref)
        self.assertFalse(result.success)
        self.assertIn("not loaded", result.error_message)

    def test_unload_adapter_missing_ref_returns_graceful_failure(self):
        mgr = _make_manager(max_ofts_per_batch=2)
        ref = _make_ref("half_present")
        # configs present but refs missing -- an inconsistent state that
        # must still fail gracefully, not assert.
        mgr.configs[ref.adapter_id] = "cfg"
        result = mgr.unload_adapter(ref)
        self.assertFalse(result.success)
        self.assertIn("not loaded", result.error_message)

    def test_load_oft_weight_to_buffer_raises_value_error_for_missing_adapter(self):
        # A bound-but-unbacked stand-in, mirroring _make_pool's pattern: the
        # real (unbound) OFTMemoryPool method is bound via MethodType, so
        # this exercises the actual production code. The oft_adapter=None
        # check fires before any R_buffer access, so no buffer state is
        # needed for this branch.
        pool = SimpleNamespace()
        pool.load_oft_weight_to_buffer = MethodType(
            OFTMemoryPool.load_oft_weight_to_buffer, pool
        )
        with self.assertRaises(ValueError):
            pool.load_oft_weight_to_buffer("some_uid", 0, None, [], None, None)


class TestResolveCommitSplit(unittest.TestCase):
    """The resolve/commit split itself (streamed_weight_loader.py): resolve
    must never mutate a buffer (no buffer_id parameter at all), and commit
    performs the deferred writes."""

    def _fake_oft_manager(self, resolve_side_effect=None):
        memory_pool = MagicMock()
        memory_pool.tp_rank = 0
        memory_pool.R_buffer = {TARGET_MODULE: {0: MagicMock(device="cpu")}}
        if resolve_side_effect is not None:
            memory_pool._resolve_oft_tensor_plan.side_effect = resolve_side_effect
        else:
            memory_pool._resolve_oft_tensor_plan.return_value = (
                TARGET_MODULE,
                None,
                False,
                None,
                1,
            )
        oft_manager = MagicMock()
        oft_manager.memory_pool = memory_pool
        oft_manager.adapter_modules = []
        return oft_manager

    def test_unresolved_name_fails_without_mutation(self):
        oft_manager = self._fake_oft_manager(
            resolve_side_effect=KeyError("no such module")
        )
        named_tensors = [("model.layers.0.self_attn.q_proj.oft_R", "T0")]
        plan, error_message = _resolve_streamed_oft_tensor_groups(
            oft_manager, named_tensors, BLOCK_SIZE
        )
        self.assertIsNone(plan)
        self.assertIn("Unresolved OFT tensor names", error_message)
        oft_manager.memory_pool.load_oft_weight_direct.assert_not_called()

    def test_embed_tokens_write_deferred_from_resolve_to_commit(self):
        oft_manager = self._fake_oft_manager()
        named_tensors = [("model.embed_tokens.oft_R", "T0")]
        plan, error_message = _resolve_streamed_oft_tensor_groups(
            oft_manager, named_tensors, BLOCK_SIZE
        )
        self.assertIsNotNone(plan, error_message)
        oft_manager.memory_pool.load_oft_weight_direct.assert_not_called()

        _, _, _, direct_writes = plan
        self.assertEqual(direct_writes, named_tensors)

        with patch("sglang.srt.oft.streamed_weight_loader.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            success, msg = _commit_streamed_oft_tensor_groups(
                oft_manager, named_tensors, plan, 3, BLOCK_SIZE, "a1", "a1"
            )
        self.assertTrue(success, msg)
        oft_manager.memory_pool.load_oft_weight_direct.assert_called_once_with(
            3, "model.embed_tokens.oft_R", "T0", BLOCK_SIZE, [], 0
        )


if __name__ == "__main__":
    unittest.main()

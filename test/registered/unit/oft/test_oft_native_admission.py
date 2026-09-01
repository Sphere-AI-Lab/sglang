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
from sglang.srt.oft.base.mem_pool import EMPTY_SLOT
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

    def test_upsert_of_disk_backed_name_keeps_it_registered(self):
        """Same bug shape at the OTHER call site that can unload a resident
        ref by name: an upsert naming an already-loaded adapter must not
        fully unload it if it turns out to be disk-backed."""
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
        ):
            result = mgr.load_adapter_from_tensors(
                new_ref, [], CONFIG_DICT, upsert=True
            )

        self.assertTrue(result.success, result.error_message)
        self.assertIn(disk_ref.adapter_id, mgr.refs)
        self.assertIn(disk_ref.adapter_id, mgr.configs)
        self.assertIn(disk_ref.adapter_id, mgr.adapters)


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

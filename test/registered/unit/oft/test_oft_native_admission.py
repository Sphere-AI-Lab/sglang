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
from sglang.srt.oft.base.manager import AdapterManager
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
    pool.prepare_oft_batch = MethodType(OFTMemoryPool.prepare_oft_batch, pool)
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


class TestPrepareOftBatchAdmission(unittest.TestCase):
    """--peft-paths (on-disk adapter preload) has been retired, along with
    the lazy per-batch admission (_acquire_buffer_slot / load_oft_weight_
    to_buffer) it depended on -- OFTMemoryPool.prepare_oft_batch's per-uid
    loop must now do exactly two things for a uid not yet resident: (1) the
    base/identity placeholder (uid=None) gets its own one-time boot
    registration (unrelated to the retired disk path, and the only reason
    prepare_oft_batch still ever assigns a fresh slot at all); (2) any real
    uid must already be resident from the native-RPC path -- a real uid
    reaching here unresident means it was never loaded, so this must fail
    loudly rather than silently proceed with a stale/missing buffer_id."""

    def test_base_placeholder_gets_boot_registered_into_first_empty_slot(self):
        pool = _make_pool(max_ofts_per_batch=2)
        pool.prepare_oft_batch(
            cur_uids={None},
            oft_adapters={},
            oft_modules=[],
            oft_refs={},
            oft_embed_tokens_module=None,
            oft_lm_head_module=None,
        )
        self.assertEqual(pool.uid_to_buffer_id[None], 0)
        self.assertEqual(pool.buffer_id_to_uid[0], None)
        pool.reset_buffer_slot_to_identity.assert_called_once_with(0)

    def test_base_placeholder_already_resident_is_not_re_registered(self):
        # Boot registration only ever fires once -- a later call with None
        # already resident (the normal case for every subsequent batch) must
        # not re-fill or re-assign its slot.
        pool = _make_pool(max_ofts_per_batch=2)
        pool.uid_to_buffer_id = {None: 0}
        pool.buffer_id_to_uid = [None, EMPTY_SLOT]
        pool.prepare_oft_batch(
            cur_uids={None},
            oft_adapters={},
            oft_modules=[],
            oft_refs={},
            oft_embed_tokens_module=None,
            oft_lm_head_module=None,
        )
        pool.reset_buffer_slot_to_identity.assert_not_called()

    def test_unresident_real_adapter_raises_loudly(self):
        # No on-disk preload path exists anymore to lazily seat a real uid
        # -- a batch naming one that was never loaded via the native-RPC
        # path must fail loudly, not silently proceed with a missing
        # buffer_id.
        pool = _make_pool(max_ofts_per_batch=2)
        pool.uid_to_buffer_id = {None: 0}
        pool.buffer_id_to_uid = [None, EMPTY_SLOT]
        with self.assertRaises(ValueError) as ctx:
            pool.prepare_oft_batch(
                cur_uids={"never_loaded"},
                oft_adapters={},
                oft_modules=[],
                oft_refs={},
                oft_embed_tokens_module=None,
                oft_lm_head_module=None,
            )
        self.assertIn("never_loaded", str(ctx.exception))
        self.assertIn("never loaded", str(ctx.exception))


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


def _make_admission_manager(max_adapters_per_batch):
    """Stand-in wiring AdapterManager.validate_batch (oft/base/manager.py) to
    a REAL AdapterMemPool-shaped memory pool -- so the admission-layer
    capacity check can be exercised against realistic pool bookkeeping.
    Boot-registers uid=None at buffer slot 0, mirroring
    OFTManager.init_memory_pool's real `fetch_new_ofts({None})` call."""
    pool = SimpleNamespace(
        max_adapters_per_batch=max_adapters_per_batch,
        uid_to_buffer_id={None: 0},
        buffer_id_to_uid=[None] + [EMPTY_SLOT] * (max_adapters_per_batch - 1),
        eviction_policy=get_eviction_policy("lru"),
    )

    mgr = SimpleNamespace(
        max_adapters_per_batch=max_adapters_per_batch,
        num_pinned=0,
        refs={},
        memory_pool=pool,
    )
    mgr.validate_batch = MethodType(AdapterManager.validate_batch, mgr)
    return mgr


class TestValidateBatchEnforcesRealAdapterCapacity(unittest.TestCase):
    """Important finding (Task 4b re-review): AdapterManager.validate_batch
    (oft/base/manager.py) validated batch size against the FULL
    max_adapters_per_batch, not max_adapters_per_batch - 1 -- even though
    buffer slot 0 is now (Task 4b's Critical eviction fix) permanently
    reserved for the base/identity placeholder and genuinely unavailable to
    real adapters. A batch referencing max_adapters_per_batch distinct real
    (non-None) adapters passed validate_batch (which thought there was
    room) but then had no real slot to seat the last one into (no on-disk
    preload path exists anymore to lazily page it in from either) --
    reaching OFTMemoryPool.prepare_oft_batch's loud ValueError with no
    handler, SIGQUITing the whole engine. This bound is still fully
    load-bearing for the native-RPC path: every real adapter it admits
    occupies a fixed slot for as long as it's resident, so the same
    real-capacity arithmetic applies regardless of how adapters get loaded."""

    def test_reviewer_repro_two_distinct_real_adapters_at_capacity_two(self):
        # Reviewer's exact repro: max_adapters_per_batch=2 (1 real slot), a
        # batch referencing 2 distinct real, unpinned adapters with no
        # base/None request mixed in. Must now be rejected at admission.
        mgr = _make_admission_manager(max_adapters_per_batch=2)
        self.assertFalse(mgr.validate_batch({"A1", "A2"}))

    def test_single_real_adapter_at_capacity_two_is_admitted(self):
        # Sanity: the fix must not be over-broad -- a batch within the real
        # (max_adapters_per_batch - 1) bound is still admitted.
        mgr = _make_admission_manager(max_adapters_per_batch=2)
        self.assertTrue(mgr.validate_batch({"A1"}))

    def test_none_in_batch_does_not_count_toward_real_capacity(self):
        # A base/no-adapter request mixed into the batch must not itself
        # consume a "real adapter" slot in the capacity arithmetic.
        mgr = _make_admission_manager(max_adapters_per_batch=2)
        self.assertTrue(mgr.validate_batch({None, "A1"}))

    def test_capacity_one_rejects_any_real_adapter(self):
        # max_adapters_per_batch=1 has zero real slots (slot 0 is the only
        # slot and it's permanently the base/identity placeholder) -- any
        # batch referencing a real adapter must be rejected, while an
        # all-base batch is still fine.
        mgr = _make_admission_manager(max_adapters_per_batch=1)
        self.assertFalse(mgr.validate_batch({"A1"}))
        self.assertTrue(mgr.validate_batch({None}))

    def test_globally_pinned_adapter_outside_this_batch_still_consumes_real_capacity(
        self,
    ):
        # A pinned adapter need not be IN this batch to consume a real slot
        # -- mem_pool_vacancy must account for num_pinned globally against
        # the real (max_adapters_per_batch - 1) bound, not the full
        # max_adapters_per_batch. 4 slots total (3 real); 2 pinned globally
        # (only "pinned_A" referenced by this batch) leaves only 1 real slot
        # free for the batch's 2 unpinned adapters.
        mgr = _make_admission_manager(max_adapters_per_batch=4)
        mgr.num_pinned = 2
        mgr.refs = {
            "pinned_A": _make_ref("pinned_A", pinned=True),
            "unpinned_1": _make_ref("unpinned_1", pinned=False),
            "unpinned_2": _make_ref("unpinned_2", pinned=False),
        }
        self.assertFalse(
            mgr.validate_batch({"pinned_A", "unpinned_1", "unpinned_2"})
        )


class TestFetchNewAdaptersAssertsRealAdapterCapacity(unittest.TestCase):
    """fetch_new_adapters's assertion (oft/base/manager.py) had the same
    off-by-one-real-slot gap as validate_batch above -- it must reject
    before ever calling _prepare_mem_pool_batch, not merely raise later."""

    def _make_manager(self, max_adapters_per_batch):
        mgr = SimpleNamespace(
            max_adapters_per_batch=max_adapters_per_batch,
            _prepare_mem_pool_batch=MagicMock(),
        )
        mgr.fetch_new_adapters = MethodType(AdapterManager.fetch_new_adapters, mgr)
        return mgr

    def test_assertion_rejects_batch_exceeding_real_capacity(self):
        mgr = self._make_manager(max_adapters_per_batch=2)
        with self.assertRaises(AssertionError):
            mgr.fetch_new_adapters({"A1", "A2"})
        mgr._prepare_mem_pool_batch.assert_not_called()

    def test_assertion_allows_batch_within_real_capacity(self):
        mgr = self._make_manager(max_adapters_per_batch=2)
        mgr.fetch_new_adapters({"A1"})  # must not raise
        mgr._prepare_mem_pool_batch.assert_called_once_with({"A1"})

    def test_none_does_not_count_toward_the_assertion(self):
        mgr = self._make_manager(max_adapters_per_batch=2)
        mgr.fetch_new_adapters({None, "A1"})  # must not raise


class TestValidateNewAdapterPinnedBoundReservesRealSlotForUnpinned(unittest.TestCase):
    """Important finding (Task 4b re-review): OFTManager.validate_new_adapter's
    anti-starvation guard (oft_manager.py) allowed pinning up to
    max_ofts_per_batch - 1 adapters -- correct back when buffer slot 0 could
    still (via the pre-Critical-fix bug) serve either the base placeholder
    or a real adapter. Now that slot 0 is permanently reserved for the base
    placeholder (Task 4b's Critical eviction fix), only max_ofts_per_batch -
    1 real slots ever exist, so allowing max_ofts_per_batch - 1 of them to
    be pinned could claim EVERY real slot, leaving zero room for any
    unpinned adapter ever. The bound must reserve one more
    (max_ofts_per_batch - 2 pinned max)."""

    def _make_manager(self, max_ofts_per_batch, num_pinned=0):
        # No `memory_pool` attribute at all -- validate_new_adapter's
        # `getattr(self, "memory_pool", None)` then short-circuits its
        # block_size-compatibility branch, so this stays a narrow unit test
        # of the pinned-bound line rather than needing a full pool double.
        mgr = SimpleNamespace(refs={}, num_pinned=num_pinned, max_ofts_per_batch=max_ofts_per_batch)
        mgr.validate_new_adapter = MethodType(OFTManager.validate_new_adapter, mgr)
        return mgr

    def _config(self):
        # target_modules as a set (not a list) skips
        # validate_model_oft_target_modules, which needs a real base_model.
        return SimpleNamespace(oft_added_tokens_size=0, target_modules={"q_proj"})

    def test_pinning_the_last_free_real_slot_is_rejected(self):
        # max_ofts_per_batch=3 -> 2 real slots. One is already pinned
        # (num_pinned=1); pinning a second would claim BOTH real slots,
        # leaving zero for any unpinned adapter. Must now raise (pre-fix,
        # the old bound of max_ofts_per_batch - 1 == 2 would have allowed
        # this: 1 >= 2 is False).
        mgr = self._make_manager(max_ofts_per_batch=3, num_pinned=1)
        with self.assertRaises(ValueError):
            mgr.validate_new_adapter(
                self._config(), _make_ref("second_pin", pinned=True)
            )

    def test_pinning_a_single_real_slot_out_of_two_still_leaves_room(self):
        # Sanity: the fix must not be over-broad -- pinning the FIRST of 2
        # real slots (num_pinned=0) still leaves 1 real slot for unpinned
        # use, so it must still be admitted.
        mgr = self._make_manager(max_ofts_per_batch=3, num_pinned=0)
        mgr.validate_new_adapter(
            self._config(), _make_ref("first_pin", pinned=True)
        )  # must not raise

    def test_capacity_two_rejects_pinning_the_only_real_slot(self):
        # max_ofts_per_batch=2 -> 1 real slot total. Pinning it would leave
        # zero room for any unpinned adapter ever -- must be rejected even
        # as the very first pinned adapter.
        mgr = self._make_manager(max_ofts_per_batch=2, num_pinned=0)
        with self.assertRaises(ValueError):
            mgr.validate_new_adapter(
                self._config(), _make_ref("only_pin", pinned=True)
            )

    def test_unpinned_adapter_never_trips_the_pinned_bound(self):
        # Sanity: the bound only gates pinned refs.
        mgr = self._make_manager(max_ofts_per_batch=2, num_pinned=0)
        mgr.validate_new_adapter(
            self._config(), _make_ref("unpinned_one", pinned=False)
        )  # must not raise


if __name__ == "__main__":
    unittest.main()

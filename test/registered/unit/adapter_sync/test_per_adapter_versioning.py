"""Per-adapter staging/activation.

The pool used to hold ONE staged version and ONE active version, and activate()
blanket-copied every buffer group into the fixed active slot. With more than one
adapter resident that is unusable: two adapters cannot sit at different versions,
and activating one would overwrite the other. These tests pin the new contract.
"""

import unittest

import torch

from sglang.srt.adapter_sync.mem_pool import AdapterMemPool
from sglang.srt.adapter_sync.versioning import VersionedStaging


class _Pool(AdapterMemPool):
    """Minimal concrete pool: three slots, one buffer group, CPU tensors."""

    def __init__(self):
        super().__init__(
            max_adapters_per_batch=3,
            dtype=torch.float32,
            tp_size=1,
            tp_rank=0,
            eviction_policy="lru",
        )
        self.active_idx, self.staging_idx = 0, 2
        self.register_buffer_group("W", {"w": (4,)}, device="cpu")
        # two resident adapters in slots 0 and 1
        for uid, slot in (("A", 0), ("B", 1)):
            self.uid_to_buffer_id[uid] = slot
            self.buffer_id_to_uid[slot] = uid

    def _fill_slot(self, slot_idx, named_tensors):
        for _, t in named_tensors:
            self.slot("W", "w", slot_idx).copy_(t)

    def val(self, uid):
        return self.slot("W", "w", self.uid_to_buffer_id[uid])[0].item()


def _tensors(v):
    return [("w", torch.full((4,), float(v)))]


class TestPerAdapterVersioning(unittest.TestCase):
    def setUp(self):
        self.p = _Pool()
        for uid in ("A", "B"):
            self.p.slot("W", "w", self.p.uid_to_buffer_id[uid]).zero_()

    def test_activate_touches_only_the_target_adapter(self):
        """The property the old blanket copy could not provide."""
        self.p.stage(7, _tensors(7), uid="A")
        self.p.activate(7, uid="A")
        self.assertEqual(self.p.val("A"), 7.0)
        self.assertEqual(self.p.val("B"), 0.0)  # untouched

    def test_adapters_hold_independent_versions(self):
        self.p.stage(7, _tensors(7), uid="A")
        self.p.activate(7, uid="A")
        self.p.stage(9, _tensors(9), uid="B")
        self.p.activate(9, uid="B")
        self.assertEqual((self.p.active_version("A"), self.p.active_version("B")), (7, 9))
        self.assertEqual((self.p.val("A"), self.p.val("B")), (7.0, 9.0))

    def test_activate_rejects_a_different_adapter_than_was_staged(self):
        """Promoting A's weights into B's slot would silently corrupt B."""
        self.p.stage(7, _tensors(7), uid="A")
        with self.assertRaises(RuntimeError) as cm:
            self.p.activate(7, uid="B")
        self.assertIn("staged_adapter_mismatch", str(cm.exception))
        self.assertEqual(self.p.val("B"), 0.0)

    def test_activate_still_rejects_a_version_mismatch(self):
        self.p.stage(7, _tensors(7), uid="A")
        with self.assertRaises(RuntimeError) as cm:
            self.p.activate(8, uid="A")
        self.assertIn("inactive_slot_busy", str(cm.exception))

    def test_staging_slot_is_released_after_activate(self):
        self.p.stage(7, _tensors(7), uid="A")
        self.p.activate(7, uid="A")
        with self.assertRaises(RuntimeError):
            self.p.activate(7, uid="A")  # nothing staged any more

    def test_single_active_path_is_unchanged(self):
        """uid=None keeps the old behaviour, so migrating srt/oft is a no-op."""
        self.p.stage(3, _tensors(3), uid=None)
        self.p.activate(3, uid=None)
        self.assertEqual(self.p.slot("W", "w", self.p.active_idx)[0].item(), 3.0)
        self.assertEqual(self.p._active_version, 3)  # compatibility shim


class _ForeignLayoutPool(VersionedStaging):
    """A pool whose weights are NOT in adapter_sync buffer groups.

    Stands in for upstream's LoRAMemoryPool, which keeps A_buffer/B_buffer dicts
    and cannot be reorganised (srt/lora is never edited). If the state machine
    works here, it works there.
    """

    def __init__(self):
        self.active_idx, self.staging_idx = 0, 2
        self.A = [torch.zeros(4) for _ in range(3)]   # one tensor per slot
        self.B = [torch.zeros(4) for _ in range(3)]
        self.uid_to_buffer_id = {"A": 0, "B": 1}
        self._init_versioning()

    def get_buffer_id(self, uid):
        return self.uid_to_buffer_id[uid]

    def _fill_slot(self, slot_idx, named_tensors):
        for name, t in named_tensors:
            (self.A if name == "a" else self.B)[slot_idx].copy_(t)

    def _copy_slot(self, src, dst):
        self.A[dst].copy_(self.A[src])
        self.B[dst].copy_(self.B[src])


class TestStateMachineIsLayoutAgnostic(unittest.TestCase):
    def test_works_over_a_foreign_buffer_layout(self):
        p = _ForeignLayoutPool()
        p.stage(5, [("a", torch.full((4,), 5.0)), ("b", torch.full((4,), 50.0))], uid="A")
        p.activate(5, uid="A")
        self.assertEqual(p.A[0][0].item(), 5.0)
        self.assertEqual(p.B[0][0].item(), 50.0)
        self.assertEqual(p.A[1][0].item(), 0.0)      # the other adapter untouched
        self.assertEqual(p.active_version("A"), 5)

    def test_mismatch_guard_holds_for_foreign_layouts_too(self):
        p = _ForeignLayoutPool()
        p.stage(5, [("a", torch.full((4,), 5.0))], uid="A")
        with self.assertRaises(RuntimeError):
            p.activate(5, uid="B")
        self.assertEqual(p.A[1][0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Staged LoRA updates over UPSTREAM's memory pool.

These exercise the two VersionedStaging primitives and the slot-reservation
trick, on instances built without upstream's full init (which needs a real base
model). They therefore cover this backend's own logic, NOT its integration with
a live LoRAMemoryPool -- that is what the GPU gate is for.
"""

import unittest

import torch

from sglang.srt.adapter_sync.backends.lora import (
    StagedLoRAManager,
    StagedLoRAMemoryPool,
)


def _pool(n_slots=2, n_layers=2, rank=4, hidden=8):
    """A StagedLoRAMemoryPool with buffers filled in by hand, N+1 slots wide."""
    p = object.__new__(StagedLoRAMemoryPool)
    p.max_loras_per_batch = n_slots
    p.active_idx, p.staging_idx = 0, n_slots        # staging is the hidden extra
    p._init_versioning()
    wide = n_slots + 1
    p.A_buffer = {"q_proj": [torch.zeros(wide, rank, hidden) for _ in range(n_layers)]}
    p.B_buffer = {"q_proj": [torch.zeros(wide, hidden, rank) for _ in range(n_layers)]}
    p.embedding_A_buffer, p.embedding_B_buffer = {}, {}
    p.lm_head_A_buffer, p.lm_head_B_buffer = {}, {}
    p.new_embeddings_buffer = {}
    p.uid_to_buffer_id = {"adapterA": 0, "adapterB": 1}
    return p


class TestSlotReservation(unittest.TestCase):
    def test_staging_slot_sits_outside_the_advertised_capacity(self):
        """Upstream picks serving slots in a nested closure scanning
        range(max_loras_per_batch); the staging slot must be beyond it."""
        p = _pool(n_slots=2)
        self.assertEqual(p.staging_idx, 2)
        self.assertGreaterEqual(p.staging_idx, p.max_loras_per_batch)
        self.assertEqual(p.available_serving_slots(), 2)   # capacity NOT reduced

    def test_buffers_are_one_slot_wider_than_advertised(self):
        p = _pool(n_slots=2)
        self.assertEqual(p.A_buffer["q_proj"][0].shape[0], 3)


class TestPlacementIsDelegatedUpstream(unittest.TestCase):
    """The backend must NOT interpret weight names or buffer layouts itself.

    The first version did, and silently skipped every q/k/v tensor: upstream
    FUSES q/k/v into one qkv_proj buffer (stacked x3) and gate/up into
    gate_up_proj (x2). Placement is now delegated to upstream's own
    load_lora_weight_to_buffer, the same routine a disk load uses, so the two
    cannot drift apart.
    """

    def test_fill_slot_calls_upstream_placement_with_the_staging_slot(self):
        p = _pool()
        seen = {}
        p.load_lora_weight_to_buffer = lambda **kw: seen.update(kw)
        sentinel = ("ADAPTER", "MODULES", "EMBED", "LMHEAD")
        p._fill_slot(p.staging_idx, sentinel)
        self.assertEqual(seen["buffer_id"], p.staging_idx)
        self.assertEqual(seen["lora_adapter"], "ADAPTER")
        self.assertEqual(seen["lora_modules"], "MODULES")

    def test_backend_no_longer_parses_weight_names(self):
        from sglang.srt.adapter_sync.backends import lora as backend
        self.assertFalse(
            hasattr(backend.StagedLoRAManager, "_resolve_named_tensors"),
            "name-driven placement was the bug; it must not come back",
        )


class TestEndToEndStaging(unittest.TestCase):
    def test_activate_promotes_into_the_named_adapters_slot_only(self):
        p = _pool()
        # stand in for upstream placement: write a marker into the target slot
        p.load_lora_weight_to_buffer = (
            lambda **kw: p.A_buffer["q_proj"][0][kw["buffer_id"]].fill_(6.0)
        )
        p.stage(11, ("ADAPTER", None, None, None), uid="adapterB")
        p.activate(11, uid="adapterB")
        self.assertEqual(p.A_buffer["q_proj"][0][1][0, 0].item(), 6.0)   # adapterB
        self.assertEqual(p.A_buffer["q_proj"][0][0][0, 0].item(), 0.0)   # adapterA untouched
        self.assertEqual(p.active_version("adapterB"), 11)
        self.assertIsNone(p.active_version("adapterA"))


if __name__ == "__main__":
    unittest.main()

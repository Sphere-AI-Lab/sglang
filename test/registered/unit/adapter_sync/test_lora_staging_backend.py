"""Staged LoRA updates over UPSTREAM's memory pool.

These exercise the two VersionedStaging primitives and the slot-reservation
trick, on instances built without upstream's full init (which needs a real base
model). They therefore cover this backend's own logic, NOT its integration with
a live LoRAMemoryPool -- that is what the GPU gate is for.
"""

import unittest

import torch

from sglang.srt.adapter_sync.backends.lora import StagedLoRAMemoryPool


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


class TestPrimitives(unittest.TestCase):
    def test_fill_slot_writes_only_the_named_buffer_and_slot(self):
        p = _pool()
        src = torch.full((2, 8), 7.0)                      # rank 2 into a rank-4 buffer
        p._fill_slot(p.staging_idx, [("q_proj", 0, "A", src)])
        staged = p.A_buffer["q_proj"][0][p.staging_idx]
        self.assertEqual(staged[0, 0].item(), 7.0)
        self.assertEqual(staged[2, 0].item(), 0.0)          # rank tail zeroed
        self.assertEqual(p.A_buffer["q_proj"][1][p.staging_idx][0, 0].item(), 0.0)  # other layer
        self.assertEqual(p.A_buffer["q_proj"][0][0][0, 0].item(), 0.0)              # other slot

    def test_fill_slot_zeroes_a_previous_occupant(self):
        """A shared staging slot must not leak the last adapter's weights."""
        p = _pool()
        p._fill_slot(p.staging_idx, [("q_proj", 0, "A", torch.full((4, 8), 9.0))])
        p._fill_slot(p.staging_idx, [("q_proj", 0, "A", torch.full((2, 8), 1.0))])
        staged = p.A_buffer["q_proj"][0][p.staging_idx]
        self.assertEqual(staged[0, 0].item(), 1.0)
        self.assertEqual(staged[3, 0].item(), 0.0)          # the 9.0s are gone

    def test_fill_slot_ignores_an_unknown_buffer_name(self):
        p = _pool()
        p._fill_slot(p.staging_idx, [("not_a_module", 0, "A", torch.ones(2, 8))])  # no raise

    def test_copy_slot_moves_every_buffer_family(self):
        p = _pool()
        for layer in (0, 1):
            p.A_buffer["q_proj"][layer][p.staging_idx].fill_(3.0)
            p.B_buffer["q_proj"][layer][p.staging_idx].fill_(4.0)
        p._copy_slot(p.staging_idx, 1)
        for layer in (0, 1):
            self.assertEqual(p.A_buffer["q_proj"][layer][1][0, 0].item(), 3.0)
            self.assertEqual(p.B_buffer["q_proj"][layer][1][0, 0].item(), 4.0)
            self.assertEqual(p.A_buffer["q_proj"][layer][0][0, 0].item(), 0.0)  # slot 0 untouched


class TestEndToEndStaging(unittest.TestCase):
    def test_activate_promotes_into_the_named_adapters_slot_only(self):
        p = _pool()
        p.stage(11, [("q_proj", 0, "A", torch.full((4, 8), 6.0))], uid="adapterB")
        p.activate(11, uid="adapterB")
        self.assertEqual(p.A_buffer["q_proj"][0][1][0, 0].item(), 6.0)   # adapterB
        self.assertEqual(p.A_buffer["q_proj"][0][0][0, 0].item(), 0.0)   # adapterA untouched
        self.assertEqual(p.active_version("adapterB"), 11)
        self.assertIsNone(p.active_version("adapterA"))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the Marlin MoE OFT runner's multi-slot (request-segmented)
rotation selection (srt/oft/oft_moe_runner_marlin.py).

Covers the two functions with real new logic added for multi-tenant Marlin
support: _select_gate_up_oft_r_all_slots (dispatch/rejection) and
_shared_r_by_slot (the per-slot shared-R invariant, generalized from a single
global invariant to one checked independently per adapter slot). Pure-Python
logic on CPU tensors -- no GPU/Marlin kernel involved.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.oft.oft_moe_runner_marlin import (
    _select_gate_up_oft_r_all_slots,
    _shared_r_by_slot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSelectGateUpOftRAllSlots(unittest.TestCase):
    def test_returns_none_when_no_multi_slot_buffer_bound(self):
        # Disk-boot (non-pool-backed) adapter: the _all_slots attrs are never
        # set, so the caller must fall back to the single-shared-R path.
        layer = SimpleNamespace()
        self.assertIsNone(_select_gate_up_oft_r_all_slots(layer))

    def test_returns_w13_all_slots_view_when_bound(self):
        w13 = torch.zeros(2, 4, 1, 8, 8)
        layer = SimpleNamespace(_oft_w13_oft_r_all_slots=w13)
        self.assertIs(_select_gate_up_oft_r_all_slots(layer), w13)

    def test_raises_when_split_w1_all_slots_present(self):
        layer = SimpleNamespace(
            _oft_w13_oft_r_all_slots=None,
            _oft_w1_oft_r_all_slots=torch.zeros(2, 4, 1, 8, 8),
        )
        with self.assertRaisesRegex(RuntimeError, "canonical OFT split-expert"):
            _select_gate_up_oft_r_all_slots(layer)

    def test_raises_when_split_w3_all_slots_present(self):
        layer = SimpleNamespace(
            _oft_w13_oft_r_all_slots=None,
            _oft_w3_oft_r_all_slots=torch.zeros(2, 4, 1, 8, 8),
        )
        with self.assertRaisesRegex(RuntimeError, "canonical OFT split-expert"):
            _select_gate_up_oft_r_all_slots(layer)


class TestSharedRBySlot(unittest.TestCase):
    def _make_oft_r(self, num_slots, num_experts, block_size=8):
        """(slot, expert, block=1, bs, bs), each slot's own R shared
        identically across its own experts but DIFFERENT across slots."""
        oft_r = torch.zeros(num_slots, num_experts, 1, block_size, block_size)
        for slot in range(num_slots):
            r = torch.eye(block_size) * (slot + 1)
            oft_r[slot, :] = r
        return oft_r

    def test_different_slots_may_hold_different_r(self):
        # This is the actual new behavior vs. the old single-slot _shared_r:
        # per-slot invariant, not a single global invariant across all slots.
        oft_r = self._make_oft_r(num_slots=3, num_experts=4)
        result = _shared_r_by_slot(oft_r, "test")
        self.assertEqual(tuple(result.shape), (3, 1, 8, 8))
        for slot in range(3):
            expected = torch.eye(8) * (slot + 1)
            torch.testing.assert_close(result[slot, 0], expected)

    def test_single_expert_per_slot_skips_invariant_check(self):
        oft_r = self._make_oft_r(num_slots=2, num_experts=1)
        result = _shared_r_by_slot(oft_r, "test")
        self.assertEqual(tuple(result.shape), (2, 1, 8, 8))

    def test_raises_when_a_slot_disagrees_across_its_own_experts(self):
        oft_r = self._make_oft_r(num_slots=2, num_experts=4)
        oft_r[0, 2] = torch.eye(8) * 99  # slot 0's expert 2 now differs
        with self.assertRaisesRegex(RuntimeError, "differs across experts"):
            _shared_r_by_slot(oft_r, "test")


if __name__ == "__main__":
    unittest.main()

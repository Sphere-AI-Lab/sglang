"""Unit tests for OFTManager's per-batch MoE multi-tenancy decision.

No GPU required: constructs weight_indices directly and calls the method
under test via a lightweight OFTManager stand-in, following this codebase's
established pattern of binding the real production method onto a minimal
double (see test_oft_native_admission.py) rather than reimplementing the
logic.
"""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.oft.oft_manager import OFTManager


class TestMoeMultiTenancyDecision(unittest.TestCase):
    def _make_tm(self):
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    def test_single_resident_adapter_yields_none(self):
        tm = self._make_tm()
        # All tokens map to the same real slot (2) or the base slot (0).
        weight_indices = [2, 2, 0, 2, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNone(result)

    def test_no_resident_adapter_yields_none(self):
        tm = self._make_tm()
        weight_indices = [0, 0, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNone(result)

    def test_two_resident_adapters_yields_tensor(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 0, 1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertTrue(
            torch.equal(result, torch.tensor([1, 2, 0, 1, 2], dtype=torch.long))
        )

    def test_three_resident_adapters_yields_tensor(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 3]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNotNone(result)


class TestPushSlotIdsOntoMoeModules(unittest.TestCase):
    def _make_tm_with_one_moe_module(self):
        moe = SimpleNamespace()
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        tm._find_fused_moe_modules = lambda: {0: moe}
        tm._push_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._push_moe_multi_tenant_slot_ids, tm
        )
        return tm, moe

    def test_pushes_none_when_no_multi_tenancy(self):
        tm, moe = self._make_tm_with_one_moe_module()
        tm._moe_multi_tenant_slot_ids = None
        tm._push_moe_multi_tenant_slot_ids()
        self.assertIsNone(moe._oft_moe_multi_tenant_slot_ids)

    def test_pushes_tensor_when_multi_tenant(self):
        tm, moe = self._make_tm_with_one_moe_module()
        slot_ids = torch.tensor([1, 2, 1], dtype=torch.long)
        tm._moe_multi_tenant_slot_ids = slot_ids
        tm._push_moe_multi_tenant_slot_ids()
        self.assertIs(moe._oft_moe_multi_tenant_slot_ids, slot_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)

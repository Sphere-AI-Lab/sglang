"""The single-adapter fast path must survive --oft-double-buffer: the
staging slot is reserved but never addressable by a request."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.oft.backend.triton_backend import TritonOFTBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSingleAdapterModeCountsOnlyAddressableSlots(unittest.TestCase):
    def _backend(self, max_ofts_per_batch, double_buffer):
        return TritonOFTBackend(
            max_ofts_per_batch=max_ofts_per_batch,
            device=torch.device("cpu"),
            server_args=SimpleNamespace(oft_double_buffer=double_buffer),
        )

    def test_sync_run_two_slots_keeps_fast_path(self):
        self.assertTrue(self._backend(2, double_buffer=False).single_adapter_mode)

    def test_double_buffer_three_slots_keeps_fast_path(self):
        # base + active + hidden staging: still one addressable adapter.
        self.assertTrue(self._backend(3, double_buffer=True).single_adapter_mode)

    def test_three_addressable_slots_is_multi_tenant(self):
        self.assertFalse(self._backend(3, double_buffer=False).single_adapter_mode)
        self.assertFalse(self._backend(4, double_buffer=True).single_adapter_mode)

    def test_backend_without_server_args_falls_back_to_slot_count(self):
        backend = TritonOFTBackend(max_ofts_per_batch=3, device=torch.device("cpu"))
        self.assertFalse(backend.single_adapter_mode)

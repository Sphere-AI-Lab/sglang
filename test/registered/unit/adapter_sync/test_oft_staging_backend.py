import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


TARGET_MODULE = "q_proj"
BLOCK_SIZE = 4


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
    fill_value so slot-isolation assertions can tell slots apart."""
    r = torch.full((BLOCK_SIZE, BLOCK_SIZE), fill_value, dtype=torch.float32)
    return {(TARGET_MODULE, 0): (r, BLOCK_SIZE, None, 1)}


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


if __name__ == "__main__":
    unittest.main()

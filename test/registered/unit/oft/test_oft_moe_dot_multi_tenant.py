"""GPU correctness tests for the per-slot-aligned tl.dot multi-tenant MoE OFT
rotation path (srt/oft/oft_moe_runners.py's _should_use_dot_multi_tenant /
_compute_dot_alignment / _oft_prerotate_multi_tenant_dot).

The key new risk this path adds over the existing shared-alignment
elementwise kernel: alignment is derived from OFTBatchInfo.seg_indptr, which
counts ORIGINAL (unexpanded) tokens per request, but the down-GEMM's A is
already expanded router_topk-fold into adjacent rows per token. Getting the
row_group_factor scaling wrong would silently rotate rows under the wrong
adapter's R. This is verified here the same way test_oft_moe_rotation_multi_slot
verifies the elementwise kernel: against the manually-expanded-slot_ids
elementwise path as the ground truth oracle.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.oft.oft_moe_runners import (
    _oft_prerotate_multi_tenant,
    _oft_prerotate_multi_tenant_dot,
    _should_use_dot_multi_tenant,
)
from sglang.srt.oft.utils import OFTBatchInfo
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-test-1-gpu-small")


class TestShouldUseDotMultiTenant(unittest.TestCase):
    def test_none_batch_info_falls_back(self):
        self.assertFalse(_should_use_dot_multi_tenant(None, max_ofts_per_batch=4))

    def test_cuda_graph_batch_falls_back_even_with_small_capacity(self):
        batch_info = SimpleNamespace(use_cuda_graph=True)
        self.assertFalse(_should_use_dot_multi_tenant(batch_info, max_ofts_per_batch=4))

    def test_large_configured_capacity_falls_back(self):
        batch_info = SimpleNamespace(use_cuda_graph=False)
        self.assertFalse(_should_use_dot_multi_tenant(batch_info, max_ofts_per_batch=64))

    def test_small_capacity_eager_batch_uses_dot_path(self):
        batch_info = SimpleNamespace(use_cuda_graph=False)
        self.assertTrue(_should_use_dot_multi_tenant(batch_info, max_ofts_per_batch=4))


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class TestDotMultiTenantMatchesElementwiseForDownGemm(unittest.TestCase):
    """The down-GEMM case: A/topk_ids already expanded router_topk-fold,
    top_k=1 for the rotation call, row_group_factor=router_topk for the dot
    path vs. slot_ids.repeat_interleave(router_topk) for the elementwise
    path -- these must produce identical rotations."""

    def test_row_group_factor_matches_manual_slot_ids_expansion(self):
        torch.manual_seed(0)
        device = "cuda"
        num_tokens, router_topk, hidden, num_experts, bs = 8, 3, 256, 8, 32
        num_blocks = hidden // bs
        max_ofts_per_batch = 4

        # 2 requests of 4 tokens each, on slots 1 and 2.
        seg_indptr = torch.tensor([0, 4, 8], device=device, dtype=torch.int32)
        weight_indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
        slot_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
        slot_ids[:4] = 1
        slot_ids[4:] = 2

        batch_info = OFTBatchInfo(
            use_cuda_graph=False,
            bs=2,
            num_segments=2,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            oft_block_sizes=torch.zeros(max_ofts_per_batch, dtype=torch.int32, device=device),
            max_len=None,
            seg_lens=None,
            permutation=None,
        )

        oft_r_all_slots = torch.randn(
            max_ofts_per_batch, num_experts, num_blocks, bs, bs,
            device=device, dtype=torch.bfloat16,
        )

        # A/topk_ids already expanded router_topk-fold (down-GEMM row layout).
        M_expanded = num_tokens * router_topk
        A = torch.randn(M_expanded, hidden, device=device, dtype=torch.bfloat16)
        topk_ids_down = torch.randint(
            0, num_experts, (M_expanded, 1), device=device, dtype=torch.int32
        )
        C = torch.empty(M_expanded, hidden, device=device, dtype=torch.bfloat16)
        topk_weights = torch.ones(M_expanded, device=device, dtype=torch.float32)
        sti0, eid0, ntpp0 = moe_align_block_size(topk_ids_down, 64, num_experts)

        # --- ground truth: elementwise path, slot_ids pre-expanded by hand ---
        expanded_slot_ids = slot_ids.repeat_interleave(router_topk)
        a_gt, *_ = _oft_prerotate_multi_tenant(
            A, oft_r_all_slots, expanded_slot_ids, C, topk_weights, topk_ids_down,
            sti0, eid0, ntpp0, 1, num_experts, 64,
        )

        # --- dot path: row_group_factor=router_topk scales seg_indptr instead ---
        a_dot, *_ = _oft_prerotate_multi_tenant_dot(
            A, oft_r_all_slots, batch_info, max_ofts_per_batch, C, topk_weights,
            topk_ids_down, sti0, eid0, ntpp0, 1, num_experts, 64,
            row_group_factor=router_topk,
        )

        torch.testing.assert_close(a_gt.float(), a_dot.float(), atol=1e-2, rtol=1e-2)

    def test_wrong_row_group_factor_would_have_mismatched(self):
        """Negative control: confirms the oracle comparison above is actually
        sharp -- using row_group_factor=1 (the gate-up default) for this
        down-GEMM-shaped input must NOT match, or the positive test above
        would pass for the wrong reason."""
        torch.manual_seed(0)
        device = "cuda"
        num_tokens, router_topk, hidden, num_experts, bs = 8, 3, 256, 8, 32
        num_blocks = hidden // bs
        max_ofts_per_batch = 4

        seg_indptr = torch.tensor([0, 4, 8], device=device, dtype=torch.int32)
        weight_indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
        slot_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
        slot_ids[:4] = 1
        slot_ids[4:] = 2

        batch_info = OFTBatchInfo(
            use_cuda_graph=False,
            bs=2,
            num_segments=2,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            oft_block_sizes=torch.zeros(max_ofts_per_batch, dtype=torch.int32, device=device),
            max_len=None,
            seg_lens=None,
            permutation=None,
        )
        oft_r_all_slots = torch.randn(
            max_ofts_per_batch, num_experts, num_blocks, bs, bs,
            device=device, dtype=torch.bfloat16,
        )
        M_expanded = num_tokens * router_topk
        A = torch.randn(M_expanded, hidden, device=device, dtype=torch.bfloat16)
        topk_ids_down = torch.randint(
            0, num_experts, (M_expanded, 1), device=device, dtype=torch.int32
        )
        C = torch.empty(M_expanded, hidden, device=device, dtype=torch.bfloat16)
        topk_weights = torch.ones(M_expanded, device=device, dtype=torch.float32)
        sti0, eid0, ntpp0 = moe_align_block_size(topk_ids_down, 64, num_experts)

        expanded_slot_ids = slot_ids.repeat_interleave(router_topk)
        a_gt, *_ = _oft_prerotate_multi_tenant(
            A, oft_r_all_slots, expanded_slot_ids, C, topk_weights, topk_ids_down,
            sti0, eid0, ntpp0, 1, num_experts, 64,
        )
        a_wrong, *_ = _oft_prerotate_multi_tenant_dot(
            A, oft_r_all_slots, batch_info, max_ofts_per_batch, C, topk_weights,
            topk_ids_down, sti0, eid0, ntpp0, 1, num_experts, 64,
            row_group_factor=1,
        )
        self.assertGreater((a_gt.float() - a_wrong.float()).abs().max().item(), 0.5)


if __name__ == "__main__":
    unittest.main()

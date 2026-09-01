"""GPU correctness oracle for the multi-slot MoE OFT rotation kernel.

Proves per-token slot selection is correct by comparing the multi-slot
kernel's output, split by each token's own slot, against running the
EXISTING single-slot kernel with only that one slot's R matrix -- the
strongest available correctness oracle, since it reuses already-trusted
kernel code as the ground truth rather than a hand-rolled reference.

The oracle is sharp on slot selection: if a row picked the wrong slot's R
(off-by-one in the slot gather, a dropped slot stride, or the slot term
missing altogether), its values would match some OTHER slot's single-slot
run, so the comparison for its own slot fails.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.oft.triton_ops import (
    apply_oft_rotation_triton,
    apply_oft_rotation_triton_multi_slot,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class TestMultiSlotRotationMatchesSingleSlotPerAdapter(unittest.TestCase):
    def _assert_matches_isolated_single_slot_runs(
        self,
        *,
        num_tokens,
        hidden,
        num_experts,
        top_k,
        bs,
        num_slots,
        block_m=64,
        slot_ids=None,
    ):
        """Run the multi-slot kernel once, then, for every slot present, run
        the single-slot kernel with just that slot's R and require the rows
        belonging to that slot to agree."""
        torch.manual_seed(0)
        device = "cuda"
        num_blocks = hidden // bs

        A = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_m, num_experts
        )
        oft_r_all_slots = torch.randn(
            num_slots,
            num_experts,
            num_blocks,
            bs,
            bs,
            device=device,
            dtype=torch.bfloat16,
        )
        if slot_ids is None:
            slot_ids = torch.randint(
                0, num_slots, (num_tokens,), device=device, dtype=torch.long
            )

        multi_out = apply_oft_rotation_triton_multi_slot(
            A,
            oft_r_all_slots,
            slot_ids,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_m=block_m,
        )
        self.assertEqual(tuple(multi_out.shape), (num_tokens * top_k, hidden))

        # Output row i is the (token i // top_k, top-k position i % top_k)
        # pair, so a per-token slot maps to top_k consecutive output rows.
        row_slots = slot_ids.repeat_interleave(top_k)
        slots_seen = 0
        for slot in range(num_slots):
            row_mask = row_slots == slot
            if not bool(row_mask.any()):
                continue
            slots_seen += 1
            single_out = apply_oft_rotation_triton(
                A,
                oft_r_all_slots[slot],
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                block_m=block_m,
            )
            torch.testing.assert_close(
                multi_out[row_mask],
                single_out[row_mask],
                atol=1e-2,
                rtol=1e-2,
                msg=f"rows on slot {slot} disagree with an isolated slot-{slot} run",
            )
        self.assertGreaterEqual(
            slots_seen, 2, "config must exercise at least two distinct slots"
        )

    def test_two_adapters_each_match_isolated_single_slot_run(self):
        # Half the tokens use slot 1, half use slot 2.
        self._assert_matches_isolated_single_slot_runs(
            num_tokens=8,
            hidden=32,
            num_experts=4,
            top_k=1,
            bs=16,
            num_slots=3,
            slot_ids=torch.tensor(
                [1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long, device="cuda"
            ),
        )

    def test_base_slot_zero_tokens_mixed_with_adapter_tokens(self):
        # Slot 0 is the identity/base slot in production; it must be selected
        # like any other slot rather than special-cased or skipped.
        self._assert_matches_isolated_single_slot_runs(
            num_tokens=12,
            hidden=64,
            num_experts=4,
            top_k=1,
            bs=16,
            num_slots=3,
            slot_ids=torch.tensor(
                [0, 1, 2, 0, 0, 2, 1, 1, 0, 2, 1, 0], dtype=torch.long, device="cuda"
            ),
        )

    def test_top_k_greater_than_one(self):
        # top_k > 1 expands each token to top_k output rows that must all take
        # the same (single, per-token) slot.
        self._assert_matches_isolated_single_slot_runs(
            num_tokens=16,
            hidden=64,
            num_experts=4,
            top_k=2,
            bs=16,
            num_slots=4,
        )

    def test_smallest_valid_block_size(self):
        # bs=4 is the smallest block size validate_oft_block_size accepts, and
        # is where the single-slot oracle itself takes its elementwise branch.
        self._assert_matches_isolated_single_slot_runs(
            num_tokens=16,
            hidden=32,
            num_experts=2,
            top_k=1,
            bs=4,
            num_slots=3,
        )

    def test_large_block_size_against_single_slot_tl_dot_path(self):
        # bs=128 puts the single-slot oracle on its tl.dot/tensor-core path,
        # so this also pins the elementwise multi-slot accumulation against it.
        self._assert_matches_isolated_single_slot_runs(
            num_tokens=32,
            hidden=256,
            num_experts=4,
            top_k=1,
            bs=128,
            num_slots=3,
        )

    def test_many_tokens_spanning_multiple_blocks_with_padding(self):
        # More tokens than one BLOCK_M, uneven per-expert counts (so
        # moe_align_block_size emits padded rows), and top_k > 1.
        self._assert_matches_isolated_single_slot_runs(
            num_tokens=200,
            hidden=64,
            num_experts=8,
            top_k=2,
            bs=32,
            num_slots=5,
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class TestMultiSlotRotationArgumentValidation(unittest.TestCase):
    def _args(self, *, num_tokens=8, hidden=32, num_experts=4, bs=16, num_slots=3):
        device = "cuda"
        A = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, 1), device=device, dtype=torch.int32
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, 64, num_experts
        )
        oft_r_all_slots = torch.randn(
            num_slots,
            num_experts,
            hidden // bs,
            bs,
            bs,
            device=device,
            dtype=torch.bfloat16,
        )
        slot_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
        return (
            A,
            oft_r_all_slots,
            slot_ids,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        )

    def test_rejects_single_slot_4d_r_tensor(self):
        A, r, slot_ids, topk_ids, sti, ei, ntpp = self._args()
        with self.assertRaisesRegex(ValueError, "must be 5D"):
            apply_oft_rotation_triton_multi_slot(
                A, r[0], slot_ids, topk_ids, sti, ei, ntpp, 1
            )

    def test_rejects_slot_ids_length_mismatch(self):
        A, r, slot_ids, topk_ids, sti, ei, ntpp = self._args()
        with self.assertRaisesRegex(ValueError, "slot_ids has 7 entries"):
            apply_oft_rotation_triton_multi_slot(
                A, r, slot_ids[:-1], topk_ids, sti, ei, ntpp, 1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

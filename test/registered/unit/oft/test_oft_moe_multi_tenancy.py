"""Unit tests for OFTManager's per-batch MoE multi-tenancy decision.

No GPU required: constructs weight_indices directly and calls the method
under test via a lightweight OFTManager stand-in, following this codebase's
established pattern of binding the real production method onto a minimal
double (see test_oft_native_admission.py) rather than reimplementing the
logic.
"""

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.oft.oft_manager import OFTManager


def _forward_batch(
    *,
    forward_mode,
    num_tokens,
    batch_size=None,
    extend_seq_lens=None,
    draft_token_num=None,
):
    """Minimal read-only ForwardBatch stand-in carrying exactly the fields
    ``generate_sequence_lengths`` and the method under test touch. Uses the
    REAL ``ForwardMode`` enum so the mode predicates are the production ones."""
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        extend_seq_lens=(
            None
            if extend_seq_lens is None
            else torch.tensor(extend_seq_lens, dtype=torch.int32)
        ),
        extend_seq_lens_cpu=extend_seq_lens,
        spec_info=(
            None
            if draft_token_num is None
            else SimpleNamespace(draft_token_num=draft_token_num)
        ),
    )


class TestMoeMultiTenancyDecision(unittest.TestCase):
    def _make_tm(self):
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    # A bare SimpleNamespace raises AttributeError on ANY field access, so the
    # single-adapter fast-path cases below also pin that the fast path reads
    # nothing off the forward batch (no per-batch tensor work on it).
    def test_single_resident_adapter_yields_none(self):
        tm = self._make_tm()
        # All requests map to the same real slot (2) or the base slot (0).
        weight_indices = [2, 2, 0, 2, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, SimpleNamespace()
        )
        self.assertIsNone(result)

    def test_no_resident_adapter_yields_none(self):
        tm = self._make_tm()
        weight_indices = [0, 0, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, SimpleNamespace()
        )
        self.assertIsNone(result)

    def test_two_resident_adapters_yields_tensor(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 0, 1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=5, batch_size=5),
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
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=3, batch_size=3),
        )
        self.assertIsNotNone(result)


class TestMoeMultiTenantSlotIdsArePerToken(unittest.TestCase):
    """The multi-tenant MoE kernel indexes slot_ids by TOKEN (MoE routing is
    per token), but ``weight_indices`` is built per REQUEST from
    ``forward_batch.adapter_ids`` (``[req.adapter_id for req in batch.reqs]``).

    Regression guard for the bug that shipped in this plan's Task 1: the
    per-request array was handed to the kernel unexpanded, so every extend /
    prefill batch with two resident adapters produced a slot tensor shorter
    than the token count and the rotation entry point rejected it. Decode-only
    batches accidentally worked because they carry exactly one token per
    request, which is why the cases below cover every mode that reaches here.
    """

    def _make_tm(self):
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    def test_extend_batch_expands_slots_to_one_entry_per_token(self):
        tm = self._make_tm()
        # 3 requests on slots 1, 2, 1 contributing 4, 1 and 3 tokens.
        weight_indices = [1, 2, 1]
        extend_seq_lens = [4, 1, 3]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(
                forward_mode=ForwardMode.EXTEND,
                num_tokens=sum(extend_seq_lens),
                batch_size=len(weight_indices),
                extend_seq_lens=extend_seq_lens,
            ),
        )
        self.assertEqual(result.shape[0], sum(extend_seq_lens))
        self.assertEqual(result.dtype, torch.long)
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor([1, 1, 1, 1, 2, 1, 1, 1], dtype=torch.long),
            )
        )

    def test_decode_batch_is_one_token_per_request(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 1]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=3, batch_size=3),
        )
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 1], dtype=torch.long)))

    def test_mixed_batch_expands_prefill_and_decode_requests(self):
        tm = self._make_tm()
        # ScheduleBatch.mix_with_running appends a 1 to extend_lens for every
        # request folded in from the running (decode) batch, so extend_seq_lens
        # stays one entry per request in MIXED mode: here a 5-token prefill
        # request on slot 1 plus a 1-token decode request on slot 2.
        weight_indices = [1, 2]
        extend_seq_lens = [5, 1]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(
                forward_mode=ForwardMode.MIXED,
                num_tokens=sum(extend_seq_lens),
                batch_size=len(weight_indices),
                extend_seq_lens=extend_seq_lens,
            ),
        )
        self.assertTrue(
            torch.equal(result, torch.tensor([1, 1, 1, 1, 1, 2], dtype=torch.long))
        )

    def test_target_verify_batch_expands_by_draft_width(self):
        tm = self._make_tm()
        # Speculative target-verify runs draft_token_num tokens per request.
        weight_indices = [1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_tokens=8,
                batch_size=2,
                draft_token_num=4,
            ),
        )
        self.assertTrue(
            torch.equal(
                result, torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
            )
        )

    def test_token_count_request_count_mismatch_fails_loud(self):
        tm = self._make_tm()
        # One token count too few for the request list: must raise rather than
        # silently emit a short (mis-indexed) per-token tensor.
        with self.assertRaisesRegex(RuntimeError, "one token count per request"):
            tm._compute_moe_multi_tenant_slot_ids(
                [1, 2, 1],
                _forward_batch(
                    forward_mode=ForwardMode.EXTEND,
                    num_tokens=5,
                    batch_size=3,
                    extend_seq_lens=[4, 1],
                ),
            )


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


class TestFullBufferBindingsInCudaGraphInit(unittest.TestCase):
    """Test full-buffer binding logic in _init_identity_expert_oft_for_cuda_graph.

    Guards against future edits that might incorrectly bind w1/w3 all-slots
    in the legacy fused branch (or vice versa), or forget to handle
    unregistered groups. This is a derived-property test: the branch-specific
    bindings are prescribed by the oft_type signal, and swapping them is a
    silent regression.
    """

    def _make_tm_for_cuda_graph_init(self, oft_type, init_w13=True, init_w2=True):
        """Create OFTManager stand-in with mocked memory_pool._groups."""
        # Create distinguishable tensor values per group, each identifiable.
        w1_tensor = torch.ones(2, 4, dtype=torch.float32) * 0.1
        w3_tensor = torch.ones(2, 4, dtype=torch.float32) * 0.3
        w13_tensor = torch.ones(2, 4, dtype=torch.float32) * 0.13
        w2_tensor = torch.ones(2, 4, dtype=torch.float32) * 0.2

        # Create a minimal MoE module stand-in.
        moe = SimpleNamespace()
        moe.hidden_size = 4096
        moe.intermediate_size_per_partition = 22016
        moe.w13_oft_r = None  # No pre-loaded legacy adapter.
        moe.w1_oft_r = None
        moe.w3_oft_r = None
        moe.w2_oft_r = None

        # Create a minimal memory_pool stand-in with _groups.
        memory_pool = SimpleNamespace()
        memory_pool._groups = {
            "w1_oft_r": {0: w1_tensor},
            "w3_oft_r": {0: w3_tensor},
            "w13_oft_r": {0: w13_tensor},
            "w2_oft_r": {0: w2_tensor},
        }
        memory_pool.active_idx = 0
        # Mock slot() and active_view() to avoid actual tensor creation.
        memory_pool.slot = mock.Mock(return_value=torch.zeros(4, 4))
        memory_pool.active_view = mock.Mock(return_value=torch.zeros(4, 4))

        # Create OFTManager stand-in.
        tm = SimpleNamespace()
        tm.oft_type = oft_type
        tm.target_modules = (
            {"gate_proj", "up_proj", "down_proj"}
            if (init_w13 and init_w2)
            else {"gate_proj", "up_proj"} if init_w13 else {"down_proj"}
        )
        tm.max_oft_block_size = 32
        tm.memory_pool = memory_pool
        tm._find_fused_moe_modules = lambda: {0: moe}

        # Bind the real method.
        tm._init_identity_expert_oft_for_cuda_graph = MethodType(
            OFTManager._init_identity_expert_oft_for_cuda_graph, tm
        )

        return tm, moe, w1_tensor, w3_tensor, w13_tensor, w2_tensor

    @mock.patch("sglang.srt.oft.oft_manager._fill_expert_oft_identity")
    def test_split_binds_w1_w3_not_w13(self, mock_fill):
        """Test canonical_oft (split) branch binds w1/w3 all-slots, not w13."""
        tm, moe, w1_tensor, w3_tensor, w13_tensor, w2_tensor = (
            self._make_tm_for_cuda_graph_init("canonical_oft", init_w13=True, init_w2=False)
        )

        tm._init_identity_expert_oft_for_cuda_graph()

        # Split branch should bind w1/w3, not w13.
        self.assertIs(moe._oft_w1_oft_r_all_slots, w1_tensor)
        self.assertIs(moe._oft_w3_oft_r_all_slots, w3_tensor)
        # w13 should not be set (or be None) in split path.
        self.assertFalse(hasattr(moe, "_oft_w13_oft_r_all_slots"))

    @mock.patch("sglang.srt.oft.oft_manager._fill_expert_oft_identity")
    def test_fused_binds_w13_not_w1_w3(self, mock_fill):
        """Test legacy fused (non-canonical) branch binds w13 all-slots, not w1/w3."""
        tm, moe, w1_tensor, w3_tensor, w13_tensor, w2_tensor = (
            self._make_tm_for_cuda_graph_init("oft", init_w13=True, init_w2=False)
        )

        tm._init_identity_expert_oft_for_cuda_graph()

        # Fused branch should bind w13, not w1/w3.
        self.assertIs(moe._oft_w13_oft_r_all_slots, w13_tensor)
        # w1/w3 should not be set (or be None) in fused path.
        self.assertFalse(hasattr(moe, "_oft_w1_oft_r_all_slots"))
        self.assertFalse(hasattr(moe, "_oft_w3_oft_r_all_slots"))

    @mock.patch("sglang.srt.oft.oft_manager._fill_expert_oft_identity")
    def test_w2_binds_in_both_split_and_fused(self, mock_fill):
        """Test w2 all-slots binding appears in both split and fused branches."""
        # Test split case.
        tm_split, moe_split, _, _, _, w2_tensor_split = (
            self._make_tm_for_cuda_graph_init("canonical_oft", init_w13=True, init_w2=True)
        )
        tm_split._init_identity_expert_oft_for_cuda_graph()
        self.assertIs(moe_split._oft_w2_oft_r_all_slots, w2_tensor_split)

        # Test fused case.
        tm_fused, moe_fused, _, _, _, w2_tensor_fused = (
            self._make_tm_for_cuda_graph_init("oft", init_w13=True, init_w2=True)
        )
        tm_fused._init_identity_expert_oft_for_cuda_graph()
        self.assertIs(moe_fused._oft_w2_oft_r_all_slots, w2_tensor_fused)

    @mock.patch("sglang.srt.oft.oft_manager._fill_expert_oft_identity")
    def test_unregistered_group_yields_none(self, mock_fill):
        """Test that unregistered groups gracefully yield None."""
        tm, moe, w1_tensor, w3_tensor, w13_tensor, w2_tensor = (
            self._make_tm_for_cuda_graph_init("canonical_oft", init_w13=True, init_w2=True)
        )
        # Remove w1_oft_r from _groups to simulate an unregistered group.
        del tm.memory_pool._groups["w1_oft_r"]

        tm._init_identity_expert_oft_for_cuda_graph()

        # Should gracefully return None from .get() instead of raising.
        self.assertIsNone(moe._oft_w1_oft_r_all_slots)
        # w3 should still work (it's registered).
        self.assertIs(moe._oft_w3_oft_r_all_slots, w3_tensor)


class TestMakeOftInvokeMultiTenantBranch(unittest.TestCase):
    def test_invoke_uses_multi_tenant_path_when_slot_ids_present(self):
        from sglang.srt.oft import oft_moe_runners

        # Real tensor (not object()) so the multi-tenant call site's B.shape[0]
        # (num_experts, mirroring _oft_prerotate's own convention) is valid;
        # dim 0 matches _oft_w13_oft_r_all_slots's expert axis (dim 1) below.
        w13_weight = torch.zeros(2, 8, 8)
        slot_ids = torch.tensor([1, 2], dtype=torch.long)
        oft_r_all_slots = torch.zeros(3, 2, 1, 4, 4)
        layer = SimpleNamespace(
            w13_weight=w13_weight,
            w2_weight=object(),
            w13_oft_r=torch.zeros(2, 1, 4, 4),
            w1_oft_r=None,
            w3_oft_r=None,
            w2_oft_r=None,
            _oft_moe_multi_tenant_slot_ids=slot_ids,
            _oft_w13_oft_r_all_slots=oft_r_all_slots,
        )
        real_invoke = unittest.mock.Mock()
        invoke = oft_moe_runners.make_oft_invoke(layer, real_invoke)

        A = torch.zeros(2, 8)
        with patch.object(
            oft_moe_runners, "_oft_prerotate_multi_tenant"
        ) as mock_multi, patch.object(oft_moe_runners, "_oft_prerotate") as mock_single:
            mock_multi.return_value = (A, A, None, None, None, None, None)
            invoke(
                A, layer.w13_weight, None, A, None, None, None,
                None, None, None, None, None, None, 1,
                {"BLOCK_SIZE_M": 32},
            )
            mock_multi.assert_called_once()
            mock_single.assert_not_called()
            call = mock_multi.call_args
            self.assertIs(call.args[1], oft_r_all_slots)
            self.assertIs(call.args[2], slot_ids)
            self.assertEqual(call.args[-2], w13_weight.shape[0])
            self.assertEqual(call.args[-1], 32)

    def test_invoke_uses_multi_tenant_path_for_down_projection(self):
        """Down-projection (w2) sub-case of the inline branch: the fused
        w13/gate-up sub-case above does not exercise is_gate_up=False, which
        selects a different attribute (_oft_w2_oft_r_all_slots)."""
        from sglang.srt.oft import oft_moe_runners

        w2_weight = torch.zeros(2, 8, 8)
        slot_ids = torch.tensor([1, 2], dtype=torch.long)
        oft_r_all_slots = torch.zeros(3, 2, 1, 4, 4)
        layer = SimpleNamespace(
            w13_weight=object(),
            w2_weight=w2_weight,
            w13_oft_r=None,
            w1_oft_r=None,
            w3_oft_r=None,
            w2_oft_r=torch.zeros(2, 1, 4, 4),
            _oft_moe_multi_tenant_slot_ids=slot_ids,
            _oft_w2_oft_r_all_slots=oft_r_all_slots,
        )
        real_invoke = unittest.mock.Mock()
        invoke = oft_moe_runners.make_oft_invoke(layer, real_invoke)

        A = torch.zeros(2, 8)
        with patch.object(
            oft_moe_runners, "_oft_prerotate_multi_tenant"
        ) as mock_multi, patch.object(oft_moe_runners, "_oft_prerotate") as mock_single:
            mock_multi.return_value = (A, A, None, None, None, None, None)
            invoke(
                A, layer.w2_weight, None, A, None, None, None,
                None, None, None, None, None, None, 1,
                {"BLOCK_SIZE_M": 32},
            )
            mock_multi.assert_called_once()
            mock_single.assert_not_called()
            call = mock_multi.call_args
            self.assertIs(call.args[1], oft_r_all_slots)
            self.assertIs(call.args[2], slot_ids)
            self.assertEqual(call.args[-2], w2_weight.shape[0])

    def test_invoke_raises_clear_error_when_all_slots_missing(self):
        """Guards the crash scenario: a legacy-fused static adapter loaded at
        boot leaves _oft_w13_oft_r_all_slots unbound
        (_init_identity_expert_oft_for_cuda_graph's short-circuit branch never
        sets it -- see oft_manager.py), so if multi-tenancy later activates for
        this layer, the direct attribute access must not surface as a bare
        AttributeError on the forward-pass hot path."""
        from sglang.srt.oft import oft_moe_runners

        w13_weight = torch.zeros(2, 8, 8)
        layer = SimpleNamespace(
            w13_weight=w13_weight,
            w2_weight=object(),
            w13_oft_r=torch.zeros(2, 1, 4, 4),
            w1_oft_r=None,
            w3_oft_r=None,
            w2_oft_r=None,
            _oft_moe_multi_tenant_slot_ids=torch.tensor([1, 2], dtype=torch.long),
            # _oft_w13_oft_r_all_slots intentionally absent.
        )
        real_invoke = unittest.mock.Mock()
        invoke = oft_moe_runners.make_oft_invoke(layer, real_invoke)

        A = torch.zeros(2, 8)
        with self.assertRaises(RuntimeError):
            invoke(
                A, layer.w13_weight, None, A, None, None, None,
                None, None, None, None, None, None, 1,
                {"BLOCK_SIZE_M": 32},
            )


class TestRunGateUpSplitMultiTenantBranch(unittest.TestCase):
    def test_selects_w1_or_w3_all_slots_per_half(self):
        """No existing test exercises _run_gate_up_split's multi-tenant
        branch: guards that the w1 half selects _oft_w1_oft_r_all_slots and
        the w3 half selects _oft_w3_oft_r_all_slots, not the other way
        around or a stale single-tenant fallback."""
        from sglang.srt.oft import oft_moe_runners

        num_experts, N, K = 2, 16, 8
        B = torch.zeros(num_experts, N, K)
        slot_ids = torch.tensor([1, 2], dtype=torch.long)
        w1_all_slots = torch.zeros(3, num_experts, 1, 4, 4)
        w3_all_slots = torch.zeros(3, num_experts, 1, 4, 4)
        layer = SimpleNamespace(
            w13_weight=B,
            w2_weight=object(),
            w13_oft_r=None,
            w1_oft_r=torch.zeros(num_experts, 1, 4, 4),
            w3_oft_r=torch.zeros(num_experts, 1, 4, 4),
            w2_oft_r=None,
            _oft_moe_multi_tenant_slot_ids=slot_ids,
            _oft_w1_oft_r_all_slots=w1_all_slots,
            _oft_w3_oft_r_all_slots=w3_all_slots,
        )
        real_invoke = unittest.mock.Mock()
        invoke = oft_moe_runners.make_oft_invoke(layer, real_invoke)

        total_tokens = 2
        A = torch.zeros(total_tokens, K)
        C = torch.zeros(total_tokens, N)

        with patch.object(
            oft_moe_runners, "_oft_prerotate_multi_tenant"
        ) as mock_multi:
            mock_multi.return_value = (A, C[:, : N // 2], None, None, None, None, None)
            invoke(
                A, B, None, C, None, None, None,
                None, None, None, None, None, None, 1,
                {"BLOCK_SIZE_M": 32}, compute_type=torch.float32,
            )

        self.assertEqual(mock_multi.call_count, 2)
        gate_call, up_call = mock_multi.call_args_list
        self.assertIs(gate_call.args[1], w1_all_slots)
        self.assertIs(gate_call.args[2], slot_ids)
        self.assertEqual(gate_call.args[-2], num_experts)
        self.assertIs(up_call.args[1], w3_all_slots)
        self.assertIs(up_call.args[2], slot_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)

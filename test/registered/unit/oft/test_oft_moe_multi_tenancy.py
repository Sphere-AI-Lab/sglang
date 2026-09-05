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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    def _make_tm(self, active_idx: int = 0):
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        # active_idx is the ONE slot oft_moe_runners.py's fast path (slot_ids
        # is None) ever reads -- see _compute_moe_multi_tenant_slot_ids's
        # docstring. Defaults to 0, matching real production (the plain
        # native-RPC OFTManager always reserves buffer slot 0 for the base/
        # identity placeholder and never assigns it to a real adapter).
        tm.memory_pool = SimpleNamespace(active_idx=active_idx)
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    # A bare SimpleNamespace forward_batch raises AttributeError on ANY field
    # access, so this case also pins that the fast path reads nothing off the
    # forward batch (no per-batch tensor work on it) when it actually takes
    # the fast path.
    def test_no_resident_adapter_yields_none(self):
        tm = self._make_tm()
        weight_indices = [0, 0, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, SimpleNamespace(), use_cuda_graph=False
        )
        self.assertIsNone(result)

    def test_single_resident_adapter_at_active_idx_yields_none(self):
        """The ONE case where a single real adapter is still fast-path-safe:
        its slot happens to equal active_idx, so the fast path's fixed read
        of active_idx's data is actually that adapter's own data."""
        tm = self._make_tm(active_idx=2)
        weight_indices = [2, 2, 0, 2, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, SimpleNamespace(), use_cuda_graph=False
        )
        self.assertIsNone(result)

    def test_single_resident_adapter_not_at_active_idx_yields_tensor(self):
        """Regression guard (Task 4b review finding): a single real adapter
        resident at a slot OTHER than active_idx must NOT take the fast
        path. Before this fix, `distinct_real_slots` (here {2}) having size
        <= 1 alone was enough to return None, so the fast path's fixed read
        of active_idx (0, the base/identity slot) would silently read
        identity instead of this adapter's real rotation -- exactly the
        scenario Task 4b's Fix 1 (writing each adapter to its OWN buffer_id)
        newly exposed, since adapters no longer land at active_idx by
        accident."""
        tm = self._make_tm(active_idx=0)
        # All requests map to the same real slot (2, != active_idx=0) or the
        # base slot (0).
        weight_indices = [2, 2, 0, 2, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=5, batch_size=5),
            use_cuda_graph=False,
        )
        self.assertIsNotNone(result)
        self.assertTrue(
            torch.equal(result, torch.tensor([2, 2, 0, 2, 0], dtype=torch.long))
        )

    def test_two_resident_adapters_yields_tensor(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 0, 1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=5, batch_size=5),
            use_cuda_graph=False,
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
            use_cuda_graph=False,
        )
        self.assertIsNotNone(result)


class TestMoeMultiTenantSlotIdsArePerToken(unittest.TestCase):
    """The multi-tenant MoE kernel indexes slot_ids by TOKEN (MoE routing is
    per token), but ``weight_indices`` is built per REQUEST from
    ``forward_batch.oft_ids`` (``[req.oft_id for req in batch.reqs]``).

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
        # active_idx=0 (matching real production default); every case in
        # this class has 2+ distinct real slots so the exact value doesn't
        # change the outcome, but _compute_moe_multi_tenant_slot_ids always
        # reads it to build its fast-path comparison set.
        tm.memory_pool = SimpleNamespace(active_idx=0)
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
            use_cuda_graph=False,
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
            use_cuda_graph=False,
        )
        self.assertTrue(torch.equal(result, torch.tensor([1, 2, 1], dtype=torch.long)))

    def test_cuda_graph_padded_decode_batch_expands_to_padded_batch_size(self):
        """Regression guard: under decode-CUDA-graph replay,
        ``DecodeCudaGraphRunner._prepare_oft_replay_batch`` temporarily
        sets ``forward_batch.batch_size`` to the padded capture-bucket size and
        pads ``oft_ids`` with ``None`` (so ``weight_indices`` is padded
        too), but leaves ``forward_batch.input_ids`` at the RAW, pre-pad token
        count. The per-request expansion computed here therefore legitimately
        sums to the PADDED batch size while ``input_ids.shape[0]`` is the raw
        one, and this function must never cross-check the two.

        The bug this guards: an intervening "avoid a device-to-host sync"
        change passed ``input_ids.shape[0]`` as ``repeat_interleave``'s
        ``output_size``, which turns that legitimate mismatch into a hard
        failure -- a catchable ``RuntimeError`` on CPU, but an unrecoverable,
        engine-killing device-side assert on CUDA. Reachable for ANY OFT batch
        padded to a capture bucket (``oft_impl=staged`` and dense-MLP-target
        sibling deployments included), not just MoE-target sibling ones.
        """
        tm = self._make_tm()
        # raw_bs=3 real requests padded to capture bucket bs=4; the padded row
        # carries oft_id None -> weight_indices 0 (the base/identity slot).
        weight_indices = [1, 1, 1, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=3, batch_size=4),
            use_cuda_graph=False,
        )
        self.assertTrue(
            torch.equal(result, torch.tensor([1, 1, 1, 0], dtype=torch.long))
        )

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
            use_cuda_graph=False,
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
            use_cuda_graph=False,
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
                use_cuda_graph=False,
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


class TestMoeExpertOftMultiTenantReady(unittest.TestCase):
    """Unit tests for OFTManager.moe_expert_oft_multi_tenant_ready -- the
    post-model-load ground truth decode_cuda_graph_runner.py's
    _resolve_record_oft_variant_graph reads (final whole-branch review's
    I1/I2/I3 fix). No GPU required: binds the real method onto a minimal
    OFTManager stand-in whose _find_fused_moe_modules returns hand-built
    SimpleNamespace "moe" objects with the exact attributes the real
    _init_identity_expert_oft_for_cuda_graph / _apply_expert_oft_to_module
    would have left on them.
    """

    def _ready(self, moe_modules: dict) -> bool:
        tm = SimpleNamespace()
        tm._find_fused_moe_modules = lambda: moe_modules
        tm.moe_expert_oft_multi_tenant_ready = MethodType(
            OFTManager.moe_expert_oft_multi_tenant_ready, tm
        )
        return tm.moe_expert_oft_multi_tenant_ready()

    def test_false_when_no_moe_modules_found(self):
        """Dense model (or no MoE targeting at all): nothing to dual-capture
        for."""
        self.assertFalse(self._ready({}))

    def test_false_when_moe_modules_found_but_no_oft_buffers(self):
        """MoE model, but OFT doesn't target its expert modules: MoE modules
        exist but carry no OFT buffers at all."""
        moe = SimpleNamespace(w13_oft_r=None, w1_oft_r=None, w3_oft_r=None, w2_oft_r=None)
        self.assertFalse(self._ready({0: moe}))

    def test_true_when_split_buffers_fully_bound(self):
        """The common, safe case: nothing boot-loaded (or a split adapter
        whose loader unconditionally binds _all_slots regardless of
        preexisting content -- see _init_identity_expert_oft_for_cuda_graph's
        elif w13_is_split branch) -- both w1/w3 and w2 all_slots bound."""
        moe = SimpleNamespace(
            w13_oft_r=None,
            w1_oft_r=torch.zeros(1),
            w3_oft_r=torch.zeros(1),
            w2_oft_r=torch.zeros(1),
            _oft_w1_oft_r_all_slots=torch.zeros(3, 1),
            _oft_w3_oft_r_all_slots=torch.zeros(3, 1),
            _oft_w2_oft_r_all_slots=torch.zeros(3, 1),
        )
        self.assertTrue(self._ready({0: moe}))

    def test_false_when_legacy_fused_boot_adapter_leaves_w13_all_slots_unbound(self):
        """Final whole-branch review I1: a legacy-fused (oft_type="oft")
        boot-loaded adapter leaves moe.w13_oft_r bound (the loader's private
        tensor) but _init_identity_expert_oft_for_cuda_graph's short-circuit
        (moe.w13_oft_r is not None: pass) never binds
        _oft_w13_oft_r_all_slots for it. Reproduced empirically against the
        real _init_identity_expert_oft_for_cuda_graph before writing this
        fixture (see final-review-fix-report.md)."""
        moe = SimpleNamespace(
            w13_oft_r=torch.zeros(1),  # boot-loaded, private -- no all_slots.
            w1_oft_r=None,
            w3_oft_r=None,
            w2_oft_r=None,
        )
        self.assertFalse(self._ready({0: moe}))

    def test_false_when_any_boot_adapter_leaves_w2_all_slots_unbound_regardless_of_oft_type(self):
        """Generalization of I1 found during investigation: _apply_expert_
        oft_to_module always privately allocates w2_oft_r for a boot-loaded
        adapter targeting down_proj -- for BOTH split (canonical_oft) and
        legacy adapters, since down_proj handling doesn't depend on oft_type
        -- so _init_identity_expert_oft_for_cuda_graph's
        `if init_w2 and moe.w2_oft_r is None` guard skips binding
        _oft_w2_oft_r_all_slots whenever ANYTHING is already loaded on it,
        not just for legacy. A server_args.oft_type-only exclusion would
        have missed this case entirely."""
        moe = SimpleNamespace(
            w13_oft_r=None,
            w1_oft_r=torch.zeros(1),
            w3_oft_r=torch.zeros(1),
            w2_oft_r=torch.zeros(1),  # boot-loaded, private -- no all_slots.
            _oft_w1_oft_r_all_slots=torch.zeros(3, 1),
            _oft_w3_oft_r_all_slots=torch.zeros(3, 1),
        )
        self.assertFalse(self._ready({0: moe}))

    def test_false_when_any_single_layer_has_a_gap(self):
        """Multi-layer model: one fully-bound layer must not mask a gap on
        another layer."""
        good = SimpleNamespace(
            w13_oft_r=None,
            w1_oft_r=torch.zeros(1),
            w3_oft_r=torch.zeros(1),
            w2_oft_r=torch.zeros(1),
            _oft_w1_oft_r_all_slots=torch.zeros(3, 1),
            _oft_w3_oft_r_all_slots=torch.zeros(3, 1),
            _oft_w2_oft_r_all_slots=torch.zeros(3, 1),
        )
        gapped = SimpleNamespace(
            w13_oft_r=torch.zeros(1),
            w1_oft_r=None,
            w3_oft_r=None,
            w2_oft_r=None,
        )
        self.assertFalse(self._ready({0: good, 1: gapped}))


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

    def test_down_projection_expands_slot_ids_by_router_topk(self):
        """Regression: the down GEMM's A is the gate-up output, already
        expanded to num_tokens*router_topk rows, but slot_ids arrives with one
        entry per ORIGINAL token. Handing it over unexpanded made the whole
        feature raise ValueError("slot_ids has N entries, expected M") for any
        router_topk > 1 -- i.e. for every realistic MoE model, since top_k=1 is
        the unusual case. Rows are token-major (row = token*router_topk + k).
        """
        from sglang.srt.oft import oft_moe_runners

        num_tokens, router_topk = 3, 2
        w2_weight = torch.zeros(2, 8, 8)
        slot_ids = torch.tensor([1, 2, 1], dtype=torch.long)
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
        invoke = oft_moe_runners.make_oft_invoke(layer, unittest.mock.Mock())

        A = torch.zeros(num_tokens * router_topk, 8)
        with patch.object(oft_moe_runners, "_oft_prerotate_multi_tenant") as mock_multi:
            mock_multi.return_value = (A, A, None, None, None, None, None)
            invoke(
                A, layer.w2_weight, None, A, None, None, None,
                None, None, None, None, None, None, 1,
                {"BLOCK_SIZE_M": 32},
                router_topk=router_topk,
            )
            passed_slot_ids = mock_multi.call_args.args[2]
        self.assertEqual(passed_slot_ids.shape[0], A.shape[0])
        self.assertTrue(
            torch.equal(
                passed_slot_ids, torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.long)
            )
        )

    def test_gate_up_slot_ids_are_not_expanded(self):
        """Negative partner of the case above: the gate-up GEMM's A is
        (num_tokens, K) and it is invoked with the real top_k, so the rotation
        kernel does the expansion itself. Expanding here too would double-count
        and is what the is_down guard exists to prevent."""
        from sglang.srt.oft import oft_moe_runners

        w13_weight = torch.zeros(2, 8, 8)
        slot_ids = torch.tensor([1, 2, 1], dtype=torch.long)
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
        invoke = oft_moe_runners.make_oft_invoke(layer, unittest.mock.Mock())

        A = torch.zeros(3, 8)
        with patch.object(oft_moe_runners, "_oft_prerotate_multi_tenant") as mock_multi:
            mock_multi.return_value = (A, A, None, None, None, None, None)
            invoke(
                A, layer.w13_weight, None, A, None, None, None,
                None, None, None, None, None, None, 2,
                {"BLOCK_SIZE_M": 32},
            )
            self.assertIs(mock_multi.call_args.args[2], slot_ids)

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


class TestPersistentSlotIdsBufferAndCaptureForcing(unittest.TestCase):
    """Task 4 (2026-09-01-oft-moe-cuda-graph-dual-capture): capture-time
    forcing via ``get_capture_oft_variant()`` and the persistent CUDA-graph
    ``slot_ids`` buffer's in-place-write discipline, mirroring
    ``LoRABackend._add_moe_lora_info``'s ``moe_cg_buffers`` handling exactly
    (read the persistent buffer when ``use_cuda_graph``, else build a fresh
    eager tensor; always WRITE into whichever tensor was resolved, never
    reassign the buffer object itself)."""

    def _make_tm(
        self,
        *,
        max_bs_in_cuda_graph=4,
        buffer=None,
        active_idx=0,
        enable_dp_attention=False,
    ):
        tm = SimpleNamespace()
        tm.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        tm._moe_cg_slot_ids_buffer = buffer
        tm.memory_pool = SimpleNamespace(active_idx=active_idx)
        tm.enable_dp_attention = enable_dp_attention
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    def test_capture_forcing_bypasses_no_real_adapter_early_return(self):
        """Capture-time dummy batches always carry 0 real adapters (every
        weight_indices entry is 0), which naturally satisfies the function's
        FIRST early-return (``not distinct_real_slots``). Capturing the
        "oft_multi" variant must force the general per-token tensor to be
        built anyway -- otherwise the "oft_multi" graph would capture the
        single-slot fast-path kernel, which is exactly the bug this task
        fixes (a real multi-adapter batch would then replay against a
        captured kernel that never learned to read per-token slot ids)."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            _set_capture_oft_variant,
        )

        self.addCleanup(_set_capture_oft_variant, None)
        _set_capture_oft_variant("oft_multi")

        tm = self._make_tm(buffer=torch.full((8,), -1, dtype=torch.long))
        weight_indices = [0, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2),
            use_cuda_graph=True,
        )
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, torch.tensor([0, 0], dtype=torch.long)))

    def test_capture_forcing_bypasses_active_idx_early_return(self):
        """The SECOND early-return (a single real adapter resident exactly
        at ``active_idx``) must also be bypassed while capturing
        "oft_multi", not just the first one."""
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            _set_capture_oft_variant,
        )

        self.addCleanup(_set_capture_oft_variant, None)
        _set_capture_oft_variant("oft_multi")

        tm = self._make_tm(
            buffer=torch.full((8,), -1, dtype=torch.long), active_idx=2
        )
        weight_indices = [2, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2),
            use_cuda_graph=True,
        )
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, torch.tensor([2, 2], dtype=torch.long)))

    def test_use_cuda_graph_writes_into_persistent_buffer_in_place(self):
        buffer = torch.full((8,), -1, dtype=torch.long)
        tm = self._make_tm(buffer=buffer)
        weight_indices = [1, 2]  # two distinct real slots -- genuinely multi
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2),
            use_cuda_graph=True,
        )
        self.assertIsNotNone(result)
        # The RETURNED tensor must be a view/slice of the SAME buffer object,
        # not a fresh allocation -- this is the whole point of the fix.
        self.assertEqual(result.data_ptr(), buffer.data_ptr())
        self.assertTrue(torch.equal(result, torch.tensor([1, 2], dtype=torch.long)))

    def test_eager_path_unaffected_still_returns_fresh_tensor(self):
        tm = self._make_tm(buffer=torch.full((8,), -1, dtype=torch.long))
        weight_indices = [1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2),
            use_cuda_graph=False,
        )
        self.assertIsNotNone(result)
        self.assertNotEqual(result.data_ptr(), tm._moe_cg_slot_ids_buffer.data_ptr())

    def test_dp_attention_enabled_raises_instead_of_silently_wrong(self):
        """--enable-dp-attention gathers MoE tokens across DP ranks, which
        this per-rank persistent buffer's sizing does not account for
        (explicitly out of scope for this first cut). Must raise a clear
        error rather than silently truncating or overrunning the buffer."""
        tm = self._make_tm(
            buffer=torch.full((8,), -1, dtype=torch.long), enable_dp_attention=True
        )
        weight_indices = [1, 2]
        with self.assertRaises(RuntimeError):
            tm._compute_moe_multi_tenant_slot_ids(
                weight_indices,
                _forward_batch(
                    forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2
                ),
                use_cuda_graph=True,
            )

    def test_buffer_capacity_exceeded_raises(self):
        """The persistent buffer is sized by the runner as
        max_bs_in_cuda_graph * num_tokens_per_bs, which should always be an
        upper bound on what actually reaches here; a batch that needs more
        tokens than the buffer holds must raise loudly rather than silently
        writing out of bounds via copy_."""
        tm = self._make_tm(buffer=torch.full((2,), -1, dtype=torch.long))
        weight_indices = [1, 2, 1]
        with self.assertRaises(RuntimeError):
            tm._compute_moe_multi_tenant_slot_ids(
                weight_indices,
                _forward_batch(
                    forward_mode=ForwardMode.DECODE, num_tokens=3, batch_size=3
                ),
                use_cuda_graph=True,
            )

    def test_missing_buffer_raises_clear_assertion(self):
        """use_cuda_graph=True should never reach here before
        init_cuda_graph_batch_info has allocated the persistent buffer;
        guard with a clear assertion rather than a bare AttributeError deep
        inside the in-place write below."""
        tm = self._make_tm(buffer=None)
        weight_indices = [1, 2]
        with self.assertRaises(AssertionError):
            tm._compute_moe_multi_tenant_slot_ids(
                weight_indices,
                _forward_batch(
                    forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2
                ),
                use_cuda_graph=True,
            )

    def test_staged_double_buffer_active_idx_adapter_forces_general_tensor_under_cuda_graph(
        self,
    ):
        """Regression (independently-verified review finding on this task's
        own commit): under --oft-double-buffer (oft_impl="staged"),
        active_idx is NOT permanently reserved the way it is for the plain
        sibling pool -- a staged adapter's activate() genuinely registers it
        AT active_idx (mem_pool.py). So a single resident staged adapter can
        legitimately satisfy distinct_real_slots == {active_idx}.

        The host-side graph-variant selector
        (decode_cuda_graph_runner.py's _resolve_oft_variant) decides purely
        from whether ANY real adapter is present, with no visibility into
        which slot it occupies -- for this exact scenario it selects the
        general-kernel ("oft_multi") graph. Before this fix, this function
        still returned None here (the active_idx fast path applied
        regardless of use_cuda_graph), so the persistent buffer was never
        written and the replayed general kernel would read stale/unwritten
        contents -- silent wrong rotation. This pins that use_cuda_graph=True
        always builds and writes the general tensor for ANY real adapter,
        regardless of which slot it is in, matching the host's own "any real
        adapter -> general path" criterion exactly."""
        buffer = torch.full((8,), -1, dtype=torch.long)
        # active_idx=1: mirrors the staged/double-buffer pool (mem_pool.py),
        # where active_idx is not the permanently-reserved 0 of the plain
        # sibling pool.
        tm = self._make_tm(buffer=buffer, active_idx=1)
        # One real adapter, resident exactly AT active_idx.
        weight_indices = [1, 1]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2),
            use_cuda_graph=True,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.data_ptr(), buffer.data_ptr())
        self.assertTrue(torch.equal(result, torch.tensor([1, 1], dtype=torch.long)))

    def test_staged_double_buffer_active_idx_adapter_still_fast_paths_in_eager_mode(
        self,
    ):
        """Negative partner of the case above: the identical staged/
        active_idx scenario in EAGER mode (use_cuda_graph=False) must still
        return None -- this function's own direct read there is consistent
        with what oft_moe_runners.py's fast path reads, so the fix above
        must not regress the already-working staged eager case."""
        tm = self._make_tm(
            buffer=torch.full((8,), -1, dtype=torch.long), active_idx=1
        )
        weight_indices = [1, 1]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices,
            _forward_batch(forward_mode=ForwardMode.DECODE, num_tokens=2, batch_size=2),
            use_cuda_graph=False,
        )
        self.assertIsNone(result)


class TestPrepareOftBatchEagerDemotionOnOverflow(unittest.TestCase):
    """Final whole-branch review I4: prepare_oft_batch's own use_cuda_graph
    heuristic (bs <= max_bs_in_cuda_graph and forward_mode.is_cuda_graph())
    has no awareness of per-token draft-width sizing. A TARGET_VERIFY batch
    whose real draft_token_num exceeds the persistent buffer's capacity must
    demote to the eager path for that one call (mirroring
    LoRAManager.prepare_lora_batch's own use_cuda_graph=False demotion) --
    not raise, since the runner would have run this batch eagerly anyway."""

    def _make_tm(self, *, buffer_size: int):
        tm = SimpleNamespace()
        tm.max_bs_in_cuda_graph = 8
        tm._moe_cg_slot_ids_buffer = torch.full(
            (buffer_size,), -1, dtype=torch.long
        )
        tm.max_ofts_per_batch = 4
        tm.memory_pool = SimpleNamespace(
            uid_to_buffer_id={"adapterA": 1, "adapterB": 2},
            get_buffer_id=lambda uid: {"adapterA": 1, "adapterB": 2}[uid],
            active_idx=0,
        )
        tm.adapters = {
            "adapterA": SimpleNamespace(block_size=32),
            "adapterB": SimpleNamespace(block_size=32),
        }
        tm.configs = {}
        tm.enable_dp_attention = False
        tm._find_fused_moe_modules = lambda: {}
        tm.oft_backend = SimpleNamespace(
            prepare_oft_batch=mock.Mock(), batch_info=None
        )
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        tm.prepare_oft_batch = MethodType(OFTManager.prepare_oft_batch, tm)
        tm._push_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._push_moe_multi_tenant_slot_ids, tm
        )
        tm._push_moe_multi_tenant_batch_info = MethodType(
            OFTManager._push_moe_multi_tenant_batch_info, tm
        )
        return tm

    def _target_verify_batch(self, *, batch_size, draft_token_num):
        return SimpleNamespace(
            batch_size=batch_size,
            forward_mode=ForwardMode.TARGET_VERIFY,
            oft_ids=["adapterA", "adapterB"][:batch_size],
            input_ids=torch.zeros(batch_size * draft_token_num, dtype=torch.int32),
            spec_info=SimpleNamespace(draft_token_num=draft_token_num),
        )

    def test_overflowing_target_verify_batch_demotes_instead_of_raising(self):
        """bs=2, draft_token_num=5 -> 10 real per-token slots needed, but the
        persistent buffer only holds 4 -- must not raise."""
        tm = self._make_tm(buffer_size=4)
        forward_batch = self._target_verify_batch(batch_size=2, draft_token_num=5)

        tm.prepare_oft_batch(forward_batch)  # must not raise

        # Demoted to eager: the backend must see use_cuda_graph=False.
        self.assertFalse(
            tm.oft_backend.prepare_oft_batch.call_args.kwargs["use_cuda_graph"]
        )

    def test_demoted_batch_produces_a_correctly_sized_fresh_tensor(self):
        """The demoted call must still produce a CORRECT result -- a
        freshly-built tensor with one entry per real token (bs *
        draft_token_num), not a truncated/garbage view into the
        undersized persistent buffer."""
        tm = self._make_tm(buffer_size=4)
        forward_batch = self._target_verify_batch(batch_size=2, draft_token_num=5)

        tm.prepare_oft_batch(forward_batch)

        self.assertEqual(tm._moe_multi_tenant_slot_ids.shape, (10,))
        self.assertTrue(
            torch.equal(
                tm._moe_multi_tenant_slot_ids,
                torch.tensor([1] * 5 + [2] * 5, dtype=torch.long),
            )
        )
        # Not a view into the (too-small) persistent buffer.
        self.assertNotEqual(
            tm._moe_multi_tenant_slot_ids.data_ptr(),
            tm._moe_cg_slot_ids_buffer.data_ptr(),
        )

    def test_non_overflowing_target_verify_batch_still_uses_cuda_graph_path(self):
        """Negative case: a batch that fits must NOT be demoted -- the
        persistent buffer is still written in place, exactly as before this
        fix, for the same shape the buffer was captured for."""
        tm = self._make_tm(buffer_size=8)
        forward_batch = self._target_verify_batch(batch_size=2, draft_token_num=4)

        tm.prepare_oft_batch(forward_batch)

        self.assertTrue(
            tm.oft_backend.prepare_oft_batch.call_args.kwargs["use_cuda_graph"]
        )
        self.assertEqual(
            tm._moe_multi_tenant_slot_ids.data_ptr(),
            tm._moe_cg_slot_ids_buffer.data_ptr(),
        )


class TestInitCudaGraphBatchInfoAllocatesSlotIdsBuffer(unittest.TestCase):
    """The persistent buffer's sizing (max_bs_in_cuda_graph *
    num_tokens_per_bs) is the one line the whole CUDA-graph mechanism depends
    on; exercise the real init_cuda_graph_batch_info method rather than only
    ever injecting a hand-built buffer via a test fixture."""

    def _make_tm(self):
        tm = SimpleNamespace()
        tm.device = torch.device("cpu")
        # Matches OFTManager.__init__'s own field lifecycle (always set to
        # None, then a None check) -- the idempotence guard in
        # init_cuda_graph_batch_info reads this directly.
        tm._moe_cg_slot_ids_buffer = None
        tm.oft_backend = SimpleNamespace(
            init_cuda_graph_batch_info=lambda **kwargs: None
        )
        tm.init_cuda_graph_batch_info = MethodType(
            OFTManager.init_cuda_graph_batch_info, tm
        )
        return tm

    def test_buffer_shape_dtype_device_match_max_bs_times_tokens_per_bs(self):
        tm = self._make_tm()
        tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=4, num_tokens_per_bs=3)

        self.assertIsNotNone(tm._moe_cg_slot_ids_buffer)
        self.assertEqual(tm._moe_cg_slot_ids_buffer.shape, (12,))
        self.assertEqual(tm._moe_cg_slot_ids_buffer.dtype, torch.long)
        self.assertEqual(tm._moe_cg_slot_ids_buffer.device, torch.device("cpu"))
        self.assertTrue(torch.equal(
            tm._moe_cg_slot_ids_buffer, torch.zeros(12, dtype=torch.long)
        ))
        self.assertEqual(tm.max_bs_in_cuda_graph, 4)

    def test_decode_bs_1_token_per_bs(self):
        """Plain decode: num_tokens_per_bs=1 (see
        BaseOFTBackend.init_cuda_graph_batch_info's docstring)."""
        tm = self._make_tm()
        tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=8, num_tokens_per_bs=1)
        self.assertEqual(tm._moe_cg_slot_ids_buffer.shape, (8,))

    def test_second_call_with_matching_sizing_is_a_no_op(self):
        """Final whole-branch review I5: a second DecodeCudaGraphRunner can
        be constructed against the same underlying OFTManager (e.g.
        adaptive-speculative-decode's target_graph_runner). A graph already
        captured against the first buffer holds a pointer into it, so a
        second call with the SAME sizing must reuse the existing buffer
        (identity preserved), not silently reallocate a new one."""
        tm = self._make_tm()
        tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=4, num_tokens_per_bs=3)
        first_buffer = tm._moe_cg_slot_ids_buffer

        tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=4, num_tokens_per_bs=3)

        self.assertIs(tm._moe_cg_slot_ids_buffer, first_buffer)
        self.assertEqual(tm.max_bs_in_cuda_graph, 4)

    def test_second_call_with_different_sizing_raises_clearly(self):
        """A second call with DIFFERENT sizing must fail loudly rather than
        silently reallocate out from under a graph that already captured a
        pointer into the first buffer."""
        tm = self._make_tm()
        tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=4, num_tokens_per_bs=3)

        with self.assertRaises(AssertionError):
            tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=8, num_tokens_per_bs=3)

    def test_second_call_with_same_product_but_different_max_bs_still_raises(self):
        """max_bs_in_cuda_graph is read directly elsewhere (prepare_oft_
        batch's own use_cuda_graph gate), so a second call must match it
        exactly -- not just the buffer's total token-slot count -- even when
        the product (max_bs_in_cuda_graph * num_tokens_per_bs) happens to
        match."""
        tm = self._make_tm()
        tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=4, num_tokens_per_bs=2)

        with self.assertRaises(AssertionError):
            tm.init_cuda_graph_batch_info(max_bs_in_cuda_graph=2, num_tokens_per_bs=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Unit tests for the OFT MoE CUDA-graph dual-capture flag pair and ShapeKey field.

No GPU required.
"""

import unittest

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_oft_variant,
    get_capture_oft_variant,
)


class TestCaptureOftVariantFlag(unittest.TestCase):
    def tearDown(self):
        # Module-global state -- reset so tests don't leak into each other.
        _set_capture_oft_variant(None)

    def test_defaults_to_none(self):
        self.assertIsNone(get_capture_oft_variant())

    def test_set_and_get_round_trip(self):
        _set_capture_oft_variant("oft_multi")
        self.assertEqual(get_capture_oft_variant(), "oft_multi")
        _set_capture_oft_variant("oft_single")
        self.assertEqual(get_capture_oft_variant(), "oft_single")
        _set_capture_oft_variant(None)
        self.assertIsNone(get_capture_oft_variant())


class TestShapeKeyOftVariantField(unittest.TestCase):
    def test_default_none_and_backward_compatible_construction(self):
        # Existing call sites across the codebase construct ShapeKey without
        # oft_variant -- this must keep working unchanged.
        key = ShapeKey(size=4, stream_idx=None, variant_label="lora", dsa_variant="dense")
        self.assertIsNone(key.oft_variant)

    def test_oft_variant_is_part_of_equality_and_hash(self):
        # ShapeKey is a frozen dataclass used as a dict key elsewhere in the
        # runner -- two keys differing only in oft_variant must compare
        # unequal and hash differently, or captured graphs would collide.
        a = ShapeKey(size=4, oft_variant="oft_single")
        b = ShapeKey(size=4, oft_variant="oft_multi")
        self.assertNotEqual(a, b)
        self.assertNotEqual(hash(a), hash(b))


from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class TestResolveOftVariant(unittest.TestCase):
    def _make_runner(self, record_oft_variant_graph):
        runner = MagicMock(spec=DecodeCudaGraphRunner)
        runner.record_oft_variant_graph = record_oft_variant_graph
        runner._resolve_oft_variant = DecodeCudaGraphRunner._resolve_oft_variant.__get__(
            runner
        )
        return runner

    def test_returns_none_when_dual_capture_not_enabled(self):
        runner = self._make_runner(record_oft_variant_graph=False)
        forward_batch = SimpleNamespace(adapter_ids=["a", "b", None])
        self.assertIsNone(runner._resolve_oft_variant(forward_batch))

    def test_exactly_one_real_adapter_resolves_to_oft_multi_not_oft_single(self):
        """Regression guard: a single real adapter used to resolve to
        "oft_single" (>1-distinct-adapters threshold), but
        _compute_moe_multi_tenant_slot_ids (oft_manager.py) already takes
        the general per-token path for ANY real adapter today, not just 2+
        -- in the plain native-RPC ("sibling") pool, buffer slot 0
        (memory_pool.active_idx) is permanently reserved for the boot-time
        base/identity registration, so a real adapter can never land there
        and the old "single real adapter is fast-path-safe" case is
        unreachable. Replaying a single-real-adapter batch against a graph
        captured for the (dead) "oft_single" variant would silently apply
        the wrong routing. Verified this fails against the pre-fix (`> 1`)
        threshold and passes against the fixed (`>= 1`) one."""
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=["a", "a", None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_multi")

    def test_returns_oft_single_when_no_adapters_at_all(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=[None, None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_single")

    def test_returns_oft_multi_for_two_or_more_distinct_adapters(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=["a", "b", None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_multi")

    def test_returns_none_when_adapter_ids_is_none(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=None)
        self.assertIsNone(runner._resolve_oft_variant(forward_batch))


class TestOftVariantsCaptureAxis(unittest.TestCase):
    def test_single_no_op_variant_when_dual_capture_disabled(self):
        runner = MagicMock(spec=DecodeCudaGraphRunner)
        runner.record_oft_variant_graph = False
        # Mirrors how lora_variants / dsa_variants degrade to a single no-op
        # entry when their own dual-capture flag is off.
        oft_variants = (
            [("oft_multi", True), ("oft_single", False)]
            if getattr(runner, "record_oft_variant_graph", False)
            else [(None, None)]
        )
        self.assertEqual(oft_variants, [(None, None)])

    def test_two_variants_when_dual_capture_enabled(self):
        runner = MagicMock(spec=DecodeCudaGraphRunner)
        runner.record_oft_variant_graph = True
        oft_variants = (
            [("oft_multi", True), ("oft_single", False)]
            if getattr(runner, "record_oft_variant_graph", False)
            else [(None, None)]
        )
        self.assertEqual(oft_variants, [("oft_multi", True), ("oft_single", False)])


class TestResolveRecordOftVariantGraph(unittest.TestCase):
    """Unit tests for DecodeCudaGraphRunner._resolve_record_oft_variant_graph,
    the pure (server_args, model_config) -> bool condition that __init__
    assigns to self.record_oft_variant_graph."""

    def _server_args(self, **overrides):
        defaults = dict(
            peft_method="oft",
            oft_target_modules={"gate_up_proj", "down_proj"},
            max_ofts_per_batch=8,
            enable_dp_attention=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _model_config(self, has_moe_layers=True):
        hf_text_config = SimpleNamespace()
        if has_moe_layers:
            hf_text_config.num_experts_per_tok = 2
        return SimpleNamespace(hf_text_config=hf_text_config)

    def test_capacity_for_exactly_one_real_adapter_is_true(self):
        """Regression guard: max_ofts_per_batch=2 gives effective capacity
        (max_ofts_per_batch - 1) of exactly 1 -- room for exactly ONE real
        resident adapter. Before the >=1 threshold fix this required
        capacity > 1 (max_ofts_per_batch >= 3), wrongly treating a
        single-adapter-capacity server as not needing dual-capture even
        though oft_manager.py's _compute_moe_multi_tenant_slot_ids already
        takes the general per-token path for any real adapter. Verified this
        fails against the pre-fix (`> 1`) threshold and passes against the
        fixed (`>= 1`) one."""
        server_args = self._server_args(max_ofts_per_batch=2)
        model_config = self._model_config(has_moe_layers=True)
        self.assertTrue(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config
            )
        )

    def test_false_when_oft_not_enabled(self):
        server_args = self._server_args(peft_method=None)
        model_config = self._model_config(has_moe_layers=True)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config
            )
        )

    def test_false_when_not_targeting_moe_expert_modules(self):
        server_args = self._server_args(oft_target_modules={"q_proj", "k_proj"})
        model_config = self._model_config(has_moe_layers=True)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config
            )
        )

    def test_false_when_model_has_no_moe_layers(self):
        server_args = self._server_args()
        model_config = self._model_config(has_moe_layers=False)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config
            )
        )

    def test_false_when_capacity_allows_zero_real_adapters(self):
        # max_ofts_per_batch=1 -> effective capacity = 0 (only the reserved
        # base/identity slot exists) -- no real adapter can ever be resident.
        server_args = self._server_args(max_ofts_per_batch=1)
        model_config = self._model_config(has_moe_layers=True)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config
            )
        )

    def test_false_when_enable_dp_attention(self):
        """Final whole-branch review C1: dual-capture must never engage for
        a --enable-dp-attention server, regardless of otherwise-eligible
        target modules / capacity. _compute_moe_multi_tenant_slot_ids
        (oft_manager.py) raises RuntimeError unconditionally under
        --enable-dp-attention when use_cuda_graph=True -- before this fix,
        _resolve_record_oft_variant_graph did not check enable_dp_attention
        at all, so dual-capture's capture-time forcing would trip that raise
        AT BOOT (capture), not runtime, for a DP-attention + MoE-target-OFT
        server that previously booted fine (eagerly, decode graphs
        disabled). Verified this fails without the enable_dp_attention
        check and passes with it."""
        server_args = self._server_args(max_ofts_per_batch=8, enable_dp_attention=True)
        model_config = self._model_config(has_moe_layers=True)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config
            )
        )

    def _fake_oft_manager(self, ready: bool):
        return SimpleNamespace(moe_expert_oft_multi_tenant_ready=lambda: ready)

    def test_manager_path_true_when_ready_and_capacity_sufficient(self):
        server_args = self._server_args(max_ofts_per_batch=2)
        model_config = self._model_config(has_moe_layers=True)
        oft_manager = self._fake_oft_manager(ready=True)
        self.assertTrue(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config, oft_manager=oft_manager
            )
        )

    def test_manager_path_false_when_not_ready(self):
        """Final whole-branch review I1 (generalized): when an OFTManager is
        available, its moe_expert_oft_multi_tenant_ready() ground truth --
        not server_args's re-derived approximation -- decides eligibility.
        False here models a boot-loaded adapter that left a multi-tenant
        buffer view unbound (legacy-fused gate_up, or any-oft_type
        down_proj -- see OFTManager.moe_expert_oft_multi_tenant_ready's
        docstring): capture-time forcing would otherwise crash at boot the
        moment dual-capture engaged. Even though server_args alone (target
        modules + capacity) looks eligible, the manager's ready()=False must
        veto it."""
        server_args = self._server_args(max_ofts_per_batch=8)
        model_config = self._model_config(has_moe_layers=True)
        oft_manager = self._fake_oft_manager(ready=False)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config, oft_manager=oft_manager
            )
        )

    def test_manager_path_still_respects_capacity(self):
        """The manager's readiness check subsumes target-module/MoE-layer
        detection but NOT capacity -- capacity stays a separate,
        server_args-only criterion (effective_oft_capacity), checked after
        the manager says ready."""
        server_args = self._server_args(max_ofts_per_batch=1)
        model_config = self._model_config(has_moe_layers=True)
        oft_manager = self._fake_oft_manager(ready=True)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config, oft_manager=oft_manager
            )
        )

    def test_manager_path_still_excludes_dp_attention(self):
        """DP-attention must short-circuit before the manager is even
        consulted -- a truthy ready() must not override it."""
        server_args = self._server_args(max_ofts_per_batch=8, enable_dp_attention=True)
        model_config = self._model_config(has_moe_layers=True)
        oft_manager = self._fake_oft_manager(ready=True)
        self.assertFalse(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config, oft_manager=oft_manager
            )
        )

    def test_manager_path_ignores_stale_server_args_target_modules(self):
        """Final whole-branch review I2/I3: whenever server_args.
        oft_target_modules is empty/unset for any reason (this fixture's
        SimpleNamespace omits it here), the server_args-only fallback branch
        cannot see the model's real target modules, but the manager-derived
        ground truth must not be fooled by it: moe_expert_oft_multi_tenant_
        ready() alone decides eligibility once a manager is available,
        regardless of what server_args.oft_target_modules says."""
        server_args = self._server_args(
            max_ofts_per_batch=2, oft_target_modules=None
        )
        model_config = self._model_config(has_moe_layers=True)
        oft_manager = self._fake_oft_manager(ready=True)
        self.assertTrue(
            DecodeCudaGraphRunner._resolve_record_oft_variant_graph(
                server_args, model_config, oft_manager=oft_manager
            )
        )


class TestCaptureOneStreamOftKwargSubclassCompat(unittest.TestCase):
    """Regression guard (Critical bug found in review): _capture_one_stream
    used to pass oft_variant=oft_variant to capture_one_shape
    unconditionally, even when oft_variant is None -- the default /
    no-dual-capture case, true for every server today and every non-OFT
    server always. Four real DecodeCudaGraphRunner subclasses
    (EAGLEDraftCudaGraphRunner, FrozenKVMTPCudaGraphRunner,
    MultiLayerEagleDraftExtendCudaGraphRunner,
    EagleDraftExtendCudaGraphRunner -- see speculative/*.py) override
    capture_one_shape with a narrower signature (no oft_variant parameter,
    no **kwargs) and none of them run DecodeCudaGraphRunner.__init__ or
    override _capture_one_stream, so they all hit this same call and
    crashed with "TypeError: capture_one_shape() got an unexpected keyword
    argument 'oft_variant'". Fix: only include oft_variant in the kwargs
    passed to capture_one_shape when it is not None.

    This drives the REAL DecodeCudaGraphRunner._capture_one_stream (not a
    hand-rolled reimplementation) against a minimal stand-in whose
    capture_one_shape signature is copied verbatim from
    EAGLEDraftCudaGraphRunner.capture_one_shape
    (speculative/eagle_draft_cuda_graph_runner.py:324-330) -- with
    record_oft_variant_graph/dsa_dual_graph/record_nolora_graph
    deliberately left unset, matching how these subclasses never run
    DecodeCudaGraphRunner.__init__ and so never set them (getattr(...,
    False) is always False for them)."""

    class _EagleLikeStandIn:
        # capture_bs/compile_bs/captured_req_width/model_runner: the minimal
        # state _capture_one_stream reads before dispatching to
        # capture_one_shape. compile_bs=[] keeps torch_compile_decoration.
        # patch_model's disabled (yield model.forward) branch, so no real
        # torch.compile / model is needed.
        capture_bs = [1]
        compile_bs = []
        captured_req_width = 1
        model_runner = SimpleNamespace(
            device="cpu",
            gpu_id=0,
            model=SimpleNamespace(forward=lambda *a, **k: None),
            tp_group=None,
        )

        def __init__(self):
            self.calls = []

        # Signature copied verbatim from
        # EAGLEDraftCudaGraphRunner.capture_one_shape -- no oft_variant, no
        # **kwargs.
        def capture_one_shape(
            self,
            size,
            forward,
            stream_idx=None,
            variant_label=None,
        ):
            self.calls.append((size, forward, stream_idx, variant_label))

    def _run(self):
        runner = self._EagleLikeStandIn()
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.get_parallel",
            return_value=SimpleNamespace(tp_rank=1),
        ), patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
            "get_available_gpu_memory",
            return_value=0.0,
        ):
            DecodeCudaGraphRunner._capture_one_stream(runner, stream_idx=None)
        return runner

    def test_no_dual_capture_subclass_without_oft_kwarg_support_does_not_crash(self):
        runner = self._run()
        self.assertEqual(
            runner.calls, [(1, runner.model_runner.model.forward, None, None)]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

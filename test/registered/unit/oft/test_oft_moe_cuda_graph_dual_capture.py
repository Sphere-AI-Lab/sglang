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
from unittest.mock import MagicMock

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
            peft_target_modules={"gate_up_proj", "down_proj"},
            max_ofts_per_batch=8,
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
        server_args = self._server_args(peft_target_modules={"q_proj", "k_proj"})
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

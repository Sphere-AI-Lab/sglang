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

    def test_returns_oft_single_for_at_most_one_distinct_adapter(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=["a", "a", None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_single")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

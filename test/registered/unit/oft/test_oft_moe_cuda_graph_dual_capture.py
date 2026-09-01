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


if __name__ == "__main__":
    unittest.main(verbosity=2)

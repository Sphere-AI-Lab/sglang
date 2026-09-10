import unittest

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


class TestPretokenizedInputIdExpansion(unittest.TestCase):
    placeholder = 99

    def test_expands_compact_placeholder_runs(self):
        self.assertEqual(
            BaseMultimodalProcessor._expand_input_ids(
                [10, self.placeholder, 11, self.placeholder, 12],
                [3, 2],
                self.placeholder,
            ),
            [
                10,
                self.placeholder,
                self.placeholder,
                self.placeholder,
                11,
                self.placeholder,
                self.placeholder,
                12,
            ],
        )

    def test_expands_adjacent_compact_placeholders(self):
        self.assertEqual(
            BaseMultimodalProcessor._expand_input_ids(
                [10, self.placeholder, self.placeholder, 11],
                [3, 2],
                self.placeholder,
            ),
            [10, *([self.placeholder] * 5), 11],
        )

    def test_preserves_hf_processor_expanded_placeholder_runs(self):
        expanded = [
            10,
            self.placeholder,
            self.placeholder,
            self.placeholder,
            11,
            self.placeholder,
            self.placeholder,
            12,
        ]
        self.assertEqual(
            BaseMultimodalProcessor._expand_input_ids(
                expanded, [3, 2], self.placeholder
            ),
            expanded,
        )

    def test_rejects_malformed_placeholder_run_length(self):
        with self.assertRaisesRegex(
            ValueError,
            "run 0 has length 2; expected compact length 1 or expanded length 3",
        ):
            BaseMultimodalProcessor._expand_input_ids(
                [10, self.placeholder, self.placeholder, 11],
                [3],
                self.placeholder,
            )

    def test_rejects_placeholder_run_count_mismatch(self):
        with self.assertRaisesRegex(
            ValueError,
            "prompt has 1 image placeholder run\\(s\\) but 2 image\\(s\\) were provided",
        ):
            BaseMultimodalProcessor._expand_input_ids(
                [10, self.placeholder, 11],
                [3, 2],
                self.placeholder,
            )


if __name__ == "__main__":
    unittest.main()

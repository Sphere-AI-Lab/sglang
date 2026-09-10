import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _append_scoring_suffix,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


def _request(**overrides):
    values = {
        "text": "prompt",
        "scoring_suffix_ids": [41, 42],
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    values.update(overrides)
    return GenerateReqInput(**values)


class TestExactScoringSuffix(unittest.TestCase):
    def test_absent_suffix_is_a_strict_noop(self):
        request = GenerateReqInput(text="prompt")
        input_ids = [10, 11]

        result = _append_scoring_suffix(request, input_ids, None, None)

        self.assertIs(result, input_ids)
        self.assertIsNone(request.scoring_suffix_ids)

    def test_appends_exact_ids_and_defers_logprob_boundary(self):
        request = _request()

        result = _append_scoring_suffix(request, [10, 11], None, None)

        self.assertEqual(result, [10, 11, 41, 42])
        self.assertEqual(request.logprob_start_len, -1)

    def test_updates_multimodal_sequence_metadata(self):
        request = _request()
        delta = torch.tensor([[2]])
        mm_inputs = SimpleNamespace(
            input_ids=[10, 11],
            padded_input_ids=[110, 111],
            mrope_positions=torch.tensor([[0, 1], [0, 1], [0, 1]]),
            mrope_position_delta=delta,
            visible_frame_counts=torch.tensor([2, 3], dtype=torch.int32),
        )

        result = _append_scoring_suffix(request, [10, 11], mm_inputs, None)

        self.assertEqual(result, [10, 11, 41, 42])
        self.assertEqual(mm_inputs.input_ids, result)
        self.assertEqual(mm_inputs.padded_input_ids, [110, 111, 41, 42])
        torch.testing.assert_close(
            mm_inputs.mrope_positions,
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
        )
        self.assertIs(mm_inputs.mrope_position_delta, delta)
        torch.testing.assert_close(
            mm_inputs.visible_frame_counts,
            torch.tensor([2, 3, 3, 3], dtype=torch.int32),
        )

    def test_exact_suffix_cannot_be_silently_auto_truncated(self):
        manager = object.__new__(TokenizerManager)
        manager.context_len = 3
        manager.num_reserved_tokens = 0
        manager.server_args = SimpleNamespace(allow_auto_truncate=True)
        manager.allow_auto_truncate = True

        with self.assertRaisesRegex(ValueError, "cannot be auto-truncated"):
            manager._validate_one_request(_request(), [10, 11, 41, 42])

    def test_rejects_ambiguous_or_unsupported_requests(self):
        cases = [
            (_request(scoring_suffix_ids=[]), "must not be empty"),
            (_request(return_logprob=False), "return_logprob=True"),
            (
                _request(multi_item_delimiter_indices=[1]),
                "does not support multi_item_delimiter_indices",
            ),
            (_request(sampling_params={"max_new_tokens": 1}), "max_new_tokens=0"),
            (_request(logprob_start_len=0), "logprob_start_len must be omitted"),
            (
                _request(text=None, input_embeds=[[0.1, 0.2]]),
                "does not support input_embeds",
            ),
            (
                _request(image_data=["encoded-image"]),
                "requires tokenizer-side multimodal processing",
            ),
        ]

        for request, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                _append_scoring_suffix(request, [10, 11], None, None)

        with self.assertRaisesRegex(ValueError, "token_type_ids"):
            _append_scoring_suffix(_request(), [10, 11], None, [0, 0])

    def test_rejects_inconsistent_multimodal_metadata_lengths(self):
        bad_padded = SimpleNamespace(
            input_ids=[10, 11],
            padded_input_ids=[110],
            mrope_positions=None,
        )
        with self.assertRaisesRegex(ValueError, "padded_input_ids length"):
            _append_scoring_suffix(_request(), [10, 11], bad_padded, None)

        bad_mrope = SimpleNamespace(
            input_ids=[10, 11],
            padded_input_ids=[110, 111],
            mrope_positions=torch.zeros((3, 1), dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "mrope_positions length"):
            _append_scoring_suffix(_request(), [10, 11], bad_mrope, None)

        bad_visible_frame_counts = SimpleNamespace(
            input_ids=[10, 11],
            padded_input_ids=[110, 111],
            mrope_positions=None,
            visible_frame_counts=torch.tensor([1], dtype=torch.int32),
        )
        with self.assertRaisesRegex(ValueError, "visible_frame_counts length"):
            _append_scoring_suffix(_request(), [10, 11], bad_visible_frame_counts, None)

    def test_rejects_multimodal_prefix_that_ends_in_a_vision_mrope_position(self):
        mm_inputs = SimpleNamespace(
            input_ids=[10, 11],
            padded_input_ids=[110, 111],
            mrope_positions=torch.tensor([[0, 1], [0, 3], [0, 2]]),
        )

        with self.assertRaisesRegex(ValueError, "end in a text position"):
            _append_scoring_suffix(_request(), [10, 11], mm_inputs, None)

    def test_rejects_multimodal_special_tokens_inside_suffix(self):
        request = _request(scoring_suffix_ids=[41, 151655, 42])
        mm_inputs = SimpleNamespace(
            input_ids=[10, 11],
            padded_input_ids=[110, 111],
            mrope_positions=None,
            im_token_id=151655,
        )

        with self.assertRaisesRegex(ValueError, "pure-text actions"):
            _append_scoring_suffix(request, [10, 11], mm_inputs, None)


if __name__ == "__main__":
    unittest.main()

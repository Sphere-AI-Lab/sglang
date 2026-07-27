"""E2E coverage for exact token suffixes on multimodal scoring requests.

The server owns text and image preprocessing, then appends scoring_suffix_ids
verbatim. A deliberately non-canonical BPE suffix proves that the sampled text
actions are not decoded and re-tokenized by the multimodal processor.
"""

import base64
import io
import unittest

import requests
from PIL import Image
from transformers import AutoProcessor

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")


def _data_uri():
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _build_noncanonical_suffix(model):
    tokenizer = AutoProcessor.from_pretrained(
        model, trust_remote_code=True, use_fast=True
    ).tokenizer

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    suffix_ids = encode("D") + encode("escribe")
    canonical_ids = encode(tokenizer.decode(suffix_ids))
    if suffix_ids == canonical_ids:
        raise AssertionError(
            "test suffix is canonical; no retokenization drift to detect"
        )
    return suffix_ids


def _score_suffix(base_url, text, suffix_ids, image):
    response = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "image_data": [image],
            "scoring_suffix_ids": suffix_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "return_logprob": True,
        },
        timeout=300,
    )
    response.raise_for_status()
    meta_info = response.json()["meta_info"]
    scored_suffix_ids = [entry[1] for entry in meta_info["input_token_logprobs"][1:]]
    return scored_suffix_ids, meta_info["prompt_tokens"]


class TestQwenVLExactScoringSuffix(CustomTestCase):
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    other_args = ["--trust-remote-code", "--mem-fraction-static", "0.7"]

    def test_multimodal_processor_preserves_exact_scoring_suffix(self):
        suffix_ids = _build_noncanonical_suffix(self.model)
        text = f"Describe the picture: {self.image_token}\nAnswer: "
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args,
        )
        try:
            scored_ids, prompt_tokens = _score_suffix(
                DEFAULT_URL_FOR_TEST, text, suffix_ids, _data_uri()
            )
        finally:
            kill_process_tree(process.pid)

        self.assertEqual(scored_ids, suffix_ids)
        self.assertGreater(prompt_tokens, len(suffix_ids))


if __name__ == "__main__":
    unittest.main()

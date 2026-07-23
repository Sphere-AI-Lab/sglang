"""E2E test for SGLANG_MM_AVOID_RETOKENIZE on the pre-tokenized VLM path.

A client may send a multimodal request as input_ids (list[int]) instead of text.
On that path the server decodes the ids back to text and the HF processor
re-tokenizes them. If the original ids were non-canonical (decode -> re-encode is
not identity), that re-tokenization drifts: the reported prompt_tokens changes.

With SGLANG_MM_AVOID_RETOKENIZE ON (default), the server keeps the user's
original tokens verbatim and preserves the image placeholder run already
expanded by the HF processor, so prompt_tokens stays faithful to what the client
sent.

We launch a real server twice with the same HF processor-generated multimodal
input_ids, then make only its text suffix non-canonical ("Describe" split into
"D"+"escribe"):

  * flag OFF -> the prompt re-tokenizes (drift): prompt_tokens shrinks by the
    drift delta.
  * flag ON  -> no drift: prompt_tokens equals the original length (with the
    image placeholder expanded).
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

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


def _test_image():
    return Image.new("RGB", (64, 64), (128, 128, 128))


def _data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _find_subsequence(sequence, subsequence):
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    raise AssertionError(f"subsequence {subsequence} not found")


def _placeholder_run_lengths(input_ids, placeholder_token_id):
    run_lengths = []
    token_idx = 0
    while token_idx < len(input_ids):
        if input_ids[token_idx] != placeholder_token_id:
            token_idx += 1
            continue
        run_start = token_idx
        while (
            token_idx < len(input_ids) and input_ids[token_idx] == placeholder_token_id
        ):
            token_idx += 1
        run_lengths.append(token_idx - run_start)
    return run_lengths


def _build_drift_prompt(model, image_token, image):
    """Return (input_ids, drift_delta).

    Start with real HF processor output so the image is represented by an
    expanded run of image-pad tokens, matching the Miles rollout payload. Then
    replace canonical "Describe" ids with the equivalent non-canonical
    "D"+"escribe" split. drift_delta is how many extra text tokens that split
    carries.
    """
    processor = AutoProcessor.from_pretrained(
        model, trust_remote_code=True, use_fast=True
    )
    tok = processor.tokenizer

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    canonical_word = enc("Describe")
    split_word = enc("D") + enc("escribe")
    drift_delta = len(split_word) - len(canonical_word)
    if drift_delta <= 0:
        raise AssertionError("chosen text split is canonical; no drift to exercise")
    if tok.decode(split_word) != tok.decode(canonical_word):
        raise AssertionError("chosen token split changes the decoded prompt text")

    processor_output = processor(
        text=[image_token + "Describe the picture."],
        images=[image],
        return_tensors="pt",
    )
    input_ids = processor_output["input_ids"][0].tolist()

    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    image_pad_run_lengths = _placeholder_run_lengths(input_ids, image_pad_id)
    if len(image_pad_run_lengths) != 1 or image_pad_run_lengths[0] <= 1:
        raise AssertionError(
            "HF processor did not produce one expanded image placeholder run: "
            f"{image_pad_run_lengths}"
        )

    word_start = _find_subsequence(input_ids, canonical_word)
    input_ids[word_start : word_start + len(canonical_word)] = split_word
    return input_ids, drift_delta


def _prompt_tokens(base_url, input_ids, image):
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "image_data": [image],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["meta_info"]["prompt_tokens"]


class TestQwenVLTokenIdRetokenize(CustomTestCase):
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    other_args = ["--trust-remote-code", "--mem-fraction-static", "0.7"]

    def test_flag_off_drifts_flag_on_does_not(self):
        image = _test_image()
        input_ids, drift_delta = _build_drift_prompt(
            self.model, self.image_token, image
        )
        self.assertGreater(drift_delta, 0, "prompt is canonical; no drift to exercise")
        image_data = _data_uri(image)

        prompt_tokens = {}
        for flag in ("0", "1"):
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self.other_args,
                env={"SGLANG_MM_AVOID_RETOKENIZE": flag},
            )
            try:
                prompt_tokens[flag] = _prompt_tokens(
                    DEFAULT_URL_FOR_TEST, input_ids, image_data
                )
            finally:
                kill_process_tree(process.pid)

        # ON keeps the user's original tokens; OFF loses the drift_delta tokens.
        pt_off, pt_on = prompt_tokens["0"], prompt_tokens["1"]
        self.assertEqual(pt_on - pt_off, drift_delta, f"on={pt_on}, off={pt_off}")


if __name__ == "__main__":
    unittest.main()

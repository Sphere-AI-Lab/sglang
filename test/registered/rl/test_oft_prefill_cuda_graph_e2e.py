"""OFT under the breakable prefill CUDA graph must match eager prefill.

Boots one engine with prefill CUDA graphs enabled and one with them disabled,
loads the same random OFT adapter (all seven dense projections) into both,
and compares prompt logprobs and greedy tokens for base-only, adapter and
mixed base/adapter requests. Both engines run with deterministic
(batch-invariant) inference, under which a no-OFT engine pair is bit-exact
graph-vs-eager on these prompts; without it, bf16 prompt logprobs vary with
batch composition (which the scheduler decides) and, on longer prompts, with
the graph's own attention path. One-element batches are therefore compared
exactly; multi-request calls, which replay through the static segment slots,
are compared on the first greedy token because their split can differ.

Run for both the single-adapter fast path (max_ofts_per_batch=2) and the
segmented kernels (max_ofts_per_batch=3). Under the fast path a mixed
base/adapter batch replays eagerly by design; both configurations still go
through the same comparison.
"""

import os
import unittest

import torch
from transformers import AutoConfig

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")

MODEL_PATH = os.environ.get("SGLANG_OFT_PCG_TEST_MODEL", "Qwen/Qwen3-0.6B")
BLOCK_SIZE = 32
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROMPTS = [
    "Hello, my name is",
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n    if n < 2:\n        return n\n    return',
    "Once upon a time, in a small village nestled between two mountains, there lived a curious child who loved to explore the forests, rivers, and caves nearby. One morning,",
]
SAMPLING = {"temperature": 0, "max_new_tokens": 4}
ADAPTER = "pcg_test_adapter"


def _adapter_tensors(seed: int) -> dict:
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    o_in = cfg.head_dim * cfg.num_attention_heads
    assert (
        hidden % BLOCK_SIZE == 0 and inter % BLOCK_SIZE == 0 and o_in % BLOCK_SIZE == 0
    )
    n = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(cfg.num_hidden_layers):
        for name, in_dim in (
            ("self_attn.q_proj", hidden),
            ("self_attn.k_proj", hidden),
            ("self_attn.v_proj", hidden),
            ("self_attn.o_proj", o_in),
            ("mlp.gate_proj", hidden),
            ("mlp.up_proj", hidden),
            ("mlp.down_proj", inter),
        ):
            tensors[f"model.layers.{layer}.{name}.oft_R"] = (
                torch.randn((in_dim // BLOCK_SIZE, n), generator=g) * 0.02
            )
    return tensors


def _collect(prefill_backend: str, max_ofts: int) -> dict:
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        enable_oft=True,
        oft_impl="sibling",
        max_oft_block_size=BLOCK_SIZE,
        oft_target_modules=TARGETS,
        max_ofts_per_batch=max_ofts,
        mem_fraction_static=0.45,
        cuda_graph_backend_prefill=prefill_backend,
        disable_radix_cache=True,
        enable_deterministic_inference=True,
        log_level="error",
    )
    try:
        result = engine.load_oft_adapter_from_tensors(
            oft_name=ADAPTER,
            tensors=_adapter_tensors(0),
            config_dict={
                "peft_type": "OFT",
                "target_modules": TARGETS,
                "oft_block_size": BLOCK_SIZE,
            },
        )
        assert result.success, result
        modes = {
            "base": [None] * len(PROMPTS),
            "adapter": [ADAPTER] * len(PROMPTS),
            "mixed": [ADAPTER, None, ADAPTER, None],
        }
        out = {}
        for mode, paths in modes.items():
            single = []
            for prompt, path in zip(PROMPTS, paths):
                # One-element batches: deterministic batch composition (the
                # scheduler cannot split them), through the list-form oft_path
                # path that mixed batches use.
                res = engine.generate(
                    [prompt],
                    SAMPLING,
                    return_logprob=True,
                    logprob_start_len=0,
                    oft_path=[path],
                )
                single.append(_summarize(res[0]))
            batched = engine.generate(
                PROMPTS,
                SAMPLING,
                return_logprob=True,
                logprob_start_len=0,
                oft_path=paths,
            )
            out[mode] = {"single": single, "batched": [_summarize(r) for r in batched]}
        return out
    finally:
        engine.shutdown()


def _summarize(res: dict) -> dict:
    meta = res["meta_info"]
    return {
        "out_ids": [t[1] for t in meta["output_token_logprobs"]],
        "in_lp": [t[0] for t in meta["input_token_logprobs"] if t[0] is not None],
    }


class _PrefillCudaGraphOFTParity(CustomTestCase):
    max_ofts = None

    @classmethod
    def setUpClass(cls):
        cls.graph = _collect("breakable", cls.max_ofts)
        cls.eager = _collect("disabled", cls.max_ofts)

    def test_single_requests_match_eager(self):
        for mode in ("base", "adapter", "mixed"):
            for i, (g, e) in enumerate(
                zip(self.graph[mode]["single"], self.eager[mode]["single"])
            ):
                with self.subTest(mode=mode, prompt=i):
                    self.assertEqual(g["out_ids"], e["out_ids"])
                    self.assertEqual(len(g["in_lp"]), len(e["in_lp"]))
                    for p, q in zip(g["in_lp"], e["in_lp"]):
                        self.assertAlmostEqual(p, q, places=3)

    def test_batched_requests_match_eager_first_token(self):
        for mode in ("base", "adapter", "mixed"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    [r["out_ids"][:1] for r in self.graph[mode]["batched"]],
                    [r["out_ids"][:1] for r in self.eager[mode]["batched"]],
                )

    def test_adapter_is_applied_under_the_graph(self):
        effect = max(
            abs(p - q)
            for g_base, g_adapter in zip(
                self.graph["base"]["single"], self.graph["adapter"]["single"]
            )
            for p, q in zip(g_base["in_lp"], g_adapter["in_lp"])
        )
        self.assertGreater(effect, 0.5)


class TestOFTPrefillCudaGraphFastPath(_PrefillCudaGraphOFTParity):
    max_ofts = 2


class TestOFTPrefillCudaGraphSegmented(_PrefillCudaGraphOFTParity):
    max_ofts = 3


del _PrefillCudaGraphOFTParity


if __name__ == "__main__":
    unittest.main()

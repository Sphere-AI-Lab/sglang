"""End-to-end GPU test: MoE expert-OFT multi-tenancy (this plan's Task 5).

1. test_lone_resident_moe_adapter_applies_correct_rotation: regression guard
   -- a single MoE-target OFT adapter resident must still have its rotation
   actually applied (this plan's Tasks 1-4b touch the exact decision/read
   path a lone MoE-target adapter goes through, so this catches a
   regression there even without a pre-recorded baseline: a real, working
   rotation must differ from the unrotated base model). NOTE: despite the
   name this test used to have, this does NOT exercise the pool's single-
   slot "fast path" (``_compute_moe_multi_tenant_slot_ids`` returning
   ``None``) -- this test's own ``_generate()`` base-model call runs FIRST
   and claims buffer slot 0 (``active_idx``) for the base/``None`` request,
   so the adapter loaded afterward always lands at slot >= 1 and every
   generate() call below actually exercises the general multi-tenant read
   path. Genuine fast-path coverage (0 real adapters resident) is exercised
   implicitly by every test's own base-model ``_generate()`` call in this
   file; a "real single adapter actually taking the fast path" scenario is
   currently impossible in this pool configuration (see final-review C1: a
   real dynamically-loaded adapter can never occupy buffer slot 0/active_idx
   in the plain native-RPC pool, so decode CUDA graphs are disabled for this
   configuration entirely -- ``peft/config.py``'s ``validate_peft_args``)
   and will get direct coverage once the ``2026-09-01-oft-moe-cuda-graph-
   dual-capture`` follow-up plan restores a working fast path for real
   adapters.
2. test_two_moe_adapters_apply_correctly: the actual fix -- two
   concurrently-resident MoE-target OFT adapters, each must produce output
   matching what that adapter alone would produce, not a shared/clobbered
   result (the original bug this plan fixes: before Tasks 1-4b, a batch
   referencing two distinct real MoE-target adapters would silently read
   whichever adapter's weights happened to be live on the shared buffer
   last).

Ported from test_oft_moe_multi_tenant_fast_path.py (this plan's Task 4b
fix-round regression test): same tiny synthetic Qwen3-MoE-architecture model
(config/tokenizer only -- ``--load-format dummy`` skips the real ~60GB weight
download, ``--json-model-override-args`` shrinks every dimension), same
sgl.Engine fixture, same native-RPC adapter loading
(``load_oft_adapter_from_tensors``) with real (large-scale, non-identity)
expert-target rotation payloads, and the same per-token-logprob comparison
technique -- greedy TEXT alone is too coarse a signal with ``--load-format
dummy``'s random/untrained weights (the model can collapse into a degenerate
repeated-token argmax loop insensitive to a real rotation), so per-token
logprobs (continuous floats straight off the model's own logits) are used
instead. setUpClass/tearDownClass and the general adapter-load/generate
calling convention additionally follow test_oft_load_from_tensor.py (the
native-RPC OFT test template both files share).

Test 2's "same output regardless of resident/concurrent context" comparison
uses EXACT equality on the full per-token logprob list (like
test_multi_adapter_concurrent_residency's exact text comparison for the
analogous "same adapter, different context" property in
test_oft_load_from_tensor.py) rather than a numeric tolerance: empirically,
this model's per-token logprobs for a genuinely different rotation (e.g.
base vs. adapted, or adapter A vs. adapter B) differ from each other by only
~1e-5 (this tiny synthetic model's untrained random weights produce a
near-uniform softmax over the full vocabulary, so even a large injected
rotation moves logprobs only slightly) -- a tolerance loose enough to be
CI-safe against any run-to-run kernel nondeterminism would also be loose
enough to silently accept the original clobbering bug (which substitutes a
same-order-of-magnitude *different* rotation, not a wildly different one).
Exact equality is also what was actually observed running this test
end-to-end: adapter A's concurrent-batch logprobs were bit-identical to its
isolated-load logprobs.
"""

import json
import unittest

import torch

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")

MODEL_PATH = "Qwen/Qwen3-30B-A3B"  # architecture only -- --load-format dummy
BLOCK_SIZE = 32
NUM_LAYERS = 2
HIDDEN_SIZE = 256
NUM_EXPERTS = 4
MOE_INTERMEDIATE_SIZE = 64
MOE_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
TEST_PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 16

MODEL_OVERRIDE = {
    "num_hidden_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    # head_dim=64 (a standard size the fused rotary-embedding kernel
    # supports) -- head_dim=16 hit a "fallback_rotary_embedding"/KV-cache
    # assertion in that kernel during test_oft_moe_multi_tenant_fast_path.py's
    # initial iteration.
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 512,
    "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": 2,
}

ENGINE_KWARGS = dict(
    model_path=MODEL_PATH,
    load_format="dummy",
    json_model_override_args=json.dumps(MODEL_OVERRIDE),
    peft_method="oft",
    oft_impl="sibling",
    max_oft_block_size=BLOCK_SIZE,
    oft_target_modules=MOE_TARGET_MODULES,
    max_loaded_ofts=4,
    max_ofts_per_batch=4,
    mem_fraction_static=0.6,
    # KNOWN LIMITATION (disclosed in _compute_moe_multi_tenant_slot_ids's
    # docstring): multi-tenant MoE OFT is not yet safe under CUDA-graph-
    # replayed decode. Both tests below exercise the eager multi-tenant MoE
    # OFT path, so this must not exercise that separate, already-disclosed
    # gap.
    disable_cuda_graph=True,
    log_level="error",
)


def _moe_config_dict() -> dict:
    return {
        "peft_type": "OFT",
        "target_modules": MOE_TARGET_MODULES,
        "oft_block_size": BLOCK_SIZE,
    }


def _compact(num_blocks: int, generator: torch.Generator) -> torch.Tensor:
    n_elements = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    # Scale 5.0 (a large rotation angle), not the small ~0.02 scale other OFT
    # test fixtures use: --load-format dummy gives every weight tiny random
    # (untrained) values, and this tiny synthetic model's logits collapse to
    # a near-degenerate, extremely peaked distribution (observed in
    # test_oft_moe_multi_tenant_fast_path.py: identical greedy output AND
    # bit-identical logprobs regardless of a small-scale rotation). A large
    # rotation is needed for its effect to be numerically visible above that
    # degenerate baseline. bfloat16 matches the model's own dtype
    # (config.torch_dtype) -- the pool's expert-OFT buffers are allocated in
    # that dtype.
    return (torch.randn((num_blocks, n_elements), generator=generator) * 5.0).to(
        torch.bfloat16
    )


def _expert_named_tensors(seed: int) -> dict:
    """MoE-target adapter payload: gate_proj/up_proj/down_proj.oft_R for
    every expert, every layer, with real (non-identity) random rotations."""
    generator = torch.Generator().manual_seed(seed)
    hidden_blocks = HIDDEN_SIZE // BLOCK_SIZE
    inter_blocks = MOE_INTERMEDIATE_SIZE // BLOCK_SIZE
    tensors = {}
    for layer_idx in range(NUM_LAYERS):
        for expert_id in range(NUM_EXPERTS):
            for proj, num_blocks in (
                ("gate_proj", hidden_blocks),
                ("up_proj", hidden_blocks),
                ("down_proj", inter_blocks),
            ):
                name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{proj}.oft_R"
                tensors[name] = _compact(num_blocks, generator)
    return tensors


class TestMoeMultiTenantEndToEnd(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(**ENGINE_KWARGS)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _generate(self, adapter_name=None):
        """Single-request generate. Returns (text, per-token output
        logprobs) -- see module docstring for why logprobs (not text) are
        this file's comparison signal."""
        output = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            adapter_path=[adapter_name] if adapter_name is not None else None,
            return_logprob=True,
        )
        text = output[0]["text"]
        logprobs = [
            entry[0] for entry in output[0]["meta_info"]["output_token_logprobs"]
        ]
        return text, logprobs

    def _generate_concurrent(self, adapter_name_a: str, adapter_name_b: str):
        """Two requests in ONE engine.generate() call -- one naming
        adapter_name_a, one naming adapter_name_b, same prompt -- so both
        land in the SAME scheduler batch/forward pass (the exact scenario
        the original bug clobbered: a batch referencing two distinct real
        MoE-target adapter slots at once). Returns ((text_a, logprobs_a),
        (text_b, logprobs_b))."""
        output = self.engine.generate(
            prompt=[TEST_PROMPT, TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            adapter_path=[adapter_name_a, adapter_name_b],
            return_logprob=True,
        )
        results = []
        for entry in output:
            text = entry["text"]
            logprobs = [
                lp_entry[0] for lp_entry in entry["meta_info"]["output_token_logprobs"]
            ]
            results.append((text, logprobs))
        return results[0], results[1]

    def test_lone_resident_moe_adapter_applies_correct_rotation(self):
        """A single resident MoE-target OFT adapter's rotation must still
        apply correctly. NOTE: this is NOT fast-path coverage -- the
        ``_generate()`` call below issues a base (adapter_name=None) request
        FIRST, which claims buffer slot 0 (active_idx) for the base/None
        request, so the adapter loaded afterward lands at slot >= 1 and this
        test actually exercises the general multi-tenant read path (see the
        module docstring's NOTE for why a real single-adapter fast-path
        scenario is currently impossible in this pool configuration)."""
        print(
            "[Test]Testing that a single resident MoE-target OFT adapter's "
            "rotation applies correctly..."
        )
        base_text, base_logprobs = self._generate()

        name = "lone_resident_adapter"
        result = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name,
            tensors=_expert_named_tensors(seed=1),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            result.success, f"Failed to load MoE-target adapter: {result.error_message}"
        )

        adapter_text, adapter_logprobs = self._generate(adapter_name=name)
        print(f"[Without OFT] {base_text} logprobs={base_logprobs}")
        print(
            f"[With single MoE-target adapter] {adapter_text} logprobs={adapter_logprobs}"
        )
        self.assertNotEqual(
            base_logprobs,
            adapter_logprobs,
            "Generation with a single resident MoE-target OFT adapter "
            "produced IDENTICAL per-token logprobs to the unadapted base "
            "output -- its expert-OFT rotation was silently skipped. This "
            "is a regression guard for this plan's Tasks 1-4b: the "
            "single-adapter code path must remain correct even though "
            "every other change in this plan targets the "
            ">1-resident-adapter case.",
        )

        # Regression guard (ported from test_oft_load_from_tensor.py's
        # test_fresh_load_and_generate): unload must complete promptly, and
        # leaves this adapter's buffer slot free for the next test.
        unload_result = self.engine.unload_oft_adapter(name)
        self.assertTrue(
            unload_result.success,
            f"Failed to unload MoE-target adapter after generate(): "
            f"{unload_result.error_message}",
        )

    def test_two_moe_adapters_apply_correctly(self):
        print(
            "[Test]Testing that two concurrently-resident MoE-target OFT "
            "adapters each produce correct, distinct output (the actual "
            "multi-tenancy fix)..."
        )
        name_a, name_b = "moe_adapter_a", "moe_adapter_b"

        # (a) Adapter A alone.
        result_a = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name_a,
            tensors=_expert_named_tensors(seed=10),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            result_a.success, f"Failed to load adapter A: {result_a.error_message}"
        )
        text_a_alone, logprobs_a_alone = self._generate(adapter_name=name_a)

        unload_a_result = self.engine.unload_oft_adapter(name_a)
        self.assertTrue(
            unload_a_result.success,
            f"Failed to unload adapter A: {unload_a_result.error_message}",
        )

        # (b) Adapter B alone -- A is unloaded, so B is the sole resident
        # real adapter here.
        result_b = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name_b,
            tensors=_expert_named_tensors(seed=20),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            result_b.success, f"Failed to load adapter B: {result_b.error_message}"
        )
        text_b_alone, logprobs_b_alone = self._generate(adapter_name=name_b)

        # Sanity: if these two "different" adapters happened to produce
        # identical output in isolation, the concurrent-residency comparison
        # below would prove nothing.
        self.assertNotEqual(
            logprobs_a_alone,
            logprobs_b_alone,
            "Adapter A and adapter B produced identical per-token logprobs "
            "in isolation -- these test adapters are not actually "
            "different, so the concurrent-residency comparison below "
            "cannot prove anything.",
        )

        # (c) Reload A so both A and B are concurrently resident, then issue
        # ONE engine.generate() call with two requests -- one naming A, one
        # naming B -- so both land in the same forward batch.
        reload_a_result = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name_a,
            tensors=_expert_named_tensors(seed=10),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            reload_a_result.success,
            f"Failed to reload adapter A: {reload_a_result.error_message}",
        )

        (text_a_concurrent, logprobs_a_concurrent), (
            text_b_concurrent,
            logprobs_b_concurrent,
        ) = self._generate_concurrent(name_a, name_b)

        print(f"[A alone] {text_a_alone} logprobs={logprobs_a_alone}")
        print(f"[B alone] {text_b_alone} logprobs={logprobs_b_alone}")
        print(
            f"[A concurrent w/ B] {text_a_concurrent} logprobs={logprobs_a_concurrent}"
        )
        print(
            f"[B concurrent w/ A] {text_b_concurrent} logprobs={logprobs_b_concurrent}"
        )

        self.assertEqual(
            logprobs_a_concurrent,
            logprobs_a_alone,
            "The request naming adapter A, issued in the SAME batch as a "
            "request naming adapter B, produced different per-token "
            "logprobs than adapter A alone -- adapter A's rotation was "
            "clobbered by adapter B's concurrent residency (the original "
            "multi-tenancy bug this plan fixes).",
        )
        self.assertEqual(
            logprobs_b_concurrent,
            logprobs_b_alone,
            "The request naming adapter B, issued in the SAME batch as a "
            "request naming adapter A, produced different per-token "
            "logprobs than adapter B alone -- adapter B's rotation was "
            "clobbered by adapter A's concurrent residency (the original "
            "multi-tenancy bug this plan fixes).",
        )


if __name__ == "__main__":
    unittest.main()

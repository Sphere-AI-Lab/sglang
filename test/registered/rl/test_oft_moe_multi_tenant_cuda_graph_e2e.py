"""End-to-end GPU test: MoE expert-OFT multi-tenancy under CUDA-graph decode
(this plan's Task 5).

1. test_single_adapter_decode_graph_unaffected: regression guard -- a single
   MoE-target OFT adapter resident, CUDA graphs enabled, must produce
   identical output to before this plan (the fast path must be untouched).
2. test_two_adapters_correct_under_cuda_graph_replay: the actual fix -- two
   concurrently-resident MoE-target OFT adapters, CUDA graphs enabled
   (NOT disabled, unlike the base plan's own e2e test), decode replay must
   apply each adapter's own rotation correctly -- this is exactly the
   scenario that silently produced wrong output before this plan.

Ported from test_oft_moe_multi_tenant_e2e.py (the base MoE-multi-tenancy
plan's own Task 5 e2e test): same tiny synthetic Qwen3-MoE-architecture model
(config/tokenizer only -- ``--load-format dummy`` skips the real weight
download, ``--json-model-override-args`` shrinks every dimension), same
sgl.Engine fixture, same native-RPC adapter loading
(``load_oft_adapter_from_tensors``) with real (large-scale, non-identity)
expert-target rotation payloads, and the same per-token-logprob comparison
technique (see that file's module docstring for why: greedy TEXT alone is too
coarse a signal with ``--load-format dummy``'s random/untrained weights).

THE ONE DELIBERATE DIFFERENCE from that file: ``ENGINE_KWARGS`` here does NOT
set ``disable_cuda_graph=True``. The base plan's own e2e test deliberately
disabled CUDA graphs to avoid the exact bug this plan (2026-09-01-oft-moe-
cuda-graph-dual-capture) fixes -- this file's whole point is to prove decode
CUDA graphs now work correctly for this configuration. Confirmed (see this
plan's Task 5 report) that oft/config.py's validate_oft_args decode-graph
guard does NOT disable decode CUDA graphs for this config: effective adapter
capacity is max_ofts_per_batch - 1 = 3 >= 1, and decode CUDA-graph capture's
default batch-size bucket list ([1, 2, 4, 8, 12, ...],
server_args.py's _generate_decode_cuda_graph_batch_sizes) includes both
batch size 1 (test 1's lone request) and batch size 2 (test 2's two
concurrent requests) without any extra engine kwargs. MAX_NEW_TOKENS is large
enough that each request's generation runs many decode steps past the first
(prefill/extend'd) token, so decode-phase CUDA graphs are actually captured
AND replayed multiple times per test, not merely captured once.
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
# Large enough that each request runs well past prefill/extend into several
# decode-phase CUDA-graph replays (not just one capture) -- OFT already
# disables prefill CUDA graphs entirely (a separate, pre-existing
# restriction), so only a multi-decode-step generation actually exercises
# the decode-graph dual-capture mechanism this plan adds.
MAX_NEW_TOKENS = 24

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
    enable_oft=True,
    oft_impl="sibling",
    max_oft_block_size=BLOCK_SIZE,
    oft_target_modules=MOE_TARGET_MODULES,
    max_loaded_ofts=4,
    max_ofts_per_batch=4,
    mem_fraction_static=0.6,
    # THE deliberate difference from test_oft_moe_multi_tenant_e2e.py: no
    # disable_cuda_graph=True here. Decode CUDA graphs stay enabled (the
    # default) -- see module docstring for why validate_oft_args's guard
    # does not disable them for this config (max_ofts_per_batch=4 gives
    # effective adapter capacity 3 >= 1), which is exactly the configuration
    # this plan's dual-capture mechanism (oft_manager.py's
    # _compute_moe_multi_tenant_slot_ids, decode_cuda_graph_runner.py's
    # _resolve_oft_variant/record_oft_variant_graph) makes safe.
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


class TestMoeMultiTenantCudaGraphEndToEnd(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(**ENGINE_KWARGS)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _generate(self, oft_name=None):
        """Single-request generate. Returns (text, per-token output
        logprobs) -- see module docstring for why logprobs (not text) are
        this file's comparison signal."""
        output = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            oft_path=[oft_name] if oft_name is not None else None,
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
        MoE-target adapter slots at once, replayed off the SAME captured
        decode CUDA graph). Returns ((text_a, logprobs_a), (text_b,
        logprobs_b))."""
        output = self.engine.generate(
            prompt=[TEST_PROMPT, TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            oft_path=[adapter_name_a, adapter_name_b],
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

    def test_single_adapter_decode_graph_unaffected(self):
        """A single resident MoE-target OFT adapter's rotation must still
        apply correctly with decode CUDA graphs enabled -- the existing
        single-slot fast path must remain correct and must not have been
        slowed or broken by this plan's capture-loop/dual-capture changes.
        MAX_NEW_TOKENS is large enough that this generate() call runs
        multiple decode-phase CUDA-graph replays, not just one."""
        print(
            "[Test]Testing that a single resident MoE-target OFT adapter's "
            "rotation applies correctly under decode CUDA-graph replay..."
        )
        base_text, base_logprobs = self._generate()

        name = "lone_resident_adapter"
        result = self.engine.load_oft_adapter_from_tensors(
            oft_name=name,
            tensors=_expert_named_tensors(seed=1),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            result.success, f"Failed to load MoE-target adapter: {result.error_message}"
        )

        adapter_text, adapter_logprobs = self._generate(oft_name=name)
        print(f"[Without OFT] {base_text} logprobs={base_logprobs}")
        print(
            f"[With single MoE-target adapter] {adapter_text} logprobs={adapter_logprobs}"
        )
        self.assertNotEqual(
            base_logprobs,
            adapter_logprobs,
            "Generation with a single resident MoE-target OFT adapter under "
            "decode CUDA-graph replay produced IDENTICAL per-token logprobs "
            "to the unadapted base output -- its expert-OFT rotation was "
            "silently skipped. This is a regression guard for this plan's "
            "capture-loop/dual-capture changes: the single-adapter code path "
            "must remain correct even though this plan targets the "
            ">1-resident-adapter case.",
        )

        # Repeatability: replaying the SAME captured decode graph again
        # (a second generate() call against the same resident adapter) must
        # reproduce identical output -- proving the graph itself, not just a
        # one-off eager-mode-lucky pass, is what is being exercised.
        adapter_text_again, adapter_logprobs_again = self._generate(oft_name=name)
        self.assertEqual(
            adapter_logprobs,
            adapter_logprobs_again,
            "Re-issuing the same request against the same lone resident "
            "adapter produced different per-token logprobs on a second "
            "decode-CUDA-graph replay -- the captured graph is not "
            "deterministic/stable across replays.",
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

    def test_two_adapters_correct_under_cuda_graph_replay(self):
        """Port of the base plan's own Task 5 two-adapter assertion
        (test_two_moe_adapters_apply_correctly), but with decode CUDA graphs
        ENABLED: before this plan, a batch referencing two distinct real
        MoE-target adapters at once would silently read whichever adapter's
        rotation the single-slot decode graph happened to have captured --
        this is exactly that scenario, and it must now pass."""
        print(
            "[Test]Testing that two concurrently-resident MoE-target OFT "
            "adapters each produce correct, distinct output under decode "
            "CUDA-graph replay (the actual multi-tenancy fix)..."
        )
        name_a, name_b = "moe_adapter_a", "moe_adapter_b"

        # (a) Adapter A alone.
        result_a = self.engine.load_oft_adapter_from_tensors(
            oft_name=name_a,
            tensors=_expert_named_tensors(seed=10),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            result_a.success, f"Failed to load adapter A: {result_a.error_message}"
        )
        text_a_alone, logprobs_a_alone = self._generate(oft_name=name_a)

        unload_a_result = self.engine.unload_oft_adapter(name_a)
        self.assertTrue(
            unload_a_result.success,
            f"Failed to unload adapter A: {unload_a_result.error_message}",
        )

        # (b) Adapter B alone -- A is unloaded, so B is the sole resident
        # real adapter here.
        result_b = self.engine.load_oft_adapter_from_tensors(
            oft_name=name_b,
            tensors=_expert_named_tensors(seed=20),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            result_b.success, f"Failed to load adapter B: {result_b.error_message}"
        )
        text_b_alone, logprobs_b_alone = self._generate(oft_name=name_b)

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
        # naming B -- so both land in the same forward batch (batch size 2,
        # within the default decode CUDA-graph capture bucket list), and
        # decode replay must apply each request's own adapter correctly.
        reload_a_result = self.engine.load_oft_adapter_from_tensors(
            oft_name=name_a,
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
            "request naming adapter B under decode CUDA-graph replay, "
            "produced different per-token logprobs than adapter A alone -- "
            "adapter A's rotation was clobbered by adapter B's concurrent "
            "residency (the original multi-tenancy-under-CUDA-graphs bug "
            "this plan fixes).",
        )
        self.assertEqual(
            logprobs_b_concurrent,
            logprobs_b_alone,
            "The request naming adapter B, issued in the SAME batch as a "
            "request naming adapter A under decode CUDA-graph replay, "
            "produced different per-token logprobs than adapter B alone -- "
            "adapter B's rotation was clobbered by adapter A's concurrent "
            "residency (the original multi-tenancy-under-CUDA-graphs bug "
            "this plan fixes).",
        )


if __name__ == "__main__":
    unittest.main()

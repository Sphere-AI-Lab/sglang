"""GPU end-to-end regression test for a Task 4b review finding: a single
MoE-target adapter resident at a buffer slot other than the memory pool's
``active_idx`` must still have its expert-OFT rotation actually applied on a
batch that references only that adapter.

Background: Task 4b fixed the expert-OFT streamed-weight write path so each
dynamically-loaded adapter writes into its OWN assigned buffer slot (not
always ``active_idx``/slot 0). That fix exposed a DIFFERENT existing bug:
``OFTManager._compute_moe_multi_tenant_slot_ids`` returned ``None`` (the
fast/single-slot read path) whenever a batch referenced at most one distinct
non-zero slot, regardless of WHICH slot that was -- but the fast-path READ
side (``oft_moe_runners.py``, when ``slot_ids is None``) always reads a fixed
module-attribute view bound to ``active_idx`` once at boot, never refreshed
per adapter. Before Task 4b's write-path fix, every adapter's real weights
silently landed at ``active_idx`` anyway, so this was accidentally correct.
After that fix, a real adapter's data sits at its OWN slot -- so a single
resident MoE-target adapter (which, in the native-RPC multi-tenant path,
never occupies buffer slot 0 -- that slot is always the base/identity
placeholder) would have its rotation silently skipped: the fast path reads
identity from slot 0 instead of that adapter's real rotation from its own
slot.

Drives the REAL forward path end to end: a real multiprocess sgl.Engine,
real native-RPC adapter loads (``load_oft_adapter_from_tensors``), and real
``generate()`` calls -- exercising ``FusedMoEWithOFT.forward`` /
``oft_moe_runners.py``'s actual dispatch, not a hand-constructed check of
pool storage (see test_oft_moe_expert_write_path.py for that level of
coverage of Task 4b's original two fixes).

Uses a tiny synthetic Qwen3-MoE-architecture model (config/tokenizer only --
``--load-format dummy`` skips the real ~60GB weight download entirely, and
``--json-model-override-args`` shrinks every dimension) so this boots and
runs in roughly the same time as the existing dense e2e OFT test
(test_oft_load_from_tensor.py), which this file's structure mirrors.
"""

import json
import unittest

import torch

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

MODEL_PATH = "Qwen/Qwen3-30B-A3B"  # architecture only -- --load-format dummy
BLOCK_SIZE = 32
NUM_LAYERS = 2
HIDDEN_SIZE = 256
NUM_EXPERTS = 4
MOE_INTERMEDIATE_SIZE = 64
# "o_proj" (dense, standalone -- not fused with q/k/v like "q_proj" would be,
# see oft/mem_pool.py's MERGED_OFT_PROJ_GROUPS) for the dense-only adapter;
# gate/up/down for the MoE-target adapter. Every layer is MoE in this
# architecture (decoder_sparse_step=1, no mlp_only_layers), so there is no
# separate dense down_proj to collide with the per-expert one.
DENSE_TARGET_MODULE = "o_proj"
MOE_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
TEST_PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 16

MODEL_OVERRIDE = {
    "num_hidden_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    # head_dim=64 (a standard size the fused rotary-embedding kernel
    # supports) -- head_dim=16 hit a "fallback_rotary_embedding"/KV-cache
    # assertion in that kernel during initial iteration on this test.
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
    oft_target_modules=[DENSE_TARGET_MODULE] + MOE_TARGET_MODULES,
    max_loaded_ofts=4,
    max_ofts_per_batch=4,
    mem_fraction_static=0.6,
    # KNOWN LIMITATION (disclosed in _compute_moe_multi_tenant_slot_ids's
    # docstring): multi-tenant MoE OFT is not yet safe under CUDA-graph-
    # replayed decode. This test is specifically about the eager multi-
    # tenant MoE OFT path, so it must not exercise that separate, already-
    # disclosed gap.
    disable_cuda_graph=True,
    log_level="error",
)


def _dense_config_dict() -> dict:
    return {
        "peft_type": "OFT",
        "target_modules": [DENSE_TARGET_MODULE],
        "oft_block_size": BLOCK_SIZE,
    }


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
    # a near-degenerate, extremely peaked distribution (observed: identical
    # greedy output AND bit-identical logprobs regardless of a small-scale
    # rotation). A large rotation is needed for its effect to be numerically
    # visible above that degenerate baseline; a real trained checkpoint
    # would not need this (see test_oft_load_from_tensor.py's 0.02 scale).
    # bfloat16 matches the model's own dtype (config.torch_dtype) -- the
    # pool's expert-OFT buffers are allocated in that dtype, and
    # apply_streamed_expert_oft's buffer-reuse validation rejects a dtype
    # mismatch (refusing to silently replace a CUDA-graph-captured buffer).
    return (torch.randn((num_blocks, n_elements), generator=generator) * 5.0).to(
        torch.bfloat16
    )


def _dense_named_tensors(seed: int) -> dict:
    """Dense-only adapter payload: o_proj.oft_R for every layer, NO expert
    tensors at all -- this adapter must never touch the expert-OFT groups."""
    generator = torch.Generator().manual_seed(seed)
    num_blocks = HIDDEN_SIZE // BLOCK_SIZE
    return {
        f"model.layers.{layer_idx}.self_attn.{DENSE_TARGET_MODULE}.oft_R": _compact(
            num_blocks, generator
        )
        for layer_idx in range(NUM_LAYERS)
    }


def _expert_named_tensors(seed: int) -> dict:
    """MoE-target adapter payload: gate_proj/up_proj/down_proj.oft_R for
    every expert, every layer -- lands in this adapter's OWN buffer slot per
    Task 4b's Fix 1, with real (non-identity) random rotations."""
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


class TestMoeMultiTenantFastPathRegression(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(**ENGINE_KWARGS)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _generate(self, oft_name=None):
        """Returns (text, per-token output logprobs). With --load-format
        dummy (random, untrained weights), greedy-decoded TEXT alone is too
        coarse a signal -- a random model can land in a degenerate repeated-
        token argmax loop that a small rotation doesn't reliably escape, even
        when it genuinely perturbs every forward pass. Per-token logprobs are
        continuous floats straight from the model's own (rotated or not)
        logits, so they change under ANY real perturbation, even one that
        doesn't flip the greedy argmax choice."""
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

    def test_single_moe_adapter_not_at_active_idx_applies_rotation(self):
        print(
            "[Test]Testing that a lone resident MoE-target adapter's "
            "rotation actually applies (multi-tenant fast-path regression)..."
        )
        base_text, base_logprobs = self._generate()

        # Loaded first: a dense-only adapter (o_proj). Its own buffer slot
        # holds real dense rotation data but IDENTITY expert-OFT buffers
        # (Task 4b's Fix 2) -- it never touches the expert groups at all.
        dense_name = "dense_only_adapter"
        dense_result = self.engine.load_oft_adapter_from_tensors(
            oft_name=dense_name,
            tensors=_dense_named_tensors(seed=1),
            config_dict=_dense_config_dict(),
        )
        self.assertTrue(
            dense_result.success,
            f"Failed to load dense-only adapter: {dense_result.error_message}",
        )

        # Loaded second: a REAL MoE-target adapter. It lands in a DIFFERENT
        # buffer slot than the dense adapter -- and, per the native-RPC
        # admission path's own invariant (buffer slot 0 is always the base/
        # identity placeholder, never assigned to a real adapter), neither
        # adapter's slot can be active_idx (0).
        moe_name = "moe_target_adapter"
        moe_result = self.engine.load_oft_adapter_from_tensors(
            oft_name=moe_name,
            tensors=_expert_named_tensors(seed=2),
            config_dict=_moe_config_dict(),
        )
        self.assertTrue(
            moe_result.success,
            f"Failed to load MoE-target adapter: {moe_result.error_message}",
        )

        # The batch below references ONLY the MoE-target adapter: exactly
        # one distinct real slot in the whole batch (the dense adapter is
        # resident but not referenced by any request here) -- but that slot
        # is NOT active_idx, so the fast path must NOT be taken.
        moe_text, moe_logprobs = self._generate(oft_name=moe_name)
        print(f"[Without OFT] {base_text} logprobs={base_logprobs}")
        print(
            f"[With sole resident MoE-target adapter] {moe_text} logprobs={moe_logprobs}"
        )
        self.assertNotEqual(
            base_logprobs,
            moe_logprobs,
            "Generation with the sole resident MoE-target adapter produced "
            "IDENTICAL per-token logprobs to the unadapted base output -- "
            "its expert-OFT rotation was silently skipped. This is the "
            "exact regression: _compute_moe_multi_tenant_slot_ids's fast "
            "path incorrectly treated 'at most one distinct real slot' as "
            "always safe, even though that slot isn't the pool's "
            "active_idx (the only slot the fast read path ever looks at).",
        )


if __name__ == "__main__":
    unittest.main()

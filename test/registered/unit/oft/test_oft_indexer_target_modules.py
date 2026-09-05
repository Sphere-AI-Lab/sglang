"""Unit tests for DSA indexer OFT target-module resolution (srt/oft/utils.py).

The DSA indexer's wq_b/wk/weights_proj are qualified with their "indexer."
parent so they don't collide with unrelated bare-name modules elsewhere (most
notably DeepSeek V4 attention's own "wq_b"). These tests pin down that
disambiguation so it can't silently regress back into a collision.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.oft.utils import get_hidden_dim, get_normalized_target_modules, get_target_module_name
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGetNormalizedTargetModules(unittest.TestCase):
    def test_bare_wq_b_is_not_remapped_to_indexer(self):
        # Unlike LoRA, OFT already uses bare "wq_b" for DeepSeek V4
        # attention's own query projection. If this were ever remapped to
        # "indexer.wq_b" (mirroring LoRA's params_mapping verbatim),
        # existing V4 attention OFT targeting would silently break.
        self.assertEqual(get_normalized_target_modules(["wq_b"]), {"wq_b"})

    def test_bare_indexer_leaves_map_to_qualified_form(self):
        self.assertEqual(
            get_normalized_target_modules(["wk", "weights_proj"]),
            {"indexer.wk", "indexer.weights_proj"},
        )

    def test_qualified_indexer_wq_b_passes_through_unchanged(self):
        # Without the DSA_INDEXER_OFT_NAMES passthrough, the generic
        # `name.split(".")[-1]` stripping would collapse this back to bare
        # "wq_b", making the indexer's query projection unreachable.
        self.assertEqual(
            get_normalized_target_modules(["indexer.wq_b"]), {"indexer.wq_b"}
        )

    def test_v4_attention_and_indexer_wq_b_coexist_when_both_requested(self):
        result = get_normalized_target_modules(["wq_b", "indexer.wq_b"])
        self.assertEqual(result, {"wq_b", "indexer.wq_b"})


class TestGetTargetModuleName(unittest.TestCase):
    def setUp(self):
        self.targets = {"wq_b", "indexer.wq_b"}

    def test_qualified_name_wins_for_indexer_module_path(self):
        # Both "wq_b" and "indexer.wq_b" substring-match this path; the
        # longer, more specific candidate must win so the indexer's buffer
        # group doesn't get resolved under the V4-attention shape spec.
        resolved = get_target_module_name(
            "model.layers.0.self_attn.indexer.wq_b", self.targets
        )
        self.assertEqual(resolved, "indexer.wq_b")

    def test_bare_name_still_resolves_for_non_indexer_module_path(self):
        resolved = get_target_module_name(
            "model.layers.0.self_attn.wq_b", self.targets
        )
        self.assertEqual(resolved, "wq_b")


class TestGetHiddenDimForIndexer(unittest.TestCase):
    def setUp(self):
        # Mirrors dsv4's C4Indexer construction: wq_b is
        # ReplicatedLinear(q_lora_rank, n_heads * head_dim) and weights_proj
        # is ReplicatedLinear(hidden_size, n_heads).
        self.config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            hidden_size=4096,
            q_lora_rank=1024,
            index_n_heads=64,
            index_head_dim=64,
            kv_lora_rank=None,
            # get_hidden_dim's head_dim fallback is computed unconditionally
            # before any module-name branch, so this must be present even
            # though the indexer branch doesn't use it.
            num_attention_heads=32,
        )
        self.base_model = SimpleNamespace()

    def test_indexer_wq_b_shape(self):
        self.assertEqual(
            get_hidden_dim("indexer.wq_b", self.config, self.base_model, layer_idx=0),
            (1024, 4096),
        )

    def test_indexer_wk_shape(self):
        self.assertEqual(
            get_hidden_dim("indexer.wk", self.config, self.base_model, layer_idx=0),
            (4096, 64),
        )

    def test_indexer_weights_proj_shape(self):
        self.assertEqual(
            get_hidden_dim(
                "indexer.weights_proj", self.config, self.base_model, layer_idx=0
            ),
            (4096, 64),
        )


if __name__ == "__main__":
    unittest.main()

"""TP and MoE coverage for native two-phase LoRA updates."""

import json
import os
import unittest

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
)
from sglang.test.test_utils import CustomTestCase

from test_lora_staged_update import (
    LORA_REPO,
    StagedLoRATestHarness,
    _versioned_tensors,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="4-gpu-h100")


def _load_adapter(repo_id):
    adapter_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(
        os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"
    ) as config_file:
        config = json.load(config_file)
    return tensors, config


class TestStagedLoRAUpdateTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)
        cls.adapter, cls.adapter_config = _load_adapter(LORA_REPO)

    def test_tp2(self):
        v1 = _versioned_tensors(self.adapter, 1)
        v2 = _versioned_tensors(self.adapter, 2)
        tp2 = StagedLoRATestHarness(
            self,
            base_gpu_id=1,
            tp_size=2,
            max_loras_per_batch=1,
        )
        try:
            tp2.stage("policy-a", 1, v1, self.adapter_config)
            tp2.activate("policy-a", 1)
            before = tp2.generate(adapter="policy-a")
            tp2.stage("policy-a", 2, v2, self.adapter_config)
            self.assertEqual(tp2.generate(adapter="policy-a"), before)
            tp2.activate("policy-a", 2)
            tp2_v2 = tp2.generate(adapter="policy-a")
            self.assertNotEqual(tp2_v2, before)
        finally:
            tp2.close()

        reference = StagedLoRATestHarness(
            self,
            # Keep the reference on the same TP layout. The synthetic v2
            # weights intentionally amplify tiny TP rounding differences, so
            # a TP=1 greedy token sequence is not a stable correctness oracle.
            base_gpu_id=1,
            tp_size=2,
            max_loras_per_batch=1,
        )
        try:
            reference.stage("policy-a", 2, v2, self.adapter_config)
            reference.activate("policy-a", 2)
            self.assertEqual(reference.generate(adapter="policy-a"), tp2_v2)
        finally:
            reference.close()

    def test_moe_sharded_placement(self):
        adapter, adapter_config = _load_adapter(MOE_LORA_PATH)
        v1 = _versioned_tensors(adapter, 1)
        v2 = _versioned_tensors(adapter, 2)
        prompt = MOE_LORA_TEST_PROMPTS[0]
        harness = StagedLoRATestHarness(
            self,
            model_path=MOE_BASE_MODEL_PATH,
            base_gpu_id=1,
            tp_size=2,
            max_loras_per_batch=1,
        )
        try:
            harness.stage("moe-policy", 1, v1, adapter_config)
            harness.activate("moe-policy", 1)
            before = harness.generate(adapter="moe-policy", prompt=prompt)
            harness.stage("moe-policy", 2, v2, adapter_config)
            self.assertEqual(
                harness.generate(adapter="moe-policy", prompt=prompt), before
            )
            harness.activate("moe-policy", 2)
            self.assertNotEqual(
                harness.generate(adapter="moe-policy", prompt=prompt), before
            )
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()

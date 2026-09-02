"""Regression coverage for graceful streamed-OFT update rejection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.oft.streamed_weight_loader import load_streamed_oft_adapter

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStreamedOFTUpdateErrors(CustomTestCase):
    def test_slot_validation_value_error_is_returned_to_the_caller(self):
        model_runner = SimpleNamespace()

        with patch(
            "sglang.srt.oft.streamed_weight_loader."
            "_ensure_streaming_oft_adapter_slot",
            side_effect=ValueError("invalid adapter slot"),
        ):
            result = load_streamed_oft_adapter(
                model_runner,
                named_tensors=[],
                adapter_config={},
                adapter_name="policy",
            )

        self.assertEqual(result, (False, "invalid adapter slot"))

    def test_dsv4_payload_without_fused_moe_target_is_returned_to_the_caller(self):
        memory_pool = SimpleNamespace(tp_rank=0, R_buffer={})
        oft_manager = SimpleNamespace(
            memory_pool=memory_pool,
            adapter_modules={},
            _find_fused_moe_modules=lambda: [],
        )
        model_runner = SimpleNamespace(oft_manager=oft_manager)
        dsv4_expert_chunk = {
            0: {
                0: {
                    "model.layers.0.mlp.experts.0.down_proj.oft_R": torch.empty(0)
                }
            }
        }

        with (
            patch(
                "sglang.srt.oft.streamed_weight_loader."
                "_ensure_streaming_oft_adapter_slot",
                return_value=(0, 32),
            ),
            patch(
                "sglang.srt.oft.streamed_weight_loader."
                "_partition_expert_oft_tensors",
                return_value=({}, dsv4_expert_chunk, []),
            ),
        ):
            result = load_streamed_oft_adapter(
                model_runner,
                named_tensors=[],
                adapter_config={"oft_block_size": 32},
                adapter_name="policy",
            )

        self.assertEqual(
            result,
            (
                False,
                "DSV4-style expert OFT adapter has no FusedMoE target "
                "(fork DeepSeekV4 model support was removed)",
            ),
        )


if __name__ == "__main__":
    unittest.main()

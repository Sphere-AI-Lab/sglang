"""Regression coverage for the retired generic OFT tensor updates."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)


def test_retired_oft_tensor_format_is_rejected_without_device_work():
    updater = SimpleNamespace(
        _assert_weight_cache_inactive=Mock(),
        custom_weight_loaders={},
        device="cuda",
        get_model=Mock(),
    )

    with (
        patch.object(
            weight_updater,
            "_unsupported_derived_weight_cache_error",
            return_value=None,
        ),
        patch.object(weight_updater, "monkey_patch_torch_reductions"),
        patch.object(torch, "get_device_module") as get_device_module,
    ):
        result = WeightUpdater.update_weights_from_tensor(
            updater, [], load_format="oft_adapter"
        )

    assert result[0] is False


class _AsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_retired_oft_tensor_format_rolls_back_tokenizer_registration():
    from sglang.srt.oft.oft_registry import OFTRegistry

    async def reject_update(_obj):
        return [
            UpdateWeightsFromTensorReqOutput(
                success=False,
                message="The oft_adapter tensor update format has been retired.",
            )
        ]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        is_pause_cond=_AsyncContext(),
        is_pause=True,
        update_weights_from_tensor_communicator=reject_update,
        mm_processor=None,
        peft_registry=OFTRegistry(),
        peft_ref_cache={},
    )
    request = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[],
        load_format="oft_adapter",
        adapter_name="adapter",
    )

    with patch(
        "sglang.srt.managers.tokenizer_control_mixin.get_parallel",
        return_value=SimpleNamespace(dp_size=1, enable_dp_attention=False),
    ):
        success, _ = asyncio.run(
            TokenizerControlMixin.update_weights_from_tensor(manager, request)
        )

    assert not success
    assert manager.peft_registry.num_registered_ofts == 0
    assert manager.peft_ref_cache == {}

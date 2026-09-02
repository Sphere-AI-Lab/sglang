"""Regression coverage for the retired generic OFT tensor updates."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

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

"""Unit coverage for canonical OFT staging and activation routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)


def _weight_updater(runner):
    return SimpleNamespace(
        _model_update_group={"sync": object()},
        device="cpu",
        get_model_runner=Mock(return_value=runner),
    )


def _stage_kwargs():
    return dict(
        names=["adapter.weight"],
        dtypes=[torch.float32],
        shapes=[(2,)],
        group_name="sync",
        load_format="oft_adapter",
        adapter_config={"target_modules": ["q_proj"], "oft_block_size": 4},
        adapter_name="policy",
        adapter_id="id-a",
        adapter_version="8",
        payload_metadata=None,
        double_buffer=True,
    )


def test_oft_stage_propagates_manager_failure():
    runner = MagicMock()
    runner.server_args.enable_lora_staging = False
    runner.server_args.peft_method = "oft"
    runner.oft_manager.stage_adapter.return_value = SimpleNamespace(
        success=False,
        error_message="OFT stage rejected",
    )
    updater = _weight_updater(runner)
    handle = MagicMock()

    with patch.object(torch.distributed, "broadcast", return_value=handle):
        result = WeightUpdater.stage_adapter(updater, **_stage_kwargs())

    assert result == (False, "OFT stage rejected")
    handle.wait.assert_called_once_with()
    call = runner.oft_manager.stage_adapter.call_args
    assert [name for name, _ in call.args[0]] == ["adapter.weight"]
    assert call.args[1:] == (_stage_kwargs()["adapter_config"], "policy", 8)
    assert call.kwargs == {"adapter_id": "id-a"}


def test_oft_activation_propagates_manager_failure():
    runner = MagicMock()
    runner.server_args.enable_lora_staging = False
    runner.server_args.peft_method = "oft"
    runner.oft_manager.activate_adapter.return_value = SimpleNamespace(
        success=False,
        error_message="OFT activation rejected",
    )
    updater = _weight_updater(runner)

    result = WeightUpdater.activate_adapter_version(
        updater,
        adapter_name="policy",
        adapter_id="id-a",
        adapter_version="8",
    )

    assert result == (False, "OFT activation rejected")
    runner.oft_manager.activate_adapter.assert_called_once_with(
        "policy", 8, adapter_id="id-a"
    )

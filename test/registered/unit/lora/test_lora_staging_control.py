"""Unit coverage for optional native staged-LoRA startup and routing."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch, sentinel

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import ActivateAdapterVersionReqInput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.server_args import ServerArgs

# The CPU login environment exposes Megatron's Python package but not its CUDA
# shared libraries. Isolate ModelRunner's optional debug integration while it
# imports, then restore sys.modules for the rest of the test process.
with patch.dict(sys.modules, {"megatron": None}):
    from sglang.srt.model_executor.model_runner import ModelRunner


def _weight_updater(runner):
    return SimpleNamespace(
        _model_update_group={"sync": sentinel.process_group},
        device="cpu",
        get_model_runner=Mock(return_value=runner),
    )


def _stage_kwargs(**overrides):
    values = dict(
        names=["__flattened__"],
        dtypes=[torch.float32],
        shapes=[(2,)],
        group_name="sync",
        load_format="lora_adapter",
        adapter_config={"target_modules": ["q_proj"], "r": 4},
        adapter_name="policy",
        adapter_id="id-a",
        adapter_version="8",
        payload_metadata={"metadata": []},
        double_buffer=True,
    )
    values.update(overrides)
    return values


class TestStagingFlagAndSelection(unittest.TestCase):
    def test_staging_defaults_off(self):
        args = ServerArgs(model_path="Qwen/Qwen3-0.6B", device="cpu")
        self.assertFalse(args.enable_lora_staging)

    def test_staging_requires_native_lora(self):
        args = ServerArgs(model_path="Qwen/Qwen3-0.6B", device="cpu")
        object.__setattr__(args, "enable_lora", False)
        object.__setattr__(args, "enable_lora_staging", True)

        with self.assertRaisesRegex(ValueError, "requires --enable-lora"):
            args.check_lora_server_args()

    def test_model_runner_selects_staged_manager(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.server_args = MagicMock(enable_lora_staging=True)

        with patch(
            "sglang.srt.adapter_sync.backends.lora.StagedLoRAManager",
            new=sentinel.staged_manager,
        ):
            self.assertIs(
                runner._get_lora_manager_class(),
                sentinel.staged_manager,
            )

    def test_model_runner_keeps_native_manager_when_staging_is_off(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.server_args = MagicMock(enable_lora_staging=False)

        self.assertIs(runner._get_lora_manager_class(), LoRAManager)


class TestWorkerForwarding(unittest.TestCase):
    def test_activation_forwards_stable_id(self):
        worker = TpModelWorker.__new__(TpModelWorker)
        worker._model_runner = MagicMock()
        worker.model_runner.weight_updater.activate_adapter_version.return_value = (
            True,
            "ok",
        )
        req = ActivateAdapterVersionReqInput(
            adapter_name="policy", adapter_id="id-a", adapter_version="8"
        )

        worker.activate_adapter_version(req)

        worker.model_runner.weight_updater.activate_adapter_version.assert_called_once_with(
            adapter_name="policy", adapter_id="id-a", adapter_version="8"
        )


class TestWeightUpdaterRouting(unittest.TestCase):
    def test_native_stage_reconstructs_payload_and_returns_manager_result(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = True
        runner.lora_manager.stage_adapter.return_value = SimpleNamespace(
            success=False, error_message="native stage rejected"
        )
        updater = _weight_updater(runner)
        handle = MagicMock()
        reconstructed = [("q_proj.lora_A.weight", torch.ones(2))]

        with (
            patch.object(torch.distributed, "broadcast", return_value=handle),
            patch.object(
                weight_updater.peft,
                "reconstruct_oft_staging",
                return_value=reconstructed,
            ) as reconstruct,
            patch.object(weight_updater.peft, "stage_adapter") as fallback,
        ):
            result = WeightUpdater.stage_adapter(updater, **_stage_kwargs())

        self.assertEqual(result, (False, "native stage rejected"))
        handle.wait.assert_called_once_with()
        reconstruct.assert_called_once()
        runner.lora_manager.stage_adapter.assert_called_once_with(
            reconstructed,
            _stage_kwargs()["adapter_config"],
            "policy",
            8,
            adapter_id="id-a",
        )
        fallback.assert_not_called()

    def test_non_native_stage_keeps_peft_fallback(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = False
        updater = _weight_updater(runner)
        handle = MagicMock()

        with (
            patch.object(torch.distributed, "broadcast", return_value=handle),
            patch.object(
                weight_updater.peft,
                "stage_adapter",
                return_value=sentinel.peft_result,
            ) as fallback,
        ):
            result = WeightUpdater.stage_adapter(
                updater, **_stage_kwargs(load_format="oft_adapter")
            )

        self.assertEqual(result, (True, "Succeeded to stage adapter online."))
        fallback.assert_called_once()
        runner.lora_manager.stage_adapter.assert_not_called()

    def test_native_activation_forwards_id_and_returns_manager_result(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = True
        runner.lora_manager.activate_adapter.return_value = SimpleNamespace(
            success=False, error_message="native activation rejected"
        )
        updater = _weight_updater(runner)

        with patch.object(weight_updater.peft, "activate_adapter") as fallback:
            result = WeightUpdater.activate_adapter_version(
                updater,
                adapter_name="policy",
                adapter_id="id-a",
                adapter_version="8",
            )

        self.assertEqual(result, (False, "native activation rejected"))
        runner.lora_manager.activate_adapter.assert_called_once_with(
            "policy", 8, adapter_id="id-a"
        )
        fallback.assert_not_called()

    def test_non_native_activation_keeps_peft_fallback(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = False
        updater = _weight_updater(runner)

        with patch.object(
            weight_updater.peft,
            "activate_adapter",
            return_value=sentinel.peft_result,
        ) as fallback:
            result = WeightUpdater.activate_adapter_version(
                updater,
                adapter_name="policy",
                adapter_id="id-a",
                adapter_version="8",
            )

        self.assertEqual(result, (True, "Succeeded to activate adapter version."))
        fallback.assert_called_once_with(runner, "policy", 8)


if __name__ == "__main__":
    unittest.main()

"""Unit coverage for optional native staged-LoRA startup and routing."""

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch, sentinel

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    ActivateAdapterVersionReqOutput,
    UpdateAdapterFromDistributedReqInput,
    UpdateAdapterFromDistributedReqOutput,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.staged_manager import LoRAStagingBackend
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.aio_rwlock import RWLock

# The CPU login environment exposes Megatron's Python package but not its CUDA
# shared libraries. Isolate ModelRunner's optional debug integration while it
# imports, then restore sys.modules for the rest of the test process.
with patch.dict(sys.modules, {"megatron": None}):
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


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


def _stage_req(name="policy", version="4", *, double_buffer=True):
    return UpdateAdapterFromDistributedReqInput(
        names=[],
        dtypes=[],
        shapes=[],
        load_format="lora_adapter",
        adapter_name=name,
        adapter_version=version,
        double_buffer=double_buffer,
    )


def _activate_req(name="policy", version="4"):
    return ActivateAdapterVersionReqInput(
        adapter_name=name,
        adapter_version=version,
        load_format="lora_adapter",
    )


def _make_tm(*, tokenizer_worker_num=1):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = SimpleNamespace(
        enable_lora_staging=True,
        dp_size=1,
        enable_dp_attention=False,
        tokenizer_worker_num=tokenizer_worker_num,
    )
    tm.lora_registry = LoRARegistry()
    tm.lora_ref_cache = {}
    tm.lora_update_lock = asyncio.Lock()
    tm.pending_lora_stage = None
    tm.failed_lora_activations = {}
    tm.model_update_lock = RWLock()
    tm.is_pause_cond = asyncio.Condition()
    tm.is_pause = False
    tm.auto_create_handle_loop = Mock()
    tm.update_adapter_from_distributed_communicator = AsyncMock(
        return_value=[
            UpdateAdapterFromDistributedReqOutput(success=True, message="staged")
        ]
    )
    tm.activate_adapter_version_communicator = AsyncMock(
        return_value=[
            ActivateAdapterVersionReqOutput(
                success=True,
                message="activated",
                active_adapter_version="4",
            )
        ]
    )
    return tm


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
            "sglang.srt.lora.staged_manager.StagedLoRAManager",
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


class TestTokenizerNativeStaging(unittest.TestCase):
    def test_stage_reuses_id_without_publishing_version(self):
        tm = _make_tm()
        old = LoRARef(
            lora_id="id-a",
            lora_name="policy",
            lora_path="__distributed__",
            pinned=True,
            version=3,
        )
        asyncio.run(tm.lora_registry.register(old))
        req = _stage_req()

        success, _ = asyncio.run(tm.update_adapter_from_distributed(req))

        self.assertTrue(success)
        self.assertEqual(req.adapter_id, "id-a")
        self.assertEqual(tm.lora_registry.get_all_adapters()["policy"].version, 3)
        self.assertEqual(tm.pending_lora_stage.version, 4)
        self.assertTrue(tm.pending_lora_stage.pinned)

    def test_first_stage_is_not_registered(self):
        tm = _make_tm()
        req = _stage_req()

        success, _ = asyncio.run(tm.update_adapter_from_distributed(req))

        self.assertTrue(success)
        self.assertEqual(tm.lora_registry.get_all_adapters(), {})
        self.assertEqual(tm.pending_lora_stage.lora_id, req.adapter_id)

    def test_same_stage_retry_reuses_pending_ref(self):
        tm = _make_tm()
        first = _stage_req()
        second = _stage_req()

        asyncio.run(tm.update_adapter_from_distributed(first))
        pending = tm.pending_lora_stage
        asyncio.run(tm.update_adapter_from_distributed(second))

        self.assertIs(tm.pending_lora_stage, pending)
        self.assertEqual(second.adapter_id, first.adapter_id)

    def test_conflicting_stage_reports_pending_identity(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))

        with self.assertRaisesRegex(
            ValueError, r"name=policy.*id=.*version=4"
        ):
            asyncio.run(
                tm.update_adapter_from_distributed(
                    _stage_req(name="other", version="5")
                )
            )

        self.assertEqual(tm.pending_lora_stage.lora_name, "policy")

    def test_concurrent_conflicting_stages_reserve_only_one_identity(self):
        tm = _make_tm()
        original = tm.lora_registry.register_or_reuse

        async def slow_register_or_reuse(*args, **kwargs):
            await asyncio.sleep(0)
            return await original(*args, **kwargs)

        async def run_concurrently():
            with patch.object(
                tm.lora_registry,
                "register_or_reuse",
                side_effect=slow_register_or_reuse,
            ):
                return await asyncio.gather(
                    tm.update_adapter_from_distributed(_stage_req()),
                    tm.update_adapter_from_distributed(
                        _stage_req(name="other", version="5")
                    ),
                    return_exceptions=True,
                )

        results = asyncio.run(run_concurrently())

        self.assertEqual(sum(isinstance(result, ValueError) for result in results), 1)
        self.assertIn(tm.pending_lora_stage.lora_name, {"policy", "other"})

    def test_activation_forwards_exact_id(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))
        expected_id = tm.pending_lora_stage.lora_id
        req = _activate_req()

        success, _ = asyncio.run(tm.activate_adapter_version(req))

        self.assertTrue(success)
        self.assertEqual(req.adapter_id, expected_id)
        forwarded = tm.activate_adapter_version_communicator.await_args.args[0]
        self.assertEqual(forwarded.adapter_id, expected_id)

    def test_successful_activation_refreshes_existing_ref(self):
        tm = _make_tm()
        old = LoRARef(
            lora_id="id-a",
            lora_name="policy",
            lora_path="__distributed__",
            pinned=True,
            version=3,
        )
        asyncio.run(tm.lora_registry.register(old))
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))

        success, _ = asyncio.run(tm.activate_adapter_version(_activate_req()))

        self.assertTrue(success)
        active = tm.lora_registry.get_all_adapters()["policy"]
        self.assertEqual((active.lora_id, active.version), ("id-a", 4))
        self.assertTrue(active.pinned)
        self.assertIsNone(tm.pending_lora_stage)

    def test_successful_activation_registers_first_ref(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))
        pending_id = tm.pending_lora_stage.lora_id

        success, _ = asyncio.run(tm.activate_adapter_version(_activate_req()))

        self.assertTrue(success)
        active = tm.lora_registry.get_all_adapters()["policy"]
        self.assertEqual((active.lora_id, active.version), (pending_id, 4))
        self.assertIs(tm.lora_ref_cache["policy"], active)

    def test_activation_failure_keeps_old_version_and_quarantines_name(self):
        tm = _make_tm()
        old = LoRARef(
            lora_id="id-a",
            lora_name="policy",
            lora_path="__distributed__",
            version=3,
        )
        asyncio.run(tm.lora_registry.register(old))
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))
        tm.activate_adapter_version_communicator = AsyncMock(
            return_value=[
                ActivateAdapterVersionReqOutput(
                    success=True,
                    message="ok",
                    active_adapter_version="4",
                ),
                ActivateAdapterVersionReqOutput(
                    success=False,
                    message="rank 1 failed",
                ),
            ]
        )

        success, message = asyncio.run(
            tm.activate_adapter_version(_activate_req())
        )

        self.assertFalse(success)
        self.assertIn("restart required", message)
        self.assertEqual(tm.lora_registry.get_all_adapters()["policy"].version, 3)
        self.assertIn("policy", tm.failed_lora_activations)
        with self.assertRaisesRegex(ValueError, "policy.*restart required"):
            LoRAStagingBackend(tm)._assert_available("policy")

    def test_quarantine_does_not_block_base_or_other_adapter(self):
        tm = _make_tm()
        backend = LoRAStagingBackend(tm)
        backend._quarantine("policy", "partial activation")

        backend._assert_available(None)
        backend._assert_available("unrelated")
        backend._assert_available([None, "unrelated"])
        with self.assertRaisesRegex(ValueError, "policy.*restart required"):
            backend._assert_available([None, "policy"])

    def test_multi_tokenizer_native_stage_is_rejected(self):
        tm = _make_tm(tokenizer_worker_num=2)

        with self.assertRaisesRegex(ValueError, "tokenizer_worker_num == 1"):
            asyncio.run(tm.update_adapter_from_distributed(_stage_req()))

        tm.update_adapter_from_distributed_communicator.assert_not_awaited()

    def test_synchronous_stage_publishes_only_matching_active_version(self):
        tm = _make_tm()
        req = _stage_req(double_buffer=False)
        tm.update_adapter_from_distributed_communicator = AsyncMock(
            return_value=[
                UpdateAdapterFromDistributedReqOutput(
                    success=True,
                    message="staged and activated",
                    staged_adapter_version="4",
                    active_adapter_version="4",
                )
            ]
        )

        success, _ = asyncio.run(tm.update_adapter_from_distributed(req))

        self.assertTrue(success)
        self.assertEqual(tm.lora_registry.get_all_adapters()["policy"].version, 4)
        self.assertIsNone(tm.pending_lora_stage)

    def test_version_disagreement_is_quarantined(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))
        tm.activate_adapter_version_communicator = AsyncMock(
            return_value=[
                ActivateAdapterVersionReqOutput(
                    success=True,
                    message="wrong version",
                    active_adapter_version="5",
                )
            ]
        )

        success, message = asyncio.run(
            tm.activate_adapter_version(_activate_req())
        )

        self.assertFalse(success)
        self.assertIn("restart required", message)
        self.assertIn("policy", tm.failed_lora_activations)


class TestResolveLoraPathRejectsQuarantine(unittest.TestCase):
    """Regression test for a bug introduced by the AdapterStagingBackend
    extraction: _resolve_lora_path (called from _validate_and_resolve_lora,
    the admission path every generate/embedding request with a lora_path
    goes through) used to call self._assert_native_lora_available directly.
    That method briefly only existed on LoRAStagingBackend after the
    extraction, so any request naming a lora_path raised AttributeError
    instead of the intended ValueError -- unconditionally, independent of
    enable_lora_staging. The fix restores an always-available
    _assert_native_lora_available on TokenizerControlMixin."""

    def _tm(self, *, enable_lora_staging):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.server_args = SimpleNamespace(enable_lora_staging=enable_lora_staging)
        tm.enable_lora = True
        tm.failed_lora_activations = {"policy": "partial activation failure"}
        return tm

    def test_resolve_lora_path_rejects_quarantined_adapter_with_staging_off(self):
        tm = self._tm(enable_lora_staging=False)
        obj = SimpleNamespace(lora_path="policy")

        with self.assertRaisesRegex(ValueError, "policy.*restart required"):
            asyncio.run(tm._resolve_lora_path(obj))

    def test_resolve_lora_path_rejects_quarantined_adapter_with_staging_on(self):
        tm = self._tm(enable_lora_staging=True)
        obj = SimpleNamespace(lora_path="policy")

        with self.assertRaisesRegex(ValueError, "policy.*restart required"):
            asyncio.run(tm._resolve_lora_path(obj))

    def test_validate_and_resolve_lora_rejects_quarantined_adapter(self):
        tm = self._tm(enable_lora_staging=False)
        obj = SimpleNamespace(lora_path="policy")

        with self.assertRaisesRegex(ValueError, "policy.*restart required"):
            asyncio.run(tm._validate_and_resolve_lora(obj))


if __name__ == "__main__":
    unittest.main()

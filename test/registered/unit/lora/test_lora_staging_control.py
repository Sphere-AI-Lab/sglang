"""Unit coverage for optional native staged-LoRA startup and routing."""

import asyncio
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch, sentinel

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.lora.staged_manager import LoRAStagingBackend
from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    ActivateAdapterVersionReqOutput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
    UpdateAdapterFromDistributedReqInput,
    UpdateAdapterFromDistributedReqOutput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.aio_rwlock import RWLock

# The CPU login environment exposes Megatron's Python package but not its CUDA
# shared libraries. Isolate ModelRunner's optional debug integration while it
# imports, then restore sys.modules for the rest of the test process.
with patch.dict(sys.modules, {"megatron": None}):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
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
        enable_lora=True,
        dp_size=1,
        enable_dp_attention=False,
        tokenizer_worker_num=tokenizer_worker_num,
        max_loaded_loras=None,
    )
    tm.lora_registry = LoRARegistry()
    tm.lora_ref_cache = {}
    tm.lora_update_lock = asyncio.Lock()
    tm.pending_lora_stage = None
    tm.failed_lora_activations = {}
    tm.failed_lora_unloads = {}
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
    tm.update_lora_adapter_communicator = AsyncMock(
        return_value=[LoRAUpdateOutput(success=True)]
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


class TestSchedulerTpAdapterConsensus(CustomTestCase):
    def setUp(self):
        super().setUp()
        parallel_config = get_context().override_server_args(
            dp_size=1,
            enable_dp_attention=False,
            enable_dp_attention_local_control_broadcast=False,
        )
        parallel_config.install()
        self.addCleanup(parallel_config.restore)

    def test_local_dp_unloads_on_different_polls_use_only_control_domain_groups(self):
        from sglang.srt.oft.io_types import OFTUpdateOutput, UnloadOFTAdapterReqInput

        # A Cartesian (DP, CP, TP) layout matches initialize_model_parallel.
        # Domain 1 has not received unload on domain 0's scheduler poll and may
        # already be entering a full-TP MLP collective. It cannot join unload.
        for method, request, output in (
            (
                "unload_lora_adapter",
                UnloadLoRAAdapterReqInput(lora_name="policy"),
                LoRAUpdateOutput,
            ),
            (
                "unload_oft_adapter",
                UnloadOFTAdapterReqInput(adapter_name="policy"),
                OFTUpdateOutput,
            ),
        ):
            for tp_size, cp_size in ((2, 2), (2, 1), (1, 2), (1, 1)):
                for raises in (False, True):
                    for joiner in (False, True):
                        with self.subTest(
                            method=method,
                            tp_size=tp_size,
                            cp_size=cp_size,
                            raises=raises,
                            joiner=joiner,
                        ):
                            width = tp_size * cp_size
                            full_tp = tuple(range(2 * width))
                            called_groups = []
                            worker_polls = []
                            local = threading.local()
                            condition = threading.Condition()
                            gathered = {}

                            def gather(outputs, value, *, group):
                                self.assertNotEqual(
                                    group,
                                    full_tp,
                                    "unload entered full TP while another DP "
                                    "domain is on its MLP poll",
                                )
                                self.assertTrue(
                                    all(rank // width == local.domain for rank in group)
                                )
                                with condition:
                                    called_groups.append(group)
                                    values = gathered.setdefault(group, {})
                                    values[local.rank] = value
                                    condition.notify_all()
                                    self.assertTrue(
                                        condition.wait_for(
                                            lambda: len(values) == len(group), timeout=2
                                        ),
                                        "control-domain ranks disagreed on "
                                        "collective membership",
                                    )
                                    outputs[:] = [values[rank] for rank in group]

                            def unload_rank(domain, rank):
                                local.rank, local.domain = rank, domain
                                offset = domain * width
                                cp_rank = (rank - offset) // tp_size
                                tp_rank = (rank - offset) % tp_size
                                scheduler = Scheduler.__new__(Scheduler)
                                scheduler.tp_cpu_group = full_tp
                                scheduler.attn_tp_cpu_group = tuple(
                                    range(
                                        offset + cp_rank * tp_size,
                                        offset + (cp_rank + 1) * tp_size,
                                    )
                                )
                                scheduler.attn_cp_cpu_group = tuple(
                                    offset + cp * tp_size + tp_rank
                                    for cp in range(cp_size)
                                )
                                scheduler.ps = SimpleNamespace(
                                    attn_tp_size=tp_size, attn_cp_size=cp_size
                                )
                                scheduler.tp_worker = Mock()

                                def unload(_):
                                    worker_polls.append(domain)
                                    if rank == offset + width - 1:
                                        if raises:
                                            raise RuntimeError(
                                                "last local rank discard failed"
                                            )
                                        return output(
                                            success=False,
                                            error_message=(
                                                "last local rank discard failed"
                                            ),
                                        )
                                    return output(success=True)

                                worker_unload = getattr(scheduler.tp_worker, method)
                                worker_unload.side_effect = unload
                                return getattr(scheduler, method)(request)

                            with (
                                patch(
                                    "sglang.srt.managers.scheduler.get_parallel",
                                    return_value=SimpleNamespace(
                                        enable_dp_attention=True,
                                        enable_dp_attention_local_control_broadcast=(
                                            not joiner
                                        ),
                                    ),
                                ),
                                patch(
                                    "sglang.srt.managers.scheduler.is_ep_scale_joiner",
                                    return_value=joiner,
                                ),
                                patch.object(
                                    torch.distributed,
                                    "get_world_size",
                                    side_effect=lambda *, group: len(group),
                                ),
                                patch.object(
                                    torch.distributed,
                                    "all_gather_object",
                                    side_effect=gather,
                                ),
                                ThreadPoolExecutor(max_workers=width) as executor,
                            ):
                                for domain in (0, 1):
                                    futures = [
                                        executor.submit(unload_rank, domain, rank)
                                        for rank in range(
                                            domain * width, (domain + 1) * width
                                        )
                                    ]
                                    results = [
                                        future.result(timeout=5) for future in futures
                                    ]
                                    self.assertTrue(
                                        all(not result.success for result in results)
                                    )
                                    for result in results:
                                        self.assertIn(
                                            "last local rank discard failed",
                                            result.error_message,
                                        )
                                    self.assertEqual(
                                        worker_polls,
                                        [0] * width + [1] * (domain * width),
                                    )

                            expected_per_rank = int(tp_size > 1) + int(cp_size > 1)
                            self.assertEqual(
                                len(called_groups), 2 * width * expected_per_rank
                            )

    def test_dp_global_control_unload_keeps_full_tp_consensus(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_cpu_group = sentinel.tp_cpu_group
        scheduler.tp_worker = Mock()
        scheduler.tp_worker.unload_lora_adapter.return_value = LoRAUpdateOutput(
            success=True
        )

        def gather(outputs, value, *, group):
            self.assertIs(group, sentinel.tp_cpu_group)
            outputs[:] = [value, (False, "other DP rank failed", None)]

        with (
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(
                    enable_dp_attention=True,
                    enable_dp_attention_local_control_broadcast=False,
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.is_ep_scale_joiner",
                return_value=False,
            ),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        ):
            result = scheduler.unload_lora_adapter(
                UnloadLoRAAdapterReqInput(lora_name="policy")
            )

        self.assertFalse(result.success)
        self.assertIn("other DP rank failed", result.error_message)

    def test_unload_reports_non_sender_rank_failure(self):
        from sglang.srt.oft.io_types import OFTUpdateOutput, UnloadOFTAdapterReqInput

        for method, request, output in (
            (
                "unload_lora_adapter",
                UnloadLoRAAdapterReqInput(lora_name="policy"),
                LoRAUpdateOutput,
            ),
            (
                "unload_oft_adapter",
                UnloadOFTAdapterReqInput(adapter_name="policy"),
                OFTUpdateOutput,
            ),
        ):
            with self.subTest(method=method):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.tp_cpu_group = sentinel.tp_cpu_group
                scheduler.tp_worker = Mock()
                getattr(scheduler.tp_worker, method).return_value = output(success=True)

                def gather_rank_results(results, local_result, *, group):
                    self.assertIs(group, sentinel.tp_cpu_group)
                    results[:] = [
                        local_result,
                        (False, "rank 1 discard failed", None),
                    ]

                with (
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(
                        torch.distributed,
                        "all_gather_object",
                        side_effect=gather_rank_results,
                    ),
                ):
                    result = getattr(scheduler, method)(request)

                self.assertFalse(result.success)
                self.assertIn("TP rank 1", result.error_message)
                self.assertIn("discard failed", result.error_message)

    def test_unload_exception_still_joins_tp_consensus(self):
        from sglang.srt.oft.io_types import UnloadOFTAdapterReqInput

        for method, request in (
            (
                "unload_lora_adapter",
                UnloadLoRAAdapterReqInput(lora_name="policy"),
            ),
            (
                "unload_oft_adapter",
                UnloadOFTAdapterReqInput(adapter_name="policy"),
            ),
        ):
            with self.subTest(method=method):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.tp_cpu_group = sentinel.tp_cpu_group
                scheduler.tp_worker = Mock()
                getattr(scheduler.tp_worker, method).side_effect = RuntimeError(
                    "discard failed"
                )

                def gather_rank_results(results, local_result, *, group):
                    results[:] = [local_result, (True, "", None)]

                with (
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(
                        torch.distributed,
                        "all_gather_object",
                        side_effect=gather_rank_results,
                    ),
                ):
                    result = getattr(scheduler, method)(request)

                self.assertFalse(result.success)
                self.assertIn("TP rank 0", result.error_message)
                self.assertIn("discard failed", result.error_message)

    def test_rank_one_stage_failure_prevents_rank_zero_activation(self):
        """A non-sender TP failure must fail the scheduler's returned result."""
        tp_worker = Mock()
        tp_worker.update_adapter_from_distributed.return_value = (
            True,
            "rank 0 staged",
        )
        updater = SchedulerWeightUpdaterManager(
            tp_worker=tp_worker,
            draft_worker=None,
            tp_cpu_group=sentinel.tp_cpu_group,
            memory_saver_adapter=Mock(),
            flush_cache=Mock(return_value=True),
            is_fully_idle=Mock(return_value=True),
        )

        def gather_rank_results(results, local_result, *, group):
            self.assertIs(group, sentinel.tp_cpu_group)
            results[:] = [local_result, (False, "rank 1 failed to stage", None)]

        with (
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=gather_rank_results,
            ),
        ):
            result = updater.update_adapter_from_distributed(
                _stage_req(double_buffer=False)
            )

        self.assertFalse(result.success)
        self.assertIn("TP rank 1", result.message)
        self.assertIsNone(result.staged_adapter_version)
        self.assertIsNone(result.active_adapter_version)
        tp_worker.activate_adapter_version.assert_not_called()


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
                weight_updater,
                "reconstruct_adapter_staging",
                return_value=reconstructed,
            ) as reconstruct,
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

    def test_non_native_stage_is_rejected_without_touching_native_manager(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = False
        updater = _weight_updater(runner)

        result = WeightUpdater.stage_adapter(
            updater, **_stage_kwargs(load_format="oft_adapter")
        )

        self.assertFalse(result[0])
        self.assertIn("native LoRA staging", result[1])
        runner.lora_manager.stage_adapter.assert_not_called()

    def test_native_activation_forwards_id_and_returns_manager_result(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = True
        runner.lora_manager.activate_adapter.return_value = SimpleNamespace(
            success=False, error_message="native activation rejected"
        )
        updater = _weight_updater(runner)

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

    def test_non_native_activation_is_rejected_without_touching_native_manager(self):
        runner = MagicMock()
        runner.server_args.enable_lora_staging = False
        updater = _weight_updater(runner)

        result = WeightUpdater.activate_adapter_version(
            updater,
            adapter_name="policy",
            adapter_id="id-a",
            adapter_version="8",
        )

        self.assertFalse(result[0])
        self.assertIn("native LoRA staging", result[1])
        runner.lora_manager.activate_adapter.assert_not_called()


class TestTokenizerNativeStaging(CustomTestCase):
    def setUp(self):
        super().setUp()
        parallel_config = get_context().override_server_args(
            dp_size=1, enable_dp_attention=False
        )
        parallel_config.install()
        self.addCleanup(parallel_config.restore)

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

    def test_failed_unload_tombstone_rejects_stage_before_reservation(self):
        from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry

        for method in ("lora", "oft"):
            with self.subTest(method=method):
                tm = _make_tm()
                request = _stage_req()
                if method == "lora":
                    ref = LoRARef(lora_id="old-id", lora_name="policy")
                    tm.failed_lora_unloads["policy"] = ref
                else:
                    request.load_format = "oft_adapter"
                    tm.server_args.enable_lora_staging = False
                    tm.server_args.peft_method = "oft"
                    tm.peft_registry = OFTRegistry()
                    tm.peft_ref_cache = {}
                    tm.peft_update_lock = asyncio.Lock()
                    tm.pending_oft_stage = None
                    tm.failed_oft_activations = {}
                    tm.failed_oft_unloads = {
                        "policy": OFTRef(adapter_id="old-id", adapter_name="policy")
                    }
                    tm.update_oft_adapter_communicator = AsyncMock()

                with self.assertRaisesRegex(ValueError, "failed unload"):
                    asyncio.run(tm.update_adapter_from_distributed(request))

                self.assertIsNone(getattr(tm, f"pending_{method}_stage"))
                tm.update_adapter_from_distributed_communicator.assert_not_awaited()

    def test_failed_unload_tombstone_is_revalidated_before_activation(self):
        from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry

        for method in ("lora", "oft"):
            with self.subTest(method=method):
                tm = _make_tm()
                stage = _stage_req()
                activation = _activate_req()
                if method == "oft":
                    stage.load_format = activation.load_format = "oft_adapter"
                    tm.server_args.enable_lora_staging = False
                    tm.server_args.peft_method = "oft"
                    tm.peft_registry = OFTRegistry()
                    tm.peft_ref_cache = {}
                    tm.peft_update_lock = asyncio.Lock()
                    tm.pending_oft_stage = None
                    tm.failed_oft_activations = {}
                    tm.failed_oft_unloads = {}

                self.assertTrue(
                    asyncio.run(tm.update_adapter_from_distributed(stage))[0]
                )
                tombstone = (
                    LoRARef(lora_id="old-id", lora_name="policy")
                    if method == "lora"
                    else OFTRef(adapter_id="old-id", adapter_name="policy")
                )
                getattr(tm, f"failed_{method}_unloads")["policy"] = tombstone

                with self.assertRaisesRegex(ValueError, "failed unload"):
                    asyncio.run(tm.activate_adapter_version(activation))

                tm.activate_adapter_version_communicator.assert_not_awaited()

    def test_same_stage_retry_reuses_pending_ref(self):
        tm = _make_tm()
        first = _stage_req()
        second = _stage_req()

        asyncio.run(tm.update_adapter_from_distributed(first))
        pending = tm.pending_lora_stage
        asyncio.run(tm.update_adapter_from_distributed(second))

        self.assertIs(tm.pending_lora_stage, pending)
        self.assertEqual(second.adapter_id, first.adapter_id)

    def test_pending_unload_invalidates_retry_queued_behind_inflight_stage(self):
        from sglang.srt.managers.communicator import FanOutCommunicator
        from sglang.srt.oft.io_types import OFTUpdateOutput, UnloadOFTAdapterReqInput
        from sglang.srt.oft.oft_registry import OFTRegistry

        async def scenario(method, replace_cancelled):
            tm = _make_tm()
            first, retry = _stage_req(), _stage_req()
            if method == "oft":
                tm.server_args.enable_lora_staging = False
                tm.server_args.peft_method = "oft"
                tm.peft_registry = OFTRegistry()
                tm.peft_ref_cache = {}
                tm.peft_update_lock = asyncio.Lock()
                tm.pending_oft_stage = None
                tm.failed_oft_activations = {}
                first.load_format = retry.load_format = "oft_adapter"
                unload_request = UnloadOFTAdapterReqInput(adapter_name="policy")
                unload = tm.unload_oft_adapter
                output = OFTUpdateOutput
            else:
                unload_request = UnloadLoRAAdapterReqInput(lora_name="policy")
                unload = tm.unload_lora_adapter
                output = LoRAUpdateOutput

            dispatched = []
            worker_stage = None
            first_started = asyncio.Event()

            def send_stage(request):
                nonlocal worker_stage
                dispatched.append(request.adapter_id)
                worker_stage = request.adapter_id
                first_started.set()
                if len(dispatched) > 1:
                    asyncio.get_running_loop().call_soon(
                        communicator.handle_recv,
                        UpdateAdapterFromDistributedReqOutput(
                            success=True, message="staged"
                        ),
                    )

            communicator = FanOutCommunicator(send_stage, fan_out=1)
            tm.update_adapter_from_distributed_communicator = communicator

            async def discard(_):
                nonlocal worker_stage
                worker_stage = None
                return [output(success=True)]

            setattr(tm, f"update_{method}_adapter_communicator", discard)
            staging = asyncio.create_task(tm.update_adapter_from_distributed(first))
            await first_started.wait()
            retrying = asyncio.create_task(tm.update_adapter_from_distributed(retry))
            await asyncio.sleep(0)
            unloading = asyncio.create_task(unload(unload_request))
            await asyncio.sleep(0)
            replacement = _stage_req()
            replacement.load_format = first.load_format
            restaging = None
            if replace_cancelled:
                restaging = asyncio.create_task(
                    tm.update_adapter_from_distributed(replacement)
                )
                await asyncio.sleep(0)
            communicator.handle_recv(
                UpdateAdapterFromDistributedReqOutput(success=True, message="staged")
            )

            self.assertTrue((await staging)[0])
            self.assertTrue((await unloading).success)
            with self.assertRaises(ValueError):
                await retrying
            if restaging is not None:
                self.assertTrue((await restaging)[0])
                self.assertEqual(dispatched, [first.adapter_id, replacement.adapter_id])
                self.assertNotEqual(first.adapter_id, replacement.adapter_id)
                self.assertEqual(worker_stage, replacement.adapter_id)
                self.assertIsNotNone(getattr(tm, f"pending_{method}_stage"))
            else:
                self.assertEqual(len(dispatched), 1)
                self.assertIsNone(worker_stage)
                self.assertIsNone(getattr(tm, f"pending_{method}_stage"))

        for method in ("lora", "oft"):
            for replace_cancelled in (False, True):
                with self.subTest(method=method, replace_cancelled=replace_cancelled):
                    asyncio.run(scenario(method, replace_cancelled))

    def test_unknown_user_unload_still_fails_before_worker_dispatch(self):
        tm = _make_tm()
        result = asyncio.run(
            tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="unknown"))
        )

        self.assertFalse(result.success)
        tm.update_lora_adapter_communicator.assert_not_awaited()

    def test_stage_cancellation_waits_for_worker_completion(self):
        async def scenario():
            tm = _make_tm()
            started, finish = asyncio.Event(), asyncio.Event()

            async def communicate(_):
                started.set()
                await finish.wait()
                return [
                    UpdateAdapterFromDistributedReqOutput(
                        success=True, message="staged"
                    )
                ]

            tm.update_adapter_from_distributed_communicator = communicate
            staging = asyncio.create_task(
                tm.update_adapter_from_distributed(_stage_req())
            )
            await started.wait()
            staging.cancel()
            await asyncio.sleep(0)
            staging.cancel()
            await asyncio.sleep(0)
            self.assertTrue(tm.lora_update_lock.locked())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await staging
            self.assertFalse(tm.lora_update_lock.locked())
            self.assertIsNotNone(tm.pending_lora_stage)

        asyncio.run(scenario())

    def test_rejects_equal_or_stale_versions_before_staging(self):
        tm = _make_tm()
        old = LoRARef(
            lora_id="id-a",
            lora_name="policy",
            lora_path="__distributed__",
            version=4,
        )
        asyncio.run(tm.lora_registry.register(old))

        for version in ("4", "3"):
            with self.assertRaisesRegex(ValueError, "newer than active version 4"):
                asyncio.run(
                    tm.update_adapter_from_distributed(_stage_req(version=version))
                )

        tm.update_adapter_from_distributed_communicator.assert_not_awaited()

    def test_conflicting_stage_reports_pending_identity(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))

        with self.assertRaisesRegex(ValueError, r"name=policy.*id=.*version=4"):
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

    def test_paused_activation_rejects_when_requests_are_retained(self):
        async def scenario():
            tm = _make_tm()
            await tm.update_adapter_from_distributed(_stage_req())
            tm.is_pause = True

            async with tm.model_update_lock.reader_lock:
                success, message = await tm.activate_adapter_version(_activate_req())

            self.assertFalse(success)
            self.assertIn("paused requests are still active", message)
            tm.activate_adapter_version_communicator.assert_not_awaited()
            self.assertIsNotNone(tm.pending_lora_stage)

        asyncio.run(scenario())

    def test_paused_activation_succeeds_after_requests_are_drained(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))
        tm.is_pause = True

        success, _ = asyncio.run(tm.activate_adapter_version(_activate_req()))

        self.assertTrue(success)
        tm.activate_adapter_version_communicator.assert_awaited_once()
        self.assertIsNone(tm.pending_lora_stage)

    def test_paused_immediate_activation_rejects_retained_requests(self):
        async def scenario():
            tm = _make_tm()
            tm.is_pause = True

            async with tm.model_update_lock.reader_lock:
                success, message = await tm.update_adapter_from_distributed(
                    _stage_req(double_buffer=False)
                )

            self.assertFalse(success)
            self.assertIn("paused requests are still active", message)
            tm.update_adapter_from_distributed_communicator.assert_not_awaited()
            self.assertIsNotNone(tm.pending_lora_stage)

        asyncio.run(scenario())

    def test_immediate_activation_cancellation_waits_for_publication(self):
        async def scenario():
            tm = _make_tm()
            update_started = asyncio.Event()
            finish_update = asyncio.Event()

            async def communicate(_):
                update_started.set()
                await finish_update.wait()
                return [
                    UpdateAdapterFromDistributedReqOutput(
                        success=True,
                        message="activated",
                        staged_adapter_version="4",
                        active_adapter_version="4",
                    )
                ]

            tm.update_adapter_from_distributed_communicator = communicate
            activation = asyncio.create_task(
                tm.update_adapter_from_distributed(_stage_req(double_buffer=False))
            )
            await update_started.wait()
            activation.cancel()
            await asyncio.sleep(0)
            activation.cancel()
            await asyncio.sleep(0)
            self.assertTrue(await tm.model_update_lock.is_locked())

            finish_update.set()
            with self.assertRaises(asyncio.CancelledError):
                await activation

            self.assertFalse(await tm.model_update_lock.is_locked())
            self.assertIsNone(tm.pending_lora_stage)

        asyncio.run(scenario())

    def test_activation_cancellation_waits_for_publication(self):
        async def scenario():
            tm = _make_tm()
            await tm.update_adapter_from_distributed(_stage_req())
            update_started = asyncio.Event()
            finish_update = asyncio.Event()

            async def communicate(_):
                update_started.set()
                await finish_update.wait()
                return [
                    ActivateAdapterVersionReqOutput(
                        success=True,
                        message="activated",
                        active_adapter_version="4",
                    )
                ]

            tm.activate_adapter_version_communicator = communicate
            activation = asyncio.create_task(
                tm.activate_adapter_version(_activate_req())
            )
            await update_started.wait()
            activation.cancel()
            await asyncio.sleep(0)
            activation.cancel()
            await asyncio.sleep(0)
            self.assertTrue(await tm.model_update_lock.is_locked())

            finish_update.set()
            with self.assertRaises(asyncio.CancelledError):
                await activation

            self.assertFalse(await tm.model_update_lock.is_locked())
            self.assertIsNone(tm.pending_lora_stage)

        asyncio.run(scenario())

    def test_oft_staging_rejects_multiple_tokenizer_workers(self):
        tm = _make_tm(tokenizer_worker_num=2)
        tm.server_args.enable_lora_staging = False
        tm.server_args.peft_method = "oft"
        req = _stage_req()
        req.load_format = "oft_adapter"

        with self.assertRaisesRegex(ValueError, "tokenizer_worker_num == 1"):
            asyncio.run(tm.update_adapter_from_distributed(req))

        tm.update_adapter_from_distributed_communicator.assert_not_awaited()

    def test_pause_cannot_freeze_requests_while_activation_is_draining(self):
        async def scenario():
            tm = _make_tm()
            await tm.update_adapter_from_distributed(_stage_req())
            waiting_for_writer = asyncio.Event()
            acquire_writer = tm.model_update_lock.acquire_writer

            async def tracked_acquire_writer():
                waiting_for_writer.set()
                await acquire_writer()

            tm.model_update_lock.acquire_writer = tracked_acquire_writer

            async with tm.model_update_lock.reader_lock:
                activation = asyncio.create_task(
                    tm.activate_adapter_version(_activate_req())
                )
                await waiting_for_writer.wait()

                pause_entered = asyncio.Event()

                async def pause():
                    async with tm.is_pause_cond:
                        pause_entered.set()
                        tm.is_pause = True

                pausing = asyncio.create_task(pause())
                await asyncio.sleep(0)
                self.assertFalse(pause_entered.is_set())

            success, _ = await activation
            await pausing
            self.assertTrue(success)
            self.assertTrue(pause_entered.is_set())

        asyncio.run(scenario())

    def test_unload_winning_race_invalidates_pending_activation(self):
        async def scenario():
            tm = _make_tm()
            active = LoRARef(
                lora_id="id-a",
                lora_name="policy",
                lora_path="__distributed__",
                version=3,
            )
            await tm.lora_registry.register(active)
            tm.lora_ref_cache["policy"] = active
            await tm.update_adapter_from_distributed(_stage_req())
            waiting_for_writer = asyncio.Event()
            acquire_writer = tm.model_update_lock.acquire_writer

            async def tracked_acquire_writer():
                waiting_for_writer.set()
                await acquire_writer()

            tm.model_update_lock.acquire_writer = tracked_acquire_writer
            async with tm.model_update_lock.reader_lock:
                activation = asyncio.create_task(
                    tm.activate_adapter_version(_activate_req())
                )
                await waiting_for_writer.wait()
                unload_result = await tm.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name="policy")
                )
                self.assertTrue(unload_result.success)
                self.assertIsNone(tm.pending_lora_stage)

            with self.assertRaisesRegex(ValueError, "no native LoRA stage is pending"):
                await activation
            tm.activate_adapter_version_communicator.assert_not_awaited()
            self.assertNotIn("policy", tm.lora_registry.get_all_adapters())

        asyncio.run(scenario())

    def test_activation_winning_race_serializes_following_unload(self):
        async def scenario():
            tm = _make_tm()
            active = LoRARef(
                lora_id="id-a",
                lora_name="policy",
                lora_path="__distributed__",
                version=3,
            )
            await tm.lora_registry.register(active)
            tm.lora_ref_cache["policy"] = active
            await tm.update_adapter_from_distributed(_stage_req())
            activation_started = asyncio.Event()
            finish_activation = asyncio.Event()
            unload_started = asyncio.Event()

            async def activate(_):
                activation_started.set()
                await finish_activation.wait()
                return [
                    ActivateAdapterVersionReqOutput(
                        success=True,
                        message="activated",
                        active_adapter_version="4",
                    )
                ]

            async def unload(_):
                unload_started.set()
                return [LoRAUpdateOutput(success=True)]

            tm.activate_adapter_version_communicator = activate
            tm.update_lora_adapter_communicator = unload
            activation = asyncio.create_task(
                tm.activate_adapter_version(_activate_req())
            )
            await activation_started.wait()
            unloading = asyncio.create_task(
                tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="policy"))
            )
            await asyncio.sleep(0)
            self.assertFalse(unload_started.is_set())

            finish_activation.set()
            self.assertTrue((await activation)[0])
            self.assertTrue((await unloading).success)
            self.assertTrue(unload_started.is_set())
            self.assertNotIn("policy", tm.lora_registry.get_all_adapters())

        asyncio.run(scenario())

    def test_failed_unload_still_invalidates_pending_activation(self):
        tm = _make_tm()
        asyncio.run(tm.update_adapter_from_distributed(_stage_req()))
        tm.update_lora_adapter_communicator = AsyncMock(
            return_value=[
                LoRAUpdateOutput(success=False, error_message="rank 1 failed")
            ]
        )

        result = asyncio.run(
            tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="policy"))
        )

        self.assertFalse(result.success)
        self.assertIsNone(tm.pending_lora_stage)
        self.assertIn("policy", tm.failed_lora_activations)
        with self.assertRaisesRegex(ValueError, "failed unload"):
            asyncio.run(tm.update_adapter_from_distributed(_stage_req()))

    def test_unload_cancelled_before_lock_keeps_adapter_available(self):
        async def scenario():
            tm = _make_tm()
            ref = LoRARef(lora_name="policy", lora_path="/disk/policy")
            await tm.lora_registry.register(ref)
            tm.lora_ref_cache["policy"] = ref
            async with tm.lora_update_lock:
                task = asyncio.create_task(
                    tm.unload_lora_adapter(
                        UnloadLoRAAdapterReqInput(lora_name="policy")
                    )
                )
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertIs(tm.lora_registry.get_all_adapters()["policy"], ref)
            self.assertIs(tm.lora_ref_cache["policy"], ref)
            self.assertEqual(tm.failed_lora_unloads, {})

        asyncio.run(scenario())

    def test_active_unload_cancellation_finishes_or_preserves_retry(self):
        async def scenario(cancel_at, worker_success):
            tm = _make_tm()
            ref = LoRARef(lora_name="policy", lora_path="/disk/policy")
            await tm.lora_registry.register(ref)
            tm.lora_ref_cache["policy"] = ref
            await tm.lora_registry.acquire_with_version("policy")
            lease_held = True
            waiting = asyncio.Event()
            dispatched = asyncio.Event()
            finish_worker = asyncio.Event()
            original_wait = tm.lora_registry.wait_for_unload
            calls = []

            async def wait_for_unload(uid):
                waiting.set()
                await original_wait(uid)

            async def communicate(obj):
                calls.append(obj.lora_id)
                dispatched.set()
                await finish_worker.wait()
                return [
                    LoRAUpdateOutput(success=True),
                    LoRAUpdateOutput(
                        success=worker_success,
                        error_message=None if worker_success else "rank 1 failed",
                    ),
                ]

            tm.lora_registry.wait_for_unload = wait_for_unload
            tm.update_lora_adapter_communicator = communicate
            task = asyncio.create_task(
                tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="policy"))
            )
            try:
                await asyncio.wait_for(waiting.wait(), 1)
                if cancel_at == "worker":
                    await tm.lora_registry.release(ref.lora_id)
                    lease_held = False
                    await asyncio.wait_for(dispatched.wait(), 1)
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    self.assertTrue(tm.lora_update_lock.locked())
                if cancel_at == "lease":
                    self.assertEqual(calls, [])
                    await tm.lora_registry.release(ref.lora_id)
                    lease_held = False
                await asyncio.wait_for(dispatched.wait(), 1)
                finish_worker.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                self.assertEqual(calls, [ref.lora_id])
                self.assertFalse(tm.lora_update_lock.locked())
                self.assertNotIn("policy", tm.lora_registry.get_all_adapters())
                if not worker_success:
                    self.assertIs(tm.failed_lora_unloads["policy"], ref)
                    self.assertIs(tm.lora_ref_cache["policy"], ref)

                    async def retry(obj):
                        self.assertEqual(obj.lora_id, ref.lora_id)
                        return [LoRAUpdateOutput(success=True)]

                    tm.update_lora_adapter_communicator = retry
                    result = await tm.unload_lora_adapter(
                        UnloadLoRAAdapterReqInput(lora_name="policy")
                    )
                    self.assertTrue(result.success)
                self.assertNotIn("policy", tm.failed_lora_unloads)
                self.assertNotIn("policy", tm.lora_ref_cache)
            finally:
                # Let protected cleanup finish even if an assertion fails.
                finish_worker.set()
                if lease_held:
                    await tm.lora_registry.release(ref.lora_id)
                    lease_held = False
                await asyncio.gather(task, return_exceptions=True)

        for cancel_at in ("lease", "worker"):
            for worker_success in (True, False):
                with self.subTest(cancel_at=cancel_at, worker_success=worker_success):
                    asyncio.run(scenario(cancel_at, worker_success))

    def test_pending_unload_cancellation_waits_for_worker_completion(self):
        async def scenario():
            tm = _make_tm()
            await tm.update_adapter_from_distributed(_stage_req())
            unload_started = asyncio.Event()
            finish_unload = asyncio.Event()

            async def communicate(_):
                unload_started.set()
                await finish_unload.wait()
                return [LoRAUpdateOutput(success=True)]

            tm.update_lora_adapter_communicator = communicate
            unloading = asyncio.create_task(
                tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="policy"))
            )
            await unload_started.wait()
            unloading.cancel()
            await asyncio.sleep(0)
            unloading.cancel()
            await asyncio.sleep(0)
            self.assertTrue(tm.lora_update_lock.locked())

            finish_unload.set()
            with self.assertRaises(asyncio.CancelledError):
                await unloading

            self.assertFalse(tm.lora_update_lock.locked())
            self.assertIsNone(tm.pending_lora_stage)

        asyncio.run(scenario())

    def test_failed_unload_retry_cancellation_keeps_operation_serialized(self):
        from sglang.srt.managers.communicator import FanOutCommunicator
        from sglang.srt.oft.io_types import OFTUpdateOutput, UnloadOFTAdapterReqInput
        from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry

        async def scenario(method):
            tm = _make_tm()
            if method == "lora":
                ref = LoRARef(lora_id="old-id", lora_name="policy")
                tm.failed_lora_unloads["policy"] = ref
                tm.lora_ref_cache["policy"] = ref
                request = UnloadLoRAAdapterReqInput(lora_name="policy")
                unload = tm.unload_lora_adapter
                output = LoRAUpdateOutput
                lifecycle_lock = tm.lora_update_lock
                fan_out = 2
            else:
                tm.server_args.peft_method = "oft"
                tm.peft_registry = OFTRegistry()
                tm.peft_ref_cache = {}
                tm.peft_update_lock = asyncio.Lock()
                tm.pending_oft_stage = None
                tm.failed_oft_activations = {}
                tm.failed_oft_unloads = {}
                ref = OFTRef(adapter_id="old-id", adapter_name="policy")
                tm.failed_oft_unloads["policy"] = ref
                tm.peft_ref_cache["policy"] = ref
                request = UnloadOFTAdapterReqInput(adapter_name="policy")
                unload = tm.unload_oft_adapter
                output = OFTUpdateOutput
                lifecycle_lock = tm.peft_update_lock
                fan_out = 1

            sends = []
            started = asyncio.Event()

            def send(obj):
                sends.append(obj)
                started.set()

            communicator = FanOutCommunicator(send, fan_out=fan_out)
            setattr(tm, f"update_{method}_adapter_communicator", communicator)
            first = asyncio.create_task(unload(request))
            await started.wait()
            first.cancel()
            await asyncio.sleep(0)
            self.assertTrue(lifecycle_lock.locked())

            second = asyncio.create_task(unload(request))
            await asyncio.sleep(0)
            self.assertEqual(len(sends), 1)

            for _ in range(fan_out):
                communicator.handle_recv(output(success=True))
            with self.assertRaises(asyncio.CancelledError):
                await first

            second_result = await second
            self.assertFalse(second_result.success)
            self.assertEqual(len(sends), 1)
            self.assertNotIn("policy", getattr(tm, f"failed_{method}_unloads"))
            cache = tm.lora_ref_cache if method == "lora" else tm.peft_ref_cache
            self.assertNotIn("policy", cache)

        for method in ("lora", "oft"):
            with self.subTest(method=method):
                asyncio.run(scenario(method))

    def test_same_name_load_is_rejected_until_pending_stage_is_unloaded(self):
        tm = _make_tm()
        self.assertTrue(
            asyncio.run(tm.update_adapter_from_distributed(_stage_req()))[0]
        )

        load_result = asyncio.run(
            tm.load_lora_adapter(
                LoadLoRAAdapterReqInput(lora_name="policy", lora_path="/disk/policy")
            )
        )

        self.assertFalse(load_result.success)
        self.assertIn("staged version 4 is pending", load_result.error_message)
        self.assertIsNotNone(tm.pending_lora_stage)
        tm.update_lora_adapter_communicator.assert_not_awaited()

        unload_result = asyncio.run(
            tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="policy"))
        )
        self.assertTrue(unload_result.success)
        self.assertIsNone(tm.pending_lora_stage)
        self.assertTrue(
            asyncio.run(tm.update_adapter_from_distributed(_stage_req()))[0]
        )

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

        success, message = asyncio.run(tm.activate_adapter_version(_activate_req()))

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

        success, message = asyncio.run(tm.activate_adapter_version(_activate_req()))

        self.assertFalse(success)
        self.assertIn("restart required", message)
        self.assertIn("policy", tm.failed_lora_activations)


class TestTokenizerStagedIdentity(unittest.TestCase):
    """Explicit IDs are assertions, never silently repaired before dispatch."""

    def _setup(self, method, *, active=True):
        tm = _make_tm()
        stage, activation = _stage_req(), _activate_req()
        if method == "lora":
            registry, cache = tm.lora_registry, tm.lora_ref_cache
            ref = LoRARef(lora_id="id-a", lora_name="policy", version=3)
        else:
            from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry

            tm.server_args.enable_lora_staging = False
            tm.server_args.peft_method = "oft"
            tm.peft_registry = registry = OFTRegistry()
            tm.peft_ref_cache = cache = {}
            tm.peft_update_lock = asyncio.Lock()
            tm.pending_oft_stage = None
            tm.failed_oft_activations = {}
            tm.failed_oft_unloads = {}
            stage.load_format = activation.load_format = "oft_adapter"
            ref = OFTRef(adapter_id="id-a", adapter_name="policy", adapter_version=3)
        if active:
            asyncio.run(registry.register(ref))
            cache["policy"] = ref
        return tm, stage, activation, registry, cache

    def _snapshot(self, tm, method, registry, cache):
        return deepcopy(
            (
                list(registry.get_all_adapters().items()),
                cache,
                getattr(tm, f"pending_{method}_stage"),
                getattr(tm, f"failed_{method}_activations"),
                getattr(tm, f"failed_{method}_unloads"),
            )
        )

    def test_active_stage_wrong_id_rejected_before_dispatch_without_mutation(self):
        for method in ("lora", "oft"):
            for requested_id in ("wrong-id", ""):
                with self.subTest(method=method, requested_id=requested_id):
                    tm, req, _, registry, cache = self._setup(method)
                    req.adapter_id = requested_id
                    before = self._snapshot(tm, method, registry, cache)

                    with self.assertRaises(ValueError) as error:
                        asyncio.run(tm.update_adapter_from_distributed(req))

                    self.assertIn(repr(requested_id), str(error.exception))
                    self.assertIn("id-a", str(error.exception))
                    self.assertEqual(req.adapter_id, requested_id)
                    self.assertEqual(
                        self._snapshot(tm, method, registry, cache), before
                    )
                    tm.update_adapter_from_distributed_communicator.assert_not_awaited()
                    tm.activate_adapter_version_communicator.assert_not_awaited()

    def test_pending_retry_wrong_id_rejected_without_second_dispatch_or_mutation(self):
        for method in ("lora", "oft"):
            with self.subTest(method=method):
                tm, req, _, registry, cache = self._setup(method)
                self.assertTrue(asyncio.run(tm.update_adapter_from_distributed(req))[0])
                pending = getattr(tm, f"pending_{method}_stage")
                before = self._snapshot(tm, method, registry, cache)
                req.adapter_id = "wrong-id"

                with self.assertRaisesRegex(ValueError, "wrong-id.*id-a"):
                    asyncio.run(tm.update_adapter_from_distributed(req))

                self.assertEqual(req.adapter_id, "wrong-id")
                self.assertIs(getattr(tm, f"pending_{method}_stage"), pending)
                self.assertEqual(self._snapshot(tm, method, registry, cache), before)
                tm.update_adapter_from_distributed_communicator.assert_awaited_once()
                tm.activate_adapter_version_communicator.assert_not_awaited()

    def test_activation_wrong_id_rejected_before_dispatch_without_mutation(self):
        for method in ("lora", "oft"):
            with self.subTest(method=method):
                tm, stage, activation, registry, cache = self._setup(method)
                self.assertTrue(
                    asyncio.run(tm.update_adapter_from_distributed(stage))[0]
                )
                pending = getattr(tm, f"pending_{method}_stage")
                before = self._snapshot(tm, method, registry, cache)
                activation.adapter_id = "wrong-id"

                with self.assertRaisesRegex(ValueError, "wrong-id.*id-a"):
                    asyncio.run(tm.activate_adapter_version(activation))

                self.assertEqual(activation.adapter_id, "wrong-id")
                self.assertIs(getattr(tm, f"pending_{method}_stage"), pending)
                self.assertEqual(self._snapshot(tm, method, registry, cache), before)
                tm.update_adapter_from_distributed_communicator.assert_awaited_once()
                tm.activate_adapter_version_communicator.assert_not_awaited()

    def test_stage_retry_and_activation_accept_omitted_or_exact_id(self):
        for method in ("lora", "oft"):
            for requested_id in (None, "id-a"):
                with self.subTest(method=method, requested_id=requested_id):
                    tm, stage, activation, registry, cache = self._setup(method)
                    stage.adapter_id = requested_id
                    self.assertTrue(
                        asyncio.run(tm.update_adapter_from_distributed(stage))[0]
                    )
                    pending = getattr(tm, f"pending_{method}_stage")
                    stage.adapter_id = requested_id
                    self.assertTrue(
                        asyncio.run(tm.update_adapter_from_distributed(stage))[0]
                    )
                    self.assertEqual(stage.adapter_id, "id-a")
                    self.assertIs(getattr(tm, f"pending_{method}_stage"), pending)
                    activation.adapter_id = requested_id

                    self.assertTrue(
                        asyncio.run(tm.activate_adapter_version(activation))[0]
                    )

                    self.assertEqual(activation.adapter_id, "id-a")
                    self.assertIs(registry.get_all_adapters()["policy"], pending)
                    self.assertIs(cache["policy"], pending)
                    self.assertIsNone(getattr(tm, f"pending_{method}_stage"))

    def test_synchronous_stage_accepts_resolved_id_and_publishes(self):
        for method in ("lora", "oft"):
            for requested_id in (None, "id-a"):
                with self.subTest(method=method, requested_id=requested_id):
                    tm, stage, _, registry, cache = self._setup(method)
                    stage.double_buffer = False
                    stage.adapter_id = requested_id
                    tm.update_adapter_from_distributed_communicator.return_value = [
                        UpdateAdapterFromDistributedReqOutput(
                            success=True,
                            message="activated",
                            active_adapter_version="4",
                        )
                    ]

                    self.assertTrue(
                        asyncio.run(tm.update_adapter_from_distributed(stage))[0]
                    )

                    self.assertEqual(stage.adapter_id, "id-a")
                    self.assertIsNone(getattr(tm, f"pending_{method}_stage"))
                    self.assertIs(
                        registry.get_all_adapters()["policy"], cache["policy"]
                    )
                    version = (
                        cache["policy"].version
                        if method == "lora"
                        else cache["policy"].adapter_version
                    )
                    self.assertEqual(version, 4)

    def test_first_stage_mints_identity_even_when_caller_supplies_id(self):
        for method in ("lora", "oft"):
            with self.subTest(method=method):
                tm, stage, _, registry, _ = self._setup(method, active=False)
                stage.adapter_id = "caller-id"

                self.assertTrue(
                    asyncio.run(tm.update_adapter_from_distributed(stage))[0]
                )

                self.assertTrue(stage.adapter_id)
                self.assertNotEqual(stage.adapter_id, "caller-id")
                self.assertEqual(registry.get_all_adapters(), {})
                self.assertIsNotNone(getattr(tm, f"pending_{method}_stage"))


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

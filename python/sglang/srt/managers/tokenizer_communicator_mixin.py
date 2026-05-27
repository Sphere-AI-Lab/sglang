from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from collections import deque
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import fastapi
import zmq

from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    LoadOFTAdapterFromTensorsReqInput,
    LoadOFTAdapterFromTensorsReqOutput,
    LoadOFTAdapterReqInput,
    LoadOFTAdapterReqOutput,
    LoRAUpdateOutput,
    OFTUpdateOutput,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UnloadOFTAdapterReqInput,
    UnloadOFTAdapterReqOutput,
    UpdateAdapterFromDistributedReqInput,
    UpdateAdapterFromDistributedReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.server_args import LoRARef, ServerArgs
from sglang.srt.utils import get_bool_env_var
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

T = TypeVar("T")

logger = logging.getLogger(__name__)


class InactiveSlotBusyError(Exception):
    """Raised when a retiring adapter version cannot drain in time.

    Surfaced as a structured 200-OK body by /update_adapter_from_distributed
    so Orbit's NCCL backend can produce a friendly retire-busy error message
    instead of a generic 500.
    """

    def __init__(
        self,
        public_name: str,
        retiring_adapter_id: str,
        in_flight: Optional[int],
    ):
        self.public_name = public_name
        self.retiring_adapter_id = retiring_adapter_id
        self.in_flight = in_flight
        super().__init__(
            f"inactive_slot_busy: public_name={public_name!r} "
            f"retiring_adapter_id={retiring_adapter_id!r} in_flight={in_flight!r}"
        )


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Future] = deque()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj:
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj:
                self._sender.send_pyobj(obj)

        await self._result_event.wait()
        result_values = copy.deepcopy(self._result_values)
        self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message


class TokenizerCommunicatorMixin:
    """Mixin class for TokenizerManager to handle communication with the scheduler."""

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        # Communicators
        self.init_weights_update_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.destroy_weights_update_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_adapter_from_distributed_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.init_weights_send_group_for_remote_instance_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.send_weights_to_remote_instance_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_tensor_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_ipc_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_weights_by_name_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.check_weights_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.slow_down_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.flush_cache_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.clear_hicache_storage_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.attach_hicache_storage_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.detach_hicache_storage_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.profile_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.expert_distribution_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_lora_adapter_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_oft_adapter_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_load_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size, mode="watching"
        )
        self.get_loads_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.dumper_control_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        self._result_dispatcher += self._get_communicator_dispatcher()

    def _get_communicator_dispatcher(self: TokenizerManager):
        return TypeBasedDispatcher(
            [
                (
                    InitWeightsUpdateGroupReqOutput,
                    self.init_weights_update_group_communicator.handle_recv,
                ),
                (
                    DestroyWeightsUpdateGroupReqOutput,
                    self.destroy_weights_update_group_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
                ),
                (
                    UpdateAdapterFromDistributedReqOutput,
                    self.update_adapter_from_distributed_communicator.handle_recv,
                ),
                (
                    InitWeightsSendGroupForRemoteInstanceReqOutput,
                    self.init_weights_send_group_for_remote_instance_communicator.handle_recv,
                ),
                (
                    SendWeightsToRemoteInstanceReqOutput,
                    self.send_weights_to_remote_instance_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromTensorReqOutput,
                    self.update_weights_from_tensor_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromIPCReqOutput,
                    self.update_weights_from_ipc_communicator.handle_recv,
                ),
                (
                    GetWeightsByNameReqOutput,
                    self.get_weights_by_name_communicator.handle_recv,
                ),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    CheckWeightsReqOutput,
                    self.check_weights_communicator.handle_recv,
                ),
                (
                    SlowDownReqOutput,
                    self.slow_down_communicator.handle_recv,
                ),
                (
                    ClearHiCacheReqOutput,
                    self.clear_hicache_storage_communicator.handle_recv,
                ),
                (
                    AttachHiCacheStorageReqOutput,
                    self.attach_hicache_storage_communicator.handle_recv,
                ),
                (
                    DetachHiCacheStorageReqOutput,
                    self.detach_hicache_storage_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    SetInternalStateReqOutput,
                    self.set_internal_state_communicator.handle_recv,
                ),
                (
                    ExpertDistributionReqOutput,
                    self.expert_distribution_communicator.handle_recv,
                ),
                (
                    LoRAUpdateOutput,
                    self.update_lora_adapter_communicator.handle_recv,
                ),
                (
                    OFTUpdateOutput,
                    self.update_oft_adapter_communicator.handle_recv,
                ),
                (
                    GetLoadReqOutput,
                    self.get_load_communicator.handle_recv,
                ),
                (
                    GetLoadsReqOutput,
                    self.get_loads_communicator.handle_recv,
                ),
                (
                    DumperControlReqOutput,
                    self.dumper_control_communicator.handle_recv,
                ),
            ]
        )

    async def flush_cache(self: TokenizerManager) -> FlushCacheReqOutput:
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    async def clear_hicache_storage(self: TokenizerManager) -> ClearHiCacheReqOutput:
        """Clear the hierarchical cache storage."""
        # Delegate to the scheduler to handle HiCacheStorage clearing
        return (await self.clear_hicache_storage_communicator(ClearHiCacheReqInput()))[
            0
        ]

    async def attach_hicache_storage(
        self: TokenizerManager,
        hicache_storage_backend: str,
        hicache_storage_backend_extra_config_json: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> AttachHiCacheStorageReqOutput:
        """Attach (enable) HiCache storage backend at runtime."""
        results = await self.attach_hicache_storage_communicator(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend=hicache_storage_backend,
                hicache_storage_backend_extra_config_json=hicache_storage_backend_extra_config_json,
                hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
                hicache_write_policy=hicache_write_policy,
            )
        )

        all_success, all_message = _Communicator.merge_results(results)
        out = AttachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            # Keep tokenizer side server_info consistent with scheduler side.
            self.server_args.hicache_storage_backend = hicache_storage_backend
            if hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    hicache_storage_backend_extra_config_json
                )
            if hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    hicache_storage_prefetch_policy
                )
            if hicache_write_policy is not None:
                self.server_args.hicache_write_policy = hicache_write_policy
        return out

    async def detach_hicache_storage(
        self: TokenizerManager,
    ) -> DetachHiCacheStorageReqOutput:
        """Detach (disable) HiCache storage backend at runtime."""
        results = await self.detach_hicache_storage_communicator(
            DetachHiCacheStorageReqInput()
        )

        all_success, all_message = _Communicator.merge_results(results)
        out = DetachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
        return out

    async def start_profile(
        self: TokenizerManager,
        output_dir: Optional[str] = None,
        start_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        profile_by_stage: bool = False,
        merge_profiles: bool = False,
        profile_prefix: Optional[str] = None,
        profile_stages: Optional[List[str]] = None,
    ):
        self.auto_create_handle_loop()
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        with_stack = False if with_stack is False or env_with_stack is False else True
        env_record_shapes: bool = get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )
        record_shapes = (record_shapes is not False) and env_record_shapes
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=merge_profiles,
            profile_prefix=profile_prefix,
            profile_stages=profile_stages,
        )
        return await self._execute_profile(req)

    async def stop_profile(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self: TokenizerManager, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def start_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.START_RECORD)
        await self.expert_distribution_communicator(req)

    async def stop_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.STOP_RECORD)
        await self.expert_distribution_communicator(req)

    async def dump_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.DUMP_RECORD)
        await self.expert_distribution_communicator(req)

    async def init_weights_update_group(
        self: TokenizerManager,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        results = await self.init_weights_update_group_communicator(obj)
        return _Communicator.merge_results(results)

    async def destroy_weights_update_group(
        self,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"

        results = await self.destroy_weights_update_group_communicator(obj)
        return _Communicator.merge_results(results)

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # For OFT adapter updates, register in tokenizer_manager BEFORE sending to
        # scheduler, so the adapter_id flows to the model_runner for consistent mapping.
        if obj.load_format == "oft_adapter" and obj.adapter_name is not None:
            from sglang.srt.oft.oft_registry import OFTRef

            adapter_name = obj.adapter_name
            if adapter_name not in self.oft_ref_cache:
                new_ref = OFTRef(
                    oft_name=adapter_name, oft_path=adapter_name, pinned=False
                )
                await self.oft_registry.register(new_ref)
                self.oft_ref_cache[adapter_name] = new_ref
            obj.adapter_id = self.oft_ref_cache[adapter_name].oft_id

        # Immediately update the weights if the engine is in paused state
        async with self.is_pause_cond:
            is_paused = self.is_pause

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            results = await self.update_weights_from_distributed_communicator(obj)

        success, message = _Communicator.merge_results(results)
        if success and obj.load_format == "oft_adapter" and obj.adapter_id is not None:
            updated_ref = await self.oft_registry.bump_version_by_id(obj.adapter_id)
            self.oft_ref_cache[updated_ref.oft_name] = updated_ref
            message += (
                f" OFT adapter {updated_ref.oft_name} version updated to "
                f"{updated_ref.oft_version}."
            )
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def update_adapter_from_distributed(
        self: TokenizerManager,
        obj: UpdateAdapterFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update adapter from distributed"

        public_name: Optional[str] = None
        slot_index: Optional[int] = None
        slot_name: Optional[str] = None

        if obj.double_buffer and obj.adapter_name is not None:
            # Double-buffer path: stage the new weights into a fresh slot whose name
            # is `{public}__orbit_slot_{index}`. The slot index ping-pongs 0/1 across
            # consecutive stages of the same public_name. The communicator (and
            # downstream model_runner) sees obj.adapter_name = slot_name, so the
            # new weights load into a brand-new adapter; activation later re-points
            # the public alias at this slot via LoRARegistry.replace.
            public_name = obj.adapter_name
            slot_index = TokenizerCommunicatorMixin._next_adapter_slot(
                self, public_name
            )
            slot_name = TokenizerCommunicatorMixin._adapter_slot_name(
                self, public_name, slot_index
            )

            # If the same slot index still has a retiring adapter from the previous
            # cycle, wait for in-flight requests to drain and unload it before
            # reusing the slot (timeout-bounded so a leaked counter doesn't hang us).
            retiring_id = self.retiring_adapter_ids.get((public_name, slot_index))
            if retiring_id is not None:
                try:
                    await TokenizerCommunicatorMixin._wait_and_unload_retiring_adapter(
                        self,
                        obj.load_format,
                        public_name,
                        retiring_id,
                        timeout_s=self.adapter_retire_timeout_s,
                    )
                except asyncio.TimeoutError as e:
                    # Spec §9: surface a structured timeout instead of letting it
                    # propagate as a bare 500. Orbit translates this to a clear
                    # retire-busy diagnostic. The retiring entry stays so the next
                    # stage can retry.
                    in_flight = TokenizerCommunicatorMixin._best_effort_in_flight_count(
                        self, obj.load_format, retiring_id
                    )
                    raise InactiveSlotBusyError(
                        public_name, retiring_id, in_flight
                    ) from e
                del self.retiring_adapter_ids[(public_name, slot_index)]

            if obj.load_format == "lora_adapter":
                from sglang.srt.lora.lora_registry import LoRARef

                slot_ref = LoRARef(
                    lora_name=slot_name, lora_path=slot_name, pinned=False
                )
                obj.adapter_id = slot_ref.lora_id
            elif obj.load_format == "oft_adapter":
                from sglang.srt.oft.oft_registry import OFTRef

                slot_ref = OFTRef(
                    oft_name=slot_name, oft_path=slot_name, pinned=False
                )
                obj.adapter_id = slot_ref.oft_id
            else:
                raise ValueError(
                    f"unknown adapter load_format={obj.load_format!r}"
                )

            obj.adapter_name = slot_name
        else:
            # Single-slot path (existing behavior). Mint a fresh ref for the public
            # name and (for LoRA) replace any prior registration in place.
            if obj.load_format == "oft_adapter" and obj.adapter_name is not None:
                from sglang.srt.oft.oft_registry import OFTRef

                adapter_name = obj.adapter_name
                if adapter_name not in self.oft_ref_cache:
                    new_ref = OFTRef(
                        oft_name=adapter_name, oft_path=adapter_name, pinned=False
                    )
                    await self.oft_registry.register(new_ref)
                    self.oft_ref_cache[adapter_name] = new_ref
                obj.adapter_id = self.oft_ref_cache[adapter_name].oft_id

            if obj.load_format == "lora_adapter" and obj.adapter_name is not None:
                from sglang.srt.lora.lora_registry import LoRARef

                adapter_name = obj.adapter_name
                new_ref = LoRARef(
                    lora_name=adapter_name,
                    lora_path=adapter_name,
                    pinned=False,
                )
                obj.adapter_id = new_ref.lora_id
                if adapter_name in self.lora_ref_cache:
                    old_id = await self.lora_registry.unregister(adapter_name)
                    await self.lora_registry.wait_for_unload(old_id)
                await self.lora_registry.register(new_ref)
                self.lora_ref_cache[adapter_name] = new_ref

        # Immediately update the adapter if the engine is in paused state
        async with self.is_pause_cond:
            is_paused = self.is_pause

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            results = await self.update_adapter_from_distributed_communicator(obj)

        success, message = _Communicator.merge_results(results)

        if obj.double_buffer:
            # Record the staged slot only after a successful load; activation will
            # later look it up by (public_name, adapter_version).
            if success and public_name is not None:
                from sglang.srt.managers.tokenizer_manager import (
                    _StagedAdapterVersion,
                )

                self.staged_adapter_versions[(public_name, obj.adapter_version)] = (
                    _StagedAdapterVersion(
                        load_format=obj.load_format,
                        public_name=public_name,
                        slot_name=slot_name,
                        adapter_id=obj.adapter_id,
                        adapter_version=obj.adapter_version,
                        slot_index=slot_index,
                    )
                )
        else:
            if (
                success
                and obj.load_format == "oft_adapter"
                and obj.adapter_id is not None
            ):
                updated_ref = await self.oft_registry.bump_version_by_id(obj.adapter_id)
                self.oft_ref_cache[updated_ref.oft_name] = updated_ref
                message += (
                    f" OFT adapter {updated_ref.oft_name} version updated to "
                    f"{updated_ref.oft_version}."
                )
            if success and obj.weight_version is not None:
                self._update_weight_version_if_provided(obj.weight_version)
                message += f" Weight version updated to {obj.weight_version}."

        return success, message

    def _adapter_slot_name(self, public_name: str, slot_index: int) -> str:
        """Slot-name format: ``{public}__orbit_slot_{index}``.

        Activation flips the public alias between two slot names (index 0 and 1)
        so that no in-flight request ever sees a half-loaded adapter.
        """
        return f"{public_name}__orbit_slot_{slot_index}"

    def _next_adapter_slot(self, public_name: str) -> int:
        """Return the next slot index (0/1) and toggle for the next call.

        Ping-pongs deterministically: 0 -> 1 -> 0 -> 1 ...
        """
        next_slot = self.adapter_version_slots.get(public_name, 0)
        self.adapter_version_slots[public_name] = 1 - next_slot
        return next_slot

    async def activate_adapter_version(
        self: TokenizerManager,
        obj: ActivateAdapterVersionReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Atomically swap the public adapter alias to a previously-staged slot.

        Looks up the staged record by (adapter_name, adapter_version), calls the
        registry's ``replace`` to re-route future acquires to the new slot's id,
        records the old id under ``retiring_adapter_ids`` so the next stage at
        the same slot drains and unloads it, and bumps the live weight_version.

        The retire is intentionally deferred to the next stage at the same slot
        (rather than fired here as a background task) to keep a single owner for
        ``wait_for_unload(old_id)``. Two callers racing on that primitive corrupt
        the registry's counter dict (one wins ``del self._counters[id]``, the
        other asserts). With two-slot ping-pong, the retiring slot is naturally
        reclaimed when the opposite slot stages back, so eager retirement saves
        nothing — memory is already bounded at exactly 2 slots by design.
        """
        staged_key = (obj.adapter_name, obj.adapter_version)
        staged = self.staged_adapter_versions.get(staged_key)
        if staged is None:
            return False, f"No staged adapter version for {staged_key}"

        if obj.load_format == "lora_adapter":
            from sglang.srt.lora.lora_registry import LoRARef

            old_id = await self.lora_registry.replace(
                LoRARef(
                    lora_id=staged.adapter_id,
                    lora_name=staged.public_name,
                    lora_path=staged.slot_name,
                    pinned=False,
                )
            )
            self.lora_ref_cache[staged.public_name] = LoRARef(
                lora_id=staged.adapter_id,
                lora_name=staged.public_name,
                lora_path=staged.slot_name,
                pinned=False,
            )
        elif obj.load_format == "oft_adapter":
            from sglang.srt.oft.oft_registry import OFTRef

            old_id = await self.oft_registry.replace(
                OFTRef(
                    oft_id=staged.adapter_id,
                    oft_name=staged.public_name,
                    oft_path=staged.slot_name,
                    pinned=False,
                )
            )
            self.oft_ref_cache[staged.public_name] = OFTRef(
                oft_id=staged.adapter_id,
                oft_name=staged.public_name,
                oft_path=staged.slot_name,
                pinned=False,
            )
        else:
            return False, f"unknown adapter load_format={obj.load_format!r}"

        if old_id is not None:
            # Track the retiring id under the slot where the OLD adapter
            # physically lives. With two-slot ping-pong, the previous activate
            # used slot ``1 - staged.slot_index``, so the retiring adapter is
            # there. The next stage that ping-pongs back to that slot looks
            # up this key and drains/unloads before reusing the slot.
            #
            # Using ``staged.slot_index`` here would be wrong: it's the slot
            # the new adapter just took, not the slot whose contents need to
            # drain. The lookup at the start of ``update_adapter_from_distributed``
            # would then miss, the next stage would silently overwrite the
            # still-loaded old slot, and the *following* stage's drain would
            # try to unload an id that the model_runner already evicted —
            # crashing the server in ``oft_manager.unload_oft_adapter``.
            retiring_slot = 1 - staged.slot_index
            self.retiring_adapter_ids[(staged.public_name, retiring_slot)] = (
                old_id
            )

        self.active_adapter_versions[staged.public_name] = staged.adapter_version
        self._update_weight_version_if_provided(obj.weight_version)
        del self.staged_adapter_versions[staged_key]
        return (
            True,
            f"Activated adapter_version={obj.adapter_version} for {obj.adapter_name}",
        )

    async def _wait_and_unload_retiring_adapter(
        self: TokenizerManager,
        load_format: str,
        public_name: str,
        adapter_id: str,
        *,
        timeout_s: float,
    ) -> None:
        """Wait for in-flight requests to release the retiring id, then unload.

        ``LoRARegistry.replace`` deletes the alias but keeps the counter so that
        in-flight requests can release their reference. After the counter hits
        zero we send the streamed-unload-by-id to the model_runner so the slot
        memory is freed; this is what allows the next stage at the same slot to
        reuse it.
        """
        if load_format == "lora_adapter":
            await asyncio.wait_for(
                self.lora_registry.wait_for_unload(adapter_id), timeout=timeout_s
            )
            result = (
                await self.update_lora_adapter_communicator(
                    UnloadLoRAAdapterReqInput(
                        lora_name=public_name, lora_id=adapter_id
                    )
                )
            )[0]
            if not result.success:
                raise RuntimeError(result.error_message)
        elif load_format == "oft_adapter":
            await asyncio.wait_for(
                self.oft_registry.wait_for_unload(adapter_id), timeout=timeout_s
            )
            result = (
                await self.update_oft_adapter_communicator(
                    UnloadOFTAdapterReqInput(oft_name=public_name, oft_id=adapter_id)
                )
            )[0]
            if not result.success:
                raise RuntimeError(result.error_message)
        else:
            raise ValueError(f"unknown adapter load_format={load_format!r}")

    def _best_effort_in_flight_count(
        self: TokenizerManager, load_format: str, adapter_id: str
    ) -> Optional[int]:
        """Read the in-flight refcount for a given adapter_id. Best-effort: returns
        None if the registry's internal counter is not introspectable.

        Used for ``inactive_slot_busy`` diagnostics; treat None as "unknown"
        downstream.
        """
        try:
            if load_format == "lora_adapter":
                counter = self.lora_registry._counters.get(adapter_id)
            elif load_format == "oft_adapter":
                counter = self.oft_registry._counters.get(adapter_id)
            else:
                return None
            if counter is None:
                return None
            return counter.value()
        except Exception:
            return None

    async def init_weights_send_group_for_remote_instance(
        self,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init_weights_send_group_for_remote_instance"
        result = (
            await self.init_weights_send_group_for_remote_instance_communicator(obj)
        )[0]
        return result.success, result.message

    async def send_weights_to_remote_instance(
        self,
        obj: SendWeightsToRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for send_weights_to_remote_instance"
        result = (await self.send_weights_to_remote_instance_communicator(obj))[0]
        return result.success, result.message

    async def update_weights_from_tensor(
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # For OFT adapter updates, register in tokenizer_manager BEFORE sending to
        # scheduler, so the adapter_id flows to the model_runner for consistent mapping.
        if obj.load_format == "oft_adapter" and obj.adapter_name is not None:
            from sglang.srt.oft.oft_registry import OFTRef

            adapter_name = obj.adapter_name
            if adapter_name not in self.oft_ref_cache:
                new_ref = OFTRef(oft_name=adapter_name, oft_path=adapter_name, pinned=False)
                await self.oft_registry.register(new_ref)
                self.oft_ref_cache[adapter_name] = new_ref
            # Pass the canonical adapter_id to the model_runner
            obj.adapter_id = self.oft_ref_cache[adapter_name].oft_id

        # Immediately update the weights if the engine is in paused state
        async with self.is_pause_cond:
            is_paused = self.is_pause

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            results = await self.update_weights_from_tensor_communicator(obj)

        success, message = _Communicator.merge_results(results)
        if success and obj.load_format == "oft_adapter" and obj.adapter_id is not None:
            updated_ref = await self.oft_registry.bump_version_by_id(obj.adapter_id)
            self.oft_ref_cache[updated_ref.oft_name] = updated_ref
            message += (
                f" OFT adapter {updated_ref.oft_name} version updated to "
                f"{updated_ref.oft_version}."
            )
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def update_weights_from_ipc(
        self,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Update weights via IPC for checkpoint-engine integration."""
        self.auto_create_handle_loop()
        try:
            # For now, we only support single data parallel instance
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            logger.info("Starting IPC weight update")
            # This means that weight sync cannot run while requests are in progress.
            async with self.model_update_lock.writer_lock:
                result = (await self.update_weights_from_ipc_communicator(obj))[0]
                success, message = result.success, result.message
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg

        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def _unload_lora_adapter_locked(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
    ) -> UnloadLoRAAdapterReqOutput:
        assert (
            self.lora_update_lock.locked()
        ), "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        await self.lora_registry.wait_for_unload(lora_id)
        result = (await self.update_lora_adapter_communicator(obj))[0]

        return result

    async def load_lora_adapter(
        self: TokenizerManager,
        obj: LoadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            async with self.lora_update_lock:
                # Generate new uniquely identifiable LoRARef object.
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                # Register the LoRA adapter only after loading is successful.
                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter

                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_lora_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter
                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_lora_adapter(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                obj.lora_name is not None
            ), "lora_name must be provided to unload LoRA adapter"

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                return await self._unload_lora_adapter_locked(obj)
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def _unload_oft_adapter_locked(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
    ) -> UnloadOFTAdapterReqOutput:
        assert (
            self.oft_update_lock.locked()
        ), "self.oft_update_lock must be locked in order for self._unload_oft_adapter_locked() to be called"

        oft_id = await self.oft_registry.unregister(obj.oft_name)
        obj.oft_id = oft_id

        await self.oft_registry.wait_for_unload(oft_id)
        result = (await self.update_oft_adapter_communicator(obj))[0]

        return result

    async def load_oft_adapter(
        self: TokenizerManager,
        obj: LoadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_oft:
                raise ValueError(
                    "OFT is not enabled. Please set `--enable-oft` to enable OFT."
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic OFT loading"
            logger.info(
                "Start load OFT adapter. OFT name=%s, path=%s",
                obj.oft_name,
                obj.oft_path,
            )

            async with self.oft_update_lock:
                new_adapter = OFTRef(
                    oft_name=obj.oft_name,
                    oft_path=obj.oft_path,
                    pinned=obj.pinned,
                )

                obj.oft_id = new_adapter.oft_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    await self.oft_registry.register(new_adapter)
                    self.oft_ref_cache[obj.oft_name] = new_adapter

                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.oft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_oft_name = await self.oft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_oft_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to evict LRU OFT adapter. "
                                f"OFT registry is: {self.oft_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used OFT adapter '{lru_oft_name}' "
                            f"(current number of adapters: {self.oft_registry.num_registered_ofts}, "
                            f"max allowed: {self.server_args.max_loaded_ofts})"
                        )

                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(oft_name=lru_oft_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter '{lru_oft_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_oft_name]

                return result
        except ValueError as e:
            return LoadOFTAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_oft_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_oft:
                raise ValueError(
                    "OFT is not enabled. Please set `--enable-oft` to enable OFT."
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic OFT loading"
            logger.info(
                "Start load OFT adapter from tensors. OFT name=%s",
                obj.oft_name,
            )

            async with self.oft_update_lock:
                new_adapter = OFTRef(
                    oft_name=obj.oft_name,
                    oft_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.oft_id = new_adapter.oft_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    await self.oft_registry.register(new_adapter)
                    self.oft_ref_cache[obj.oft_name] = new_adapter
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.oft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_oft_name = await self.oft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_oft_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to evict LRU OFT adapter. "
                                f"OFT registry is: {self.oft_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used OFT adapter '{lru_oft_name}' "
                            f"(current number of adapters: {self.oft_registry.num_registered_ofts}, "
                            f"max allowed: {self.server_args.max_loaded_ofts})"
                        )

                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(oft_name=lru_oft_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter '{lru_oft_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_oft_name]

                return result
        except ValueError as e:
            return LoadOFTAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_oft_adapter(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadOFTAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_oft:
                raise ValueError(
                    "OFT is not enabled. Please set `--enable-oft` to enable OFT."
                )

            assert (
                obj.oft_name is not None
            ), "oft_name must be provided to unload OFT adapter"

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic OFT loading"
            logger.info(
                "Start unload OFT adapter. OFT name=%s",
                obj.oft_name,
            )

            async with self.oft_update_lock:
                return await self._unload_oft_adapter_locked(obj)
        except ValueError as e:
            return UnloadOFTAdapterReqOutput(success=False, error_message=str(e))

    async def get_weights_by_name(
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self: TokenizerManager,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self: TokenizerManager,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def check_weights(
        self: TokenizerManager,
        obj: CheckWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> CheckWeightsReqOutput:
        self.auto_create_handle_loop()
        results = await self.check_weights_communicator(obj)
        return _Communicator.merge_results(results)

    async def slow_down(
        self: TokenizerManager,
        obj: SlowDownReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.slow_down_communicator(obj)

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
        req = GetInternalStateReq()
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
        return [res.updated for res in responses]

    async def dumper_control(
        self: TokenizerManager, obj: DumperControlReqInput
    ) -> List[DumperControlReqOutput]:
        return await self.dumper_control_communicator(obj)

    async def get_load(self: TokenizerManager) -> List[GetLoadReqOutput]:
        req = GetLoadReqInput()
        return await self.get_load_communicator(req)

    async def get_loads(
        self: TokenizerManager,
        include: Optional[List[str]] = None,
        dp_rank: Optional[int] = None,
    ) -> List[GetLoadsReqOutput]:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            include: List of sections to include. Options: core, memory, spec, lora, disagg, queues, all
            dp_rank: Optional filter for specific DP rank

        Returns:
            List of GetLoadsReqOutput, one per scheduler (filtered by dp_rank if specified)
        """
        req = GetLoadsReqInput(
            include=include if include else ["all"],
            dp_rank=dp_rank,
        )
        results = await self.get_loads_communicator(req)

        # Filter by dp_rank if specified
        if dp_rank is not None:
            results = [r for r in results if r.dp_rank == dp_rank]

        return results

    async def open_session(
        self, obj: OpenSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(
        self, obj: CloseSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    def _update_weight_version_if_provided(self, weight_version: Optional[str]) -> None:
        """Update weight version if provided."""
        if weight_version is not None:
            self.server_args.weight_version = weight_version

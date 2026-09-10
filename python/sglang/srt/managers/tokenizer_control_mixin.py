from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import fastapi

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    ActivateAdapterVersionReqOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    CudaMemoryPeakRankResult,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DiscardAdapterStageReqInput,
    DiscardAdapterStageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    EndWeightUpdateReqInput,
    EndWeightUpdateReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromDistributedReqInput,
    LoadLoRAAdapterFromDistributedReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    LoRAUpdateOutput,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    PullWeightsReqInput,
    PullWeightsReqOutput,
    ReadCudaMemoryPeakReqInput,
    ReadCudaMemoryPeakReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResetCudaMemoryPeakReqInput,
    ResetCudaMemoryPeakReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateAdapterFromDistributedReqInput,
    UpdateAdapterFromDistributedReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import LoRARef, ServerArgs
from sglang.srt.utils import (
    get_bool_env_var,
    normalize_serialized_named_tensor_payloads,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Declarative spec: (attr_name_prefix, response_type[, mode])
# Each entry creates self.{prefix}_communicator and registers
# response_type -> communicator.handle_recv in the dispatch table.
_COMMUNICATOR_SPECS = [
    ("init_weights_update_group", InitWeightsUpdateGroupReqOutput),
    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),
    ("update_weights_from_distributed", UpdateWeightsFromDistributedReqOutput),
    ("update_adapter_from_distributed", UpdateAdapterFromDistributedReqOutput),
    ("activate_adapter_version", ActivateAdapterVersionReqOutput),
    ("discard_adapter_stage", DiscardAdapterStageReqOutput),
    (
        "init_weights_send_group_for_remote_instance",
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    ("send_weights_to_remote_instance", SendWeightsToRemoteInstanceReqOutput),
    ("update_weights_from_tensor", UpdateWeightsFromTensorReqOutput),
    ("update_weights_from_ipc", UpdateWeightsFromIPCReqOutput),
    ("update_weight_version", UpdateWeightVersionReqOutput),
    ("get_weights_by_name", GetWeightsByNameReqOutput),
    ("release_memory_occupation", ReleaseMemoryOccupationReqOutput),
    ("resume_memory_occupation", ResumeMemoryOccupationReqOutput),
    ("reset_cuda_memory_peak", ResetCudaMemoryPeakReqOutput),
    ("read_cuda_memory_peak", ReadCudaMemoryPeakReqOutput),
    ("check_weights", CheckWeightsReqOutput),
    ("pull_weights", PullWeightsReqOutput),
    ("slow_down", SlowDownReqOutput),
    ("flush_cache", FlushCacheReqOutput),
    ("add_external_corpus", AddExternalCorpusReqOutput),
    ("remove_external_corpus", RemoveExternalCorpusReqOutput),
    ("list_external_corpora", ListExternalCorporaReqOutput),
    ("clear_hicache_storage", ClearHiCacheReqOutput),
    ("attach_hicache_storage", AttachHiCacheStorageReqOutput),
    ("detach_hicache_storage", DetachHiCacheStorageReqOutput),
    ("profile", ProfileReqOutput),
    ("get_internal_state", GetInternalStateReqOutput),
    ("set_internal_state", SetInternalStateReqOutput),
    ("expert_distribution", ExpertDistributionReqOutput),
    ("begin_weight_update", BeginWeightUpdateReqOutput),
    ("end_weight_update", EndWeightUpdateReqOutput),
    ("update_lora_adapter", LoRAUpdateOutput),
    ("dumper_control", DumperControlReqOutput),
    ("scale_elastic_ep", ScaleElasticEPReqOutput),
]


def _merge_lora_update_results(results: List[LoRAUpdateOutput]) -> LoRAUpdateOutput:
    """Merge the per-rank replies of a LoRA load/unload fan-out into one result.

    The operation succeeded only if every rank succeeded. Reporting a partial
    failure as success would let the tokenizer-side LoRA registry drift from
    the ranks that failed, so failures win: their deduplicated error messages
    are joined, and loaded_adapters reflects the first failed rank.
    """
    failed = [r for r in results if not r.success]
    if not failed:
        return results[0]
    error_messages = list(
        dict.fromkeys(r.error_message for r in failed if r.error_message)
    )
    return LoRAUpdateOutput(
        success=False,
        error_message=" | ".join(error_messages),
        loaded_adapters=failed[0].loaded_adapters,
    )


class TokenizerControlMixin:
    """Mixin for TokenizerManager's control-plane operations (weights, cache, lora,
    profile, internal state, etc.) -- everything that talks to the scheduler via
    FanOutCommunicator, as opposed to data-plane inference requests multiplexed by rid.
    """

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        dispatch_pairs = []
        # Canonical OFT IPC types are resolved lazily: sglang.srt.oft.io_types
        # imports managers.io_struct, so a module-level import here would close
        # the very cycle that io_struct.__getattr__ exists to avoid.
        from sglang.srt.oft.io_types import OFTUpdateOutput

        self._communicator_specs = list(_COMMUNICATOR_SPECS) + [
            ("update_oft_adapter", OFTUpdateOutput),
        ]
        for spec in self._communicator_specs:
            name, resp_type = spec[0], spec[1]
            mode = spec[2] if len(spec) > 2 else "queueing"
            comm = FanOutCommunicator(
                self._dispatch_to_scheduler,
                get_parallel().dp_size,
                mode,
            )
            setattr(self, f"{name}_communicator", comm)
            dispatch_pairs.append((resp_type, comm.handle_recv))
        self._result_dispatcher += TypeBasedDispatcher(dispatch_pairs)

    def update_control_communicator_fan_out(self: TokenizerManager, worker_count: int):
        primary_group_control = (
            get_parallel().enable_dp_attention
            and not get_parallel().enable_dp_attention_local_control_broadcast
        )
        if primary_group_control:
            control_fan_out = (
                worker_count + self.server_args.tp_size - 1
            ) // self.server_args.tp_size
        else:
            control_fan_out = worker_count

        for spec in getattr(self, "_communicator_specs", _COMMUNICATOR_SPECS):
            getattr(self, f"{spec[0]}_communicator").set_fan_out(worker_count)

        self.get_internal_state_communicator.set_fan_out(control_fan_out)

    async def add_external_corpus(
        self: TokenizerManager, obj: AddExternalCorpusReqInput
    ) -> AddExternalCorpusReqOutput:
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        truncated = False
        try:
            if not obj.corpus_id:
                import uuid

                obj.corpus_id = uuid.uuid4().hex
            if obj.file_path is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    iter_external_corpus_chunks,
                )

                max_tokens = (
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                obj.token_chunks = list(
                    iter_external_corpus_chunks(
                        obj.file_path, self.tokenizer, max_tokens
                    )
                )
            elif obj.documents is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    SEPARATOR_TOKEN,
                )

                max_tokens = (
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                token_chunks = []
                total_tokens = 0
                has_prev = False
                for doc in obj.documents:
                    if not doc:
                        continue
                    token_ids = list(
                        self.tokenizer.encode(doc, add_special_tokens=False)
                    )
                    if not token_ids:
                        continue
                    if has_prev:
                        token_ids = [SEPARATOR_TOKEN] + token_ids
                    if total_tokens + len(token_ids) > max_tokens:
                        truncated = True
                        break
                    token_chunks.append(token_ids)
                    total_tokens += len(token_ids)
                    has_prev = True
                obj.token_chunks = token_chunks
            else:
                return AddExternalCorpusReqOutput(
                    success=False,
                    message="Either file_path or documents must be provided.",
                )
            obj.file_path = None
            obj.documents = None
            results = await self.add_external_corpus_communicator(obj)
            all_success, all_message = FanOutCommunicator.merge_results(results)
            if truncated and all_success:
                all_message += f" (truncated: exceeded {max_tokens} token limit)"
            return AddExternalCorpusReqOutput(
                success=all_success,
                corpus_id=results[0].corpus_id if all_success else "",
                message=all_message,
                loaded_token_count=results[0].loaded_token_count if all_success else 0,
            )
        except Exception as e:
            return AddExternalCorpusReqOutput(success=False, message=str(e))

    async def remove_external_corpus(
        self: TokenizerManager, corpus_id: str
    ) -> RemoveExternalCorpusReqOutput:
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.remove_external_corpus_communicator(
            RemoveExternalCorpusReqInput(corpus_id=corpus_id)
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        return RemoveExternalCorpusReqOutput(success=all_success, message=all_message)

    async def list_external_corpora(
        self: TokenizerManager,
    ) -> ListExternalCorporaReqOutput:
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.list_external_corpora_communicator(
            ListExternalCorporaReqInput()
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        # Merge corpus token counts from all DP ranks (each rank loads the same set).
        corpus_token_counts = results[0].corpus_token_counts if all_success else {}
        return ListExternalCorporaReqOutput(
            success=all_success,
            corpus_token_counts=corpus_token_counts,
            message=all_message,
        )

    async def flush_cache(
        self: TokenizerManager, timeout_s: Optional[float] = None
    ) -> FlushCacheReqOutput:
        self.auto_create_handle_loop()
        result = (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
        )[0]
        if result.success and self.mm_processor is not None:
            self.mm_processor.clear_preprocess_cache()
        return result

    async def clear_hicache_storage(self: TokenizerManager) -> ClearHiCacheReqOutput:
        """Clear the hierarchical cache storage."""
        self.auto_create_handle_loop()
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
        self.auto_create_handle_loop()
        results = await self.attach_hicache_storage_communicator(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend=hicache_storage_backend,
                hicache_storage_backend_extra_config_json=hicache_storage_backend_extra_config_json,
                hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
                hicache_write_policy=hicache_write_policy,
            )
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = AttachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            # Keep tokenizer side server_info consistent with scheduler side.
            hicache_fields = {"hicache_storage_backend": hicache_storage_backend}
            if hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    hicache_storage_backend_extra_config_json
                )
            if hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    hicache_storage_prefetch_policy
                )
            if hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = hicache_write_policy
            self.record_config_updates("tokenizer.attach_hicache", **hicache_fields)
        return out

    async def detach_hicache_storage(
        self: TokenizerManager,
    ) -> DetachHiCacheStorageReqOutput:
        """Detach (disable) HiCache storage backend at runtime."""
        self.auto_create_handle_loop()
        results = await self.detach_hicache_storage_communicator(
            DetachHiCacheStorageReqInput()
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = DetachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            self.record_config_updates(
                "tokenizer.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
        return out

    async def start_profile(
        self: TokenizerManager,
        req: Optional[ProfileReq] = None,
    ):
        self.auto_create_handle_loop()
        req = req or ProfileReq()
        req.req_type = ProfileReqType.START_PROFILE
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        req.with_stack = (
            False if req.with_stack is False or env_with_stack is False else True
        )
        env_record_shapes: bool = get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )
        req.record_shapes = (req.record_shapes is not False) and env_record_shapes
        req.profile_id = req.profile_id or str(time.time())
        return await self._execute_profile(req)

    async def stop_profile(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
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
            get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        results = await self.init_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def destroy_weights_update_group(
        self: TokenizerManager,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"

        results = await self.destroy_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def _weight_update_session_call(
        self: TokenizerManager, communicator, obj
    ) -> Tuple[bool, str]:
        """Run one weight-update session RPC under the same pause-aware locking as
        update_weights_from_distributed: while the engine is paused the writer lock
        is already held by whoever paused it, so taking it again would deadlock."""
        self.auto_create_handle_loop()
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await communicator(obj)
        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def begin_weight_update(
        self: TokenizerManager,
        obj: BeginWeightUpdateReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        return await self._weight_update_session_call(
            self.begin_weight_update_communicator, obj
        )

    async def end_weight_update(
        self: TokenizerManager,
        obj: EndWeightUpdateReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        return await self._weight_update_session_call(
            self.end_weight_update_communicator, obj
        )

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_distributed_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_distributed_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.flush_cache and self.mm_processor is not None:
            self.mm_processor.clear_preprocess_cache()
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    def _assert_native_lora_available(self, lora_path) -> None:
        """Reject adapters quarantined by a partial activation failure."""
        names = [lora_path] if isinstance(lora_path, str) else (lora_path or [])
        for name in names:
            if name in getattr(self, "failed_lora_unloads", {}):
                raise ValueError(
                    f"LoRA adapter '{name}' is unavailable after a failed unload; "
                    "retry unload before loading or serving it"
                )
            if name in self.failed_lora_activations:
                raise ValueError(
                    f"LoRA adapter '{name}' is unavailable after a partial "
                    "activation failure; restart required"
                )

    def _staging_backend_for(self, obj):
        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        backend = get_staging_backend(self, obj)
        if (
            backend is not None
            and obj.load_format == "oft_adapter"
            and self.server_args.tokenizer_worker_num > 1
        ):
            raise ValueError(
                "OFT staging requires tokenizer_worker_num == 1 because adapter "
                "activation cannot globally drain requests across tokenizer workers."
            )
        return backend

    async def _run_adapter_activation_safely(self, backend, obj, operation):
        """Run an irreversible adapter activation as one admission transaction."""
        from sglang.srt.adapter_sync.tokenizer_backend import (
            finish_irreversible_update,
        )

        # A paused request keeps its reader lock for its full response lifetime.
        # Even retract mode can retain an old versioned radix prefix, so only a
        # genuinely drained paused engine may activate without corrupting KV.
        async with self.is_pause_cond:
            if self.is_pause and await self.model_update_lock.is_locked():
                return (
                    False,
                    "Cannot activate adapter weights while paused requests are "
                    "still active; continue generation or abort those requests "
                    "before retrying.",
                )
            async with self.model_update_lock.writer_lock:
                if backend is None:
                    return await finish_irreversible_update(operation)
                async with backend.lifecycle_lock:
                    # Unload or another lifecycle operation may have completed
                    # while activation waited for inference to drain.
                    backend.prepare_activation(obj)
                    return await finish_irreversible_update(operation)

    async def _rollback_failed_adapter_stage(self, backend, obj, stage_message):
        """Complete rollback under the caller's lifecycle lock and cancel shield."""
        try:
            discard = DiscardAdapterStageReqInput(
                load_format=obj.load_format,
                adapter_name=obj.adapter_name,
                adapter_id=obj.adapter_id,
                adapter_version=obj.adapter_version,
            )
            results = await self.discard_adapter_stage_communicator(discard)
            success, message = FanOutCommunicator.merge_results(results)
            if not results or not success:
                raise RuntimeError(message or "No stage rollback responses")
            backend.clear_stage_reservation(discard)
        except Exception as error:
            failure = (
                f"{stage_message}; stage rollback failed: {error}; restart required"
            )
            backend._quarantine(obj.adapter_name, failure)
            return False, failure
        return False, f"{stage_message}; stage rollback succeeded"

    async def update_adapter_from_distributed(
        self: TokenizerManager,
        obj: UpdateAdapterFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Stage native LoRA or canonical OFT adapter weights."""
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for adapter staging"

        from sglang.srt.oft import tokenizer_hooks as oft_tokenizer_hooks

        backend = self._staging_backend_for(obj)
        if backend is not None:
            await backend.reserve_stage(obj)
        else:
            await oft_tokenizer_hooks.register_oft_ref(self, obj)

        if obj.double_buffer:
            if backend is not None:
                from sglang.srt.adapter_sync.tokenizer_backend import (
                    finish_irreversible_update,
                )

                reserved_id = obj.adapter_id
                async with backend.lifecycle_lock:
                    # A retry may have queued behind an in-flight stage while
                    # unload cancelled its reservation. Revalidate before any
                    # dispatch, including the ID if a new reservation replaced it.
                    backend.prepare_activation(obj)
                    if obj.adapter_id != reserved_id:
                        raise ValueError("Adapter stage reservation was cancelled")

                    async def stage_and_finish():
                        try:
                            results = (
                                await self.update_adapter_from_distributed_communicator(
                                    obj
                                )
                            )
                            success, message = FanOutCommunicator.merge_results(results)
                            if results and success:
                                return success, message
                            message = message or "No adapter stage responses"
                        except Exception as error:
                            message = f"Adapter stage failed: {error}"
                        return await self._rollback_failed_adapter_stage(
                            backend, obj, message
                        )

                    # Keep unload excluded until every worker has replied even
                    # when the HTTP caller is cancelled during the fan-out.
                    return await finish_irreversible_update(stage_and_finish)

            results = await self.update_adapter_from_distributed_communicator(obj)
            success, message = FanOutCommunicator.merge_results(results)
        else:

            async def stage_activate_and_publish():
                results = await self.update_adapter_from_distributed_communicator(obj)
                if backend is not None:
                    # Missing replies or a completed stage can hide activation.
                    # Only unanimous pre-activation failure is safe to discard.
                    if results and all(
                        not r.success
                        and r.staged_adapter_version is None
                        and r.active_adapter_version is None
                        for r in results
                    ):
                        _, message = FanOutCommunicator.merge_results(results)
                        return await self._rollback_failed_adapter_stage(
                            backend, obj, message
                        )
                    return await backend.finish_activation(obj, results)
                success, message = FanOutCommunicator.merge_results(results)
                message += await oft_tokenizer_hooks.bump_oft_version(
                    self, obj, success
                )
                return success, message

            return await self._run_adapter_activation_safely(
                backend, obj, stage_activate_and_publish
            )

        message += await oft_tokenizer_hooks.bump_oft_version(self, obj, success)
        return success, message

    async def activate_adapter_version(
        self: TokenizerManager,
        obj: ActivateAdapterVersionReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Drain admission and activate native LoRA or canonical OFT."""
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for adapter activation"

        backend = self._staging_backend_for(obj)

        async def activate_and_publish():
            results = await self.activate_adapter_version_communicator(obj)
            if backend is not None:
                return await backend.finish_activation(obj, results)
            return FanOutCommunicator.merge_results(results)

        return await self._run_adapter_activation_safely(
            backend, obj, activate_and_publish
        )

    async def init_weights_send_group_for_remote_instance(
        self: TokenizerManager,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            get_parallel().dp_size == 1
        ), "dp_size must be 1 for init_weights_send_group_for_remote_instance"
        result = (
            await self.init_weights_send_group_for_remote_instance_communicator(obj)
        )[0]
        return result.success, result.message

    async def send_weights_to_remote_instance(
        self: TokenizerManager,
        obj: SendWeightsToRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            get_parallel().dp_size == 1
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
            get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
            obj.serialized_named_tensors
        )

        from sglang.srt.oft import tokenizer_hooks as oft_tokenizer_hooks

        newly_registered_oft_ref = await oft_tokenizer_hooks.register_oft_ref(self, obj)

        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_tensor_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_tensor_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.flush_cache and self.mm_processor is not None:
            self.mm_processor.clear_preprocess_cache()
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."
        message += await oft_tokenizer_hooks.bump_oft_version(self, obj, success)
        if not success and newly_registered_oft_ref:
            await oft_tokenizer_hooks.rollback_oft_ref(self, obj.adapter_name)

        return success, message

    async def update_weights_from_ipc(
        self: TokenizerManager,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Update weights via IPC for checkpoint-engine integration."""
        self.auto_create_handle_loop()
        try:
            # For now, we only support single data parallel instance
            assert (
                get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            logger.info("Starting IPC weight update")

            async with self.is_pause_cond:
                is_paused = self.is_pause
                if is_paused:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message

            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg

        if success and obj.flush_cache and self.mm_processor is not None:
            self.mm_processor.clear_preprocess_cache()
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

        pending = self.pending_lora_stage
        active = self.lora_registry.get_all_adapters().get(obj.lora_name)
        failed_unloads = getattr(self, "failed_lora_unloads", None)
        if failed_unloads is None:
            failed_unloads = self.failed_lora_unloads = {}
        failed_unload = failed_unloads.get(obj.lora_name)
        cancelling_pending = pending is not None and pending.lora_name == obj.lora_name

        async def dispatch_unload():
            try:
                return _merge_lora_update_results(
                    await self.update_lora_adapter_communicator(obj)
                )
            except Exception as error:
                return LoRAUpdateOutput(success=False, error_message=str(error))

        async def cancel_pending_on_workers():
            from sglang.srt.adapter_sync.tokenizer_backend import (
                finish_irreversible_update,
            )

            async def dispatch_and_validate():
                try:
                    result = _merge_lora_update_results(
                        await self.update_lora_adapter_communicator(obj)
                    )
                except Exception as error:
                    result = LoRAUpdateOutput(success=False, error_message=str(error))
                if not result.success:
                    failure = (
                        f"LoRA adapter '{obj.lora_name}' is quarantined because "
                        "its staged unload did not succeed on every worker; "
                        "restart required"
                    )
                    self.failed_lora_activations[obj.lora_name] = failure
                    failed_unloads[obj.lora_name] = active or pending
                    return LoRAUpdateOutput(
                        success=False,
                        error_message=(
                            f"{result.error_message or 'worker unload failed'} | "
                            f"{failure}"
                        ),
                        loaded_adapters=result.loaded_adapters,
                    )
                return result

            return await finish_irreversible_update(dispatch_and_validate)

        if active is None and failed_unload is not None:
            obj.lora_id = failed_unload.lora_id

            from sglang.srt.adapter_sync.tokenizer_backend import (
                finish_irreversible_update,
            )

            async def retry_and_finalize():
                result = await dispatch_unload()
                if result.success:
                    failed_unloads.pop(obj.lora_name, None)
                    self.failed_lora_activations.pop(obj.lora_name, None)
                    self.lora_ref_cache.pop(obj.lora_name, None)
                return result

            return await finish_irreversible_update(retry_and_finalize)

        if active is None and cancelling_pending:
            # A freshly staged adapter is intentionally absent from the serving
            # registry. Unload acts as cancellation of that pending transaction.
            obj.lora_id = pending.lora_id
            self.pending_lora_stage = None
            return await cancel_pending_on_workers()

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        await self.lora_registry.wait_for_unload(lora_id)
        if cancelling_pending:
            # Once unload fan-out begins, any worker may discard this stage.
            # It is no longer safe to activate even if another worker fails.
            self.pending_lora_stage = None
            return await cancel_pending_on_workers()

        result = await dispatch_unload()
        if not result.success:
            failed_unloads[obj.lora_name] = active
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

            assert (
                get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for dynamic lora loading"
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            async with self.lora_update_lock:
                self._ensure_no_pending_lora_load(obj.lora_name)
                # Generate new uniquely identifiable LoRARef object.
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                obj.lora_id = new_adapter.lora_id
                result = _merge_lora_update_results(
                    await self.update_lora_adapter_communicator(obj)
                )

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

    def _validate_lora_upsert_supported(
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromDistributedReqInput,
    ) -> None:
        """Upsert resolves lora_name -> lora_id through this process's registry.

        With multiple tokenizer workers each HTTP worker process holds its own
        registry, so the resolution depends on which worker the router picks:
        a worker that never served the original load would mint a fresh id and
        die on the backend duplicate check. Fail loudly instead.
        """
        if obj.upsert and self.server_args.tokenizer_worker_num > 1:
            raise ValueError(
                "LoRA upsert is not supported with tokenizer_worker_num > 1: "
                "each HTTP worker resolves lora_name against its own registry, "
                "making upsert nondeterministic across workers."
            )

    def _ensure_no_pending_lora_load(self, lora_name: str) -> None:
        if lora_name in getattr(self, "failed_lora_unloads", {}):
            raise ValueError(
                f"Cannot load LoRA adapter '{lora_name}' after a failed unload; "
                "retry unload first."
            )
        pending = self.pending_lora_stage
        if pending is not None and pending.lora_name == lora_name:
            raise ValueError(
                f"Cannot load LoRA adapter '{lora_name}' while staged version "
                f"{pending.version} is pending; activate or unload it first."
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
                get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for dynamic lora loading"
            if obj.upsert:
                # In-place refresh is only wired up on the from_distributed
                # route (the disaggregated RL weight-sync path). Reject
                # explicitly instead of dying later on the duplicate check
                # with a fresh uuid.
                raise ValueError(
                    "upsert is not supported on the from_tensors route; use "
                    "/load_lora_adapter_from_distributed to refresh an adapter in place."
                )
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",
                obj.lora_name,
            )

            obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
                obj.serialized_named_tensors
            )

            async with self.lora_update_lock:
                self._ensure_no_pending_lora_load(obj.lora_name)
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path="__tensor__",
                    pinned=obj.pinned,
                    reloadable=False,
                )
                obj.lora_id = new_adapter.lora_id
                result = _merge_lora_update_results(
                    await self.update_lora_adapter_communicator(obj)
                )

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

    async def load_lora_adapter_from_distributed(
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromDistributedReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterFromDistributedReqOutput:
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
                "Start load Lora adapter from distributed. Lora name=%s, group=%s",
                obj.lora_name,
                obj.group_name,
            )

            self._validate_lora_upsert_supported(obj)
            candidate = LoRARef(
                lora_name=obj.lora_name,
                lora_path="__distributed__",
                pinned=obj.pinned,
                reloadable=False,
            )

            async def update_and_publish(new_adapter, reused):
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]
                if result.success:
                    if reused:
                        await self.lora_registry.refresh(new_adapter)
                    else:
                        await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter
                return result

            # Once a refresh has been dispatched it cannot be recalled. Keep
            # shielding it through repeated caller cancellations so admission
            # resumes only after backend completion and registry publication.
            async def finish_despite_cancellation(new_adapter):
                task = asyncio.create_task(update_and_publish(new_adapter, reused=True))
                caller_cancelled = False
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        if task.cancelled():
                            raise
                        caller_cancelled = True
                result = task.result()
                if caller_cancelled:
                    raise asyncio.CancelledError
                return result

            while True:
                # Preflight without the lifecycle lock. If the adapter exists,
                # take locks in the same order as inference (model -> LoRA
                # lifecycle), avoiding a cycle with implicit adapter reload.
                needs_drain = obj.upsert and (
                    await self.lora_registry.get_lora_id(obj.lora_name) is not None
                )
                if not needs_drain:
                    async with self.lora_update_lock:
                        self._ensure_no_pending_lora_load(obj.lora_name)
                        new_adapter, reused = (
                            await self.lora_registry.register_or_reuse(
                                candidate,
                                upsert=obj.upsert,
                                bump_version=True,
                            )
                        )
                        if not reused:
                            # A fresh adapter is invisible until this finishes
                            # and need not stop unrelated inference.
                            result = await update_and_publish(new_adapter, reused=False)
                            break
                    # Another load published this name after preflight. Retry
                    # outside the lifecycle lock through the draining path.
                    continue

                retry_as_fresh = False
                async with self.is_pause_cond:
                    if self.is_pause:
                        raise ValueError(
                            "Cannot upsert an existing LoRA adapter while "
                            "generation is paused; continue generation before "
                            "retrying."
                        )
                    async with self.model_update_lock.writer_lock:
                        async with self.lora_update_lock:
                            self._ensure_no_pending_lora_load(obj.lora_name)
                            # Re-resolve after draining because a concurrent
                            # unload may have removed the adapter. If so, retry
                            # it as a fresh load without the global writer.
                            new_adapter, reused = (
                                await self.lora_registry.register_or_reuse(
                                    candidate,
                                    upsert=obj.upsert,
                                    bump_version=True,
                                )
                            )
                            if reused:
                                result = await finish_despite_cancellation(new_adapter)
                            else:
                                retry_as_fresh = True
                if retry_as_fresh:
                    continue
                break

            async with self.lora_update_lock:
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
            return LoadLoRAAdapterFromDistributedReqOutput(
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

            assert (
                get_parallel().dp_size == 1 or get_parallel().enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for dynamic lora loading"
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            from sglang.srt.adapter_sync.tokenizer_backend import (
                finish_irreversible_update,
            )

            async def unload_and_finalize():
                result = await self._unload_lora_adapter_locked(obj)
                # Explicit unload is a DELETE: drop the reload-catalog entry too.
                # The max_loaded_loras LRU loop calls _unload_lora_adapter_locked
                # directly — an EVICT — and must keep the entry so disk-backed
                # adapters can be implicitly reloaded later.
                if result.success:
                    self.lora_ref_cache.pop(obj.lora_name, None)
                return result

            async with self.lora_update_lock:
                # Keep unregister, lease drainage, worker cleanup, and catalog
                # finalization serialized through repeated caller cancellation.
                return await finish_irreversible_update(unload_and_finalize)
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def get_weights_by_name(
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if get_parallel().dp_size == 1:
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

    async def pull_weights(
        self: TokenizerManager,
        obj: PullWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        results = await self.pull_weights_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def check_weights(
        self: TokenizerManager,
        obj: CheckWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str, Optional[List[Dict]], Optional[str]]:
        self.auto_create_handle_loop()
        results = await self.check_weights_communicator(obj)
        success, message = FanOutCommunicator.merge_results(results)
        ranks: Optional[List[Dict]] = None
        per_engine_checksum: Optional[str] = None
        if any(r.payload is not None for r in results):
            rank_infos: List[ChecksumInfo] = []
            for r in results:
                if r.payload is not None:
                    rank_infos.extend(r.payload)
            h = hashlib.sha256()
            for info in rank_infos:
                h.update(info.per_gpu_checksum.encode())
            per_engine_checksum = h.hexdigest()
            ranks = [msgspec_to_builtins(info) for info in rank_infos]
        return success, message, ranks, per_engine_checksum

    async def slow_down(
        self: TokenizerManager,
        obj: SlowDownReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.slow_down_communicator(obj)

    async def reset_cuda_memory_peak(
        self: TokenizerManager, sample_id: str
    ) -> ResetCudaMemoryPeakReqOutput:
        return await self._cuda_memory_peak_control(sample_id, "reset")

    async def read_cuda_memory_peak(
        self: TokenizerManager, sample_id: str
    ) -> ReadCudaMemoryPeakReqOutput:
        return await self._cuda_memory_peak_control(sample_id, "read")

    async def _cuda_memory_peak_control(self: TokenizerManager, sample_id, operation):
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        # The tokenizer parent has configuration but no live PP process group.
        if self.server_args.dp_size != 1 or self.server_args.pp_size != 1:
            raise ValueError("CUDA peak control supports only DP1/PP1")
        request_type, output_type = (
            (ResetCudaMemoryPeakReqInput, ResetCudaMemoryPeakReqOutput)
            if operation == "reset"
            else (ReadCudaMemoryPeakReqInput, ReadCudaMemoryPeakReqOutput)
        )
        self.auto_create_handle_loop()
        responses = await getattr(self, f"{operation}_cuda_memory_peak_communicator")(
            request_type(sample_id=sample_id)
        )
        if not isinstance(responses, list) or len(responses) != 1:
            raise RuntimeError("expected exactly one CUDA peak TP consensus response")
        result = responses[0]
        if not isinstance(result, output_type):
            raise RuntimeError("invalid CUDA peak consensus response type")
        if result.success is not True:
            raise RuntimeError(f"CUDA peak {operation} failed: {result.message}")
        if result.sample_id != sample_id or result.operation != operation:
            raise RuntimeError("CUDA peak consensus sample/operation mismatch")
        if (
            not isinstance(result.ranks, list)
            or len(result.ranks) != self.server_args.tp_size
        ):
            raise RuntimeError("incomplete CUDA peak TP rank evidence")
        for rank, row in enumerate(result.ranks):
            if (
                not isinstance(row, CudaMemoryPeakRankResult)
                or type(row.rank) is not int
                or row.rank != rank
                or row.sample_id != sample_id
                or row.operation != operation
                or row.success is not True
            ):
                raise RuntimeError("invalid CUDA peak TP rank identity or outcome")
            if operation == "read":
                if (
                    type(row.allocated_bytes) is not int
                    or type(row.reserved_bytes) is not int
                    or not 0 <= row.allocated_bytes <= row.reserved_bytes
                ):
                    raise RuntimeError("invalid CUDA peak TP metrics")
            elif row.allocated_bytes is not None or row.reserved_bytes is not None:
                raise RuntimeError("unexpected CUDA reset metrics")
        return result

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
        self.auto_create_handle_loop()
        req = GetInternalStateReq()
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        self.auto_create_handle_loop()
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
        return [res.updated for res in responses]

    async def dumper_control(
        self: TokenizerManager, obj: DumperControlReqInput
    ) -> List[DumperControlReqOutput]:
        self.auto_create_handle_loop()
        return await self.dumper_control_communicator(obj)

    async def get_loads(
        self: TokenizerManager,
        include: Optional[List[str]] = None,
        dp_rank: Optional[int] = None,
    ) -> List[LoadSnapshot]:
        """
        Get load snapshots for /v1/loads endpoint.

        Args:
            include: List of sections to include. Options: core, memory, spec, lora, disagg, queues, all
            dp_rank: Optional filter for specific DP rank

        Returns:
            List of LoadSnapshot, one per scheduler (filtered by dp_rank if specified)
        """
        self.auto_create_handle_loop()
        if dp_rank is not None and (
            dp_rank < 0 or dp_rank >= self.elastic_worker_count
        ):
            return []

        reader = self.load_snapshot_reader
        if dp_rank is not None:
            load = reader.read(dp_rank)
            results = [load] if load is not None else []
        else:
            results = reader.read_all()

        return results

    async def open_session(
        self: TokenizerManager,
        obj: OpenSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        if obj.streaming:
            if not self.server_args.enable_streaming_session:
                raise ValueError(
                    "Streaming sessions are disabled. "
                    "Please relaunch with --enable-streaming-session."
                )

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        future = asyncio.Future()
        self.session_futures[obj.session_id] = future
        self._dispatch_to_scheduler(obj)

        try:
            return await future
        finally:
            self.session_futures.pop(obj.session_id, None)

    async def close_session(
        self: TokenizerManager,
        obj: CloseSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        await self._async_dispatch_to_scheduler(obj)

    async def update_weight_version(
        self: TokenizerManager, obj: UpdateWeightVersionReqInput
    ) -> None:
        self.auto_create_handle_loop()
        await self.update_weight_version_communicator(obj)
        self._update_weight_version_if_provided(obj.new_version)

    def _update_weight_version_if_provided(
        self: TokenizerManager, weight_version: Optional[str]
    ) -> None:
        """Update weight version if provided."""
        if weight_version is not None:
            self.record_config_updates(
                "tokenizer.weight_version", weight_version=weight_version
            )

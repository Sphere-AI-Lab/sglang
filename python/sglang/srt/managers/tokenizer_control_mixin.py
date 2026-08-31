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
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
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
    LoadOFTAdapterFromDistributedReqInput,
    LoadOFTAdapterFromDistributedReqOutput,
    LoadOFTAdapterFromTensorsReqInput,
    LoadOFTAdapterFromTensorsReqOutput,
    LoRAUpdateOutput,
    OFTUpdateOutput,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    PullWeightsReqInput,
    PullWeightsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
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
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.oft.oft_registry import OFTRef
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
    ("update_oft_adapter", OFTUpdateOutput),
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


def _merge_oft_update_results(results: List[OFTUpdateOutput]) -> OFTUpdateOutput:
    """Merge per-rank replies of an OFT load/unload fan-out into one result.
    Mirrors _merge_lora_update_results exactly: any rank failing wins."""
    failed = [r for r in results if not r.success]
    if not failed:
        return results[0]
    error_messages = list(
        dict.fromkeys(r.error_message for r in failed if r.error_message)
    )
    return OFTUpdateOutput(
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
        for spec in _COMMUNICATOR_SPECS:
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

        for spec in _COMMUNICATOR_SPECS:
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
        """Reject a request naming an adapter quarantined by a partial native
        LoRA activation failure. Runs unconditionally for every generate/
        embedding request with a lora_path (see _resolve_lora_path in
        tokenizer_manager.py) -- not gated on enable_lora_staging, since
        self.failed_lora_activations is initialized unconditionally in
        __init__ and a previously staged-and-quarantined adapter name must
        stay rejected regardless of the server's current staging config."""
        names = [lora_path] if isinstance(lora_path, str) else (lora_path or [])
        for name in names:
            if name in self.failed_lora_activations:
                raise ValueError(
                    f"LoRA adapter '{name}' is unavailable after a partial "
                    "activation failure; restart required"
                )

    def _staging_backend_for(self, obj):
        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        return get_staging_backend(self, obj)

    async def update_adapter_from_distributed(
        self: TokenizerManager,
        obj: UpdateAdapterFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Double-buffer PEFT STAGE over NCCL.

        double_buffer=True: LOCK-FREE stage into the reserved staging slot while
        generation continues (overlaps decode); no writer_lock. double_buffer=
        False: the synchronous distributed path stages then ACTIVATEs-in-place in
        the scheduler in one round-trip, so we hold model_update_lock.writer_lock
        (drain-to-idle, mirror update_weights_from_distributed) around it."""
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update adapter from distributed"

        from sglang.srt.peft import tokenizer_hooks as peft_tokenizer_hooks

        backend = self._staging_backend_for(obj)
        if backend is not None:
            await backend.reserve_stage(obj)
        else:
            # The existing PEFT path remains register-before-dispatch.
            await peft_tokenizer_hooks.register_peft_ref(self, obj)

        if obj.double_buffer:
            results = await self.update_adapter_from_distributed_communicator(obj)
            success, message = FanOutCommunicator.merge_results(results)
            if backend is not None:
                return success, message
        else:
            # Hold is_pause_cond while updating to prevent unpause from racing.
            async with self.is_pause_cond:
                is_paused = self.is_pause
                if is_paused:
                    results = await self.update_adapter_from_distributed_communicator(
                        obj
                    )
                    if backend is not None:
                        backend_result = await backend.finish_activation(obj, results)
            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    results = await self.update_adapter_from_distributed_communicator(
                        obj
                    )
                    if backend is not None:
                        backend_result = await backend.finish_activation(obj, results)
            if backend is not None:
                return backend_result
            success, message = FanOutCommunicator.merge_results(results)

        message += await peft_tokenizer_hooks.bump_peft_version(self, obj, success)
        return success, message

    async def activate_adapter_version(
        self: TokenizerManager,
        obj: ActivateAdapterVersionReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Double-buffer PEFT ACTIVATE (the drained atomic swap). The drain lives
        HERE: model_update_lock.writer_lock waits for all in-flight generation
        reader_locks to release (drain running_batch to empty) and blocks new
        admission -- exactly what update_weights_from_disk/from_distributed use.
        Only THEN is the activate control request sent to the scheduler (a simple
        staging->active flip, since the batch is already drained); releasing the
        lock on return resumes admission."""
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for activate adapter version"

        backend = self._staging_backend_for(obj)
        if backend is not None:
            backend.prepare_activation(obj)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.activate_adapter_version_communicator(obj)
                if backend is not None:
                    backend_result = await backend.finish_activation(obj, results)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.activate_adapter_version_communicator(obj)
                if backend is not None:
                    backend_result = await backend.finish_activation(obj, results)

        if backend is not None:
            return backend_result

        success, message = FanOutCommunicator.merge_results(results)
        return success, message

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

        # PEFT register-before-dispatch: mint/lookup the streamed peft adapter's ref
        # (LoRA or OFT -- single-active, routed by the active peft registry) and set
        # obj.adapter_id, so a later generate request naming this adapter resolves
        # against tm.peft_ref_cache. Triggered by obj.adapter_name; without it,
        # streamed adapters loaded into the scheduler were never registered
        # tokenizer-side -> generate 400s with "never been loaded".
        from sglang.srt.peft import tokenizer_hooks as peft_tokenizer_hooks

        await peft_tokenizer_hooks.register_peft_ref(self, obj)

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
        message += await peft_tokenizer_hooks.bump_peft_version(self, obj, success)

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

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        await self.lora_registry.wait_for_unload(lora_id)
        result = _merge_lora_update_results(
            await self.update_lora_adapter_communicator(obj)
        )

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

            async with self.lora_update_lock:
                self._validate_lora_upsert_supported(obj)
                # With upsert, a same-name adapter keeps its lora_id so the
                # backend refreshes it in place instead of failing the
                # duplicate check; otherwise this resolves to a fresh ref.
                new_adapter, reused = await self.lora_registry.register_or_reuse(
                    LoRARef(
                        lora_name=obj.lora_name,
                        lora_path="__distributed__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                    upsert=obj.upsert,
                )
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                if result.success:
                    if reused:
                        await self.lora_registry.refresh(new_adapter)
                    else:
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

            async with self.lora_update_lock:
                result = await self._unload_lora_adapter_locked(obj)
                # Explicit unload is a DELETE: drop the reload-catalog entry too.
                # The max_loaded_loras LRU loop calls _unload_lora_adapter_locked
                # directly — an EVICT — and must keep the entry so disk-backed
                # adapters can be implicitly reloaded later.
                if result.success:
                    self.lora_ref_cache.pop(obj.lora_name, None)
                return result
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def load_oft_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (
                self.server_args.peft_method == "oft"
                and self.server_args.oft_impl == "sibling"
            ):
                raise ValueError(
                    "Native OFT adapter loading requires --peft-method oft "
                    "--oft-impl sibling."
                )
            obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
                obj.serialized_named_tensors
            )
            async with self.peft_update_lock:
                # Built inline (not via obj.to_ref()): to_ref() passes
                # obj.adapter_id through explicitly, which is None on a fresh
                # load and would short-circuit OFTRef's default_factory,
                # tripping its "adapter_id cannot be None" guard. Mirrors
                # load_lora_adapter_from_distributed's LoRARef(...) construction.
                new_ref, reused = await self.peft_registry.resolve_or_reuse(
                    OFTRef(
                        adapter_name=obj.adapter_name,
                        adapter_path="__tensor__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                    upsert=obj.upsert,
                )
                obj.adapter_id = new_ref.adapter_id
                results = await self.update_oft_adapter_communicator(obj)
                result = _merge_oft_update_results(results)

                if result.success:
                    if reused:
                        await self.peft_registry.refresh(new_ref)
                    else:
                        await self.peft_registry.register(new_ref)
                    self.peft_ref_cache[obj.adapter_name] = new_ref
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.peft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.peft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.peft_registry.get_all_adapters()}"
                            )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(adapter_name=lru_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter "
                                f"'{lru_name}': {unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_name]
                return result
        except ValueError as e:
            return LoadOFTAdapterFromTensorsReqOutput(
                success=False, error_message=str(e)
            )

    async def load_oft_adapter_from_distributed(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromDistributedReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromDistributedReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (
                self.server_args.peft_method == "oft"
                and self.server_args.oft_impl == "sibling"
            ):
                raise ValueError(
                    "Native OFT adapter loading requires --peft-method oft "
                    "--oft-impl sibling."
                )
            async with self.peft_update_lock:
                # See load_oft_adapter_from_tensors: built inline rather than
                # via obj.to_ref(), which would pass the not-yet-minted
                # obj.adapter_id (None) straight through and trip OFTRef's
                # "adapter_id cannot be None" guard instead of minting a
                # fresh id. Mirrors load_lora_adapter_from_distributed's
                # LoRARef(...) construction.
                new_ref, reused = await self.peft_registry.resolve_or_reuse(
                    OFTRef(
                        adapter_name=obj.adapter_name,
                        adapter_path="__distributed__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                    upsert=obj.upsert,
                )
                obj.adapter_id = new_ref.adapter_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    if reused:
                        await self.peft_registry.refresh(new_ref)
                    else:
                        await self.peft_registry.register(new_ref)
                    self.peft_ref_cache[obj.adapter_name] = new_ref
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.peft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.peft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.peft_registry.get_all_adapters()}"
                            )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(adapter_name=lru_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter "
                                f"'{lru_name}': {unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_name]
                return result
        except ValueError as e:
            return LoadOFTAdapterFromDistributedReqOutput(
                success=False, error_message=str(e)
            )

    async def _unload_oft_adapter_locked(
        self: TokenizerManager, obj: UnloadOFTAdapterReqInput
    ) -> UnloadOFTAdapterReqOutput:
        """Caller must hold peft_update_lock. Unregisters + tells the
        scheduler to free GPU state; does NOT touch peft_ref_cache (the
        caller decides evict-vs-delete semantics, mirroring
        _unload_lora_adapter_locked)."""
        adapter_id = await self.peft_registry.unregister(obj.adapter_name)
        obj.adapter_id = adapter_id
        result = (await self.update_oft_adapter_communicator(obj))[0]
        await self.peft_registry.wait_for_unload(adapter_id)
        return result

    async def unload_oft_adapter(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadOFTAdapterReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (
                self.server_args.peft_method == "oft"
                and self.server_args.oft_impl == "sibling"
            ):
                raise ValueError(
                    "Native OFT adapter loading requires --peft-method oft "
                    "--oft-impl sibling."
                )
            async with self.peft_update_lock:
                result = await self._unload_oft_adapter_locked(obj)
                # Explicit unload is a DELETE: drop the ref_cache entry too
                # (mirrors unload_lora_adapter's explicit-vs-evict distinction).
                if result.success:
                    self.peft_ref_cache.pop(obj.adapter_name, None)
                return result
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

"""OFT tokenizer-manager handlers — the serving-API seam for OFT adapters.

``OFTTokenizerMixin`` holds the dedicated OFT async handlers moved
verbatim out of ``sglang.srt.managers.tokenizer_communicator_mixin``.
``TokenizerManager`` mixes this in as a base class, so the bodies below
resolve ``self.server_args``, ``self.peft_registry``, ``self.peft_update_lock``,
``self.peft_ref_cache``, and ``self.update_oft_adapter_communicator`` at
runtime exactly as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import fastapi

from sglang.srt.oft.io_types import (
    LoadOFTAdapterFromDistributedReqInput,
    LoadOFTAdapterFromDistributedReqOutput,
    LoadOFTAdapterFromTensorsReqInput,
    LoadOFTAdapterFromTensorsReqOutput,
    LoadOFTAdapterReqInput,
    LoadOFTAdapterReqOutput,
    OFTUpdateOutput,
    UnloadOFTAdapterReqInput,
    UnloadOFTAdapterReqOutput,
)
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.utils import normalize_serialized_named_tensor_payloads

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _merge_oft_update_results(results: List[OFTUpdateOutput]) -> OFTUpdateOutput:
    """Collapse worker replies, with any worker failure winning."""
    failed = [result for result in results if not result.success]
    if not failed:
        return results[0]
    messages = list(
        dict.fromkeys(result.error_message for result in failed if result.error_message)
    )
    return OFTUpdateOutput(
        success=False,
        error_message=" | ".join(messages),
        loaded_adapters=failed[0].loaded_adapters,
        previous_adapter_preserved=all(
            not result.success
            and result.previous_adapter_preserved
            and not result.inconsistent_update
            for result in results
        ),
        inconsistent_update=(
            any(result.inconsistent_update for result in results)
            or any(result.success for result in results)
        ),
    )


class OFTTokenizerMixin:
    """Mixin class for TokenizerManager to handle OFT adapter loading/unloading."""

    def _ensure_oft_load_is_not_quarantined(
        self: TokenizerManager, adapter_name: str
    ) -> None:
        if adapter_name in getattr(self, "failed_oft_unloads", {}):
            raise ValueError(
                f"OFT adapter '{adapter_name}' is unavailable after a failed "
                "unload; retry unload before loading or serving it"
            )
        failure = self.failed_oft_activations.get(adapter_name)
        if failure is not None:
            raise ValueError(
                f"OFT adapter '{adapter_name}' is quarantined after an "
                f"inconsistent update; restart required: {failure}"
            )

    def _ensure_no_pending_oft_load(self, adapter_name: str) -> None:
        self._ensure_oft_load_is_not_quarantined(adapter_name)
        pending = self.pending_oft_stage
        if pending is not None and pending.adapter_name == adapter_name:
            raise ValueError(
                f"Cannot load OFT adapter '{adapter_name}' while staged version "
                f"{pending.adapter_version} is pending; activate or unload it first."
            )

    async def _prepare_oft_wire_load(
        self: TokenizerManager,
        obj,
        candidate: OFTRef,
    ):
        """Resolve identity under the lifecycle lock after any model drain."""

        self._ensure_oft_load_is_not_quarantined(obj.adapter_name)
        replaced_ref = self.peft_registry.get_all_adapters().get(obj.adapter_name)
        new_ref, reused = await self.peft_registry.resolve_or_reuse(
            candidate,
            upsert=obj.upsert,
        )
        obj.adapter_id = new_ref.adapter_id
        obj.adapter_version = new_ref.adapter_version
        return new_ref, replaced_ref if reused else None

    async def _finish_oft_wire_load(
        self: TokenizerManager,
        obj,
        new_ref: OFTRef,
        replaced_ref: Optional[OFTRef],
        results: List[OFTUpdateOutput],
    ) -> OFTUpdateOutput:
        """Publish unanimous success or quarantine a divergent/failed upsert."""
        result = _merge_oft_update_results(results)
        if result.success:
            if replaced_ref is None:
                await self.peft_registry.register(new_ref)
            else:
                await self.peft_registry.refresh(new_ref)
            self.peft_ref_cache[obj.adapter_name] = new_ref
            return result

        partial_failure = result.inconsistent_update
        previous_preserved = (
            replaced_ref is not None
            and bool(results)
            and not partial_failure
            and all(
                getattr(item, "previous_adapter_preserved", False) for item in results
            )
        )
        if previous_preserved:
            return result

        if replaced_ref is not None or partial_failure:
            failure = (
                f"OFT adapter '{obj.adapter_name}' is quarantined because its "
                "update did not succeed consistently on every worker; restart required"
            )
            self.failed_oft_activations[obj.adapter_name] = failure
            self.peft_ref_cache.pop(obj.adapter_name, None)
            if replaced_ref is not None:
                adapter_id = await self.peft_registry.unregister(obj.adapter_name)
                await self.peft_registry.wait_for_unload(adapter_id)
            return OFTUpdateOutput(
                success=False,
                error_message=f"{result.error_message} | {failure}",
                loaded_adapters=result.loaded_adapters,
            )
        return result

    async def _run_oft_wire_load(
        self: TokenizerManager,
        obj,
        candidate: OFTRef,
    ) -> OFTUpdateOutput:
        """Drain existing inference before mutating an adapter in place."""
        from sglang.srt.adapter_sync.tokenizer_backend import (
            finish_irreversible_update,
        )

        async def dispatch_and_finish(new_ref, replaced_ref):
            try:
                results = await self.update_oft_adapter_communicator(obj)
            except Exception as error:
                results = [
                    OFTUpdateOutput(
                        success=False,
                        error_message=str(error),
                        inconsistent_update=True,
                    )
                ]
            result = await self._finish_oft_wire_load(
                obj,
                new_ref,
                replaced_ref,
                results,
            )
            if result.success:
                await self._enforce_oft_registry_limit(result)
            return result

        while True:
            # Preflight without the lifecycle lock, matching inference's
            # model -> OFT lock order for implicit adapter reloads.
            _, needs_drain = await self.peft_registry.resolve_or_reuse(
                candidate, upsert=obj.upsert
            )
            if not needs_drain:
                async with self.peft_update_lock:
                    self._ensure_no_pending_oft_load(obj.adapter_name)
                    new_ref, replaced_ref = await self._prepare_oft_wire_load(
                        obj, candidate
                    )
                    if replaced_ref is None:
                        return await finish_irreversible_update(
                            lambda: dispatch_and_finish(new_ref, replaced_ref)
                        )
                # A concurrent load published this name after preflight.
                continue

            async with self.is_pause_cond:
                if self.is_pause:
                    raise ValueError(
                        "Cannot upsert an existing OFT adapter while "
                        "generation is paused; continue generation before retrying."
                    )
                async with self.model_update_lock.writer_lock:
                    async with self.peft_update_lock:
                        self._ensure_no_pending_oft_load(obj.adapter_name)
                        new_ref, replaced_ref = await self._prepare_oft_wire_load(
                            obj, candidate
                        )
                        if replaced_ref is not None:
                            # Dispatch and publication retain both locks through
                            # repeated caller cancellation. Waiting for admission
                            # above is still cancellable without changing state.
                            return await finish_irreversible_update(
                                lambda: dispatch_and_finish(new_ref, replaced_ref)
                            )
            # An unload won the race. Retry as a fresh load without the writer.

    async def _run_oft_path_load(
        self: TokenizerManager,
        obj: LoadOFTAdapterReqInput,
        new_ref: OFTRef,
    ) -> OFTUpdateOutput:
        """Publish and enforce limits as one cancellation-resistant mutation."""
        from sglang.srt.adapter_sync.tokenizer_backend import (
            finish_irreversible_update,
        )

        async def dispatch_publish_and_enforce():
            try:
                results = await self.update_oft_adapter_communicator(obj)
            except Exception as error:
                results = [
                    OFTUpdateOutput(
                        success=False,
                        error_message=str(error),
                        inconsistent_update=True,
                    )
                ]
            result = await self._finish_oft_wire_load(obj, new_ref, None, results)
            if result.success:
                await self._enforce_oft_registry_limit(result)
            return result

        return await finish_irreversible_update(dispatch_publish_and_enforce)

    async def _enforce_oft_registry_limit(
        self: TokenizerManager,
        result: OFTUpdateOutput,
    ) -> None:
        limit = self.server_args.max_loaded_ofts
        if limit is None:
            return
        while self.peft_registry.num_registered_ofts > limit:
            lru_name = await self.peft_registry.lru_oft_name(exclude_pinned=True)
            if lru_name is None:
                raise ValueError(
                    "Didn't find an OFT adapter eligible for LRU eviction. "
                    f"Loaded adapters: {self.peft_registry.get_all_adapters()}"
                )
            unload_result = await self._unload_oft_adapter_locked(
                UnloadOFTAdapterReqInput(adapter_name=lru_name)
            )
            if not unload_result.success:
                raise ValueError(
                    f"Error while unloading LRU OFT adapter {lru_name!r}: "
                    f"{unload_result.error_message}"
                )
            if result.loaded_adapters is not None:
                result.loaded_adapters.pop(lru_name, None)

    async def _unload_oft_adapter_locked(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
    ) -> UnloadOFTAdapterReqOutput:
        assert (
            self.peft_update_lock.locked()
        ), "self.peft_update_lock must be locked in order for self._unload_oft_adapter_locked() to be called"

        pending = self.pending_oft_stage
        active = self.peft_registry.get_all_adapters().get(obj.adapter_name)
        failed_unloads = getattr(self, "failed_oft_unloads", None)
        if failed_unloads is None:
            failed_unloads = self.failed_oft_unloads = {}
        failed_unload = failed_unloads.get(obj.adapter_name)
        cancelling_pending = (
            pending is not None and pending.adapter_name == obj.adapter_name
        )

        async def dispatch_unload():
            try:
                return _merge_oft_update_results(
                    await self.update_oft_adapter_communicator(obj)
                )
            except Exception as error:
                return OFTUpdateOutput(success=False, error_message=str(error))

        async def cancel_pending_on_workers():
            from sglang.srt.adapter_sync.tokenizer_backend import (
                finish_irreversible_update,
            )

            async def dispatch_and_validate():
                try:
                    result = _merge_oft_update_results(
                        await self.update_oft_adapter_communicator(obj)
                    )
                except Exception as error:
                    result = OFTUpdateOutput(success=False, error_message=str(error))
                if not result.success:
                    failure = (
                        f"OFT adapter '{obj.adapter_name}' is quarantined because "
                        "its staged unload did not succeed on every worker; "
                        "restart required"
                    )
                    self.failed_oft_activations[obj.adapter_name] = failure
                    failed_unloads[obj.adapter_name] = active or pending
                    return OFTUpdateOutput(
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
            obj.adapter_id = failed_unload.adapter_id

            from sglang.srt.adapter_sync.tokenizer_backend import (
                finish_irreversible_update,
            )

            async def retry_and_finalize():
                result = await dispatch_unload()
                if result.success:
                    failed_unloads.pop(obj.adapter_name, None)
                    self.failed_oft_activations.pop(obj.adapter_name, None)
                    self.peft_ref_cache.pop(obj.adapter_name, None)
                return result

            return await finish_irreversible_update(retry_and_finalize)

        if active is None and cancelling_pending:
            # A freshly staged adapter is intentionally absent from the serving
            # registry. Unload acts as cancellation of that pending transaction.
            obj.adapter_id = pending.adapter_id
            self.pending_oft_stage = None
            return await cancel_pending_on_workers()

        adapter_id = await self.peft_registry.unregister(obj.adapter_name)
        await self.peft_registry.wait_for_unload(adapter_id)
        obj.adapter_id = adapter_id

        if cancelling_pending:
            # Once unload fan-out begins, any worker may discard this stage.
            # It is no longer safe to activate even if another worker fails.
            self.pending_oft_stage = None
            return await cancel_pending_on_workers()

        result = await dispatch_unload()
        if not result.success:
            failed_unloads[obj.adapter_name] = active
        return result

    async def load_oft_adapter(
        self: TokenizerManager,
        obj: LoadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.peft_method == "oft":
                raise ValueError(
                    "OFT is not enabled. Please set `--peft-method oft` to enable OFT."
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic OFT loading"
            logger.info(
                "Start load OFT adapter. OFT name=%s, path=%s",
                obj.adapter_name,
                obj.adapter_path,
            )

            async with self.peft_update_lock:
                self._ensure_no_pending_oft_load(obj.adapter_name)
                new_adapter = OFTRef(
                    adapter_name=obj.adapter_name,
                    adapter_path=obj.adapter_path,
                    pinned=obj.pinned,
                )

                obj.adapter_id = new_adapter.adapter_id
                return await self._run_oft_path_load(obj, new_adapter)
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
            if not self.server_args.peft_method == "oft":
                raise ValueError(
                    "OFT is not enabled. Please set `--peft-method oft` to enable OFT."
                )
            obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
                obj.serialized_named_tensors
            )
            logger.info(
                "Start load OFT adapter from tensors. OFT name=%s",
                obj.adapter_name,
            )

            return await self._run_oft_wire_load(
                obj,
                OFTRef(
                    adapter_name=obj.adapter_name,
                    adapter_path="__tensor__",
                    pinned=obj.pinned,
                    reloadable=False,
                ),
            )
        except ValueError as e:
            return LoadOFTAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_oft_adapter_from_distributed(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromDistributedReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromDistributedReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.peft_method == "oft":
                raise ValueError(
                    "OFT is not enabled. Please set `--peft-method oft` to enable OFT."
                )
            logger.info(
                "Start load OFT adapter from distributed. OFT name=%s, group=%s",
                obj.adapter_name,
                obj.group_name,
            )

            return await self._run_oft_wire_load(
                obj,
                OFTRef(
                    adapter_name=obj.adapter_name,
                    adapter_path="__distributed__",
                    pinned=obj.pinned,
                    reloadable=False,
                ),
            )
        except ValueError as e:
            return LoadOFTAdapterFromDistributedReqOutput(
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
            if not self.server_args.peft_method == "oft":
                raise ValueError(
                    "OFT is not enabled. Please set `--peft-method oft` to enable OFT."
                )

            assert (
                obj.adapter_name is not None
            ), "adapter_name must be provided to unload OFT adapter"

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic OFT loading"
            logger.info(
                "Start unload OFT adapter. OFT name=%s",
                obj.adapter_name,
            )

            from sglang.srt.adapter_sync.tokenizer_backend import (
                finish_irreversible_update,
            )

            async def unload_and_finalize():
                result = await self._unload_oft_adapter_locked(obj)
                if result.success:
                    self.peft_ref_cache.pop(obj.adapter_name, None)
                return result

            async with self.peft_update_lock:
                # Unregister, lease drainage, worker fan-out, and catalog cleanup
                # form one mutation. Keep the lock until it finishes even when
                # the HTTP caller is cancelled repeatedly.
                return await finish_irreversible_update(unload_and_finalize)
        except ValueError as e:
            return UnloadOFTAdapterReqOutput(success=False, error_message=str(e))

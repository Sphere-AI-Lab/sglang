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
    )


class OFTTokenizerMixin:
    """Mixin class for TokenizerManager to handle OFT adapter loading/unloading."""

    def _ensure_oft_load_is_not_quarantined(
        self: TokenizerManager, adapter_name: str
    ) -> None:
        failure = self.failed_oft_activations.get(adapter_name)
        if failure is not None:
            raise ValueError(
                f"OFT adapter '{adapter_name}' is quarantined after an "
                f"inconsistent update; restart required: {failure}"
            )

    async def _prepare_oft_wire_load(
        self: TokenizerManager,
        obj,
        candidate: OFTRef,
    ):
        """Resolve identity and drain an existing adapter before an upsert."""
        self._ensure_oft_load_is_not_quarantined(obj.adapter_name)
        new_ref, reused = await self.peft_registry.resolve_or_reuse(
            candidate,
            upsert=obj.upsert,
        )
        if reused:
            adapter_id = await self.peft_registry.unregister(obj.adapter_name)
            self.peft_ref_cache.pop(obj.adapter_name, None)
            await self.peft_registry.wait_for_unload(adapter_id)

        obj.adapter_id = new_ref.adapter_id
        obj.adapter_version = new_ref.adapter_version
        return new_ref, reused

    async def _finish_oft_wire_load(
        self: TokenizerManager,
        obj,
        new_ref: OFTRef,
        reused: bool,
        results: List[OFTUpdateOutput],
    ) -> OFTUpdateOutput:
        """Publish unanimous success or quarantine a divergent/failed upsert."""
        result = _merge_oft_update_results(results)
        if result.success:
            await self.peft_registry.register(new_ref)
            self.peft_ref_cache[obj.adapter_name] = new_ref
            return result

        partial_failure = any(item.success for item in results) and any(
            not item.success for item in results
        )
        if reused or partial_failure:
            failure = (
                f"OFT adapter '{obj.adapter_name}' is quarantined because its "
                "update did not succeed consistently on every worker; restart required"
            )
            self.failed_oft_activations[obj.adapter_name] = failure
            self.peft_ref_cache.pop(obj.adapter_name, None)
            return OFTUpdateOutput(
                success=False,
                error_message=f"{result.error_message} | {failure}",
                loaded_adapters=result.loaded_adapters,
            )
        return result

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

        adapter_id = await self.peft_registry.unregister(obj.adapter_name)
        await self.peft_registry.wait_for_unload(adapter_id)
        obj.adapter_id = adapter_id

        return _merge_oft_update_results(
            await self.update_oft_adapter_communicator(obj)
        )

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
                new_adapter = OFTRef(
                    adapter_name=obj.adapter_name,
                    adapter_path=obj.adapter_path,
                    pinned=obj.pinned,
                )

                obj.adapter_id = new_adapter.adapter_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    await self.peft_registry.register(new_adapter)
                    self.peft_ref_cache[obj.adapter_name] = new_adapter

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

            async with self.peft_update_lock:
                new_ref, reused = await self._prepare_oft_wire_load(
                    obj,
                    OFTRef(
                        adapter_name=obj.adapter_name,
                        adapter_path="__tensor__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                )
                result = await self._finish_oft_wire_load(
                    obj,
                    new_ref,
                    reused,
                    await self.update_oft_adapter_communicator(obj),
                )

                if result.success:
                    await self._enforce_oft_registry_limit(result)

                return result
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

            async with self.peft_update_lock:
                new_ref, reused = await self._prepare_oft_wire_load(
                    obj,
                    OFTRef(
                        adapter_name=obj.adapter_name,
                        adapter_path="__distributed__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                )
                result = await self._finish_oft_wire_load(
                    obj,
                    new_ref,
                    reused,
                    await self.update_oft_adapter_communicator(obj),
                )

                if result.success:
                    await self._enforce_oft_registry_limit(result)

                return result
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

            async with self.peft_update_lock:
                result = await self._unload_oft_adapter_locked(obj)
                if result.success:
                    self.peft_ref_cache.pop(obj.adapter_name, None)
                return result
        except ValueError as e:
            return UnloadOFTAdapterReqOutput(success=False, error_message=str(e))

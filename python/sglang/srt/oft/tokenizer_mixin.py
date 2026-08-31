"""OFT tokenizer-manager handlers — the serving-API seam for OFT adapters.

``OFTTokenizerMixin`` holds the dedicated OFT async handlers moved
verbatim out of ``sglang.srt.managers.tokenizer_communicator_mixin``.
``TokenizerManager`` mixes this in as a base class, so the bodies below
resolve ``self.server_args``, ``self.oft_registry``, ``self.oft_update_lock``,
``self.oft_ref_cache``, and ``self.update_oft_adapter_communicator`` at
runtime exactly as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import fastapi

from sglang.srt.oft.io_types import (
    LoadOFTAdapterFromTensorsReqInput,
    LoadOFTAdapterFromTensorsReqOutput,
    LoadOFTAdapterReqInput,
    LoadOFTAdapterReqOutput,
    UnloadOFTAdapterReqInput,
    UnloadOFTAdapterReqOutput,
)
from sglang.srt.oft.oft_registry import OFTRef

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OFTTokenizerMixin:
    """Mixin class for TokenizerManager to handle OFT adapter loading/unloading."""

    async def _unload_oft_adapter_locked(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
    ) -> UnloadOFTAdapterReqOutput:
        assert (
            self.oft_update_lock.locked()
        ), "self.oft_update_lock must be locked in order for self._unload_oft_adapter_locked() to be called"

        adapter_id = await self.oft_registry.unregister(obj.adapter_name)
        obj.adapter_id = adapter_id

        await self.oft_registry.wait_for_unload(adapter_id)
        result = (await self.update_oft_adapter_communicator(obj))[0]

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

            async with self.oft_update_lock:
                new_adapter = OFTRef(
                    adapter_name=obj.adapter_name,
                    adapter_path=obj.adapter_path,
                    pinned=obj.pinned,
                )

                obj.adapter_id = new_adapter.adapter_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    await self.oft_registry.register(new_adapter)
                    self.oft_ref_cache[obj.adapter_name] = new_adapter

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

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic OFT loading"
            logger.info(
                "Start load OFT adapter from tensors. OFT name=%s",
                obj.adapter_name,
            )

            async with self.oft_update_lock:
                new_adapter = OFTRef(
                    adapter_name=obj.adapter_name,
                    adapter_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.adapter_id = new_adapter.adapter_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    await self.oft_registry.register(new_adapter)
                    self.oft_ref_cache[obj.adapter_name] = new_adapter

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

            async with self.oft_update_lock:
                return await self._unload_oft_adapter_locked(obj)
        except ValueError as e:
            return UnloadOFTAdapterReqOutput(success=False, error_message=str(e))

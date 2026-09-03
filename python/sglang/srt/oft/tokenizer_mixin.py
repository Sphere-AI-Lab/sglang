"""Tokenizer-side OFT adapter registry hooks.

``OFTTokenizerMixin`` holds the OFT-specific tokenizer-manager handlers;
``TokenizerManager`` mixes this in as a base class, so the bodies below
resolve ``self.server_args``, ``self.oft_registry``, ``self.oft_ref_cache``,
and ``self.oft_update_lock`` at runtime -- the same shape as LoRA's own
``self.lora_registry``/``self.lora_update_lock`` and its
``_resolve_lora_path``/``_finalize_lora_lease`` methods, so the two stay easy
to compare side by side when porting a future upstream LoRA change over to
OFT.

Single-active invariant: the engine boots with ``enable_lora`` XOR
``enable_oft``, so there is exactly one active OFT registry / ref
cache. These hooks never branch on ``load_format`` or adapter type -- the
ref class is fixed at init via ``_mint_ref``. The AdapterRegistry base
(oft/base/registry.py) provides the register/acquire/LRU behaviour.

Import-light by design (registry classes imported lazily): the tokenizer-manager
boot chain is deep. See commit c42b88a1c for the cycle this avoids.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

import fastapi

from sglang.srt.managers.io_struct import (
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


def _propagate_id_to_cached_sub_objs(obj, *, field, resolved):
    """Push the resolved adapter id into sub-objects already memoized by
    ``GenerateReqInput.__getitem__`` (``obj._sub_obj_cache``).

    ``_init_req_state`` materializes ``obj[i]`` for every request in a batch
    BEFORE this resolver runs, so an id set on ``obj`` alone never reaches the
    cached copies -- ``_handle_batch_request`` then tokenizes those stale
    copies and every batched request lands on the scheduler with a ``None`` id
    (OFT: identity slot applied, output silently equals the base model; lora:
    radix ``extra_key`` loses the adapter id). Mirrors the legacy propagation
    in ``TokenizerManager._resolve_lora_path``.
    """
    for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
        setattr(
            sub_obj, field, resolved[i] if isinstance(resolved, list) else resolved
        )


class OFTTokenizerMixin:
    """Mixin class for TokenizerManager to handle OFT adapter registry/lease."""

    def _mint_ref(self: TokenizerManager, name):
        """Build the OFT AdapterRef for ``name`` (path == name for streamed
        adapters). Matches the registry class built in init_tokenizer_oft.

        reloadable=False: a streamed adapter has no on-disk artifact to reload
        from, mirroring staged_manager.py's LoRARef construction for its own
        streamed/staged adapters."""
        from sglang.srt.oft.oft_registry import OFTRef

        return OFTRef(
            oft_name=name, oft_path=name, pinned=False, reloadable=False
        )

    def _request_oft_path(self: TokenizerManager, obj):
        """The peft adapter path a request carries (single-active: at most one of
        oft_path/lora_path is set). oft_path/lora_path are declared on both
        GenerateReqInput and EmbeddingReqInput with default None, so plain
        attribute access is the contract (mirrors _finalize_lora_lease's
        lora_path/lora_id None checks)."""
        if obj.oft_path:
            return obj.oft_path
        return obj.lora_path

    def init_tokenizer_oft(self: TokenizerManager):
        """OFT registry bootstrap (replaces the former init_oft / init_lora split for
        the single-active peft path). Sets up the OFT registry + ref cache."""
        self.oft_update_lock = asyncio.Lock()
        self.oft_ref_cache = {}
        self.oft_registry = None
        self._logged_oft_base_only_request = False

        if self.enable_oft:
            from sglang.srt.oft.oft_registry import OFTRegistry

            self.oft_registry = OFTRegistry()
            logger.info(
                "event=oft_tokenizer_registry_initialized initial_adapters=%s",
                sorted(self.oft_ref_cache.keys()),
            )

    async def register_oft_ref(self: TokenizerManager, obj) -> bool:
        """Register-before-dispatch for a streamed OFT adapter: mint the
        ref in the registry and set obj.adapter_id.

        Triggered by ``obj.adapter_name`` being set -- orbit populates it only on
        adapter loads (update_weights_from_tensor), not on base-weight updates --
        so no load_format branch is needed.

        Returns True if this call newly minted and registered the ref (the name
        was not already known to this tokenizer), False if it resolved an
        existing one. A caller whose backend dispatch can still fail after this
        returns uses that to decide whether a failure should roll the
        registration back via ``rollback_oft_ref`` -- otherwise it would
        incorrectly tear down an adapter that was already loaded and serving
        before this request.
        """
        if obj.adapter_name is None or self.oft_registry is None:
            return False
        name = obj.adapter_name
        newly_registered = name not in self.oft_ref_cache
        if newly_registered:
            ref = self._mint_ref(name)
            await self.oft_registry.register(ref)
            self.oft_ref_cache[name] = ref
        obj.adapter_id = self.oft_ref_cache[name].oft_id
        return newly_registered

    async def rollback_oft_ref(self: TokenizerManager, name):
        """Undo a ``register_oft_ref`` registration after the backend load it was
        staged for turned out to fail.

        Only call this when ``register_oft_ref`` reported ``True`` (newly
        registered) for this same name -- otherwise this would incorrectly tear
        down an adapter that was already loaded and serving before this request.

        Without this, a failed streamed load (e.g. the retired
        ``load_format="oft_adapter"`` path's graceful reject) leaves a
        registered-but-not-actually-resident name behind: a later ``/generate``
        naming it passes the tokenizer-side registry check and reaches the
        GPU-side code with no matching adapter there, instead of a clean
        "adapter not found" rejection.
        """
        if self.oft_registry is None:
            return
        adapter_id = await self.oft_registry.unregister(name)
        await self.oft_registry.wait_for_unload(adapter_id)
        self.oft_ref_cache.pop(name, None)

    async def bump_oft_version(self: TokenizerManager, obj, success):
        """Bump the adapter version after a successful update (OFT tracks
        versions via bump_version_by_id). Returns a message suffix."""
        if not (
            success and obj.adapter_name is not None and self.oft_registry is not None
        ):
            return ""
        reg = self.oft_registry
        if obj.adapter_id is not None and hasattr(reg, "bump_version_by_id"):
            updated = await reg.bump_version_by_id(obj.adapter_id)
            self.oft_ref_cache[updated.oft_name] = updated
            return (
                f" PEFT adapter {updated.oft_name} version updated to "
                f"{updated.version}."
            )
        return ""

    async def resolve_oft_path(self: TokenizerManager, obj):
        """Per-request OFT adapter resolver: acquire the adapter id (+ version),
        and reload any dynamically-evicted OFT adapter."""
        path = self._request_oft_path(obj)
        unique_paths = {path} if isinstance(path, str) else set(path)

        # Mirrors TokenizerManager._resolve_lora_path's max_loaded_loras guard:
        # reject a request naming more unique adapters than the tokenizer-side
        # registry can hold, before touching the registry at all.
        if (
            self.server_args.max_loaded_ofts is not None
            and len(unique_paths) > self.server_args.max_loaded_ofts
        ):
            raise ValueError(
                f"Received request with {len(unique_paths)} unique OFT adapters "
                f"requested but max loaded ofts is {self.server_args.max_loaded_ofts}"
            )

        # Reload adapters that were dynamically evicted (OFT eviction; no-op single-active).
        unregistered = await self.oft_registry.get_unregistered_adapters(unique_paths)
        for oft_path in unregistered:
            if oft_path is None:
                continue
            if oft_path not in self.oft_ref_cache:
                raise ValueError(
                    f"Got PEFT adapter that has never been loaded: {oft_path}\n"
                    f"All loaded adapters: {self.oft_ref_cache.keys()}."
                )
            ref = self.oft_ref_cache[oft_path]
            if not ref.reloadable:
                raise ValueError(
                    f"OFT adapter '{oft_path}' was loaded dynamically (via "
                    "tensors/distributed, or streamed via "
                    "update_weights_from_tensor) and was evicted from the "
                    "registry; it has no on-disk artifact to reload from and "
                    "must be re-loaded via a fresh "
                    "load_oft_adapter_from_tensors/_from_distributed call, or "
                    "re-streamed by the trainer."
                )
            # Mirrors TokenizerManager._resolve_lora_path's implicit reload of
            # a disk/HF-path adapter LRU-evicted from the tokenizer registry.
            logger.info(f"Reloading evicted adapter: {oft_path}")
            load_result = await self.load_oft_adapter(
                LoadOFTAdapterReqInput(
                    oft_name=ref.oft_name,
                    oft_path=ref.oft_path,
                    pinned=ref.pinned,
                )
            )
            if (
                not load_result.success
                and "already loaded" not in load_result.error_message
            ):
                raise ValueError(
                    f"Failed to implicitly load OFT adapter {oft_path}: "
                    f"{load_result.error_message}"
                )

        oft_id, oft_version = await self.oft_registry.acquire_with_version(path)
        # Set the request-side id/version fields the scheduler reads.
        obj.oft_id = oft_id
        obj.oft_version = oft_version
        _propagate_id_to_cached_sub_objs(obj, field="oft_id", resolved=oft_id)
        # The version needs the same propagation as the id: batched sub-objects
        # are materialized before this resolver runs, so a version set only on
        # the parent never reaches the tokenized requests built from them.
        _propagate_id_to_cached_sub_objs(
            obj, field="oft_version", resolved=oft_version
        )

    async def maybe_resolve_oft_path(self: TokenizerManager, obj):
        """Request-intake resolve for the active peft method. Routes a named request
        through resolve_oft_path; logs the base-only (no adapter path) case once.

        The one production call site already guards on self.enable_oft before
        calling this, so self.oft_registry always exists by then (set in
        init_tokenizer_oft); this check is belt-and-suspenders for any other
        caller."""
        if not self.enable_oft:
            return
        if self._request_oft_path(obj):
            await self.resolve_oft_path(obj)
        elif not self._logged_oft_base_only_request:
            logger.info(
                "event=oft_request_base_only message='OFT enabled but this "
                "request carries no adapter path; using base/identity slot.'"
            )
            self._logged_oft_base_only_request = True

    def finalize_oft_lease(self: TokenizerManager, state) -> None:
        """Release the request's OFT adapter lease exactly once, however it
        terminates: normal finish, scheduler abort echo (queued / tokenizer-held /
        disagg), status-code abort, or a failed dispatch. Mirrors
        TokenizerManager._finalize_lora_lease exactly, for the OFT registry
        path: without this release, oft_registry.wait_for_unload (called by
        unload_oft_adapter and the max_loaded_ofts LRU-eviction loop) would
        block forever on any adapter that ever served a request.

        ``oft_path``/``oft_id`` are declared on both GenerateReqInput and
        EmbeddingReqInput with default None, so plain attribute access is the
        contract (mirrors _finalize_lora_lease's lora_path/lora_id None
        checks exactly). State-derived checks come first: a request without a
        lease has nothing to release, whatever the server config says, and
        only then does the enable-check (self.enable_oft, mirroring
        _finalize_lora_lease's `not self.enable_lora`) gate the actual
        registry touch.
        """
        if state is None or state.oft_lease_released:
            return
        if state.obj.oft_path is None or state.obj.oft_id is None:
            return
        if not self.enable_oft:
            return
        state.oft_lease_released = True
        asyncio.create_task(self.oft_registry.release(state.obj.oft_id))

    async def load_oft_adapter(
        self: TokenizerManager,
        obj: LoadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterReqOutput:
        """Mirrors TokenizerManager.load_lora_adapter exactly: a disk/HF-path
        adapter load, producing a ref that defaults to reloadable=True (unlike
        the tensor/distributed routes, which explicitly pin reloadable=False)
        -- this is what makes resolve_oft_path's implicit-reload branch for an
        evicted adapter reachable for OFT, the same way it already is for
        LoRA."""
        self.auto_create_handle_loop()

        try:
            if not (self.enable_oft and self.server_args.oft_impl == "sibling"):
                raise ValueError(
                    "Native OFT adapter loading requires --enable-oft "
                    "--oft-impl sibling."
                )
            logger.info(
                "Start load OFT adapter. Adapter name=%s, path=%s",
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
                result = _merge_oft_update_results(
                    await self.update_oft_adapter_communicator(obj)
                )

                if result.success:
                    await self.oft_registry.register(new_adapter)
                    self.oft_ref_cache[obj.oft_name] = new_adapter

                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.oft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.oft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.oft_registry.get_all_adapters()}"
                            )
                        logger.info(
                            f"Unloading least recently used OFT adapter '{lru_name}' "
                            f"(current number of adapters: {self.oft_registry.num_registered_ofts}, "
                            f"max allowed: {self.server_args.max_loaded_ofts})"
                        )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(oft_name=lru_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter "
                                f"'{lru_name}': {unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_name]

                return result
        except ValueError as e:
            return LoadOFTAdapterReqOutput(success=False, error_message=str(e))

    async def load_oft_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (self.enable_oft and self.server_args.oft_impl == "sibling"):
                raise ValueError(
                    "Native OFT adapter loading requires --enable-oft "
                    "--oft-impl sibling."
                )
            obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
                obj.serialized_named_tensors
            )
            logger.info(
                "Start load OFT adapter from tensors. Adapter name=%s",
                obj.oft_name,
            )
            async with self.oft_update_lock:
                # Built inline (not via obj.to_ref()): to_ref() passes
                # obj.oft_id through explicitly, which is None on a fresh
                # load and would short-circuit OFTRef's default_factory,
                # tripping its "oft_id cannot be None" guard. Mirrors
                # load_lora_adapter_from_distributed's LoRARef(...) construction.
                new_ref, reused = await self.oft_registry.resolve_or_reuse(
                    OFTRef(
                        oft_name=obj.oft_name,
                        oft_path="__tensor__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                    upsert=obj.upsert,
                )
                obj.oft_id = new_ref.oft_id
                results = await self.update_oft_adapter_communicator(obj)
                result = _merge_oft_update_results(results)

                if result.success:
                    if reused:
                        await self.oft_registry.refresh(new_ref)
                    else:
                        await self.oft_registry.register(new_ref)
                    self.oft_ref_cache[obj.oft_name] = new_ref
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.oft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.oft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.oft_registry.get_all_adapters()}"
                            )
                        logger.info(
                            f"Unloading least recently used OFT adapter '{lru_name}' "
                            f"(current number of adapters: {self.oft_registry.num_registered_ofts}, "
                            f"max allowed: {self.server_args.max_loaded_ofts})"
                        )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(oft_name=lru_name)
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
            if not (self.enable_oft and self.server_args.oft_impl == "sibling"):
                raise ValueError(
                    "Native OFT adapter loading requires --enable-oft "
                    "--oft-impl sibling."
                )
            logger.info(
                "Start load OFT adapter from distributed. Adapter name=%s, group=%s",
                obj.oft_name,
                obj.group_name,
            )
            async with self.oft_update_lock:
                # See load_oft_adapter_from_tensors: built inline rather than
                # via obj.to_ref(), which would pass the not-yet-minted
                # obj.oft_id (None) straight through and trip OFTRef's
                # "oft_id cannot be None" guard instead of minting a
                # fresh id. Mirrors load_lora_adapter_from_distributed's
                # LoRARef(...) construction.
                new_ref, reused = await self.oft_registry.resolve_or_reuse(
                    OFTRef(
                        oft_name=obj.oft_name,
                        oft_path="__distributed__",
                        pinned=obj.pinned,
                        reloadable=False,
                    ),
                    upsert=obj.upsert,
                )
                obj.oft_id = new_ref.oft_id
                # Merge (not [0]): unlike LoRA's from_distributed route, this
                # handler has no dp_size == 1 guard, so a non-rank-0 failure
                # must not be silently reported as success.
                result = _merge_oft_update_results(
                    await self.update_oft_adapter_communicator(obj)
                )

                if result.success:
                    if reused:
                        await self.oft_registry.refresh(new_ref)
                    else:
                        await self.oft_registry.register(new_ref)
                    self.oft_ref_cache[obj.oft_name] = new_ref
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.oft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.oft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.oft_registry.get_all_adapters()}"
                            )
                        logger.info(
                            f"Unloading least recently used OFT adapter '{lru_name}' "
                            f"(current number of adapters: {self.oft_registry.num_registered_ofts}, "
                            f"max allowed: {self.server_args.max_loaded_ofts})"
                        )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(oft_name=lru_name)
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
        """Caller must hold oft_update_lock. Unregisters + tells the
        scheduler to free GPU state; does NOT touch oft_ref_cache (the
        caller decides evict-vs-delete semantics, mirroring
        _unload_lora_adapter_locked)."""
        # Unregister the OFT adapter from the registry to stop new requests
        # for this adapter from being started.
        oft_id = await self.oft_registry.unregister(obj.oft_name)

        # Initiate the actual unloading operation at the backend processes
        # only after all ongoing requests using this adapter are finished.
        await self.oft_registry.wait_for_unload(oft_id)
        obj.oft_id = oft_id
        result = _merge_oft_update_results(
            await self.update_oft_adapter_communicator(obj)
        )

        return result

    async def unload_oft_adapter(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadOFTAdapterReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (self.enable_oft and self.server_args.oft_impl == "sibling"):
                raise ValueError(
                    "Native OFT adapter loading requires --enable-oft "
                    "--oft-impl sibling."
                )
            logger.info(
                "Start unload OFT adapter. Adapter name=%s",
                obj.oft_name,
            )
            async with self.oft_update_lock:
                result = await self._unload_oft_adapter_locked(obj)
                # Explicit unload is a DELETE: drop the ref_cache entry too
                # (mirrors unload_lora_adapter's explicit-vs-evict distinction).
                if result.success:
                    self.oft_ref_cache.pop(obj.oft_name, None)
                return result
        except ValueError as e:
            return UnloadOFTAdapterReqOutput(success=False, error_message=str(e))

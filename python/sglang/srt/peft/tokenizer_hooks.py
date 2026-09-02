"""Tokenizer-side PEFT adapter registry hooks for the single-active OFT peft
method.

Single-active invariant: the engine boots with ``enable_peft_lora`` XOR
``enable_oft``, so there is exactly ONE active peft registry / ref_cache. These
hooks route to that active registry and never branch on ``load_format`` or
adapter type -- the ref class is fixed once at init via a ref factory. The
AdapterRegistry base (peft/base/registry.py) provides the register/acquire/LRU
behaviour.

Import-light by design (registry classes imported lazily): the tokenizer-manager
boot chain is deep. See commit c42b88a1c for the cycle this avoids.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


def _peft_kind(tm):
    """The active single-active peft method: "oft", or None."""
    return tm.server_args.peft_method


def _mint_ref(tm, name):
    """Build the OFT AdapterRef for ``name`` (path == name for streamed
    adapters). Matches the registry class built in init_tokenizer_peft.

    reloadable=False: a streamed adapter has no on-disk artifact to reload
    from, mirroring staged_manager.py's LoRARef construction for its own
    streamed/staged adapters."""
    from sglang.srt.oft.oft_registry import OFTRef

    return OFTRef(
        adapter_name=name, adapter_path=name, pinned=False, reloadable=False
    )


def _request_peft_path(obj):
    """The peft adapter path a request carries (single-active: at most one of
    adapter_path/lora_path is set)."""
    if getattr(obj, "adapter_path", None):
        return obj.adapter_path
    return getattr(obj, "lora_path", None)


def init_tokenizer_peft(tm):
    """PEFT registry bootstrap (replaces the former init_oft / init_lora split for
    the single-active peft path). Sets up the one active peft registry + ref cache."""
    kind = _peft_kind(tm)
    tm.peft_kind = kind
    tm.peft_update_lock = asyncio.Lock()
    tm.peft_ref_cache = {}
    tm.peft_registry = None
    tm._logged_peft_base_only_request = False

    if kind == "oft":
        from sglang.srt.oft.oft_registry import OFTRegistry

        tm.peft_registry = OFTRegistry()

    if kind is not None:
        logger.info(
            "event=peft_tokenizer_registry_initialized kind=%s initial_adapters=%s",
            kind,
            sorted(tm.peft_ref_cache.keys()),
        )


async def register_peft_ref(tm, obj) -> bool:
    """Register-before-dispatch for a streamed peft adapter (LoRA or OFT): mint the
    ref in the active registry and set obj.adapter_id.

    Triggered by ``obj.adapter_name`` being set -- orbit populates it only on peft
    adapter loads (update_weights_from_tensor), not on base-weight updates -- so no
    load_format branch is needed.

    Returns True if this call newly minted and registered the ref (the name
    was not already known to this tokenizer), False if it resolved an
    existing one. A caller whose backend dispatch can still fail after this
    returns uses that to decide whether a failure should roll the
    registration back via ``rollback_peft_ref`` -- otherwise it would
    incorrectly tear down an adapter that was already loaded and serving
    before this request.
    """
    if obj.adapter_name is None or tm.peft_registry is None:
        return False
    name = obj.adapter_name
    newly_registered = name not in tm.peft_ref_cache
    if newly_registered:
        ref = _mint_ref(tm, name)
        await tm.peft_registry.register(ref)
        tm.peft_ref_cache[name] = ref
    obj.adapter_id = tm.peft_ref_cache[name].adapter_id
    return newly_registered


async def rollback_peft_ref(tm, name):
    """Undo a ``register_peft_ref`` registration after the backend load it was
    staged for turned out to fail.

    Only call this when ``register_peft_ref`` reported ``True`` (newly
    registered) for this same name -- otherwise this would incorrectly tear
    down an adapter that was already loaded and serving before this request.

    Without this, a failed streamed load (e.g. the retired
    ``load_format="oft_adapter"`` path's graceful reject) leaves a
    registered-but-not-actually-resident name behind: a later ``/generate``
    naming it passes the tokenizer-side registry check and reaches the
    GPU-side code with no matching adapter there, instead of a clean
    "adapter not found" rejection.
    """
    if tm.peft_registry is None:
        return
    adapter_id = await tm.peft_registry.unregister(name)
    await tm.peft_registry.wait_for_unload(adapter_id)
    tm.peft_ref_cache.pop(name, None)


async def bump_peft_version(tm, obj, success):
    """Bump the adapter version after a successful update (OFT tracks
    versions via bump_version_by_id). Returns a message suffix."""
    if not (success and obj.adapter_name is not None and tm.peft_registry is not None):
        return ""
    reg = tm.peft_registry
    if obj.adapter_id is not None and hasattr(reg, "bump_version_by_id"):
        updated = await reg.bump_version_by_id(obj.adapter_id)
        tm.peft_ref_cache[updated.adapter_name] = updated
        return (
            f" PEFT adapter {updated.adapter_name} version updated to "
            f"{updated.adapter_version}."
        )
    return ""


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


async def resolve_peft_path(tm, obj):
    """Per-request peft adapter resolver: acquire the adapter id (+ version),
    and reload any dynamically-evicted OFT adapter."""
    path = _request_peft_path(obj)
    unique_paths = {path} if isinstance(path, str) else set(path)

    # Reload adapters that were dynamically evicted (OFT eviction; no-op single-active).
    unregistered = await tm.peft_registry.get_unregistered_adapters(unique_paths)
    for adapter_path in unregistered:
        if adapter_path is None:
            continue
        if adapter_path not in tm.peft_ref_cache:
            raise ValueError(
                f"Got PEFT adapter that has never been loaded: {adapter_path}\n"
                f"All loaded adapters: {tm.peft_ref_cache.keys()}."
            )
        ref = tm.peft_ref_cache[adapter_path]
        if not ref.reloadable:
            raise ValueError(
                f"OFT adapter '{adapter_path}' was loaded dynamically (via "
                "tensors/distributed, or streamed via "
                "update_weights_from_tensor) and was evicted from the "
                "registry; it has no on-disk artifact to reload from and "
                "must be re-loaded via a fresh "
                "load_oft_adapter_from_tensors/_from_distributed call, or "
                "re-streamed by the trainer."
            )

    adapter_id, adapter_version = await tm.peft_registry.acquire_with_version(path)
    # Set the request-side id/version fields the scheduler reads.
    obj.adapter_id = adapter_id
    obj.adapter_version = adapter_version
    _propagate_id_to_cached_sub_objs(obj, field="adapter_id", resolved=adapter_id)
    # The version needs the same propagation as the id: batched sub-objects
    # are materialized before this resolver runs, so a version set only on
    # the parent never reaches the tokenized requests built from them.
    _propagate_id_to_cached_sub_objs(
        obj, field="adapter_version", resolved=adapter_version
    )


async def maybe_resolve_peft_path(tm, obj):
    """Request-intake resolve for the active peft method. Routes a named request
    through resolve_peft_path; logs the base-only (no adapter path) case once."""
    if tm.peft_registry is None:
        return
    if _request_peft_path(obj):
        await resolve_peft_path(tm, obj)
    elif not tm._logged_peft_base_only_request:
        logger.info(
            "event=peft_request_base_only kind=%s message='peft enabled but this "
            "request carries no adapter path; using base/identity slot.'",
            tm.peft_kind,
        )
        tm._logged_peft_base_only_request = True


def finalize_peft_lease(tm, state) -> None:
    """Release the request's peft (OFT) adapter lease exactly once, however it
    terminates: normal finish, scheduler abort echo (queued / tokenizer-held /
    disagg), status-code abort, or a failed dispatch. Mirrors
    TokenizerManager._finalize_lora_lease exactly, for the single-active
    peft_registry path: without this release, peft_registry.wait_for_unload
    (called by unload_oft_adapter and the max_loaded_ofts LRU-eviction loop)
    would block forever on any adapter that ever served a request.

    ``adapter_id`` lives only on GenerateReqInput (peft has no embedding
    support, see generate_request's isinstance guard), hence the getattr
    instead of direct attribute access -- state.obj may be an
    EmbeddingReqInput, which never declares the field.
    """
    if state is None or state.peft_lease_released:
        return
    adapter_id = getattr(state.obj, "adapter_id", None)
    if adapter_id is None or tm.peft_registry is None:
        return
    state.peft_lease_released = True
    asyncio.create_task(tm.peft_registry.release(adapter_id))

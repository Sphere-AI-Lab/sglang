"""Tokenizer-side PEFT adapter registry hooks — unified across the single-active
peft methods (LoRA and OFT).

Single-active invariant: the engine boots with ``enable_peft_lora`` XOR
``enable_oft``, so there is exactly ONE active peft registry / ref_cache. These
hooks route to that active registry and never branch on ``load_format`` or adapter
type -- the ``LoRARef`` vs ``OFTRef`` choice is fixed once at init via a ref
factory. The AdapterRegistry base (peft/base/registry.py) provides all the
register/acquire/LRU behaviour for both.

Import-light by design (registry classes imported lazily): the tokenizer-manager
boot chain is deep. See commit c42b88a1c for the cycle this avoids.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


def _peft_kind(tm):
    """The active single-active peft method: "oft", "lora", or None."""
    return tm.server_args.peft_method


def _mint_ref(tm, name):
    """Build the active kind's AdapterRef for ``name`` (path == name for
    streamed adapters). The OFT ref class follows --oft-impl so minted refs
    match the registry built in init_tokenizer_peft."""
    if tm.peft_kind == "oft":
        if tm.server_args.oft_impl in ("sibling", "staged"):
            from sglang.srt.oft.oft_registry import OFTRef
        else:
            from sglang.srt.peft.oft.oft_registry import OFTRef

        return OFTRef(adapter_name=name, adapter_path=name, pinned=False)
    from sglang.srt.peft.lora.lora_registry import LoRARef

    return LoRARef(adapter_name=name, adapter_path=name, pinned=False)


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
        # Registry class follows --oft-impl (byte-identical twins); its ctor
        # asserts the refs are its own OFTRef class, which validate_peft_args
        # normalized with the same dispatch.
        if tm.server_args.oft_impl in ("sibling", "staged"):
            from sglang.srt.oft.oft_registry import OFTRegistry
        else:
            from sglang.srt.peft.oft.oft_registry import OFTRegistry

        tm.peft_registry = OFTRegistry(tm.server_args.peft_paths)
        for ref in tm.server_args.peft_paths or []:
            tm.peft_ref_cache[ref.adapter_name] = ref
    elif kind == "lora":
        from sglang.srt.peft.lora.lora_registry import LoRARef, LoRARegistry

        # peft_paths is {name: path} (or None/[] at identity boot).
        raw = tm.server_args.peft_paths
        initial = (
            [LoRARef(adapter_name=n, adapter_path=p, pinned=False) for n, p in raw.items()]
            if isinstance(raw, dict)
            else list(raw or [])
        )
        tm.peft_registry = LoRARegistry(initial)
        for ref in initial:
            tm.peft_ref_cache[ref.adapter_name] = ref

    if kind is not None:
        logger.info(
            "event=peft_tokenizer_registry_initialized kind=%s initial_adapters=%s",
            kind,
            sorted(tm.peft_ref_cache.keys()),
        )


async def register_peft_ref(tm, obj):
    """Register-before-dispatch for a streamed peft adapter (LoRA or OFT): mint the
    ref in the active registry and set obj.adapter_id.

    Triggered by ``obj.adapter_name`` being set -- orbit populates it only on peft
    adapter loads (update_weights_from_tensor), not on base-weight updates -- so no
    load_format branch is needed.
    """
    if obj.adapter_name is None or tm.peft_registry is None:
        return
    name = obj.adapter_name
    if name not in tm.peft_ref_cache:
        ref = _mint_ref(tm, name)
        await tm.peft_registry.register(ref)
        tm.peft_ref_cache[name] = ref
    obj.adapter_id = tm.peft_ref_cache[name].adapter_id


async def bump_peft_version(tm, obj, success):
    """Bump the adapter version after a successful update. OFT tracks versions;
    LoRA has none (no bump_version_by_id) -> no-op. Returns a message suffix."""
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
    """Per-request peft adapter resolver: acquire the adapter id (+ version), and
    for OFT reload any dynamically-evicted adapter (single-active LoRA never
    evicts). Generic over the active registry."""
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
        if tm.peft_kind == "oft":
            from sglang.srt.peft.io_types import LoadOFTAdapterReqInput

            ref = tm.peft_ref_cache[adapter_path]
            logger.info(f"Reloading evicted adapter: {adapter_path}")
            load_result = await tm.load_oft_adapter(
                LoadOFTAdapterReqInput(
                    adapter_name=ref.adapter_name, adapter_path=ref.adapter_path, pinned=ref.pinned
                )
            )
            if (
                not load_result.success
                and "already loaded" not in load_result.error_message
            ):
                raise ValueError(
                    f"Failed to implicitly load OFT adapter {adapter_path}: "
                    f"{load_result.error_message}"
                )

    adapter_id = await tm.peft_registry.acquire(path)
    # Set the kind's request-side id/version fields the scheduler reads.
    if tm.peft_kind == "oft":
        obj.adapter_id = adapter_id
        adapter_version = await tm.peft_registry.get_version_by_id(adapter_id)
        obj.adapter_version = adapter_version
        _propagate_id_to_cached_sub_objs(obj, field="adapter_id", resolved=adapter_id)
        # The version needs the same propagation as the id: batched sub-objects
        # are materialized before this resolver runs, so a version set only on
        # the parent never reaches the tokenized requests built from them.
        _propagate_id_to_cached_sub_objs(
            obj, field="adapter_version", resolved=adapter_version
        )
    else:
        obj.lora_id = adapter_id
        _propagate_id_to_cached_sub_objs(obj, field="lora_id", resolved=adapter_id)


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

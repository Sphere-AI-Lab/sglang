"""Import-light tokenizer-side registry hooks for canonical OFT."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def _mint_ref(name):
    from sglang.srt.oft.oft_registry import OFTRef

    return OFTRef(adapter_name=name, adapter_path=name, pinned=False)


def _request_oft_path(obj):
    return getattr(obj, "adapter_path", None)


def finalize_oft_lease(tm, state):
    """Release a request's canonical OFT registry lease exactly once."""
    if state is None or state.oft_lease_released:
        return
    obj = state.obj
    if getattr(obj, "adapter_path", None) is None:
        return
    adapter_id = getattr(obj, "adapter_id", None)
    if adapter_id is None:
        return
    if getattr(tm, "peft_kind", None) != "oft" or tm.peft_registry is None:
        return
    state.oft_lease_released = True
    asyncio.create_task(tm.peft_registry.release(adapter_id))


def init_tokenizer_oft(tm):
    """Initialize the canonical OFT registry and tokenizer-side caches."""
    tm.peft_kind = tm.server_args.peft_method
    tm.peft_update_lock = asyncio.Lock()
    tm.peft_ref_cache = {}
    tm.peft_registry = None
    tm._logged_peft_base_only_request = False

    if tm.peft_kind != "oft":
        return

    from sglang.srt.oft.oft_registry import OFTRegistry

    initial = list(tm.server_args.peft_paths or [])
    tm.peft_registry = OFTRegistry(initial)
    for ref in initial:
        tm.peft_ref_cache[ref.adapter_name] = ref
    logger.info(
        "event=oft_tokenizer_registry_initialized initial_adapters=%s",
        sorted(tm.peft_ref_cache),
    )


async def register_oft_ref(tm, obj):
    """Register a streamed OFT adapter before dispatch and attach its ID."""
    if obj.adapter_name is None or tm.peft_registry is None:
        return
    name = obj.adapter_name
    if name not in tm.peft_ref_cache:
        ref = _mint_ref(name)
        await tm.peft_registry.register(ref)
        tm.peft_ref_cache[name] = ref
    obj.adapter_id = tm.peft_ref_cache[name].adapter_id


async def bump_oft_version(tm, obj, success):
    """Publish the next OFT version after all worker activations succeed."""
    if not (success and obj.adapter_name is not None and tm.peft_registry is not None):
        return ""
    updated = await tm.peft_registry.bump_version_by_id(obj.adapter_id)
    tm.peft_ref_cache[updated.adapter_name] = updated
    return (
        f" OFT adapter {updated.adapter_name} version updated to "
        f"{updated.adapter_version}."
    )


def _propagate_id_to_cached_sub_objs(obj, *, field, resolved):
    for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
        setattr(
            sub_obj,
            field,
            resolved[i] if isinstance(resolved, list) else resolved,
        )


async def resolve_oft_path(tm, obj):
    """Resolve a request's OFT path to its current adapter ID and version."""
    path = _request_oft_path(obj)
    unique_paths = {path} if isinstance(path, str) else set(path)

    from sglang.srt.oft.io_types import LoadOFTAdapterReqInput

    unregistered = await tm.peft_registry.get_unregistered_adapters(unique_paths)
    for adapter_path in unregistered:
        if adapter_path is None:
            continue
        if adapter_path not in tm.peft_ref_cache:
            raise ValueError(
                f"Got OFT adapter that has never been loaded: {adapter_path}\n"
                f"All loaded adapters: {tm.peft_ref_cache.keys()}."
            )
        ref = tm.peft_ref_cache[adapter_path]
        logger.info("Reloading evicted OFT adapter: %s", adapter_path)
        load_result = await tm.load_oft_adapter(
            LoadOFTAdapterReqInput(
                adapter_name=ref.adapter_name,
                adapter_path=ref.adapter_path,
                pinned=ref.pinned,
            )
        )
        if not load_result.success and "already loaded" not in load_result.error_message:
            raise ValueError(
                f"Failed to implicitly load OFT adapter {adapter_path}: "
                f"{load_result.error_message}"
            )

    adapter_id = await tm.peft_registry.acquire(path)
    obj.adapter_id = adapter_id
    adapter_version = await tm.peft_registry.get_version_by_id(adapter_id)
    obj.adapter_version = adapter_version
    _propagate_id_to_cached_sub_objs(obj, field="adapter_id", resolved=adapter_id)
    _propagate_id_to_cached_sub_objs(
        obj,
        field="adapter_version",
        resolved=adapter_version,
    )


async def maybe_resolve_oft_path(tm, obj):
    """Resolve an OFT request when the canonical registry is active."""
    if tm.peft_registry is None:
        return
    if _request_oft_path(obj):
        await resolve_oft_path(tm, obj)
    elif not tm._logged_peft_base_only_request:
        logger.info(
            "event=oft_request_base_only message='OFT is enabled but this request "
            "has no adapter path; using the base identity slot.'"
        )
        tm._logged_peft_base_only_request = True

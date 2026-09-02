import asyncio
import base64
from types import SimpleNamespace

from sglang.srt.oft.io_types import (
    LoadOFTAdapterFromDistributedReqInput,
    LoadOFTAdapterFromTensorsReqInput,
    OFTUpdateOutput,
    UnloadOFTAdapterReqInput,
)
from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry
from sglang.srt.oft.tokenizer_mixin import OFTTokenizerMixin


def _tensor_request(name="adapter", *, upsert=False, payloads=None):
    return LoadOFTAdapterFromTensorsReqInput(
        adapter_name=name,
        config_dict={"target_modules": ["q_proj"], "r": 8},
        serialized_named_tensors=payloads or [],
        upsert=upsert,
    )


def _distributed_request(name="adapter", *, upsert=False):
    return LoadOFTAdapterFromDistributedReqInput(
        adapter_name=name,
        config_dict={"target_modules": ["q_proj"], "r": 8},
        names=[],
        dtypes=[],
        shapes=[],
        upsert=upsert,
    )


def _handler(*, max_loaded_ofts=None, preloaded=None, responses=None):
    handler = OFTTokenizerMixin()
    handler.server_args = SimpleNamespace(
        peft_method="oft",
        dp_size=1,
        max_loaded_ofts=max_loaded_ofts,
    )
    handler.auto_create_handle_loop = lambda: None
    handler.peft_update_lock = asyncio.Lock()
    handler.peft_registry = OFTRegistry()
    handler.peft_ref_cache = {}
    preloaded = dict(preloaded or {})
    observed_payloads = []

    async def communicate(obj):
        if responses is not None:
            return responses
        if isinstance(obj, UnloadOFTAdapterReqInput):
            return [OFTUpdateOutput(success=True)]
        observed_payloads.append(getattr(obj, "serialized_named_tensors", None))
        loaded = dict(preloaded)
        loaded[obj.adapter_name] = obj.adapter_id
        return [OFTUpdateOutput(success=True, loaded_adapters=loaded)]

    handler.update_oft_adapter_communicator = communicate
    handler.observed_payloads = observed_payloads
    return handler


def test_tensor_load_normalizes_payload_before_registering():
    encoded = base64.b64encode(b"rank payload").decode()
    handler = _handler()
    request = _tensor_request(payloads=[encoded])

    result = asyncio.run(handler.load_oft_adapter_from_tensors(request))

    assert result.success
    assert handler.observed_payloads == [[b"rank payload"]]
    assert handler.peft_registry.get_all_adapters()["adapter"].adapter_id == request.adapter_id


def test_distributed_load_registers_adapter():
    handler = _handler()
    request = _distributed_request()

    result = asyncio.run(handler.load_oft_adapter_from_distributed(request))

    assert result.success
    registered = handler.peft_registry.get_all_adapters()["adapter"]
    assert registered.adapter_id == request.adapter_id
    assert registered.adapter_path == "__distributed__"
    assert registered.reloadable is False


def test_tensor_upsert_reuses_id_and_bumps_version():
    handler = _handler()
    existing = OFTRef(
        adapter_name="adapter",
        adapter_path="__tensor__",
        reloadable=False,
        adapter_version=4,
    )
    asyncio.run(handler.peft_registry.register(existing))
    handler.peft_ref_cache["adapter"] = existing
    request = _tensor_request(upsert=True)

    result = asyncio.run(handler.load_oft_adapter_from_tensors(request))

    assert result.success
    updated = handler.peft_registry.get_all_adapters()["adapter"]
    assert request.adapter_id == existing.adapter_id
    assert updated.adapter_id == existing.adapter_id
    assert updated.adapter_version == 5


def test_tensor_load_reports_any_rank_failure_without_registering():
    handler = _handler(
        responses=[
            OFTUpdateOutput(success=True, loaded_adapters={"adapter": "id"}),
            OFTUpdateOutput(success=False, error_message="rank 1 failed"),
        ]
    )

    result = asyncio.run(
        handler.load_oft_adapter_from_tensors(_tensor_request())
    )

    assert not result.success
    assert result.error_message == "rank 1 failed"
    assert handler.peft_registry.num_registered_ofts == 0


def test_explicit_unload_removes_reload_catalog_entry():
    handler = _handler()
    ref = OFTRef(adapter_name="adapter", adapter_path="/disk/adapter")
    asyncio.run(handler.peft_registry.register(ref))
    handler.peft_ref_cache["adapter"] = ref

    result = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )

    assert result.success
    assert handler.peft_registry.num_registered_ofts == 0
    assert "adapter" not in handler.peft_ref_cache


def test_unload_waits_for_active_leases_before_backend_removal():
    handler = _handler()
    ref = OFTRef(adapter_name="adapter", adapter_path="/disk/adapter")
    asyncio.run(handler.peft_registry.register(ref))
    handler.peft_ref_cache["adapter"] = ref
    call_order = []
    real_wait_for_unload = handler.peft_registry.wait_for_unload

    async def tracking_wait_for_unload(adapter_id):
        call_order.append("wait_for_unload")
        return await real_wait_for_unload(adapter_id)

    async def tracking_communicator(obj):
        call_order.append("communicator")
        return [OFTUpdateOutput(success=True)]

    handler.peft_registry.wait_for_unload = tracking_wait_for_unload
    handler.update_oft_adapter_communicator = tracking_communicator

    result = asyncio.run(
        handler.unload_oft_adapter(
            UnloadOFTAdapterReqInput(adapter_name="adapter")
        )
    )

    assert result.success
    assert call_order == ["wait_for_unload", "communicator"]


def test_unload_reports_any_rank_failure():
    handler = _handler(
        responses=[
            OFTUpdateOutput(success=True),
            OFTUpdateOutput(success=False, error_message="rank 1 failed"),
        ]
    )
    ref = OFTRef(adapter_name="adapter", adapter_path="/disk/adapter")
    asyncio.run(handler.peft_registry.register(ref))
    handler.peft_ref_cache["adapter"] = ref

    result = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )

    assert not result.success
    assert result.error_message == "rank 1 failed"
    assert "adapter" in handler.peft_ref_cache


def test_tensor_load_evicts_old_disk_adapter_at_registry_limit():
    old_ref = OFTRef(adapter_name="old", adapter_path="/disk/old")
    handler = _handler(
        max_loaded_ofts=1,
        preloaded={"old": old_ref.adapter_id},
    )
    asyncio.run(handler.peft_registry.register(old_ref))
    handler.peft_ref_cache["old"] = old_ref

    result = asyncio.run(
        handler.load_oft_adapter_from_tensors(_tensor_request("new"))
    )

    assert result.success
    assert set(handler.peft_registry.get_all_adapters()) == {"new"}
    assert set(result.loaded_adapters) == {"new"}
    assert handler.peft_ref_cache["old"] is old_ref

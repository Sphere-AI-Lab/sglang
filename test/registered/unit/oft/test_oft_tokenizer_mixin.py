import asyncio
import base64
import contextlib
from types import SimpleNamespace

import pytest

from sglang.srt.oft.io_types import (
    LoadOFTAdapterFromDistributedReqInput,
    LoadOFTAdapterFromTensorsReqInput,
    LoadOFTAdapterReqInput,
    OFTUpdateOutput,
    UnloadOFTAdapterReqInput,
)
from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry
from sglang.srt.oft.tokenizer_mixin import OFTTokenizerMixin
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    handler.model_update_lock = RWLock()
    handler.is_pause_cond = asyncio.Condition()
    handler.is_pause = False
    handler.peft_registry = OFTRegistry()
    handler.peft_ref_cache = {}
    handler.failed_oft_activations = {}
    handler.failed_oft_unloads = {}
    handler.pending_oft_stage = None
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
    assert (
        handler.peft_registry.get_all_adapters()["adapter"].adapter_id
        == request.adapter_id
    )


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
    assert request.adapter_version == 5
    assert updated.adapter_id == existing.adapter_id
    assert updated.adapter_version == 5


@pytest.mark.parametrize("route", ("tensors", "distributed"))
def test_wire_upsert_preserves_old_state_while_waiting_for_inference(route):
    async def scenario():
        handler = _handler()
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path="__tensor__",
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing
        leased_id, leased_version = await handler.peft_registry.acquire_with_version(
            "adapter"
        )
        communicator_started = asyncio.Event()

        async def communicate(obj):
            communicator_started.set()
            return [OFTUpdateOutput(success=True)]

        handler.update_oft_adapter_communicator = communicate
        waiting_for_drain = asyncio.Event()
        acquire_writer = handler.model_update_lock.acquire_writer
        wait_for_unload = handler.peft_registry.wait_for_unload

        async def tracked_writer():
            waiting_for_drain.set()
            await acquire_writer()

        async def tracked_unload(adapter_id):
            waiting_for_drain.set()
            await wait_for_unload(adapter_id)

        handler.model_update_lock.acquire_writer = tracked_writer
        handler.peft_registry.wait_for_unload = tracked_unload
        request = _tensor_request if route == "tensors" else _distributed_request
        load = getattr(handler, f"load_oft_adapter_from_{route}")
        async with handler.model_update_lock.reader_lock:
            update = asyncio.create_task(load(request(upsert=True)))
            try:
                await asyncio.wait_for(waiting_for_drain.wait(), 1)
                assert (
                    handler.peft_registry.get_all_adapters().get("adapter") is existing
                )
                assert handler.peft_ref_cache.get("adapter") is existing
                assert not communicator_started.is_set()
                await asyncio.wait_for(handler.peft_update_lock.acquire(), 1)
                handler.peft_update_lock.release()
            finally:
                await handler.peft_registry.release(leased_id)
        result = await update

        assert result.success
        assert communicator_started.is_set()
        assert leased_version == 4
        updated = handler.peft_registry.get_all_adapters()["adapter"]
        assert updated.adapter_id == existing.adapter_id
        assert updated.adapter_version == 5

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ("tensors", "distributed"))
def test_wire_upsert_rejects_paused_generation_without_mutation(route):
    async def scenario():
        handler = _handler()
        existing = OFTRef(adapter_name="adapter", adapter_version=4)
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing
        handler.is_pause = True
        request = _tensor_request if route == "tensors" else _distributed_request
        load = getattr(handler, f"load_oft_adapter_from_{route}")

        result = await load(request(upsert=True))

        assert not result.success
        assert "paused" in result.error_message
        assert handler.peft_registry.get_all_adapters()["adapter"] is existing
        assert handler.peft_ref_cache["adapter"] is existing
        assert handler.observed_payloads == []

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ("tensors", "distributed"))
@pytest.mark.parametrize("outcome", ("success", "preserved", "unknown"))
def test_cancelled_wire_upsert_holds_admission_through_publication(route, outcome):
    async def scenario():
        handler = _handler()
        existing = OFTRef(adapter_name="adapter", adapter_version=4)
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing
        dispatched = asyncio.Event()
        finish_backend = asyncio.Event()
        publishing = asyncio.Event()
        finish_publication = asyncio.Event()
        reader_entered = asyncio.Event()

        async def communicate(obj):
            dispatched.set()
            await finish_backend.wait()
            return [
                OFTUpdateOutput(
                    success=outcome == "success",
                    previous_adapter_preserved=outcome == "preserved",
                )
            ]

        finish = handler._finish_oft_wire_load

        async def gated_publication(*args):
            publishing.set()
            await finish_publication.wait()
            return await finish(*args)

        async def inference():
            async with handler.model_update_lock.reader_lock:
                reader_entered.set()

        handler.update_oft_adapter_communicator = communicate
        handler._finish_oft_wire_load = gated_publication
        request = _tensor_request if route == "tensors" else _distributed_request
        update = asyncio.create_task(
            getattr(handler, f"load_oft_adapter_from_{route}")(request(upsert=True))
        )
        reader = None
        try:
            await asyncio.wait_for(dispatched.wait(), 1)
            update.cancel()
            await asyncio.sleep(0)
            update.cancel()
            await asyncio.sleep(0)
            reader = asyncio.create_task(inference())
            await asyncio.sleep(0)
            assert not reader_entered.is_set()
            assert handler.peft_update_lock.locked()
            finish_backend.set()
            await asyncio.wait_for(publishing.wait(), 1)
            update.cancel()
            await asyncio.sleep(0)
            assert not reader_entered.is_set()
            assert handler.peft_update_lock.locked()
            finish_publication.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(update, 1)
            await asyncio.wait_for(reader, 1)

            refs = handler.peft_registry.get_all_adapters()
            if outcome == "success":
                assert refs["adapter"].adapter_id == existing.adapter_id
                assert refs["adapter"].adapter_version == 5
                assert handler.peft_ref_cache["adapter"] is refs["adapter"]
            elif outcome == "preserved":
                assert refs["adapter"] is existing
                assert handler.peft_ref_cache["adapter"] is existing
            else:
                assert "adapter" not in refs
                assert "adapter" not in handler.peft_ref_cache
                assert "adapter" in handler.failed_oft_activations
        finally:
            finish_backend.set()
            finish_publication.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(update, 1)
            if reader is not None:
                await asyncio.wait_for(reader, 1)

    asyncio.run(scenario())


def test_distributed_partial_upsert_failure_quarantines_adapter():
    async def scenario():
        handler = _handler(
            responses=[
                OFTUpdateOutput(success=True),
                OFTUpdateOutput(success=False, error_message="rank 1 failed"),
            ]
        )
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path="__distributed__",
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing

        result = await handler.load_oft_adapter_from_distributed(
            _distributed_request(upsert=True)
        )

        assert not result.success
        assert "rank 1 failed" in result.error_message
        assert "adapter" not in handler.peft_registry.get_all_adapters()
        assert "adapter" not in handler.peft_ref_cache

        retry = await handler.load_oft_adapter_from_distributed(
            _distributed_request(upsert=True)
        )
        assert not retry.success
        assert "quarantined" in retry.error_message

    asyncio.run(scenario())


def test_unanimous_failed_upsert_restores_preserved_adapter():
    async def scenario():
        old_path = "__tensor__"
        handler = _handler(
            responses=[
                OFTUpdateOutput(
                    success=False,
                    error_message="rank 0 rejected payload",
                    loaded_adapters={"adapter": old_path},
                    previous_adapter_preserved=True,
                ),
                OFTUpdateOutput(
                    success=False,
                    error_message="rank 1 rejected payload",
                    loaded_adapters={"adapter": old_path},
                    previous_adapter_preserved=True,
                ),
            ]
        )
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path=old_path,
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing

        result = await handler.load_oft_adapter_from_tensors(
            _tensor_request(upsert=True)
        )

        assert not result.success
        assert "rejected payload" in result.error_message
        assert "adapter" not in handler.failed_oft_activations
        restored = handler.peft_registry.get_all_adapters()["adapter"]
        assert restored is existing
        assert handler.peft_ref_cache["adapter"] is existing
        leased_id, leased_version = await handler.peft_registry.acquire_with_version(
            "adapter"
        )
        assert leased_id == existing.adapter_id
        assert leased_version == 4
        await handler.peft_registry.release(leased_id)

    asyncio.run(scenario())


def test_failed_upsert_quarantines_when_workers_do_not_confirm_preservation():
    async def scenario():
        old_path = "__tensor__"
        handler = _handler(
            responses=[
                OFTUpdateOutput(
                    success=False,
                    error_message="rollback failed; worker restart required",
                    loaded_adapters={"adapter": old_path},
                    previous_adapter_preserved=False,
                )
            ]
        )
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path=old_path,
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing

        result = await handler.load_oft_adapter_from_tensors(
            _tensor_request(upsert=True)
        )

        assert not result.success
        assert "quarantined" in result.error_message
        assert "adapter" in handler.failed_oft_activations
        assert "adapter" not in handler.peft_registry.get_all_adapters()
        assert "adapter" not in handler.peft_ref_cache

    asyncio.run(scenario())


def test_upsert_communication_failure_quarantines_unpublished_adapter():
    async def scenario():
        handler = _handler()
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path="__tensor__",
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing

        async def fail_communication(obj):
            raise RuntimeError("worker channel failed")

        handler.update_oft_adapter_communicator = fail_communication
        result = await handler.load_oft_adapter_from_tensors(
            _tensor_request(upsert=True)
        )

        assert not result.success
        assert "worker channel failed" in result.error_message
        assert "quarantined" in result.error_message
        assert "adapter" in handler.failed_oft_activations
        assert "adapter" not in handler.peft_registry.get_all_adapters()
        assert "adapter" not in handler.peft_ref_cache

    asyncio.run(scenario())


def test_cancelled_tensor_upsert_finishes_before_next_update_is_dispatched():
    from sglang.srt.managers.communicator import FanOutCommunicator

    async def scenario():
        handler = _handler()
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path="__tensor__",
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing

        sends = []
        first_started = asyncio.Event()

        def send(obj):
            sends.append(obj)
            first_started.set()

        communicator = FanOutCommunicator(send, fan_out=1)
        handler.update_oft_adapter_communicator = communicator
        first = asyncio.create_task(
            handler.load_oft_adapter_from_tensors(_tensor_request(upsert=True))
        )
        second = None
        try:
            await first_started.wait()
            first.cancel()
            await asyncio.sleep(0)
            assert handler.peft_update_lock.locked()

            second = asyncio.create_task(
                handler.load_oft_adapter_from_tensors(_tensor_request(upsert=True))
            )
            await asyncio.sleep(0)
            assert len(sends) == 1

            communicator.handle_recv(
                OFTUpdateOutput(
                    success=False,
                    error_message="rank 0 rejected payload",
                    loaded_adapters={"adapter": "__tensor__"},
                    previous_adapter_preserved=True,
                )
            )
            with contextlib.suppress(asyncio.CancelledError):
                await first

            for _ in range(100):
                if len(sends) == 2:
                    break
                await asyncio.sleep(0)
            assert len(sends) == 2
            communicator.handle_recv(OFTUpdateOutput(success=True))
            result = await second

            assert result.success
            registered = handler.peft_registry.get_all_adapters()["adapter"]
            assert registered.adapter_id == existing.adapter_id
            assert registered.adapter_version == 5
        finally:
            for task in (first, second):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ("tensors", "distributed"))
def test_cancelled_wire_upsert_waiting_for_inference_leaves_state_untouched(route):
    async def scenario():
        handler = _handler()
        existing = OFTRef(
            adapter_name="adapter",
            adapter_path="__tensor__",
            reloadable=False,
            adapter_version=4,
        )
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing
        leased_id, _ = await handler.peft_registry.acquire_with_version("adapter")

        waiting_for_writer = asyncio.Event()
        acquire_writer = handler.model_update_lock.acquire_writer

        async def tracked_writer():
            waiting_for_writer.set()
            await acquire_writer()

        handler.model_update_lock.acquire_writer = tracked_writer
        request = _tensor_request if route == "tensors" else _distributed_request
        load = getattr(handler, f"load_oft_adapter_from_{route}")
        async with handler.model_update_lock.reader_lock:
            update = asyncio.create_task(load(request(upsert=True)))
            await asyncio.wait_for(waiting_for_writer.wait(), 1)
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(update, 1)
            assert handler.peft_registry.get_all_adapters()["adapter"] is existing
            assert handler.peft_ref_cache["adapter"] is existing
            assert handler.observed_payloads == []
            assert not handler.peft_update_lock.locked()
            await handler.peft_registry.release(leased_id)

        assert (await load(request(upsert=True))).success
        assert handler.peft_registry.get_all_adapters()["adapter"].adapter_version == 5

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ("tensors", "distributed"))
@pytest.mark.parametrize("upsert", (False, True))
def test_fresh_wire_load_does_not_drain_unrelated_inference(route, upsert):
    async def scenario():
        handler = _handler()
        request = _tensor_request if route == "tensors" else _distributed_request
        load = getattr(handler, f"load_oft_adapter_from_{route}")
        async with handler.model_update_lock.reader_lock:
            result = await asyncio.wait_for(load(request(upsert=upsert)), 1)
        assert result.success
        assert handler.peft_registry.get_all_adapters()["adapter"].adapter_version == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ("tensors", "distributed"))
def test_removed_wire_adapter_retries_as_fresh_without_model_writer(route):
    async def scenario():
        handler = _handler()
        existing = OFTRef(adapter_name="adapter", adapter_version=4)
        await handler.peft_registry.register(existing)
        handler.peft_ref_cache["adapter"] = existing
        waiting_for_writer = asyncio.Event()
        acquire_writer = handler.model_update_lock.acquire_writer

        async def tracked_writer():
            waiting_for_writer.set()
            await acquire_writer()

        handler.model_update_lock.acquire_writer = tracked_writer
        communicate = handler.update_oft_adapter_communicator

        async def require_fresh_dispatch(obj):
            if not isinstance(obj, UnloadOFTAdapterReqInput):
                assert not await handler.model_update_lock.is_locked()
                assert obj.adapter_id != existing.adapter_id
                assert obj.adapter_version == 1
            return await communicate(obj)

        handler.update_oft_adapter_communicator = require_fresh_dispatch
        request = _tensor_request if route == "tensors" else _distributed_request
        async with handler.model_update_lock.reader_lock:
            update = asyncio.create_task(
                getattr(handler, f"load_oft_adapter_from_{route}")(request(upsert=True))
            )
            await asyncio.wait_for(waiting_for_writer.wait(), 1)
            result = await asyncio.wait_for(
                handler.unload_oft_adapter(
                    UnloadOFTAdapterReqInput(adapter_name="adapter")
                ),
                1,
            )
            assert result.success

        assert (await asyncio.wait_for(update, 1)).success

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ("tensors", "distributed"))
def test_concurrent_wire_load_after_preflight_retries_through_reader_drain(route):
    async def scenario():
        handler = _handler()
        existing = OFTRef(adapter_name="adapter", adapter_version=4)
        preflight_complete = asyncio.Event()
        waiting_for_writer = asyncio.Event()
        resolve = handler.peft_registry.resolve_or_reuse
        acquire_writer = handler.model_update_lock.acquire_writer

        async def tracked_resolve(*args, **kwargs):
            result = await resolve(*args, **kwargs)
            preflight_complete.set()
            return result

        async def tracked_writer():
            waiting_for_writer.set()
            await acquire_writer()

        handler.peft_registry.resolve_or_reuse = tracked_resolve
        handler.model_update_lock.acquire_writer = tracked_writer
        request = _tensor_request if route == "tensors" else _distributed_request
        async with handler.model_update_lock.reader_lock:
            async with handler.peft_update_lock:
                update = asyncio.create_task(
                    getattr(handler, f"load_oft_adapter_from_{route}")(
                        request(upsert=True)
                    )
                )
                await asyncio.wait_for(preflight_complete.wait(), 1)
                await handler.peft_registry.register(existing)
                handler.peft_ref_cache["adapter"] = existing
            await asyncio.wait_for(waiting_for_writer.wait(), 1)
            assert handler.peft_registry.get_all_adapters()["adapter"] is existing
            assert handler.peft_ref_cache["adapter"] is existing
            assert handler.observed_payloads == []
            assert not handler.peft_update_lock.locked()

        assert (await asyncio.wait_for(update, 1)).success
        assert handler.peft_registry.get_all_adapters()["adapter"].adapter_version == 5

    asyncio.run(scenario())


def test_tensor_load_reports_any_rank_failure_without_registering():
    handler = _handler(
        responses=[
            OFTUpdateOutput(success=True, loaded_adapters={"adapter": "id"}),
            OFTUpdateOutput(success=False, error_message="rank 1 failed"),
        ]
    )

    result = asyncio.run(handler.load_oft_adapter_from_tensors(_tensor_request()))

    assert not result.success
    assert "rank 1 failed" in result.error_message
    assert "quarantined" in result.error_message
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


def test_unknown_user_unload_still_fails_before_worker_dispatch():
    from unittest.mock import AsyncMock

    handler = _handler()
    handler.update_oft_adapter_communicator = AsyncMock()
    result = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="unknown"))
    )

    assert not result.success
    handler.update_oft_adapter_communicator.assert_not_awaited()


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
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
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


def test_failed_unload_is_quarantined_and_retryable_with_same_id():
    handler = _handler()
    ref = OFTRef(
        adapter_id="id-a",
        adapter_name="adapter",
        adapter_path="__distributed__",
        reloadable=False,
    )
    asyncio.run(handler.peft_registry.register(ref))
    handler.peft_ref_cache["adapter"] = ref
    responses = iter(
        [
            [
                OFTUpdateOutput(success=True),
                OFTUpdateOutput(success=False, error_message="rank 1 failed"),
            ],
            [OFTUpdateOutput(success=True), OFTUpdateOutput(success=True)],
        ]
    )
    dispatched_ids = []

    async def communicate(obj):
        dispatched_ids.append(obj.adapter_id)
        return next(responses)

    handler.update_oft_adapter_communicator = communicate

    first = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )

    assert not first.success
    assert "adapter" not in handler.peft_registry.get_all_adapters()
    assert handler.failed_oft_unloads["adapter"] is ref
    assert "adapter" in handler.peft_ref_cache
    blocked_load = asyncio.run(
        handler.load_oft_adapter(
            LoadOFTAdapterReqInput(adapter_name="adapter", adapter_path="/disk/adapter")
        )
    )
    assert not blocked_load.success
    assert "failed unload" in blocked_load.error_message

    retry = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )

    assert retry.success
    assert dispatched_ids == ["id-a", "id-a"]
    assert "adapter" not in handler.failed_oft_unloads
    assert "adapter" not in handler.peft_ref_cache


def test_successful_unload_invalidates_same_adapter_pending_stage():
    handler = _handler()
    ref = OFTRef(adapter_name="adapter", adapter_path="/disk/adapter")
    asyncio.run(handler.peft_registry.register(ref))
    handler.peft_ref_cache["adapter"] = ref
    handler.pending_oft_stage = OFTRef(
        adapter_id=ref.adapter_id,
        adapter_name="adapter",
        adapter_path="__distributed__",
        adapter_version=2,
    )

    result = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )

    assert result.success
    assert handler.pending_oft_stage is None


def test_failed_unload_still_invalidates_same_adapter_pending_stage():
    handler = _handler(
        responses=[OFTUpdateOutput(success=False, error_message="rank 1 failed")]
    )
    ref = OFTRef(adapter_name="adapter", adapter_path="/disk/adapter")
    asyncio.run(handler.peft_registry.register(ref))
    handler.peft_ref_cache["adapter"] = ref
    handler.pending_oft_stage = OFTRef(
        adapter_id=ref.adapter_id,
        adapter_name="adapter",
        adapter_path="__distributed__",
        adapter_version=2,
    )

    result = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )

    assert not result.success
    assert handler.pending_oft_stage is None
    assert "adapter" in handler.failed_oft_activations

    restage = asyncio.run(handler.load_oft_adapter_from_tensors(_tensor_request()))
    assert not restage.success
    assert "failed unload" in restage.error_message


def test_same_name_load_is_rejected_until_pending_stage_is_unloaded():
    handler = _handler()
    handler.pending_oft_stage = OFTRef(
        adapter_name="adapter",
        adapter_path="__distributed__",
        adapter_version=2,
    )

    load_result = asyncio.run(
        handler.load_oft_adapter(
            LoadOFTAdapterReqInput(adapter_name="adapter", adapter_path="/disk/adapter")
        )
    )

    assert not load_result.success
    assert "staged version 2 is pending" in load_result.error_message
    assert handler.pending_oft_stage is not None

    unload_result = asyncio.run(
        handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
    )
    assert unload_result.success
    assert handler.pending_oft_stage is None

    retry_result = asyncio.run(
        handler.load_oft_adapter(
            LoadOFTAdapterReqInput(adapter_name="adapter", adapter_path="/disk/adapter")
        )
    )
    assert retry_result.success


def test_tensor_load_evicts_old_disk_adapter_at_registry_limit():
    old_ref = OFTRef(adapter_name="old", adapter_path="/disk/old")
    handler = _handler(
        max_loaded_ofts=1,
        preloaded={"old": old_ref.adapter_id},
    )
    asyncio.run(handler.peft_registry.register(old_ref))
    handler.peft_ref_cache["old"] = old_ref

    result = asyncio.run(handler.load_oft_adapter_from_tensors(_tensor_request("new")))

    assert result.success
    assert set(handler.peft_registry.get_all_adapters()) == {"new"}
    assert set(result.loaded_adapters) == {"new"}
    assert handler.peft_ref_cache["old"] is old_ref


def test_path_load_evicts_old_disk_adapter_at_registry_limit():
    old_ref = OFTRef(adapter_name="old", adapter_path="/disk/old")
    handler = _handler(
        max_loaded_ofts=1,
        preloaded={"old": old_ref.adapter_id},
    )
    asyncio.run(handler.peft_registry.register(old_ref))
    handler.peft_ref_cache["old"] = old_ref

    result = asyncio.run(
        handler.load_oft_adapter(
            LoadOFTAdapterReqInput(
                adapter_name="new",
                adapter_path="/disk/new",
            )
        )
    )

    assert result.success
    assert set(handler.peft_registry.get_all_adapters()) == {"new"}
    assert set(result.loaded_adapters) == {"new"}
    assert handler.peft_ref_cache["old"] is old_ref


def test_cancelled_path_load_finishes_registry_limit_eviction():
    from sglang.srt.managers.communicator import FanOutCommunicator

    async def scenario():
        old_ref = OFTRef(adapter_name="old", adapter_path="/disk/old")
        handler = _handler(max_loaded_ofts=1)
        await handler.peft_registry.register(old_ref)
        handler.peft_ref_cache["old"] = old_ref
        sends = []
        communicator = FanOutCommunicator(sends.append, fan_out=1)
        handler.update_oft_adapter_communicator = communicator

        loading = asyncio.create_task(
            handler.load_oft_adapter(
                LoadOFTAdapterReqInput(
                    adapter_name="new",
                    adapter_path="/disk/new",
                )
            )
        )
        try:
            for _ in range(100):
                if len(sends) == 1:
                    break
                await asyncio.sleep(0)
            assert len(sends) == 1
            new_id = sends[0].adapter_id
            communicator.handle_recv(
                OFTUpdateOutput(
                    success=True,
                    loaded_adapters={"old": old_ref.adapter_id, "new": new_id},
                )
            )

            for _ in range(100):
                if len(sends) == 2:
                    break
                await asyncio.sleep(0)
            assert len(sends) == 2
            assert isinstance(sends[1], UnloadOFTAdapterReqInput)

            loading.cancel()
            await asyncio.sleep(0)
            assert handler.peft_update_lock.locked()
            communicator.handle_recv(
                OFTUpdateOutput(
                    success=True,
                    loaded_adapters={"new": new_id},
                )
            )
            try:
                await loading
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("caller cancellation was not propagated")

            assert not handler.peft_update_lock.locked()
            assert set(handler.peft_registry.get_all_adapters()) == {"new"}
            assert handler.peft_ref_cache["old"] is old_ref
        finally:
            if not loading.done():
                loading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await loading

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("cancel_at", ["lease", "worker"])
@pytest.mark.parametrize("worker_success", [True, False])
def test_cancelled_unload_finishes_cleanup_or_preserves_retry(
    cancel_at, worker_success
):
    async def scenario():
        handler = _handler()
        ref = OFTRef(adapter_name="adapter", adapter_path="/disk/adapter")
        await handler.peft_registry.register(ref)
        handler.peft_ref_cache["adapter"] = ref
        await handler.peft_registry.acquire_with_version("adapter")
        waiting = asyncio.Event()
        dispatched = asyncio.Event()
        finish_worker = asyncio.Event()
        original_wait = handler.peft_registry.wait_for_unload
        calls = []

        async def wait_for_unload(uid):
            waiting.set()
            await original_wait(uid)

        async def communicate(obj):
            calls.append(obj.adapter_id)
            dispatched.set()
            await finish_worker.wait()
            return [
                OFTUpdateOutput(
                    success=worker_success,
                    error_message=None if worker_success else "worker failed",
                )
            ]

        handler.peft_registry.wait_for_unload = wait_for_unload
        handler.update_oft_adapter_communicator = communicate
        task = asyncio.create_task(
            handler.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="adapter"))
        )
        await asyncio.wait_for(waiting.wait(), 1)
        if cancel_at == "worker":
            await handler.peft_registry.release(ref.adapter_id)
            await asyncio.wait_for(dispatched.wait(), 1)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert handler.peft_update_lock.locked()
        if cancel_at == "lease":
            assert not calls
            await handler.peft_registry.release(ref.adapter_id)
        await asyncio.wait_for(dispatched.wait(), 1)
        finish_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert calls == [ref.adapter_id]
        assert not handler.peft_update_lock.locked()
        assert handler.peft_registry.num_registered_ofts == 0
        if worker_success:
            assert "adapter" not in handler.peft_ref_cache
            assert "adapter" not in handler.failed_oft_unloads
        else:
            assert handler.failed_oft_unloads["adapter"] is ref
            worker_success_result = OFTUpdateOutput(success=True)

            async def retry(obj):
                assert obj.adapter_id == ref.adapter_id
                return [worker_success_result]

            handler.update_oft_adapter_communicator = retry
            assert (
                await handler.unload_oft_adapter(
                    UnloadOFTAdapterReqInput(adapter_name="adapter")
                )
            ).success
            assert "adapter" not in handler.peft_ref_cache
            assert "adapter" not in handler.failed_oft_unloads

    asyncio.run(scenario())


@pytest.mark.parametrize("mechanism", ["path", "tensor", "distributed"])
def test_aggregated_partial_fresh_load_quarantines_name(mechanism):
    async def scenario():
        handler = _handler(
            responses=[
                OFTUpdateOutput(
                    success=False,
                    error_message="TP rank 1 failed",
                    inconsistent_update=True,
                )
            ]
        )
        if mechanism == "path":
            request = LoadOFTAdapterReqInput(
                adapter_name="adapter", adapter_path="/fixture"
            )
            load = handler.load_oft_adapter
        elif mechanism == "tensor":
            request, load = _tensor_request(), handler.load_oft_adapter_from_tensors
        else:
            request, load = (
                _distributed_request(),
                handler.load_oft_adapter_from_distributed,
            )
        result = await load(request)
        assert not result.success
        assert "quarantined" in result.error_message
        assert handler.peft_registry.num_registered_ofts == 0
        retry = await load(request)
        assert not retry.success
        assert "quarantined" in retry.error_message

    asyncio.run(scenario())

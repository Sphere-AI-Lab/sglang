"""A cancelled client must not outlive its scheduler abort protection."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("adapter_kind", ["lora", "oft"])
@pytest.mark.parametrize("partial_batch", [False, True])
def test_cancelled_dispatch_retains_protection_until_scheduler_ack(
    adapter_kind, partial_batch
):
    async def scenario():
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.server_args = SimpleNamespace(
            language_only=False, peft_method=adapter_kind, tokenizer_worker_num=1
        )
        tm.enable_lora = adapter_kind == "lora"
        tm.peft_kind = adapter_kind
        tm.enable_metrics = False
        tm.tokenizer = None
        tm.model_update_lock = RWLock()
        tm.is_pause_cond = asyncio.Condition()
        tm.is_pause = False
        tm.rid_to_state = {}
        tm.auto_create_handle_loop = Mock()
        tm._set_default_priority = Mock()
        tm.request_logger = Mock()
        tm.config_value = lambda name: "1"
        registry = SimpleNamespace(release=AsyncMock())
        tm.lora_registry = tm.peft_registry = registry
        dispatch = asyncio.Event()
        abort = asyncio.Event()
        messages = []
        rids = ["sent", "not-sent"] if partial_batch else ["sent"]
        obj = SimpleNamespace(
            rid=rids if partial_batch else rids[0],
            is_single=not partial_batch,
            return_prompt_token_ids=False,
            normalize_batch_and_arguments=lambda: None,
        )

        def initialize(*args):
            for rid in rids:
                request = SimpleNamespace(
                    rid=rid,
                    stream=False,
                    return_logprob=False,
                    lora_path="policy" if adapter_kind == "lora" else None,
                    lora_id=rid if adapter_kind == "lora" else None,
                    adapter_path="policy" if adapter_kind == "oft" else None,
                    adapter_id=rid if adapter_kind == "oft" else None,
                )
                tm.rid_to_state[rid] = ReqState(
                    [], False, asyncio.Event(), request, Mock()
                )

        def send(request):
            tm.rid_to_state["sent"].dispatched = True
            messages.append("generate")
            dispatch.set()

        async def wait(*args):
            await asyncio.Event().wait()
            yield None

        async def partial(*args):
            send(None)
            async for item in wait():
                yield item

        def ipc(request):
            messages.append(request)
            abort.set()

        tm._init_req_state = initialize
        tm._validate_and_resolve_lora = AsyncMock()
        tm._tokenize_one_request = AsyncMock(return_value=obj)
        tm._send_one_request = send
        tm._wait_one_response = wait
        tm._handle_batch_request = partial
        tm._dispatch_to_scheduler = ipc
        generation = tm.generate_request(obj)
        task = asyncio.create_task(anext(generation))
        await dispatch.wait()
        state = tm.rid_to_state["sent"]
        task.cancel()
        try:
            await asyncio.wait_for(abort.wait(), 1)
            assert isinstance(messages[-1], AbortReq)
            assert messages[-1].rid == "sent"
            assert tm.rid_to_state["sent"] is state
            assert await tm.model_update_lock.is_locked()
            assert "sent" not in [
                call.args[0] for call in registry.release.await_args_list
            ]
            if partial_batch:
                assert "not-sent" not in tm.rid_to_state
            # Repeated cancellation must not interrupt the scheduler drain.
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert await tm.model_update_lock.is_locked()
        finally:
            tm._handle_abort_req(AbortReq(rid="sent"))
            try:
                await asyncio.wait_for(task, 1)
            except asyncio.CancelledError:
                pass
            await generation.aclose()
        await asyncio.sleep(0)
        assert not tm.rid_to_state
        assert not await tm.model_update_lock.is_locked()
        released = [call.args[0] for call in registry.release.await_args_list]
        assert sorted(released) == sorted(rids)

    asyncio.run(scenario())


def test_batched_dispatch_records_each_scheduler_owned_request():
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.rid_to_state = {}
    tm._abort_instead_of_dispatch = lambda obj: False
    tm.cuda_vmm_feature_transport = Mock()
    tm._dispatch_to_scheduler = Mock()
    requests = []
    for rid in ("a", "b"):
        obj = SimpleNamespace(
            rid=rid, mm_inputs=None, time_stats=Mock(), wrap_pickle_fields=Mock()
        )
        requests.append(obj)
        tm.rid_to_state[rid] = ReqState([], False, asyncio.Event(), obj, Mock())

    tm._send_batch_request(requests)

    assert all(getattr(tm.rid_to_state[rid], "dispatched", False) for rid in ("a", "b"))


@pytest.mark.parametrize("cancel_during_prefix", [True, False])
def test_parallel_sampling_tracks_generated_requests_for_cancellation(
    cancel_during_prefix,
):
    async def scenario():
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.rid_to_state = {}
        tm._finalize_lora_lease = Mock()
        tm._finalize_oft_lease = Mock()
        ready = asyncio.Event()
        next_id = iter(["prefix", "choice-0", "choice-1"])

        class Request:
            rid = "parent"
            batch_size = 1
            parallel_sample_num = 2
            stream = False
            is_single = False
            return_prompt_token_ids = False

            def __getitem__(self, index):
                return self

            def regenerate_rid(self):
                self.rid = next(next_id)
                return self.rid

        obj = Request()

        def initialize(request):
            tm.rid_to_state[request.rid] = ReqState(
                [], False, asyncio.Event(), request, Mock()
            )

        initialize(obj)
        tm._init_req_state = initialize
        tm._tokenize_one_request = AsyncMock(
            return_value=SimpleNamespace(
                sampling_params=SimpleNamespace(max_new_tokens=2), mm_inputs=None
            )
        )

        def send(request):
            tm.rid_to_state[request.rid].dispatched = True

        tm._send_one_request = send

        async def wait(request, *args):
            if request.rid == "prefix" and not cancel_during_prefix:
                tm.rid_to_state.pop(request.rid)
                yield None
            else:
                ready.set()
                await asyncio.Event().wait()
                yield None

        tm._wait_one_response = wait
        child_rids = []
        stream = tm._handle_batch_request(obj, None, child_rids)
        task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.aclose()
        expected = ["prefix"] if cancel_during_prefix else ["choice-0", "choice-1"]
        aborted = []

        def abort(rid):
            aborted.append(rid)
            state = tm.rid_to_state.pop(rid)
            state.finished = True
            state.event.set()

        tm.abort_request = abort
        # generate_request retains the original IDs separately from children.
        await tm._abort_and_drain_pending_req_states(
            SimpleNamespace(is_single=False, rid=["parent"]), child_rids
        )
        assert aborted == expected
        assert not tm.rid_to_state

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def _generation_at_yield(*, batch=False, finished=True, removed=True, child=False):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = SimpleNamespace(language_only=False, peft_method=None)
    tm.enable_lora = True
    tm.tokenizer = None
    tm.model_update_lock = RWLock()
    tm.is_pause_cond = asyncio.Condition()
    tm.is_pause = False
    tm.rid_to_state = {}
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = Mock()
    tm._validate_and_resolve_lora = AsyncMock()
    tm._finalize_lora_lease = Mock()
    tm._finalize_oft_lease = Mock()
    tm.abort_request = Mock()
    rids = ["a", "b"] if batch else ["a"]
    obj = SimpleNamespace(
        rid=rids if batch else "a",
        is_single=not batch,
        return_prompt_token_ids=False,
        normalize_batch_and_arguments=lambda: None,
    )

    def initialize(*args):
        for rid in rids:
            tm.rid_to_state[rid] = SimpleNamespace(
                dispatched=True, finished=finished, event=asyncio.Event()
            )

    async def response(*args):
        if removed:
            tm.rid_to_state.clear()
        if child:
            args[-1].append("child")
            tm.rid_to_state = {
                "child": SimpleNamespace(
                    dispatched=True, finished=False, event=asyncio.Event()
                )
            }
        yield {"text": "result"}

    tm._init_req_state = initialize
    tm._tokenize_one_request = AsyncMock(return_value=obj)
    tm._send_one_request = Mock()
    tm._wait_one_response = tm._handle_batch_request = response
    return tm, tm.generate_request(obj)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("removed", [False, True])
def test_completed_generator_closes_without_running_event_loop(batch, removed):
    # Sync Engine.generate() returns after one __anext__, outside its loop.
    loop = asyncio.new_event_loop()
    tm, generation = _generation_at_yield(batch=batch, removed=removed)
    try:
        assert loop.run_until_complete(anext(generation)) == {"text": "result"}
        with pytest.raises(StopIteration):
            generation.aclose().send(None)
        assert not loop.run_until_complete(tm.model_update_lock.is_locked())
    finally:
        loop.run_until_complete(generation.aclose())
        loop.close()


@pytest.mark.parametrize("child", [False, True])
def test_closing_active_stream_drains_before_releasing_reader_lock(child):
    async def scenario():
        tm, generation = _generation_at_yield(
            batch=child, finished=False, removed=False, child=child
        )
        await anext(generation)
        state = tm.rid_to_state["child" if child else "a"]
        aborted = asyncio.Event()
        tm.abort_request = lambda rid: aborted.set()
        closing = asyncio.create_task(generation.aclose())
        try:
            await asyncio.wait_for(aborted.wait(), 1)
            assert not closing.done()
            assert await tm.model_update_lock.is_locked()
        finally:
            state.finished = True
            state.event.set()
            await asyncio.wait_for(closing, 1)
        assert not await tm.model_update_lock.is_locked()

    asyncio.run(scenario())

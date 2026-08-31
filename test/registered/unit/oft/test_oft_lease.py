"""Task 8 regression for exactly-once canonical OFT request leases."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager


def _state(adapter_path="policy-a", adapter_id="adapter-id-a"):
    obj = SimpleNamespace(
        adapter_path=adapter_path,
        adapter_id=adapter_id,
        rid="request-a",
    )
    return ReqState([], False, asyncio.Event(), obj, Mock())


def test_task8_oft_lease_is_released_exactly_once():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.peft_kind = "oft"
    manager.peft_registry = SimpleNamespace(release=AsyncMock())
    state = _state()

    async def run():
        for _ in range(3):
            manager._finalize_oft_lease(state)
        await asyncio.sleep(0)

    asyncio.run(run())

    manager.peft_registry.release.assert_awaited_once_with("adapter-id-a")
    assert state.oft_lease_released


def test_task8_oft_pre_acquire_failure_releases_nothing():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.peft_kind = "oft"
    manager.peft_registry = SimpleNamespace(release=AsyncMock())
    state = _state(adapter_id=None)

    async def run():
        manager._finalize_oft_lease(state)
        await asyncio.sleep(0)

    asyncio.run(run())

    manager.peft_registry.release.assert_not_awaited()
    assert not state.oft_lease_released

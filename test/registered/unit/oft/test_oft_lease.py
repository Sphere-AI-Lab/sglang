"""Task 8 regression for exactly-once canonical OFT request leases."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.oft import tokenizer_hooks


def _state(adapter_path="policy-a", adapter_id="adapter-id-a"):
    obj = SimpleNamespace(
        adapter_path=adapter_path,
        adapter_id=adapter_id,
        rid="request-a",
    )
    return SimpleNamespace(obj=obj, oft_lease_released=False)


def test_task8_oft_lease_is_released_exactly_once():
    manager = SimpleNamespace(
        peft_kind="oft",
        peft_registry=SimpleNamespace(release=AsyncMock()),
    )
    state = _state()

    async def run():
        for _ in range(3):
            tokenizer_hooks.finalize_oft_lease(manager, state)
        await asyncio.sleep(0)

    asyncio.run(run())

    manager.peft_registry.release.assert_awaited_once_with("adapter-id-a")
    assert state.oft_lease_released


def test_task8_oft_pre_acquire_failure_releases_nothing():
    manager = SimpleNamespace(
        peft_kind="oft",
        peft_registry=SimpleNamespace(release=AsyncMock()),
    )
    state = _state(adapter_id=None)

    async def run():
        tokenizer_hooks.finalize_oft_lease(manager, state)
        await asyncio.sleep(0)

    asyncio.run(run())

    manager.peft_registry.release.assert_not_awaited()
    assert not state.oft_lease_released

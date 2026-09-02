import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry
from sglang.srt.oft.tokenizer_hooks import resolve_oft_path


def test_evicted_wire_adapter_requires_a_fresh_native_load():
    async def run():
        registry = OFTRegistry()
        ref = OFTRef(
            adapter_name="adapter",
            adapter_path="__tensor__",
            reloadable=False,
        )
        await registry.register(ref)
        adapter_id = await registry.unregister(ref.adapter_name)
        await registry.wait_for_unload(adapter_id)

        async def reject_disk_load(_request):
            raise AssertionError("wire-only adapter reached the disk loader")

        tokenizer_manager = SimpleNamespace(
            peft_registry=registry,
            peft_ref_cache={ref.adapter_name: ref},
            load_oft_adapter=reject_disk_load,
        )
        request = SimpleNamespace(
            adapter_path=ref.adapter_name,
            adapter_id=None,
            adapter_version=None,
        )

        with pytest.raises(ValueError, match="has no on-disk artifact to reload"):
            await resolve_oft_path(tokenizer_manager, request)

    asyncio.run(run())


def test_resolver_uses_one_atomic_id_and_version_snapshot():
    class VersionBumpingAcquireRegistry(OFTRegistry):
        async def acquire(self, name):
            adapter_id = await super().acquire(name)
            await self.bump_version_by_id(adapter_id)
            return adapter_id

    async def run():
        registry = VersionBumpingAcquireRegistry()
        ref = OFTRef(
            adapter_name="adapter",
            adapter_path="/disk/adapter",
            adapter_version=7,
        )
        await registry.register(ref)
        tokenizer_manager = SimpleNamespace(
            peft_registry=registry,
            peft_ref_cache={ref.adapter_name: ref},
        )
        request = SimpleNamespace(
            adapter_path=ref.adapter_name,
            adapter_id=None,
            adapter_version=None,
        )

        await resolve_oft_path(tokenizer_manager, request)

        assert request.adapter_id == ref.adapter_id
        assert request.adapter_version == 7
        await registry.release(request.adapter_id)

    asyncio.run(run())

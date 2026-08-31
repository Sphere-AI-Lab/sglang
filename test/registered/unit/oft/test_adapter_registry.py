import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveOrReuse(unittest.IsolatedAsyncioTestCase):
    async def test_no_upsert_returns_unchanged(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        ref = AdapterRef(adapter_name="a", adapter_path="a")
        resolved, reused = await registry.resolve_or_reuse(ref, upsert=False)
        self.assertIs(resolved, ref)
        self.assertFalse(reused)

    async def test_upsert_with_no_existing_returns_unchanged(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        ref = AdapterRef(adapter_name="a", adapter_path="a")
        resolved, reused = await registry.resolve_or_reuse(ref, upsert=True)
        self.assertIs(resolved, ref)
        self.assertFalse(reused)

    async def test_upsert_with_existing_reuses_id(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        existing = AdapterRef(adapter_name="a", adapter_path="old")
        await registry.register(existing)
        new_ref = AdapterRef(adapter_name="a", adapter_path="new")
        resolved, reused = await registry.resolve_or_reuse(new_ref, upsert=True)
        self.assertTrue(reused)
        self.assertEqual(resolved.adapter_id, existing.adapter_id)
        self.assertEqual(resolved.adapter_path, "new")
        # resolve_or_reuse must not mutate the registry itself.
        self.assertIs(registry.get_all_adapters()["a"], existing)

    async def test_upsert_preserve_pinned(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        existing = AdapterRef(adapter_name="a", adapter_path="old", pinned=True)
        await registry.register(existing)
        new_ref = AdapterRef(adapter_name="a", adapter_path="new", pinned=False)
        resolved, _ = await registry.resolve_or_reuse(
            new_ref, upsert=True, preserve_pinned=True
        )
        self.assertTrue(resolved.pinned)


class TestRefresh(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_updates_registered_entry(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        existing = AdapterRef(adapter_name="a", adapter_path="old")
        await registry.register(existing)
        updated = AdapterRef(
            adapter_id=existing.adapter_id, adapter_name="a", adapter_path="new"
        )
        await registry.refresh(updated)
        self.assertEqual(registry.get_all_adapters()["a"].adapter_path, "new")

    async def test_refresh_asserts_matching_id(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        existing = AdapterRef(adapter_name="a", adapter_path="old")
        await registry.register(existing)
        wrong_id_ref = AdapterRef(adapter_name="a", adapter_path="new")
        with self.assertRaises(AssertionError):
            await registry.refresh(wrong_id_ref)

    async def test_refresh_asserts_already_registered(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        ref = AdapterRef(adapter_name="a", adapter_path="a")
        with self.assertRaises(AssertionError):
            await registry.refresh(ref)


class TestAcquireWithVersion(unittest.IsolatedAsyncioTestCase):
    async def test_single_name_returns_id_and_version(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        ref = AdapterRef(adapter_name="a", adapter_path="a", adapter_version=3)
        await registry.register(ref)
        uid, version = await registry.acquire_with_version("a")
        self.assertEqual(uid, ref.adapter_id)
        self.assertEqual(version, 3)

    async def test_list_of_names_returns_parallel_lists(self):
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        a = AdapterRef(adapter_name="a", adapter_path="a", adapter_version=1)
        b = AdapterRef(adapter_name="b", adapter_path="b", adapter_version=2)
        await registry.register(a)
        await registry.register(b)
        uids, versions = await registry.acquire_with_version(["a", None, "b"])
        self.assertEqual(uids, [a.adapter_id, None, b.adapter_id])
        self.assertEqual(versions, [1, None, 2])


if __name__ == "__main__":
    unittest.main()

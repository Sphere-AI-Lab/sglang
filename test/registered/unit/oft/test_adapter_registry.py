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

    async def test_upsert_bumps_adapter_version(self):
        """Regression guard for C2: an in-place upsert must bump
        adapter_version past the existing entry's, so the radix cache key
        (which is extended with the adapter's version) changes across the
        refresh. Without this, repeated in-place upserts of the same
        adapter name never change the radix key, and a prompt re-served
        after an upsert could be served from a stale KV prefix cached under
        the pre-upsert weights."""
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        existing = AdapterRef(adapter_name="a", adapter_path="old", adapter_version=1)
        await registry.register(existing)
        new_ref = AdapterRef(adapter_name="a", adapter_path="new")
        resolved, reused = await registry.resolve_or_reuse(new_ref, upsert=True)
        self.assertTrue(reused)
        self.assertEqual(resolved.adapter_version, existing.adapter_version + 1)

        # A second round of upserts must keep bumping past the CURRENT
        # registered version, not the original one.
        await registry.refresh(resolved)
        second_new_ref = AdapterRef(adapter_name="a", adapter_path="newer")
        resolved_again, reused_again = await registry.resolve_or_reuse(
            second_new_ref, upsert=True
        )
        self.assertTrue(reused_again)
        self.assertEqual(resolved_again.adapter_version, resolved.adapter_version + 1)

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

    async def test_acquire_with_version_rejects_invalid_input_type(self):
        """Verify acquire_with_version rejects non-str/list inputs like acquire() does."""
        from sglang.srt.oft.base.registry import AdapterRegistry

        registry = AdapterRegistry()
        with self.assertRaises(TypeError):
            await registry.acquire_with_version(123)

    async def test_acquire_with_version_single_name_with_nones_in_list(self):
        """Verify single-name path returns scalar tuple, not list."""
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        ref = AdapterRef(adapter_name="a", adapter_path="a", adapter_version=5)
        await registry.register(ref)
        # Single string input should return scalar tuple
        uid, version = await registry.acquire_with_version("a")
        self.assertIsInstance(uid, str)
        self.assertIsInstance(version, int)
        self.assertEqual(uid, ref.adapter_id)
        self.assertEqual(version, 5)

    async def test_acquire_with_version_counter_incremented(self):
        """Verify counter is incremented as part of atomic acquisition.

        Regression guard: prior to the atomicity fix, the counter increment
        happened outside the writer lock, creating a race where unload could
        delete the counter before increment ran.
        """
        from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry

        registry = AdapterRegistry()
        ref = AdapterRef(adapter_name="a", adapter_path="a", adapter_version=1)
        await registry.register(ref)

        # Acquire the adapter
        uid, version = await registry.acquire_with_version("a")

        # Verify the counter is tracking (by acquiring again and ensuring
        # we can decrement the same number of times)
        await registry.acquire("a")

        # Release both acquisitions - if counter wasn't properly incremented,
        # this would fail
        await registry.release(uid)
        await registry.release(uid)


if __name__ == "__main__":
    unittest.main()

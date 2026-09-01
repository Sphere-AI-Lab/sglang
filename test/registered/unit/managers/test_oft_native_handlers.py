# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for the native OFT adapter-loading tokenizer-manager handlers
(``load_oft_adapter_from_tensors``/``_from_distributed``/``unload_oft_adapter``),
mirroring test/registered/unit/lora/test_lora_upsert.py and
test/registered/unit/lora/test_lora_lease.py's mocked-TokenizerManager
approach. Uses a real OFTRegistry (not a mock) so the registry lock, LRU
eviction loop, and id/version bookkeeping are actually exercised -- only the
scheduler-facing communicator is mocked.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    LoadOFTAdapterFromDistributedReqInput,
    LoadOFTAdapterFromTensorsReqInput,
    OFTUpdateOutput,
    UnloadOFTAdapterReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.oft.oft_registry import OFTRef, OFTRegistry
from sglang.srt.peft.tokenizer_hooks import finalize_peft_lease

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONFIG_DICT = {"target_modules": ["q_proj"], "r": 8}


def _make_communicator(preloaded: dict = None) -> AsyncMock:
    """Fan-out communicator stub: for a load request, echoes back a
    loaded_adapters map merging any preloaded entries with this request's own
    (name -> id), mimicking the scheduler reporting the full resident set;
    for an unload request only success matters to the caller."""
    preloaded = dict(preloaded or {})

    async def _side_effect(obj):
        if isinstance(obj, UnloadOFTAdapterReqInput):
            return [OFTUpdateOutput(success=True)]
        merged = dict(preloaded)
        merged[obj.adapter_name] = obj.adapter_id
        return [OFTUpdateOutput(success=True, loaded_adapters=merged)]

    return AsyncMock(side_effect=_side_effect)


def _make_tokenizer_manager(
    peft_method: str = "oft",
    oft_impl: str = "sibling",
    max_loaded_ofts=None,
) -> TokenizerManager:
    """TokenizerManager with only the fields the OFT handler path reads."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.peft_method = peft_method
    tm.server_args.oft_impl = oft_impl
    tm.server_args.max_loaded_ofts = max_loaded_ofts
    tm.auto_create_handle_loop = Mock()
    tm.peft_update_lock = asyncio.Lock()
    tm.peft_registry = OFTRegistry()
    tm.peft_ref_cache = {}
    tm.update_oft_adapter_communicator = _make_communicator()
    return tm


def _make_tensors_req(
    adapter_name: str = "a", upsert: bool = False, pinned: bool = False
) -> LoadOFTAdapterFromTensorsReqInput:
    return LoadOFTAdapterFromTensorsReqInput(
        adapter_name=adapter_name,
        config_dict=CONFIG_DICT,
        serialized_named_tensors=[],
        pinned=pinned,
        upsert=upsert,
    )


def _make_distributed_req(
    adapter_name: str = "a", upsert: bool = False, pinned: bool = False
) -> LoadOFTAdapterFromDistributedReqInput:
    return LoadOFTAdapterFromDistributedReqInput(
        adapter_name=adapter_name,
        config_dict=CONFIG_DICT,
        names=[],
        dtypes=[],
        shapes=[],
        pinned=pinned,
        upsert=upsert,
    )


class TestFreshLoadRegisters(CustomTestCase):
    def test_from_tensors_fresh_load_registers(self):
        tm = _make_tokenizer_manager()

        obj = _make_tensors_req("a")
        result = asyncio.run(tm.load_oft_adapter_from_tensors(obj))

        self.assertTrue(result.success)
        self.assertIsNotNone(obj.adapter_id)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 1)
        self.assertEqual(
            tm.peft_registry.get_all_adapters()["a"].adapter_id, obj.adapter_id
        )
        self.assertIs(tm.peft_ref_cache["a"], tm.peft_registry.get_all_adapters()["a"])

    def test_from_distributed_fresh_load_registers(self):
        tm = _make_tokenizer_manager()

        obj = _make_distributed_req("a")
        result = asyncio.run(tm.load_oft_adapter_from_distributed(obj))

        self.assertTrue(result.success)
        self.assertIsNotNone(obj.adapter_id)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 1)
        self.assertEqual(tm.peft_ref_cache["a"].adapter_id, obj.adapter_id)


class TestUpsertReusesId(CustomTestCase):
    def test_from_distributed_upsert_reuses_existing_id(self):
        tm = _make_tokenizer_manager()
        existing = OFTRef(
            adapter_name="a", adapter_path="__distributed__", reloadable=False
        )
        asyncio.run(tm.peft_registry.register(existing))

        obj = _make_distributed_req("a", upsert=True)
        result = asyncio.run(tm.load_oft_adapter_from_distributed(obj))

        self.assertTrue(result.success)
        self.assertEqual(obj.adapter_id, existing.adapter_id)
        # Refreshed in place, not re-registered as a second entry.
        self.assertEqual(tm.peft_registry.num_registered_ofts, 1)
        self.assertEqual(tm.peft_ref_cache["a"].adapter_id, existing.adapter_id)

    def test_from_tensors_upsert_reuses_existing_id(self):
        # Unlike LoRA's from_tensors route (which rejects upsert outright),
        # OFT's from_tensors handler resolves upsert the same way
        # from_distributed does -- this pins that intentional behavior.
        tm = _make_tokenizer_manager()
        existing = OFTRef(
            adapter_name="a", adapter_path="__tensor__", reloadable=False
        )
        asyncio.run(tm.peft_registry.register(existing))

        obj = _make_tensors_req("a", upsert=True)
        result = asyncio.run(tm.load_oft_adapter_from_tensors(obj))

        self.assertTrue(result.success)
        self.assertEqual(obj.adapter_id, existing.adapter_id)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 1)

    def test_from_distributed_upsert_bumps_adapter_version(self):
        """Regression guard for C2: the native handler's upsert path must
        bump adapter_version on the registered/cached ref -- otherwise the
        radix cache key never changes across in-place refreshes and a
        request could be served from a stale, pre-refresh KV prefix."""
        tm = _make_tokenizer_manager()
        existing = OFTRef(
            adapter_name="a",
            adapter_path="__distributed__",
            reloadable=False,
            adapter_version=1,
        )
        asyncio.run(tm.peft_registry.register(existing))

        result = asyncio.run(
            tm.load_oft_adapter_from_distributed(_make_distributed_req("a", upsert=True))
        )

        self.assertTrue(result.success)
        updated = tm.peft_registry.get_all_adapters()["a"]
        self.assertEqual(updated.adapter_version, existing.adapter_version + 1)
        self.assertEqual(tm.peft_ref_cache["a"].adapter_version, updated.adapter_version)

    def test_from_tensors_upsert_bumps_adapter_version(self):
        tm = _make_tokenizer_manager()
        existing = OFTRef(
            adapter_name="a",
            adapter_path="__tensor__",
            reloadable=False,
            adapter_version=1,
        )
        asyncio.run(tm.peft_registry.register(existing))

        result = asyncio.run(
            tm.load_oft_adapter_from_tensors(_make_tensors_req("a", upsert=True))
        )

        self.assertTrue(result.success)
        updated = tm.peft_registry.get_all_adapters()["a"]
        self.assertEqual(updated.adapter_version, existing.adapter_version + 1)

    def test_non_upsert_duplicate_fails(self):
        tm = _make_tokenizer_manager()
        asyncio.run(
            tm.peft_registry.register(
                OFTRef(adapter_name="a", adapter_path="__tensor__")
            )
        )

        result = asyncio.run(
            tm.load_oft_adapter_from_tensors(_make_tensors_req("a", upsert=False))
        )

        self.assertFalse(result.success)
        self.assertIn("already exists", result.error_message)


class TestLRUEviction(CustomTestCase):
    def test_eviction_fires_when_max_loaded_ofts_exceeded(self):
        tm = _make_tokenizer_manager(max_loaded_ofts=1)
        old_ref = OFTRef(adapter_name="old", adapter_path="/old")
        asyncio.run(tm.peft_registry.register(old_ref))
        tm.peft_ref_cache["old"] = old_ref
        tm.update_oft_adapter_communicator = _make_communicator(
            {"old": old_ref.adapter_id}
        )

        result = asyncio.run(
            tm.load_oft_adapter_from_tensors(_make_tensors_req("new"))
        )

        self.assertTrue(result.success)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 1)
        registered = tm.peft_registry.get_all_adapters()
        self.assertNotIn("old", registered)
        self.assertIn("new", registered)
        # The eviction loop must scrub the evicted name from the reported
        # loaded_adapters map, or the caller reports a since-unloaded adapter
        # as still resident.
        self.assertNotIn("old", result.loaded_adapters)
        self.assertIn("new", result.loaded_adapters)
        # EVICT (LRU) must keep the ref_cache entry so a disk-backed adapter
        # can still be implicitly reloaded later -- mirrors LoRA's
        # _unload_lora_adapter_locked contract.
        self.assertIn("old", tm.peft_ref_cache)

    def test_no_eviction_when_under_limit(self):
        tm = _make_tokenizer_manager(max_loaded_ofts=2)
        old_ref = OFTRef(adapter_name="old", adapter_path="/old")
        asyncio.run(tm.peft_registry.register(old_ref))
        tm.peft_ref_cache["old"] = old_ref
        tm.update_oft_adapter_communicator = _make_communicator(
            {"old": old_ref.adapter_id}
        )

        result = asyncio.run(
            tm.load_oft_adapter_from_tensors(_make_tensors_req("new"))
        )

        self.assertTrue(result.success)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 2)
        self.assertIn("old", tm.peft_registry.get_all_adapters())


class TestUnloadDeleteVsEvictSemantics(CustomTestCase):
    def test_explicit_unload_drops_ref_cache_entry(self):
        tm = _make_tokenizer_manager()
        ref = OFTRef(adapter_name="a", adapter_path="/x")
        asyncio.run(tm.peft_registry.register(ref))
        tm.peft_ref_cache["a"] = ref

        result = asyncio.run(
            tm.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="a"))
        )

        self.assertTrue(result.success)
        self.assertNotIn("a", tm.peft_ref_cache)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 0)


class TestUnloadWaitsBeforeCommunicatorDispatch(CustomTestCase):
    """Regression guard: _unload_oft_adapter_locked must wait for in-flight
    leases to drain (wait_for_unload) BEFORE telling the backend to free GPU
    state (the communicator dispatch) -- mirrors _unload_lora_adapter_locked's
    ordering. An earlier version of this code dispatched the communicator
    first and called wait_for_unload after, which could free backend state
    while a request was still in flight against it."""

    def test_wait_for_unload_precedes_communicator_call(self):
        tm = _make_tokenizer_manager()
        ref = OFTRef(adapter_name="a", adapter_path="/x")
        asyncio.run(tm.peft_registry.register(ref))
        tm.peft_ref_cache["a"] = ref

        call_order = []
        real_wait_for_unload = tm.peft_registry.wait_for_unload

        async def _tracking_wait_for_unload(uid):
            call_order.append("wait_for_unload")
            return await real_wait_for_unload(uid)

        tm.peft_registry.wait_for_unload = _tracking_wait_for_unload

        async def _tracking_communicator(obj):
            call_order.append("communicator")
            return [OFTUpdateOutput(success=True)]

        tm.update_oft_adapter_communicator = AsyncMock(
            side_effect=_tracking_communicator
        )

        result = asyncio.run(
            tm.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="a"))
        )

        self.assertTrue(result.success)
        self.assertEqual(call_order, ["wait_for_unload", "communicator"])


class TestMultiRankFailureNotSwallowed(CustomTestCase):
    """Regression guard: unload and load_from_distributed used to take only
    rank 0's reply ([0]) with no dp_size==1 guard backing that shortcut (unlike
    LoRA's from_distributed route), so a failure on any later rank was
    silently reported as success. Both must merge all ranks' replies instead."""

    def test_unload_reports_non_first_rank_failure(self):
        tm = _make_tokenizer_manager()
        ref = OFTRef(adapter_name="a", adapter_path="/x")
        asyncio.run(tm.peft_registry.register(ref))
        tm.peft_ref_cache["a"] = ref
        tm.update_oft_adapter_communicator = AsyncMock(
            return_value=[
                OFTUpdateOutput(success=True),
                OFTUpdateOutput(success=False, error_message="rank1 oom"),
            ]
        )

        result = asyncio.run(
            tm.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="a"))
        )

        self.assertFalse(result.success)
        self.assertIn("rank1 oom", result.error_message)

    def test_load_from_distributed_reports_non_first_rank_failure(self):
        tm = _make_tokenizer_manager()
        tm.update_oft_adapter_communicator = AsyncMock(
            return_value=[
                OFTUpdateOutput(success=True, loaded_adapters={"a": "id"}),
                OFTUpdateOutput(success=False, error_message="rank1 oom"),
            ]
        )

        result = asyncio.run(
            tm.load_oft_adapter_from_distributed(_make_distributed_req("a"))
        )

        self.assertFalse(result.success)
        self.assertIn("rank1 oom", result.error_message)


class TestFinalizePeftLeaseIdempotency(CustomTestCase):
    """Regression guard for finalize_peft_lease's idempotency contract: every
    terminal request path (normal finish, abort echo, status-code abort,
    failed dispatch) calls it, and more than one of those can legitimately
    fire for the same state (see the "idempotency is what prevents the
    counter from going negative here" comment at the
    _handle_abort_finish_reason call site). A future 5th terminal path that
    forgets the peft_lease_released guard would double-release the lease,
    driving the registry's usage counter negative -- which hangs
    wait_for_unload forever just as surely as never releasing at all (it
    waits for exactly zero, per ConcurrentCounter.wait_for_zero)."""

    def _finalize(self, tm, state, times=1):
        async def run():
            for _ in range(times):
                finalize_peft_lease(tm, state)
            # Let the create_task'd release coroutine run.
            await asyncio.sleep(0)

        asyncio.run(run())

    def test_double_finalize_releases_exactly_once(self):
        tm = SimpleNamespace(peft_registry=MagicMock())
        tm.peft_registry.release = AsyncMock()
        state = SimpleNamespace(
            peft_lease_released=False, obj=SimpleNamespace(adapter_id="adapter-1")
        )

        self._finalize(tm, state, times=3)

        tm.peft_registry.release.assert_awaited_once_with("adapter-1")
        self.assertTrue(state.peft_lease_released)


class TestFinalizePeftLeaseNoOp(CustomTestCase):
    """finalize_peft_lease must be a safe no-op whenever there is no lease to
    release, rather than raising or releasing something bogus."""

    def test_no_op_when_state_is_none(self):
        # A request whose peft acquire failed before state existed (or was
        # already dropped by another terminal path) has nothing to release.
        tm = SimpleNamespace(peft_registry=MagicMock())
        tm.peft_registry.release = AsyncMock()

        self._run(finalize_peft_lease, tm, None)

        tm.peft_registry.release.assert_not_awaited()

    def test_no_op_when_adapter_id_is_none(self):
        # Base-only request (peft enabled, no adapter named): adapter_id
        # stays None, so there is no lease to release.
        tm = SimpleNamespace(peft_registry=MagicMock())
        tm.peft_registry.release = AsyncMock()
        state = SimpleNamespace(
            peft_lease_released=False, obj=SimpleNamespace(adapter_id=None)
        )

        self._run(finalize_peft_lease, tm, state)

        tm.peft_registry.release.assert_not_awaited()
        self.assertFalse(state.peft_lease_released)

    def test_no_op_when_request_type_has_no_adapter_id_field(self):
        # EmbeddingReqInput never declares adapter_id at all (peft has no
        # embedding support -- generate_request only calls
        # maybe_resolve_peft_path under isinstance(obj, GenerateReqInput)),
        # so state.obj can genuinely lack the attribute. Must getattr-guard
        # this rather than assume the field exists.
        tm = SimpleNamespace(peft_registry=MagicMock())
        tm.peft_registry.release = AsyncMock()
        state = SimpleNamespace(
            peft_lease_released=False, obj=EmbeddingReqInput(text="hi")
        )

        self._run(finalize_peft_lease, tm, state)

        tm.peft_registry.release.assert_not_awaited()

    @staticmethod
    def _run(fn, *args):
        async def run():
            fn(*args)
            await asyncio.sleep(0)

        asyncio.run(run())


class TestRegisterPeftRefRollback(CustomTestCase):
    """Regression guard for C1c: register_peft_ref (used by the retired
    load_format="oft_adapter" streamed-update path, tokenizer_control_mixin
    .update_weights_from_tensor) must report whether it newly minted a
    registration, so a caller whose backend load subsequently fails can roll
    it back via rollback_peft_ref. Without this, a failed streamed load
    leaves a registered-but-not-actually-resident name behind: a later
    /generate naming it passes the tokenizer-side registry check and
    reaches the GPU-side code with no matching adapter there, instead of a
    clean "adapter not found" rejection."""

    @staticmethod
    def _make_tm():
        return SimpleNamespace(peft_registry=OFTRegistry(), peft_ref_cache={})

    def test_register_peft_ref_reports_newly_registered(self):
        from sglang.srt.peft.tokenizer_hooks import register_peft_ref

        tm = self._make_tm()
        obj = SimpleNamespace(adapter_name="a", adapter_id=None)
        newly_registered = asyncio.run(register_peft_ref(tm, obj))
        self.assertTrue(newly_registered)
        self.assertIn("a", tm.peft_ref_cache)
        self.assertIsNotNone(obj.adapter_id)

    def test_register_peft_ref_reports_not_new_for_existing_name(self):
        from sglang.srt.peft.tokenizer_hooks import register_peft_ref

        tm = self._make_tm()
        obj1 = SimpleNamespace(adapter_name="a", adapter_id=None)
        asyncio.run(register_peft_ref(tm, obj1))

        obj2 = SimpleNamespace(adapter_name="a", adapter_id=None)
        newly_registered = asyncio.run(register_peft_ref(tm, obj2))
        self.assertFalse(newly_registered)
        self.assertEqual(obj2.adapter_id, obj1.adapter_id)

    def test_rollback_peft_ref_removes_newly_registered_name(self):
        from sglang.srt.peft.tokenizer_hooks import (
            register_peft_ref,
            rollback_peft_ref,
        )

        tm = self._make_tm()
        obj = SimpleNamespace(adapter_name="a", adapter_id=None)
        newly_registered = asyncio.run(register_peft_ref(tm, obj))
        self.assertTrue(newly_registered)

        asyncio.run(rollback_peft_ref(tm, obj.adapter_name))

        self.assertNotIn("a", tm.peft_ref_cache)
        self.assertEqual(tm.peft_registry.num_registered_ofts, 0)

    def test_rollback_gate_does_not_apply_to_a_previously_existing_adapter(self):
        """Guards the exact regression this fix must avoid: a caller must
        only roll back when register_peft_ref reported newly_registered=
        True. A second registration attempt for an already-loaded name
        resolves the EXISTING ref (newly_registered=False) precisely so a
        caller never rolls back an adapter that was already loaded and
        serving before this request."""
        from sglang.srt.peft.tokenizer_hooks import register_peft_ref

        tm = self._make_tm()
        obj1 = SimpleNamespace(adapter_name="a", adapter_id=None)
        asyncio.run(register_peft_ref(tm, obj1))

        obj2 = SimpleNamespace(adapter_name="a", adapter_id=None)
        newly_registered = asyncio.run(register_peft_ref(tm, obj2))
        self.assertFalse(newly_registered)


class TestWrongPeftConfigRejected(CustomTestCase):
    """The native OFT RPC handlers must reject loudly -- before touching the
    lock or the communicator -- unless the engine actually booted with
    --peft-method oft --oft-impl sibling."""

    def test_load_from_tensors_rejects_wrong_peft_method(self):
        tm = _make_tokenizer_manager(peft_method="lora")

        result = asyncio.run(tm.load_oft_adapter_from_tensors(_make_tensors_req()))

        self.assertFalse(result.success)
        self.assertIn("--peft-method oft", result.error_message)
        tm.update_oft_adapter_communicator.assert_not_awaited()

    def test_load_from_distributed_rejects_wrong_oft_impl(self):
        tm = _make_tokenizer_manager(oft_impl="staged")

        result = asyncio.run(
            tm.load_oft_adapter_from_distributed(_make_distributed_req())
        )

        self.assertFalse(result.success)
        self.assertIn("--oft-impl sibling", result.error_message)
        tm.update_oft_adapter_communicator.assert_not_awaited()

    def test_unload_rejects_wrong_peft_method(self):
        tm = _make_tokenizer_manager(peft_method=None)

        result = asyncio.run(
            tm.unload_oft_adapter(UnloadOFTAdapterReqInput(adapter_name="a"))
        )

        self.assertFalse(result.success)
        self.assertIn("Native OFT adapter loading requires", result.error_message)
        tm.update_oft_adapter_communicator.assert_not_awaited()


class TestResolvePeftPathRejectsEvictedDiskBackedAdapter(CustomTestCase):
    """Regression guard: resolve_peft_path's disk-backed-adapter reload
    branch used to call tm.load_oft_adapter, which does not exist anywhere
    on TokenizerManager -- OFT never got LoRA's equivalent disk-path reload
    RPC (only the wire-load load_oft_adapter_from_tensors/_from_distributed
    RPCs are wired up). So a disk-backed (--peft-paths) adapter whose
    tokenizer-side registry entry was LRU-evicted (--max-loaded-ofts) and
    then re-requested crashed with AttributeError instead of a clear,
    catchable error. Constructs the post-eviction state directly (a
    reloadable ref present in peft_ref_cache but absent from peft_registry)
    rather than driving a full eviction cycle."""

    def test_evicted_disk_backed_adapter_raises_value_error(self):
        from sglang.srt.peft.tokenizer_hooks import resolve_peft_path

        ref = OFTRef(adapter_name="a", adapter_path="/disk/a", pinned=False)
        tm = SimpleNamespace(
            peft_kind="oft",
            peft_registry=OFTRegistry(),
            peft_ref_cache={"a": ref},
        )
        obj = SimpleNamespace(adapter_path="a", lora_path=None)

        with self.assertRaisesRegex(ValueError, "evicted"):
            asyncio.run(resolve_peft_path(tm, obj))


if __name__ == "__main__":
    unittest.main(verbosity=2)

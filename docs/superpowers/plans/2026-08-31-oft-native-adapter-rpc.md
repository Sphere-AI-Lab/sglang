# Native OFT Adapter RPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give single-active ("sibling") OFT its own native adapter-loading RPC surface — `load_oft_adapter_from_tensors`, `load_oft_adapter_from_distributed`, `unload_oft_adapter` — at full functional parity with native LoRA's equivalent, so it no longer depends on the shared `srt/peft` integration layer for its live weight-sync path.

**Architecture:** Mirror LoRA's native mechanism file-by-file (wire types in `io_struct.py`, handlers in `tokenizer_control_mixin.py`, dispatch in `scheduler.py`/`tp_worker.py`, GPU-side admission in `oft_manager.py`, Engine/HTTP surface), extending the `AdapterRegistry`/`AdapterRef` base classes that `OFTRegistry` already uses with the two-phase resolve/commit and atomic acquire-with-version primitives LoRA's registry has and OFT's doesn't yet.

**Tech Stack:** Python, asyncio, msgspec (wire types), PyTorch (tensor broadcast/serialization), pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-08-31-oft-native-adapter-rpc-design.md`

## Global Constraints

- Subclass/extend, don't rewrite: `AdapterRegistry`'s existing `acquire()`/`replace()` are left untouched — new methods are added alongside, not refactored in.
- No `added_tokens_config` on any new OFT wire type — OFT's rotation-based adapters never touch token embeddings (per spec Non-goals).
- `StagedOFTManager`/`stage_adapter`/`activate_adapter_version` are untouched by this plan — the new RPC is gated to `oft_impl == "sibling"` only.
- Any edit to `scheduler.py`, `tokenizer_control_mixin.py` (mixed into `TokenizerManager`), or `model_runner.py` requires reading the `large-class-style` skill first (repo rule, `.claude/rules/modify-component-must-read.md`).
- Every task's tests must run via `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest ...` — `/sgl-workspace/sglang` is a stale unrelated install that silently wins without `PYTHONPATH=python`.
- Retiring the old shared `srt/peft` streamed-loader mechanism happens only in Task 9, gated on Task 8's GPU evidence passing review — do not delete it speculatively.
- Do not push without explicit approval (standing rule for this repo/session).

---

### Task 1: Extend `AdapterRef`/`AdapterRegistry` with `reloadable`, two-phase resolve/commit, and atomic acquire-with-version

**Files:**
- Modify: `python/sglang/srt/oft/base/registry.py`
- Test: `test/registered/unit/oft/test_adapter_registry.py` (new)

**Interfaces:**
- Consumes: nothing new (pure extension of existing `AdapterRef`/`AdapterRegistry`).
- Produces: `AdapterRef.reloadable: bool` (default `True`); `AdapterRegistry.resolve_or_reuse(ref, upsert=False, *, preserve_pinned=False) -> Tuple[AdapterRef, bool]`; `AdapterRegistry.refresh(ref) -> None`; `AdapterRegistry.acquire_with_version(name) -> Tuple[str, int] | Tuple[List[Optional[str]], List[Optional[int]]]`. `OFTRegistry` inherits all of these directly (no new aliases — see spec's naming-symmetry note; base already has 6 of 9 methods unaliased).

- [ ] **Step 1: Read the exact LoRA equivalents before writing anything**

Read `python/sglang/srt/lora/lora_registry.py` lines 28-52 (`LoRARef`, including its `reloadable` field and comment) and lines 142-252 (`register_or_reuse`, `refresh`, `_lookup_refs_for_admission`, `_increment_ref_counters`, `_acquire_refs`, `acquire_with_version`). These are the exact behaviors to generalize onto `AdapterRef`/`AdapterRegistry`.

- [ ] **Step 2: Write the failing unit tests**

Create `test/registered/unit/oft/test_adapter_registry.py`:

```python
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
```

- [ ] **Step 2b: Run the tests to verify they fail**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_adapter_registry.py -v`
Expected: FAIL — `AttributeError: 'AdapterRegistry' object has no attribute 'resolve_or_reuse'` (and similarly for `refresh`/`acquire_with_version`), and `AdapterRef(...)` construction itself still succeeds (since `reloadable` isn't referenced by these tests directly, only exercised implicitly).

- [ ] **Step 3: Add `reloadable` to `AdapterRef`**

In `python/sglang/srt/oft/base/registry.py`, inside the `AdapterRef` dataclass (after the existing `pinned`/`adapter_version` fields):

```python
    # False for adapters whose weights arrived over the wire (no on-disk
    # artifact to reload from): they must never be LRU-evicted nor
    # implicitly reloaded. Mirrors LoRARef.reloadable exactly.
    reloadable: bool = True
```

- [ ] **Step 4: Add `resolve_or_reuse` and `refresh` to `AdapterRegistry`**

Add directly below the existing `replace()` method (leave `replace()` itself untouched):

```python
    async def resolve_or_reuse(
        self,
        ref: AdapterRef,
        upsert: bool = False,
        *,
        preserve_pinned: bool = False,
    ) -> "Tuple[AdapterRef, bool]":
        """Resolve which identity a load request should use.

        Returns ``(ref, reused)``. With ``upsert`` and a same-name adapter
        already registered, the returned ref adopts the existing
        ``adapter_id`` (``reused=True``) so the backend refreshes that
        adapter in place; otherwise ``ref`` is returned unchanged
        (``reused=False``). Nothing is registered here: the caller commits
        the resolved ref with ``register``/``refresh`` once the backend load
        succeeded, keeping failed loads invisible to the registry.
        """
        if not upsert:
            return ref, False
        async with self._registry_lock.reader_lock:
            existing = self._registry.get(ref.adapter_name)
            if existing is None:
                return ref, False
            updates = {"adapter_id": existing.adapter_id}
            if preserve_pinned:
                updates["pinned"] = existing.pinned
            return replace(ref, **updates), True

    async def refresh(self, ref: AdapterRef):
        """Replace a registered adapter's ref after a successful upsert.

        Keeps the id (asserted) while adopting the new path/pinned metadata,
        and counts as a use for LRU ordering.
        """
        async with self._registry_lock.writer_lock:
            existing = self._registry.get(ref.adapter_name)
            assert existing is not None and existing.adapter_id == ref.adapter_id, (
                f"refresh() must target a registered adapter with the same "
                f"adapter_id; got {ref}, registered: {existing}"
            )
            self._registry[ref.adapter_name] = ref
            self._registry.move_to_end(ref.adapter_name)
```

Note `replace` (the dataclasses function, imported at the top of the file already per the existing `bump_version_by_id` usage in `oft_registry.py`) is used here as the copy-with-overrides helper — this is `dataclasses.replace`, unrelated to `AdapterRegistry.replace()` the method.

- [ ] **Step 5: Add `_acquire_refs` and `acquire_with_version`**

Add directly below `acquire()`:

```python
    async def _acquire_refs(
        self, name: Union[str, List[Optional[str]]]
    ) -> Union[Optional[AdapterRef], List[Optional[AdapterRef]]]:
        """Atomically snapshot the matching AdapterRef(s) and start tracking
        usage, in one lock acquisition (unlike acquire(), which only returns
        ids and requires a separate call for version)."""

        def _lookup(n: Optional[str]) -> Optional[AdapterRef]:
            if n is None:
                return None
            ref = self._registry.get(n)
            if ref is None:
                raise ValueError(
                    f"The following requested adapters are not loaded: {n}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(n)
            return ref

        if isinstance(name, str) or name is None:
            async with self._registry_lock.writer_lock:
                ref = _lookup(name)
            if ref is not None:
                await self._counters[ref.adapter_id].increment(notify_all=False)
            return ref
        async with self._registry_lock.writer_lock:
            refs = [_lookup(n) for n in name]
        await asyncio.gather(
            *[
                self._counters[ref.adapter_id].increment(notify_all=False)
                for ref in refs
                if ref is not None
            ]
        )
        return refs

    async def acquire_with_version(
        self, name: Union[str, List[Optional[str]]]
    ) -> Union[
        Tuple[Optional[str], Optional[int]],
        Tuple[List[Optional[str]], List[Optional[int]]],
    ]:
        """Acquire request leases and atomically snapshot ids and versions."""
        if isinstance(name, str):
            ref = await self._acquire_refs(name)
            return (ref.adapter_id if ref else None), (
                ref.adapter_version if ref else None
            )
        refs = await self._acquire_refs(name)
        ids = [ref.adapter_id if ref else None for ref in refs]
        versions = [ref.adapter_version if ref else None for ref in refs]
        return ids, versions
```

Add `import asyncio` and `Tuple` to the existing `typing` import at the top of the file if not already present (check first — `asyncio` is not currently imported in `registry.py`, `Tuple` is not currently in the `typing` import either).

- [ ] **Step 6: Fix the stale "shared by OFT, LoRA" docstring**

`AdapterRef`'s docstring currently says *"Generic adapter-reference record shared by the single-active peft methods (OFT, LoRA)... Subclasses (OFTRef, LoRARef) inherit this identity vocabulary"* — this is false today: `LoRARef` is an independent `msgspec.Struct`, not a subclass of `AdapterRef`. Update the docstring to describe current reality:

```python
@dataclass(frozen=True)
class AdapterRef:
    """Generic adapter-reference record. Originally intended to be shared
    with LoRA (see the historical note in git blame), but LoRARef evolved
    independently as its own msgspec.Struct — today only OFTRef subclasses
    this. Holds the unified adapter identity; AdapterRegistry accesses
    adapters only through these members.

    The unique ``adapter_id`` eliminates conflicts from reused names/paths and
    can be used to generate deterministic cache keys (e.g. radix cache)."""
```

Apply the same correction to `AdapterRegistry`'s class docstring (drop the implied LoRA usage, state it's OFT's registry base today).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_adapter_registry.py -v`
Expected: PASS, all cases.

- [ ] **Step 8: Run the broader existing suite to confirm no regression**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/ -k "oft or peft or adapter_sync" -q --ignore=test/registered/unit/mem_cache/test_umbp_store.py`
Expected: same pass/fail counts as before this task (1 pre-existing unrelated failure, `test_soften_never_crashes`, per this plan's own history — confirm this is still the only failure, not a new one).

- [ ] **Step 9: Commit**

```bash
git add python/sglang/srt/oft/base/registry.py test/registered/unit/oft/test_adapter_registry.py
git commit -m "feat(oft): extend AdapterRegistry with resolve_or_reuse/refresh/acquire_with_version"
```

---

### Task 2: New wire types in `io_struct.py`

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LoadOFTAdapterFromTensorsReqInput`, `LoadOFTAdapterFromDistributedReqInput`, `UnloadOFTAdapterReqInput` (each with `.to_ref() -> OFTRef`), `OFTUpdateOutput`, and its 4 aliases `LoadOFTAdapterReqOutput = UnloadOFTAdapterReqOutput = LoadOFTAdapterFromTensorsReqOutput = LoadOFTAdapterFromDistributedReqOutput = OFTUpdateOutput`.

- [ ] **Step 1: Read the exact LoRA types to mirror**

Read `python/sglang/srt/managers/io_struct.py` lines 2450-2519 (`UnloadLoRAAdapterReqInput`, `LoadLoRAAdapterFromTensorsReqInput`, `LoadLoRAAdapterFromDistributedReqInput`, `LoRAUpdateOutput`, and the 4-way alias line).

- [ ] **Step 2: Add the new types**

Add immediately after the existing LoRA adapter type block (after line ~2519's alias statement):

```python
class UnloadOFTAdapterReqInput(BaseReq, kw_only=True):
    adapter_name: str
    adapter_id: Optional[str] = None

    def to_ref(self) -> OFTRef:
        return OFTRef(adapter_id=self.adapter_id, adapter_name=self.adapter_name)


class LoadOFTAdapterFromTensorsReqInput(BaseReq, kw_only=True):
    adapter_name: str
    # The PEFT adapter_config.json, already JSON.
    config_dict: Dict[str, Any]
    # One serialized copy of the adapter tensors per TP rank; each rank
    # deserializes only its own copy.
    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
    pinned: bool = False
    adapter_id: Optional[str] = None
    load_format: Optional[str] = None
    # If already loaded, refresh weights in place instead of failing.
    upsert: bool = False

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            adapter_path="__tensor__",
            pinned=self.pinned,
            reloadable=False,
        )


class LoadOFTAdapterFromDistributedReqInput(BaseReq, kw_only=True):
    adapter_name: str
    config_dict: Dict[str, Any]
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str = "weight_update_group"
    pinned: bool = False
    adapter_id: Optional[str] = None
    # If already loaded, refresh weights in place instead of failing.
    upsert: bool = False

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            adapter_path="__distributed__",
            pinned=self.pinned,
            reloadable=False,
        )


class OFTUpdateOutput(BaseReq, kw_only=True):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, Union[str, OFTRef]]] = None


LoadOFTAdapterReqOutput = UnloadOFTAdapterReqOutput = (
    LoadOFTAdapterFromTensorsReqOutput
) = LoadOFTAdapterFromDistributedReqOutput = OFTUpdateOutput
```

Add `from sglang.srt.oft.oft_registry import OFTRef` to the file's import block if not already imported (check first with `grep -n "OFTRef" python/sglang/srt/managers/io_struct.py`).

- [ ] **Step 3: Verify it imports cleanly**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -c "from sglang.srt.managers.io_struct import LoadOFTAdapterFromTensorsReqInput, LoadOFTAdapterFromDistributedReqInput, UnloadOFTAdapterReqInput, OFTUpdateOutput; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/managers/io_struct.py
git commit -m "feat(oft): add native adapter-loading wire types mirroring LoRA's"
```

---

### Task 3: `--max-loaded-ofts` flag + validation

**Files:**
- Modify: `python/sglang/srt/peft/config.py`

**Interfaces:**
- Consumes: `OFT_IMPL_CHOICES`, `max_ofts_per_batch`, `validate_peft_args` (all pre-existing in this file).
- Produces: `PEFTArgs.max_loaded_ofts: Optional[int]`, validated in `validate_peft_args`.

- [ ] **Step 1: Read the exact LoRA equivalent**

Read `python/sglang/srt/server_args.py` lines 2932-2936 (`max_loaded_loras` field definition) and lines 9552-9560 (its validation block: `max_loaded_loras >= max_loras_per_batch`, `len(lora_paths) <= max_loaded_loras`).

- [ ] **Step 2: Add the field**

In `python/sglang/srt/peft/config.py`, immediately after the existing `max_ofts_per_batch` field (line ~67):

```python
    max_loaded_ofts: A[
        Optional[int],
        "If specified, limits the maximum number of OFT adapters loaded in "
        "the tokenizer-side registry at a time (CPU-side bookkeeping — "
        "independent of --max-ofts-per-batch's GPU-resident batch capacity). "
        "Must be >= --max-ofts-per-batch.",
        NS("lora"),
    ] = None
```

- [ ] **Step 3: Add the validation**

In `validate_peft_args`, immediately after the existing `assert server_args.max_ofts_per_batch > 0` line (line ~237):

```python
    if server_args.max_loaded_ofts is not None:
        assert server_args.max_loaded_ofts >= server_args.max_ofts_per_batch, (
            "max_loaded_ofts should be greater than or equal to "
            "max_ofts_per_batch. "
            f"max_loaded_ofts={server_args.max_loaded_ofts}, "
            f"max_ofts_per_batch={server_args.max_ofts_per_batch}"
        )
        if server_args.peft_paths:
            assert len(server_args.peft_paths) <= server_args.max_loaded_ofts, (
                "The number of OFT paths should not exceed max_loaded_ofts. "
                f"max_loaded_ofts={server_args.max_loaded_ofts}, "
                f"peft_paths={len(server_args.peft_paths)}"
            )
```

- [ ] **Step 4: Write a unit test**

Add to `test/registered/unit/peft/test_peft_config.py` (check this file exists first — it was referenced in Task 8b's review; append to it rather than creating a new file):

```python
    def test_max_loaded_ofts_must_be_at_least_max_ofts_per_batch(self):
        from types import SimpleNamespace
        from sglang.srt.peft.config import validate_peft_args

        args = SimpleNamespace(
            peft_method="oft",
            oft_impl="sibling",
            max_ofts_per_batch=4,
            max_loaded_ofts=2,
            peft_paths=None,
            enable_lora=False,
            peft_target_modules=["all"],
            oft_type="canonical_oft",
            peft_double_buffer=False,
            max_oft_chunk_size=16,
        )
        with self.assertRaises(AssertionError):
            validate_peft_args(args)
```

(Adjust the `SimpleNamespace` fields to match whatever minimal set `validate_peft_args` actually reads by this point in the file — read the full function body first, since it may reference more fields than listed here; add whichever are missing.)

- [ ] **Step 5: Run the test**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/peft/test_peft_config.py -v -k max_loaded_ofts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/peft/config.py test/registered/unit/peft/test_peft_config.py
git commit -m "feat(oft): add --max-loaded-ofts flag with validation"
```

---

### Task 4: Tokenizer-manager handlers

**Read the `large-class-style` skill before starting this task** (repo rule for `TokenizerManager` edits).

**Files:**
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py`

**Interfaces:**
- Consumes: `tm.peft_registry` (`OFTRegistry`, already constructed in `init_tokenizer_peft`), `tm.peft_ref_cache`, `tm.peft_update_lock` (all pre-existing, currently idle for this purpose); `resolve_or_reuse`/`refresh`/`acquire_with_version` from Task 1; wire types from Task 2; `server_args.max_loaded_ofts` from Task 3.
- Produces: `TokenizerManager.load_oft_adapter_from_tensors`, `.load_oft_adapter_from_distributed`, `.unload_oft_adapter` (all `async def(self, obj, request=None) -> OFTUpdateOutput`); a new `update_oft_adapter_communicator` (via `_COMMUNICATOR_SPECS`).

- [ ] **Step 1: Read the exact LoRA equivalents**

Read `python/sglang/srt/managers/tokenizer_control_mixin.py` lines 106-150 (`_COMMUNICATOR_SPECS` table and `_merge_lora_update_results`), lines 850-1045 (the three LoRA handler methods in full), and `python/sglang/srt/peft/tokenizer_hooks.py` in full (the existing `tm.peft_registry`/`tm.peft_ref_cache`/`tm.peft_update_lock` construction and usage — `init_tokenizer_peft`, `register_peft_ref`, `resolve_peft_path`).

- [ ] **Step 2: Register the new communicator**

Add to `_COMMUNICATOR_SPECS` (immediately after the existing `("update_lora_adapter", LoRAUpdateOutput)` line):

```python
    ("update_oft_adapter", OFTUpdateOutput),
```

- [ ] **Step 3: Add the result-merge helper**

Add immediately after `_merge_lora_update_results`:

```python
def _merge_oft_update_results(results: List[OFTUpdateOutput]) -> OFTUpdateOutput:
    """Merge per-rank replies of an OFT load/unload fan-out into one result.
    Mirrors _merge_lora_update_results exactly: any rank failing wins."""
    failed = [r for r in results if not r.success]
    if not failed:
        return results[0]
    error_messages = list(
        dict.fromkeys(r.error_message for r in failed if r.error_message)
    )
    return OFTUpdateOutput(
        success=False,
        error_message=" | ".join(error_messages),
        loaded_adapters=failed[0].loaded_adapters,
    )
```

- [ ] **Step 4: Add the three handler methods**

Add near the end of `TokenizerControlMixin` (alongside where `register_peft_ref`/`resolve_peft_path` are called from — these new methods are new entry points, separate from those hooks):

```python
    async def load_oft_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (
                self.server_args.peft_method == "oft"
                and self.server_args.oft_impl == "sibling"
            ):
                raise ValueError(
                    "Native OFT adapter loading requires --peft-method oft "
                    "--oft-impl sibling."
                )
            obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
                obj.serialized_named_tensors
            )
            async with self.peft_update_lock:
                new_ref, reused = await self.peft_registry.resolve_or_reuse(
                    obj.to_ref(), upsert=obj.upsert
                )
                obj.adapter_id = new_ref.adapter_id
                results = await self.update_oft_adapter_communicator(obj)
                result = _merge_oft_update_results(results)

                if result.success:
                    if reused:
                        await self.peft_registry.refresh(new_ref)
                    else:
                        await self.peft_registry.register(new_ref)
                    self.peft_ref_cache[obj.adapter_name] = new_ref
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.peft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.peft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.peft_registry.get_all_adapters()}"
                            )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(adapter_name=lru_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter "
                                f"'{lru_name}': {unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_name]
                return result
        except ValueError as e:
            return LoadOFTAdapterFromTensorsReqOutput(
                success=False, error_message=str(e)
            )

    async def load_oft_adapter_from_distributed(
        self: TokenizerManager,
        obj: LoadOFTAdapterFromDistributedReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadOFTAdapterFromDistributedReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (
                self.server_args.peft_method == "oft"
                and self.server_args.oft_impl == "sibling"
            ):
                raise ValueError(
                    "Native OFT adapter loading requires --peft-method oft "
                    "--oft-impl sibling."
                )
            async with self.peft_update_lock:
                new_ref, reused = await self.peft_registry.resolve_or_reuse(
                    obj.to_ref(), upsert=obj.upsert
                )
                obj.adapter_id = new_ref.adapter_id
                result = (await self.update_oft_adapter_communicator(obj))[0]

                if result.success:
                    if reused:
                        await self.peft_registry.refresh(new_ref)
                    else:
                        await self.peft_registry.register(new_ref)
                    self.peft_ref_cache[obj.adapter_name] = new_ref
                if self.server_args.max_loaded_ofts is not None:
                    while (
                        self.peft_registry.num_registered_ofts
                        > self.server_args.max_loaded_ofts
                    ):
                        lru_name = await self.peft_registry.lru_oft_name(
                            exclude_pinned=True
                        )
                        if lru_name is None:
                            raise ValueError(
                                "Didn't find any OFT adapters when trying to "
                                "evict LRU OFT adapter. OFT registry is: "
                                f"{self.peft_registry.get_all_adapters()}"
                            )
                        unload_result = await self._unload_oft_adapter_locked(
                            UnloadOFTAdapterReqInput(adapter_name=lru_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU OFT adapter "
                                f"'{lru_name}': {unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_name]
                return result
        except ValueError as e:
            return LoadOFTAdapterFromDistributedReqOutput(
                success=False, error_message=str(e)
            )

    async def _unload_oft_adapter_locked(
        self: TokenizerManager, obj: UnloadOFTAdapterReqInput
    ) -> UnloadOFTAdapterReqOutput:
        """Caller must hold peft_update_lock. Unregisters + tells the
        scheduler to free GPU state; does NOT touch peft_ref_cache (the
        caller decides evict-vs-delete semantics, mirroring
        _unload_lora_adapter_locked)."""
        adapter_id = await self.peft_registry.unregister(obj.adapter_name)
        obj.adapter_id = adapter_id
        result = (await self.update_oft_adapter_communicator(obj))[0]
        await self.peft_registry.wait_for_unload(adapter_id)
        return result

    async def unload_oft_adapter(
        self: TokenizerManager,
        obj: UnloadOFTAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadOFTAdapterReqOutput:
        self.auto_create_handle_loop()
        try:
            if not (
                self.server_args.peft_method == "oft"
                and self.server_args.oft_impl == "sibling"
            ):
                raise ValueError(
                    "Native OFT adapter loading requires --peft-method oft "
                    "--oft-impl sibling."
                )
            async with self.peft_update_lock:
                result = await self._unload_oft_adapter_locked(obj)
                # Explicit unload is a DELETE: drop the ref_cache entry too
                # (mirrors unload_lora_adapter's explicit-vs-evict distinction).
                if result.success:
                    self.peft_ref_cache.pop(obj.adapter_name, None)
                return result
        except ValueError as e:
            return UnloadOFTAdapterReqOutput(success=False, error_message=str(e))
```

Add the necessary imports (`LoadOFTAdapterFromTensorsReqInput`, `LoadOFTAdapterFromTensorsReqOutput`, `LoadOFTAdapterFromDistributedReqInput`, `LoadOFTAdapterFromDistributedReqOutput`, `UnloadOFTAdapterReqInput`, `UnloadOFTAdapterReqOutput`, `OFTUpdateOutput`) to this file's existing `io_struct` import block.

- [ ] **Step 5: Fix `resolve_peft_path`'s reload guard for non-reloadable refs**

In `python/sglang/srt/peft/tokenizer_hooks.py`'s `resolve_peft_path`, the block that implicitly reloads dynamically-evicted adapters (`for adapter_path in unregistered: ...`) currently assumes every ref has a real disk path. Add a guard before the reload attempt:

```python
        if adapter_path not in tm.peft_ref_cache:
            raise ValueError(
                f"Got PEFT adapter that has never been loaded: {adapter_path}\n"
                f"All loaded adapters: {tm.peft_ref_cache.keys()}."
            )
        ref = tm.peft_ref_cache[adapter_path]
        if not ref.reloadable:
            raise ValueError(
                f"OFT adapter '{adapter_path}' was loaded dynamically (via "
                "tensors/distributed) and was evicted from the registry; it "
                "has no on-disk artifact to reload from and must be "
                "re-loaded via a fresh load_oft_adapter_from_tensors/"
                "_from_distributed call."
            )
        if tm.peft_kind == "oft":
```

(This replaces the existing `if tm.peft_kind == "oft":` check — insert the new `ref`/`reloadable` guard between the existing `adapter_path not in tm.peft_ref_cache` check and the existing `if tm.peft_kind == "oft":` line; read the surrounding function body first since the existing code already assigns `ref = tm.peft_ref_cache[adapter_path]` slightly later — reuse that assignment rather than duplicating it, adjust order accordingly.)

- [ ] **Step 5b: Close `resolve_peft_path`'s id+version TOCTOU gap (per spec, "AdapterRegistry extension" section)**

`resolve_peft_path` currently resolves the adapter id and version as two
separate calls under two separate lock acquisitions:

```python
    adapter_id = await tm.peft_registry.acquire(path)
    ...
    adapter_version = await tm.peft_registry.get_version_by_id(adapter_id)
```

(`python/sglang/srt/peft/tokenizer_hooks.py:156` and `:159`, with the
`obj.adapter_id = adapter_id` / `_propagate_id_to_cached_sub_objs` calls
in between). Replace both calls with Task 1's new atomic primitive:

```python
    adapter_id, adapter_version = await tm.peft_registry.acquire_with_version(path)
```

placed where the original `acquire()` call was; delete the now-redundant
`get_version_by_id` call. Everything else in the function (the
`obj.adapter_id = adapter_id` assignment, both
`_propagate_id_to_cached_sub_objs` calls) stays unchanged — only the
resolution of `adapter_id`/`adapter_version` themselves changes from two
non-atomic calls to one atomic call.

- [ ] **Step 6: Write handler unit tests with a mocked communicator**

Create `test/registered/unit/managers/test_oft_native_handlers.py` mirroring whatever test file (if any) covers `load_lora_adapter_from_tensors`'s handler logic with a mocked `TokenizerManager` — search first: `grep -rln "load_lora_adapter_from_tensors" test/registered/unit/` — if one exists, model this file on it directly (same mocking approach for `update_oft_adapter_communicator`, `peft_registry`, `peft_ref_cache`, `peft_update_lock`); if none exists, write a minimal `unittest.IsolatedAsyncioTestCase` that constructs a bare object with just the attributes these methods read (`server_args`, `peft_registry`, `peft_ref_cache`, `peft_update_lock`, `update_oft_adapter_communicator` as an `AsyncMock`), covering: fresh load succeeds and registers; upsert reuses id; LRU eviction fires when `max_loaded_ofts` exceeded; wrong `peft_method`/`oft_impl` rejects with a clear error.

- [ ] **Step 7: Run the new tests**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/managers/test_oft_native_handlers.py -v`
Expected: PASS.

- [ ] **Step 8: Run the broader regression sweep**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/adapter_sync/ test/registered/unit/peft/ test/registered/unit/lora/ -q`
Expected: same pass/fail counts as this plan's Task 1/3 baseline (8 pre-existing unrelated LoRA isolation failures).

- [ ] **Step 9: Commit**

```bash
git add python/sglang/srt/managers/tokenizer_control_mixin.py python/sglang/srt/peft/tokenizer_hooks.py test/registered/unit/managers/test_oft_native_handlers.py
git commit -m "feat(oft): add native load_oft_adapter_from_tensors/_from_distributed/unload handlers"
```

---

### Task 5: Scheduler dispatch + `tp_worker` forwarding

**Read the `large-class-style` skill before starting this task.**

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/tp_worker.py`

**Interfaces:**
- Consumes: wire types from Task 2; `ModelRunner.load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter` from Task 6 (forward reference — write this task's dispatch first, Task 6 supplies what it calls into; the two can be developed in either order since they only share the method *names*, not runtime coupling until both exist).
- Produces: `Scheduler.load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter`; `TpModelWorker.load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter`.

- [ ] **Step 1: Read the exact LoRA equivalents**

Read `python/sglang/srt/managers/scheduler.py` lines 4964-4985 (the three LoRA scheduler methods and their dispatch-table registration around line 1651-1656) and `python/sglang/srt/managers/tp_worker.py` lines 235-305 (the three LoRA `tp_worker` methods in full, including the `expected_checksums` sha256 verification block and the distributed-broadcast receive loop).

- [ ] **Step 2: Add scheduler dispatch entries**

In `scheduler.py`'s dispatch table (near the existing `LoadLoRAAdapterFromTensorsReqInput`/`LoadLoRAAdapterFromDistributedReqInput` entries around line 1651-1656), add:

```python
                    LoadOFTAdapterFromTensorsReqInput,
                    ...
                    LoadOFTAdapterFromDistributedReqInput,
```

(Match the exact tuple/list shape the existing LoRA entries use — read the surrounding 10 lines to see whether it's a list of `(type, handler_name)` pairs or something else, and mirror precisely.) Add the corresponding import lines alongside the existing `LoadLoRAAdapterFromTensorsReqInput` import block.

- [ ] **Step 3: Add scheduler handler methods**

Add near the existing `load_lora_adapter_from_tensors`/`load_lora_adapter_from_distributed` methods (around line 4964):

```python
    def load_oft_adapter_from_tensors(
        self, recv_req: LoadOFTAdapterFromTensorsReqInput
    ) -> LoadOFTAdapterFromTensorsReqOutput:
        """In-place loading a new OFT adapter from serialized tensors."""
        result = self.tp_worker.load_oft_adapter_from_tensors(recv_req)
        return result

    def load_oft_adapter_from_distributed(
        self, recv_req: LoadOFTAdapterFromDistributedReqInput
    ) -> LoadOFTAdapterFromDistributedReqOutput:
        """In-place loading a new OFT adapter broadcast over a process group."""
        result = self.tp_worker.load_oft_adapter_from_distributed(recv_req)
        return result

    def unload_oft_adapter(
        self, recv_req: UnloadOFTAdapterReqInput
    ) -> UnloadOFTAdapterReqOutput:
        result = self.tp_worker.unload_oft_adapter(recv_req)
        return result
```

- [ ] **Step 4: Add `tp_worker` methods**

Add near the existing `unload_lora_adapter`/`load_lora_adapter_from_tensors`/`load_lora_adapter_from_distributed` methods in `tp_worker.py` (around line 240):

```python
    def unload_oft_adapter(self, recv_req: UnloadOFTAdapterReqInput):
        result = self.model_runner.unload_oft_adapter(recv_req.to_ref())
        return result

    def load_oft_adapter_from_tensors(
        self, recv_req: LoadOFTAdapterFromTensorsReqInput
    ):
        data = self._deserialize_own_rank(recv_req.serialized_named_tensors)
        result = self.model_runner.load_oft_adapter_from_tensors(
            recv_req.to_ref(),
            data,
            recv_req.config_dict,
            upsert=recv_req.upsert,
        )
        return result

    def load_oft_adapter_from_distributed(
        self, recv_req: LoadOFTAdapterFromDistributedReqInput
    ):
        result = self.model_runner.load_oft_adapter_from_distributed(
            recv_req.to_ref(),
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.config_dict,
            recv_req.group_name,
            upsert=recv_req.upsert,
        )
        return result
```

(This intentionally omits the `expected_checksums`/sha256 verification block LoRA's `load_lora_adapter_from_tensors` has and the `flattened_bucket` branch — neither is part of the spec's scope; if a future need arises, add them then rather than speculatively now.)

Add the necessary imports to both files.

- [ ] **Step 5: Verify imports and dispatch wiring**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -c "import sglang.srt.managers.scheduler; import sglang.srt.managers.tp_worker; print('ok')"`
Expected: `ok` (no import errors; this doesn't exercise runtime dispatch, just confirms the new code parses and imports cleanly — full dispatch is exercised in Task 8's GPU tests).

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/tp_worker.py
git commit -m "feat(oft): wire native adapter RPC dispatch through scheduler/tp_worker"
```

---

### Task 6: `OFTManager`/`ModelRunner` GPU-side admission

**Read the `large-class-style` skill before starting this task** (this task edits `model_runner.py`, a frozen core file per repo rules).

**Files:**
- Modify: `python/sglang/srt/oft/oft_manager.py`
- Modify: `python/sglang/srt/oft/base/manager.py` (docstring fix only)
- Modify: `python/sglang/srt/model_executor/model_runner.py`

**Interfaces:**
- Consumes: `OFTRef` (existing), `_ensure_streaming_oft_adapter_slot`'s primitives (`memory_pool.allocate_buffer_slot`, `register_streamed_adapter`, `unload_streamed_adapter` — read `python/sglang/srt/oft/streamed_weight_loader.py:308-386` for their exact signatures before writing).
- Produces: `OFTManager.load_adapter_from_tensors(ref, tensors, config_dict, upsert=False)`, `OFTManager.load_adapter_from_distributed(ref, names, dtypes, shapes, config_dict, group_name, upsert=False)`, `OFTManager.unload_adapter(ref)` (may already exist via `AdapterManager.unload_adapter` — check first); `ModelRunner.load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter`, mirroring `ModelRunner`'s LoRA methods at lines 1287-1366.

- [ ] **Step 1: Read the exact LoRA equivalents and the existing streamed-loader primitives**

Read `python/sglang/srt/model_executor/model_runner.py` lines 1283-1366 in full (`load_lora_adapter`, `load_lora_adapter_from_tensors`, `load_lora_adapter_from_distributed`, `unload_lora_adapter`) and `python/sglang/srt/oft/streamed_weight_loader.py` lines 308-386 (`_ensure_streaming_oft_adapter_slot`, `load_streamed_oft_adapter`) — the new `OFTManager` methods reuse the same primitives this function calls, but drop its single-active `other_adapters` rejection (the whole point of this plan).

- [ ] **Step 2: Add `OFTManager.load_adapter_from_tensors`/`_from_distributed`**

Add to `python/sglang/srt/oft/oft_manager.py`, near the existing streaming-adapter methods:

```python
    def load_adapter_from_tensors(self, ref, named_tensors, config_dict, *, upsert=False):
        """Native-RPC admission path: like _ensure_streaming_oft_adapter_slot,
        but multi-tenant (no single-active restriction) since this serves
        the new native load_oft_adapter_from_tensors RPC, not the legacy
        srt/peft streamed path. Capacity is still bounded by
        max_ofts_per_batch via the memory pool's own admission."""
        try:
            existing_id = None
            for ref_id, existing_ref in list(self.refs.items()):
                if existing_ref.adapter_name == ref.adapter_name:
                    existing_id = ref_id
                    break
            if existing_id is not None:
                if not upsert:
                    return self._make_update_result(
                        success=False,
                        error_message=(
                            f"OFT adapter '{ref.adapter_name}' is already "
                            "loaded; pass upsert=True to refresh it in place."
                        ),
                    )
                self.unload_streamed_adapter(self.refs[existing_id])
            buffer_id = self.memory_pool.allocate_buffer_slot()
            self.memory_pool.reset_buffer_slot_to_identity(buffer_id)
            result = self.register_streamed_adapter(ref, buffer_id, config_dict)
            if not result.success:
                return result
            self._load_streamed_weights(named_tensors, buffer_id, config_dict)
        except Exception as e:
            return self._make_update_result(success=False, error_message=str(e))
        return self._make_update_result(success=True)

    def load_adapter_from_distributed(
        self, ref, names, dtypes, shapes, config_dict, group_name, *, upsert=False
    ):
        """Receives the adapter's tensors over the process group, then
        delegates to load_adapter_from_tensors for admission."""
        update_groups = self.weight_updater._model_update_group
        assert group_name in update_groups, (
            f"Group {group_name} not in {list(update_groups.keys())}. "
            "Please call `init_weights_update_group` first."
        )
        try:
            tensors = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight, src=0, group=update_groups[group_name], async_op=True
                    )
                )
                tensors.append((name, weight))
            for handle in handles:
                handle.wait()
        except Exception as e:
            return self._make_update_result(
                success=False,
                error_message=f"Failed to receive OFT adapter weights: {e}.",
            )
        return self.load_adapter_from_tensors(
            ref, tensors, config_dict, upsert=upsert
        )
```

Before finalizing this step, verify the exact names of the primitives referenced here (`self.memory_pool.allocate_buffer_slot`, `self.memory_pool.reset_buffer_slot_to_identity`, `self.register_streamed_adapter`, `self.unload_streamed_adapter`, and whatever internal helper actually writes tensors into the buffer — `_load_streamed_weights` above is a placeholder name; find the real one by reading how `load_streamed_oft_adapter` in `streamed_weight_loader.py` does the actual tensor-to-buffer write after calling `_ensure_streaming_oft_adapter_slot`, past line 386, and call that same logic/helper here instead of inventing a new name) — **do not leave `_load_streamed_weights` as a stub; replace it with the real call** before moving to Step 3.

Confirm whether `OFTManager` already has an `unload_adapter` suitable for a dynamically-loaded (streamed, `ref.adapter_path in ("__tensor__", "__distributed__")`) ref via the inherited `AdapterManager.unload_adapter` — its body branches on `ref.adapter_id not in self.adapters` to call `self._unload_streamed_adapter(stored_ref)`, which should already cover this case; if so, no new `OFTManager`-level unload method is needed beyond what `AdapterManager` already provides.

- [ ] **Step 3: Fix `AdapterManager`'s stale docstring**

In `python/sglang/srt/oft/base/manager.py`, update the class docstring:

```python
class AdapterManager:
    """Generic lifecycle/utility methods for adapter managers. Originally
    intended to be shared with LoRA (several method docstrings below still
    describe LoRA-specific behavior), but LoRAManager evolved independently
    and does not subclass this — today only OFTManager does."""
```

Leave the per-method docstrings that describe both LoRA and OFT behavior (e.g. `_make_streamed_ref`'s) as-is — they're historically accurate design notes explaining *why* the hook signature is shaped the way it is, not claims about current class relationships.

- [ ] **Step 4: Add `ModelRunner` forwarding methods**

Add to `python/sglang/srt/model_executor/model_runner.py`, immediately after the existing `unload_lora_adapter` method (after line 1366):

```python
    def load_oft_adapter_from_tensors(
        self, oft_ref, tensors, config_dict, *, upsert: bool = False
    ):
        logger.info(f"OFT adapter loading from tensors starts: {oft_ref}.")
        result = self.oft_manager.load_adapter_from_tensors(
            oft_ref, tensors, config_dict, upsert=upsert
        )
        logger.info(f"OFT adapter loading from tensors completes: {oft_ref}.")
        return result

    def load_oft_adapter_from_distributed(
        self, oft_ref, names, dtypes, shapes, config_dict, group_name, *, upsert: bool = False
    ):
        logger.info(f"OFT adapter loading from distributed starts: {oft_ref}.")
        result = self.oft_manager.load_adapter_from_distributed(
            oft_ref, names, dtypes, shapes, config_dict, group_name, upsert=upsert
        )
        logger.info(f"OFT adapter loading from distributed completes: {oft_ref}.")
        return result

    def unload_oft_adapter(self, oft_ref):
        return self.oft_manager.unload_adapter(oft_ref)
```

- [ ] **Step 5: Verify imports**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -c "import sglang.srt.oft.oft_manager; import sglang.srt.model_executor.model_runner; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/oft/oft_manager.py python/sglang/srt/oft/base/manager.py python/sglang/srt/model_executor/model_runner.py
git commit -m "feat(oft): add multi-tenant native adapter admission to OFTManager/ModelRunner"
```

---

### Task 7: Engine + HTTP surface

**Files:**
- Modify: `python/sglang/srt/entrypoints/engine.py`
- Modify: `python/sglang/srt/entrypoints/http_server.py`

**Interfaces:**
- Consumes: wire types from Task 2; `tokenizer_manager.load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter` from Task 4.
- Produces: `Engine.load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter`; `POST /load_oft_adapter_from_tensors`, `/load_oft_adapter_from_distributed`, `/unload_oft_adapter`.

- [ ] **Step 1: Read the exact LoRA equivalents**

Read `python/sglang/srt/entrypoints/engine.py` lines 1542-1631 (the four LoRA Engine methods) and `python/sglang/srt/entrypoints/http_server.py` lines 1661-1705 (the four LoRA HTTP routes).

- [ ] **Step 2: Add Engine methods**

Add to `engine.py`, immediately after the existing `unload_lora_adapter`/`async_unload_lora_adapter` methods:

```python
    def load_oft_adapter_from_tensors(
        self,
        adapter_name: str,
        tensors: Union[Dict[str, torch.Tensor], List[SerializedTensorPayload]],
        config_dict: Dict,
        load_format: Optional[str] = None,
    ):
        serialized_named_tensors = self._serialize_tensors_per_rank(
            tensors, load_format
        )
        req = LoadOFTAdapterFromTensorsReqInput(
            adapter_name=adapter_name,
            config_dict=config_dict,
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.load_oft_adapter_from_tensors(req, None)
        )

    def load_oft_adapter_from_distributed(
        self,
        adapter_name: str,
        config_dict: Dict,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        pinned: bool = False,
    ):
        """Load a new OFT adapter whose weights are broadcast over a
        process group. The weight-update group must already be initialized
        via `init_weights_update_group`."""
        req = LoadOFTAdapterFromDistributedReqInput(
            adapter_name=adapter_name,
            config_dict=config_dict,
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            pinned=pinned,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.load_oft_adapter_from_distributed(req, None)
        )

    def unload_oft_adapter(self, adapter_name: str):
        """Unload an OFT adapter without re-launching the engine."""
        obj = UnloadOFTAdapterReqInput(adapter_name=adapter_name)
        return self.loop.run_until_complete(
            self.tokenizer_manager.unload_oft_adapter(obj, None)
        )
```

(Omitting the disk-based `load_oft_adapter`/`async_*` variants and `added_tokens_config` — the former already exists via the existing `--peft-paths` static-loading mechanism, per this plan's non-goals.)

- [ ] **Step 3: Add HTTP routes**

Add to `http_server.py`, immediately after the existing `/unload_lora_adapter` route:

```python
@app.api_route("/load_oft_adapter_from_tensors", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def load_oft_adapter_from_tensors(
    obj: Annotated[LoadOFTAdapterFromTensorsReqInput, Body()], request: Request
):
    """Load a new OFT adapter from tensors without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_oft_adapter_from_tensors(
        obj, request
    )
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/load_oft_adapter_from_distributed", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def load_oft_adapter_from_distributed(
    obj: Annotated[LoadOFTAdapterFromDistributedReqInput, Body()], request: Request
):
    """Load a new OFT adapter broadcast over a process group without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_oft_adapter_from_distributed(
        obj, request
    )
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)


@app.api_route("/unload_oft_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def unload_oft_adapter(
    obj: Annotated[UnloadOFTAdapterReqInput, Body()], request: Request
):
    """Unload an OFT adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.unload_oft_adapter(obj, request)
    status_code = HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST
    return ORJSONResponse(msgspec_to_builtins(result), status_code=status_code)
```

Add the necessary imports to both files.

- [ ] **Step 4: Verify imports**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -c "import sglang.srt.entrypoints.engine; import sglang.srt.entrypoints.http_server; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/entrypoints/engine.py python/sglang/srt/entrypoints/http_server.py
git commit -m "feat(oft): add Engine methods and HTTP routes for native adapter RPC"
```

---

### Task 8: GPU integration tests (GPU gate)

**Files:**
- Test: `test/registered/rl/test_oft_load_from_tensor.py` (new)

**Interfaces:**
- Consumes: everything from Tasks 1-7, end to end.
- Produces: GPU-verified evidence this whole mechanism works, gating Task 9.

- [ ] **Step 1: Read the reference file**

Read `test/registered/rl/test_lora_load_from_tensor.py` in full — this is the direct template. Also read `test/registered/lora/test_oft_staged_update.py`'s `_oft_named_tensors`/`_write_local_oft_adapter`/`_adapter_config_dict` helpers (BLOCK_SIZE=32, TARGET_MODULE="down_proj", MODEL_PATH="Qwen/Qwen3-0.6B") for synthesizing OFT tensors, since (per that file's own docstring) no reusable small real OFT HF adapter repo exists.

- [ ] **Step 2: Write the test file**

Create `test/registered/rl/test_oft_load_from_tensor.py` covering, using `sgl.Engine` booted with `peft_method="oft", oft_impl="sibling", max_oft_block_size=32, peft_target_modules=["down_proj"], max_loaded_ofts=4, max_ofts_per_batch=4`:

1. `test_fresh_load_and_generate` — `engine.load_oft_adapter_from_tensors(...)` succeeds; `engine.generate(..., adapter_path="name")` (confirm the exact per-request field name by reading `GenerateReqInput`/`_request_peft_path` first, per this plan's Task 4 note — do not assume `lora_path`) produces output without crashing.
2. `test_upsert_refresh` — load once, then load again with `upsert=True` and different tensor values; assert success and that the adapter is still resolvable/generatable afterward.
3. `test_lru_eviction_past_max_loaded_ofts` — load `max_loaded_ofts + 1` distinctly-named adapters; assert the registry never exceeds `max_loaded_ofts` and the least-recently-used one was evicted (check via `engine.tokenizer_manager.peft_registry.get_all_adapters()` or an equivalent introspection point — find the right one by checking what `test_lora_load_from_tensor.py`'s own LRU eviction test inspects).
4. `test_multi_adapter_concurrent_residency` — load 2 distinctly-named adapters (within `max_ofts_per_batch`) without unloading either; issue `/generate` requests against both in the same test and confirm both succeed (this is the capability newly unlocked by Task 6's single-active-restriction removal — if this fails, that's a Task 6 bug, not a test-design issue, so investigate the actual admission path before assuming test error).

- [ ] **Step 3: Run on real GPU**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/rl/test_oft_load_from_tensor.py -v`
Expected: all 4 tests PASS on real GPU hardware. If any fails, treat it as a genuine bug in Tasks 1-7's implementation (fix the implementation, not the test) unless the test itself is provably wrong (e.g. wrong field name) — in which case fix the test and re-run, documenting which.

- [ ] **Step 4: Register for CI**

Add `register_cuda_ci(est_time=..., stage=..., runner_config="1-gpu-large")` at the top of the file, matching `test_lora_load_from_tensor.py`'s registration line exactly in shape (adjust `est_time` based on actual observed runtime from Step 3).

- [ ] **Step 5: Commit**

```bash
git add test/registered/rl/test_oft_load_from_tensor.py
git commit -m "test(oft): add GPU integration tests for native adapter RPC"
```

---

### Task 9: Retire the old shared `srt/peft` streamed-loader mechanism (FINAL — gated on Task 8)

**Global Constraint reminder: this task only proceeds after Task 8's GPU tests pass and this plan's own review confirms the new path is solid. Do not start this task speculatively.**

**Files:**
- Modify: `python/sglang/srt/peft/integration.py` (remove `maybe_load_adapter_format`'s `oft_adapter` branch, `reconstruct_oft_staging`'s OFT-only callers if now unused — check first)
- Modify: `python/sglang/srt/oft/streamed_weight_loader.py` (remove `load_streamed_oft_adapter`, `_ensure_streaming_oft_adapter_slot` if no longer called anywhere)
- Modify: `python/sglang/srt/model_executor/model_runner_components/weight_updater.py` (remove the `update_weights_from_tensor` fallback branch that called into the now-deleted mechanism, once confirmed no longer reachable)

**Context for whoever picks up this task:** separately from this plan, a critical bug was found and fixed in the OLD mechanism during this plan's motivating investigation — `_ensure_streaming_oft_adapter_slot`'s `ValueError` guards were propagating uncaught past a `try/except RuntimeError`-only block, crashing the entire engine process on a malformed streamed update. That fix already landed independently; it is not blocking this task, but it's good context for why replacing this mechanism was worth doing at all — the old path's failure mode wasn't just "less capable," it was "actively dangerous."

- [ ] **Step 1: Grep for every remaining reference to the old mechanism**

Run: `cd /workspace/sglang-spherelab && grep -rn "load_streamed_oft_adapter\|_ensure_streaming_oft_adapter_slot\|maybe_load_adapter_format" python/sglang/ test/`

- [ ] **Step 2: Check for external callers before deleting anything**

Per the spec's own open question: grep any deployment/orchestration code outside this repo that might call the OLD wire shape (`update_weights_from_tensor` with `load_format="oft_adapter"` + `adapter_config`/`adapter_name`/`adapter_id` fields) directly — if such an external caller exists (e.g. an RL training harness), this task needs a coordinated cutover, not a silent deletion. Flag this explicitly rather than assuming it's safe.

- [ ] **Step 3: Delete the old mechanism**

Following the same rigor as the prior plan's Task 8a/8b (verify byte-identical or safely-superseded before deleting, run the full test suite before and after, confirm no remaining references).

- [ ] **Step 4: Run the full regression suite**

Run: `cd /workspace/sglang-spherelab && PYTHONPATH=python python3 -m pytest test/registered/unit/ -k "oft or peft or adapter_sync" -q --ignore=test/registered/unit/mem_cache/test_umbp_store.py`

- [ ] **Step 5: Commit**

```bash
git commit -m "cleanup: retire srt/peft's shared OFT streamed-loader mechanism, superseded by native RPC"
```

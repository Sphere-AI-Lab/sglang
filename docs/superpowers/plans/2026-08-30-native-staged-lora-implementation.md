# Native Staged LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional two-phase stage/activate workflow to the corrected branch's native `--enable-lora` stack while preserving the existing one-step LoRA APIs and the existing OFT implementation.

**Architecture:** A staging-specific `LoRAManager` subclass builds adapters through the corrected branch's native tensor loader and places them in one hidden GPU slot owned by a staging-specific `LoRAMemoryPool` subclass. The tokenizer manager publishes a new `LoRARef.version` only after all scheduler workers activate successfully; request admission snapshots the ID and version together, and the radix key contains both. If workers disagree during activation, the tokenizer quarantines that adapter before releasing its writer lock, so the server fails closed for that adapter while base and unrelated adapters remain available.

**Tech Stack:** Python 3.12, PyTorch 2.11/CUDA 13, `msgspec`, `asyncio`, SGLang native LoRA, NCCL, `unittest`/`pytest`, Slurm

**Spec:** `docs/superpowers/specs/2026-08-30-native-staged-lora-design.md`

## Global Constraints

- Source behavior is pinned at `32af48d8df1e1999136a83f6993e25b43a9bfb24`.
- Target base is pinned at `1c15111fba954babe2b9742caefcbedb22ce306c`.
- Work only in `/data/home/zeju/miles-orbit-dev/sphere-lab/sglang/.worktrees/staged-lora-native` on `codex/port-staged-lora-native`.
- Native staging is enabled only by `--enable-lora --enable-lora-staging`; ordinary `--enable-lora` behavior remains unchanged.
- Keep the existing one-step native LoRA load/upsert API and its current cache-flush behavior.
- Do not restore `--peft-method lora` or `--lora-impl sibling` as the native serving interface.
- Do not change the OFT manager, OFT memory pool, or OFT stage/activate routing.
- Allocate exactly one hidden physical LoRA slot beyond `max_loras_per_batch`; never expose that slot to routing, LRU eviction, or batch metadata.
- Permit at most one outstanding staged `(lora_id, version)` per model runner.
- Stage must not mutate `configs`, `loras`, `lora_refs`, pinned accounting, serving-slot maps, registry visibility, or active versions.
- A same-name staged refresh must preserve the existing adapter's `pinned` value; a first-time streamed adapter is unpinned.
- Publish a tokenizer-registry version only after every worker reports activation success.
- Aggregate activation failure must quarantine the affected adapter before admission resumes. Quarantined adapter requests fail with a restart-required error; base and unrelated adapters remain admissible.
- Append `lora_version` as the final declared field of each array-like tokenizer IPC struct. Never insert it beside `lora_id`, because that would shift older positional payloads.
- A recoverable resident-slot activation failure must restore the previous adapter with the native placement routine; a failed restore is a hard worker-restart error.
- GPU checks run under Slurm, never on the login node, and use the existing shared environment `/data/home/zeju/miles-orbit-dev/envs/candidate` with worktree source supplied through `PYTHONPATH`.
- Preserve all unrelated branch changes and do not delete remote files, jobs, worktrees, or sessions.

## Pre-implementation CPU baseline

Recorded on 2026-08-30 before Task 1 source or test changes. The bounded baseline command was:

```bash
env PYTHONPATH="$PWD/python" \
  /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest -q \
  test/registered/unit/lora/test_lora_lease.py \
  test/registered/unit/managers/test_msgpack_ipc_roundtrip.py
```

Result: `3 failed, 20 passed, 16 subtests passed` in 61.48 seconds. The three failures already exist at target base `1c15111fba954babe2b9742caefcbedb22ce306c`:

1. `TestUnloadRefCacheCleanup.test_explicit_unload_drops_ref_cache_entry`
2. `TestUnloadRefCacheCleanup.test_failed_unload_keeps_ref_cache_entry`

   Both reach `tokenizer_control_mixin.py:1024` without a published parallel runtime configuration and fail because `ParallelContext` has no `dp_size` attribute.

3. `TestMsgpackIpcRoundtrip.test_check_weights_mirrors_match_pydantic_models`

   The existing `ParallelismInfo` IPC model contains `role`, while its Pydantic mirror does not.

These failures are outside the staged-LoRA port and are not part of Tasks 1-5. During implementation, the regression gate is: all new staged-LoRA tests pass, all previously passing focused tests remain passing, and any full-suite rerun has exactly this same three-test failure set with no new failure. Fixing the baseline failures requires separate scope and evidence.

---

## File map

| File | Responsibility |
|---|---|
| `python/sglang/srt/lora/lora_registry.py` | Store active LoRA version and atomically acquire `(lora_id, version)` with the request lease. |
| `python/sglang/srt/managers/io_struct.py` | Carry `lora_version` through batched input and tokenizer-to-scheduler IPC. |
| `python/sglang/srt/managers/tokenizer_manager.py` | Resolve and propagate native LoRA ID/version at request admission. |
| `python/sglang/srt/managers/schedule_batch.py` | Add native LoRA identity/version to the radix-cache key. |
| `python/sglang/srt/managers/scheduler.py` | Pass `lora_version` from tokenized requests into `Req`. |
| `python/sglang/srt/adapter_sync/backends/lora.py` | Implement the native hidden-slot pool and two-phase manager using current native placement. |
| `python/sglang/srt/server_args.py` | Define and validate `--enable-lora-staging`. |
| `python/sglang/srt/model_executor/model_runner.py` | Select `StagedLoRAManager` only when the new flag is enabled. |
| `python/sglang/srt/model_executor/model_runner_components/weight_updater.py` | Route native LoRA stage/activate before the existing PEFT fallback. |
| `python/sglang/srt/managers/tp_worker.py` | Preserve `adapter_id` through the activation call. |
| `python/sglang/srt/managers/tokenizer_control_mixin.py` | Reserve staged native identity, publish only after aggregate success, and quarantine partial activation failures. |
| `test/registered/unit/lora/test_lora_versioning.py` | Protect wire compatibility, atomic admission, propagation, and cache isolation. |
| `test/registered/unit/adapter_sync/test_lora_staging_backend.py` | Protect hidden-slot allocation, native placement, idempotency, conflict, activation, and rollback. |
| `test/registered/unit/lora/test_lora_staging_control.py` | Protect CLI selection, worker routing, and tokenizer-side publication rules. |
| `test/registered/lora/test_lora_staged_update.py` | Exercise a TP=1 server plus a separate NCCL trainer rank, cache isolation, multi-tenancy, slot pressure, and CUDA graphs on a two-GPU runner. |
| `test/registered/lora/test_lora_staged_update_tp.py` | Exercise a TP=2 server plus a separate trainer rank and MoE sharded placement on a four-GPU CI runner. |

### Task 1: Version native LoRA request identity and radix keys

**Files:**
- Modify: `python/sglang/srt/lora/lora_registry.py:25-220`
- Modify: `python/sglang/srt/managers/io_struct.py:240-260, 880-915, 959-1020, 1125-1360`
- Modify: `python/sglang/srt/managers/tokenizer_manager.py:1435-1480, 3420-3510`
- Modify: `python/sglang/srt/managers/schedule_batch.py:807-960`
- Modify: `python/sglang/srt/managers/scheduler.py:2440-2460, 2900-2920`
- Create: `test/registered/unit/lora/test_lora_versioning.py`

**Interfaces:**
- Produces: `LoRARef.version: int = 0` as the final array-like field.
- Produces: `LoRARegistry.acquire_with_version(name) -> (lora_id, version)` for a scalar and parallel ID/version lists for a list.
- Produces: `GenerateReqInput.lora_version`, `EmbeddingReqInput.lora_version`, `TokenizedGenerateReqInput.lora_version`, and `TokenizedEmbeddingReqInput.lora_version`.
- Produces: `Req.lora_version` and `_extend_lora_extra_key(extra_key, lora_id, lora_version)`.
- Preserves: `LoRARegistry.acquire()` returns only IDs for existing callers.

- [ ] **Step 1: Write failing registry and wire-compatibility tests**

Create `test_lora_versioning.py` with CPU registration and these concrete cases:

```python
import asyncio
import unittest

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import _extend_lora_extra_key


class TestLoRARefVersion(CustomTestCase):
    def test_old_array_payload_defaults_version_to_zero(self):
        old = ["id-a", "adapter-a", "/adapter/a", False, True]
        decoded = msgspec.json.decode(
            msgspec.json.encode(old), type=LoRARef
        )
        self.assertEqual(decoded.version, 0)

    def test_acquire_snapshots_id_and_version_under_one_lock(self):
        registry = LoRARegistry()
        ref = LoRARef(
            lora_id="id-a",
            lora_name="adapter-a",
            lora_path="/adapter/a",
            pinned=False,
            version=7,
        )
        asyncio.run(registry.register(ref))
        self.assertEqual(
            asyncio.run(registry.acquire_with_version("adapter-a")),
            ("id-a", 7),
        )
        asyncio.run(registry.release("id-a"))

    def test_tokenized_version_fields_are_wire_compatible_trailing_fields(self):
        for req_type in (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput):
            self.assertEqual(msgspec.structs.fields(req_type)[-1].name, "lora_version")


class TestLoRARadixIdentity(CustomTestCase):
    def test_base_key_is_unchanged(self):
        self.assertEqual(_extend_lora_extra_key("tenant", None, None), "tenant")

    def test_lora_key_contains_id_and_version(self):
        self.assertEqual(
            _extend_lora_extra_key("tenant", "id-a", 3),
            "tenant|lora:id-a:v3",
        )

    def test_versions_never_share_a_key(self):
        self.assertNotEqual(
            _extend_lora_extra_key(None, "id-a", 3),
            _extend_lora_extra_key(None, "id-a", 4),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and confirm the missing interfaces fail**

Run from the remote worktree with the shared environment:

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/unit/lora/test_lora_versioning.py
```

Expected: import or attribute failures for `version`, `acquire_with_version`, and `_extend_lora_extra_key`.

- [ ] **Step 3: Add the trailing version field and one-lock admission helper**

Append the field after `reloadable`:

```python
version: int = 0
```

Refactor the existing lookup/increment body into the following complete helpers. Both public methods call `_acquire_refs` directly, so neither can increment a lease twice:

```python
def _lookup_refs_for_admission(
    self, lora_name: Union[str, List[Optional[str]]]
) -> List[Optional[LoRARef]]:
    if isinstance(lora_name, str):
        names = [lora_name]
    elif isinstance(lora_name, list):
        names = lora_name
    else:
        raise TypeError("lora_name must be either a string or a list of strings.")

    refs = []
    for name in names:
        if name is None:
            refs.append(None)
            continue
        ref = self._registry.get(name)
        if ref is None:
            raise ValueError(
                f"The following requested LoRA adapters are not loaded: {name}\n"
                f"Loaded adapters: {self._registry.keys()}."
            )
        self._registry.move_to_end(name)
        refs.append(ref)
    return refs

async def _increment_ref_counters(
    self, refs: List[Optional[LoRARef]]
) -> None:
    await asyncio.gather(
        *[
            self._counters[ref.lora_id].increment(notify_all=False)
            for ref in refs
            if ref is not None
        ]
    )

async def _acquire_refs(
    self, lora_name: Union[str, List[Optional[str]]]
) -> List[Optional[LoRARef]]:
    async with self._registry_lock.writer_lock:
        refs = self._lookup_refs_for_admission(lora_name)
        await self._increment_ref_counters(refs)
        return refs

async def acquire(self, lora_name):
    refs = await self._acquire_refs(lora_name)
    ids = [ref.lora_id if ref is not None else None for ref in refs]
    return ids[0] if isinstance(lora_name, str) else ids

async def acquire_with_version(self, lora_name):
    refs = await self._acquire_refs(lora_name)
    ids = [ref.lora_id if ref is not None else None for ref in refs]
    versions = [ref.version if ref is not None else None for ref in refs]
    if isinstance(lora_name, str):
        return ids[0], versions[0]
    return ids, versions
```

Extend `register_or_reuse` with a keyword-only `preserve_pinned: bool = False`. Existing one-step upsert callers keep the default; native staging passes `preserve_pinned=True`:

```python
async def register_or_reuse(
    self,
    lora_ref: LoRARef,
    upsert: bool = False,
    *,
    preserve_pinned: bool = False,
) -> Tuple[LoRARef, bool]:
    if not upsert:
        return lora_ref, False
    async with self._registry_lock.reader_lock:
        existing = self._registry.get(lora_ref.lora_name)
        if existing is None:
            return lora_ref, False
        updates = {"lora_id": existing.lora_id}
        if preserve_pinned:
            updates["pinned"] = existing.pinned
        return replace(lora_ref, **updates), True
```

- [ ] **Step 4: Carry `lora_version` through every request shape**

Add scalar-or-list `lora_version` fields beside `lora_id` in the public dataclasses `GenerateReqInput` and `EmbeddingReqInput`. Normalize them when batching and copy the indexed value in each `__getitem__`. In the array-like `TokenizedGenerateReqInput` and `TokenizedEmbeddingReqInput`, append `lora_version: Optional[int] = None` as the final declared field after every existing field; do not insert it beside `lora_id`.

In `_resolve_lora_path`, set both results and propagate both into `_sub_obj_cache`:

```python
obj.lora_id, obj.lora_version = await self.lora_registry.acquire_with_version(
    obj.lora_path
)
for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
    is_batch = isinstance(obj.lora_id, list)
    sub_obj.lora_id = obj.lora_id[i] if is_batch else obj.lora_id
    sub_obj.lora_version = (
        obj.lora_version[i] if isinstance(obj.lora_version, list) else obj.lora_version
    )
```

Set `lora_version=obj.lora_version` in both tokenized-request constructors in `_tokenize_one_request`. In `Scheduler._process_input_requests`, pass `recv_req.lora_version` to `Req(..., lora_version=...)` for generate and embedding requests. Add `lora_version: Optional[int]` to `Req.__init__` and store it as `self.lora_version`.

- [ ] **Step 5: Extend the native LoRA radix key**

Add a focused helper and call it before the existing OFT extension:

```python
def _extend_lora_extra_key(extra_key, lora_id, lora_version) -> str:
    if lora_id is None:
        return extra_key
    version = 0 if lora_version is None else lora_version
    return (extra_key or "") + f"|lora:{lora_id}:v{version}"
```

Remove the current bare `+ lora_id` concatenation. Keep the OFT `|oft:<adapter-id>:v<version>` helper unchanged.

- [ ] **Step 6: Run versioning, lease, and IPC regression tests**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest -q \
  test/registered/unit/lora/test_lora_versioning.py \
  test/registered/unit/lora/test_lora_lease.py \
  test/registered/unit/managers/test_msgpack_ipc_roundtrip.py
```

Expected: the new versioning suite passes, every previously passing lease and IPC test remains passing, and the full command reproduces only the three recorded pre-implementation failures.

Observed after Task 1 implementation: the new suite passed all 8 tests. The full command produced `3 failed, 28 passed, 16 subtests passed`; the failure names and signatures exactly matched the recorded baseline. A second run excluding only those three tests produced `28 passed, 3 deselected, 16 subtests passed`.

- [ ] **Step 7: Commit the versioned identity unit**

```bash
git add python/sglang/srt/lora/lora_registry.py \
  python/sglang/srt/managers/io_struct.py \
  python/sglang/srt/managers/tokenizer_manager.py \
  python/sglang/srt/managers/schedule_batch.py \
  python/sglang/srt/managers/scheduler.py \
  test/registered/unit/lora/test_lora_versioning.py
git commit -m "feat(lora): version request cache identity"
```

### Task 2: Replace the legacy staged backend with native placement

**Files:**
- Modify: `python/sglang/srt/adapter_sync/backends/lora.py:1-220`
- Modify: `test/registered/unit/adapter_sync/test_lora_staging_backend.py:1-180`

**Interfaces:**
- Produces: `PendingLoRAStage(ref, config, adapter, old_ref, old_config, old_adapter)`.
- Produces: `StagedLoRAMemoryPool.stage(uid, version, adapter, lora_modules, embed_module, lm_head_module)`.
- Produces: `StagedLoRAMemoryPool.activate(uid, version, destination)`.
- Produces: `StagedLoRAMemoryPool.discard_stage(uid, version)` and `staged_identity()`.
- Produces: `StagedLoRAManager.stage_adapter(named_tensors, config, name, version, adapter_id=None) -> LoRAUpdateOutput`.
- Produces: `StagedLoRAManager.activate_adapter(name, version, adapter_id=None) -> LoRAUpdateOutput`.

- [ ] **Step 1: Rewrite the backend tests around native placement**

Retain hidden-capacity assertions and replace name-driven `_fill_slot` tests with a mocked native-loader contract:

```python
def test_stage_calls_native_loader_for_hidden_slot(self):
    pool = _pool(n_slots=2)
    adapter = MagicMock()
    pool.load_lora_weight_to_buffer = MagicMock()

    pool.stage(
        uid="id-a",
        version=4,
        adapter=adapter,
        lora_modules=[{}],
        embed_module=None,
        lm_head_module=None,
    )

    pool.load_lora_weight_to_buffer.assert_called_once_with(
        "id-a", 2, adapter, [{}], None, None
    )
    self.assertEqual(pool.staged_identity(), ("id-a", 4))


def test_same_identity_retry_is_idempotent(self):
    pool = _pool()
    pool.load_lora_weight_to_buffer = MagicMock()
    args = dict(
        uid="id-a",
        version=4,
        adapter=MagicMock(),
        lora_modules=[{}],
        embed_module=None,
        lm_head_module=None,
    )
    pool.stage(**args)
    pool.stage(**args)
    pool.load_lora_weight_to_buffer.assert_called_once()


def test_conflicting_stage_is_rejected(self):
    pool = _pool()
    pool.load_lora_weight_to_buffer = MagicMock()
    pool.stage("id-a", 4, MagicMock(), [{}], None, None)
    with self.assertRaisesRegex(ValueError, "id-a.*4"):
        pool.stage("id-b", 5, MagicMock(), [{}], None, None)
```

Add `test_stage_preserves_active_state` that snapshots `configs`, `loras`, `lora_refs`, `num_pinned_loras`, `uid_to_buffer_id`, and `buffer_id_to_uid` before stage and compares every snapshot after stage. Add `test_existing_pinned_adapter_stays_pinned_after_activation` with an old ref whose `pinned=True`; assert the pending ref and committed ref remain pinned and `num_pinned_loras` is unchanged.

- [ ] **Step 2: Run the rewritten backend test and confirm legacy behavior fails it**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/unit/adapter_sync/test_lora_staging_backend.py
```

Expected: failures because the existing backend accepts resolved A/B rows and does not build a native `LoRAAdapter` or retain pending manager state.

- [ ] **Step 3: Keep the hidden-slot allocation trick but remove manual tensor placement**

Keep the `init_buffers` N+1 allocation override. Replace `VersionedStaging` and `_fill_slot` with explicit staged metadata. `stage()` must call only the native loader and record metadata only after it succeeds:

```python
def stage(self, uid, version, adapter, lora_modules, embed_module, lm_head_module):
    current = self.staged_identity()
    if current == (uid, version):
        return
    if current is not None:
        raise ValueError(
            f"staging slot already holds uid={current[0]} version={current[1]}"
        )
    self.load_lora_weight_to_buffer(
        uid,
        self.staging_idx,
        adapter,
        lora_modules,
        embed_module,
        lm_head_module,
    )
    self._staged_uid = uid
    self._staged_version = version
```

`activate()` must validate exact identity, require `0 <= destination < max_loras_per_batch`, require `destination != staging_idx`, and copy every dense, MoE, embedding, lm-head, and added-token slot family. It must not update `uid_to_buffer_id`, `buffer_id_to_uid`, or eviction policy.

- [ ] **Step 4: Build and retain a native adapter without committing it**

Define `PendingLoRAStage` as a frozen dataclass. In `StagedLoRAManager.stage_adapter`:

1. Resolve `uid = adapter_id or name` and `version = int(version)`.
2. Return success without work when pending identity matches.
3. Reject a different pending identity.
4. Build `LoRAConfig.from_dict`. Resolve `old_ref = self.lora_refs.get(uid)` and build the pending `LoRARef` with `pinned=old_ref.pinned if old_ref is not None else False`, `reloadable=False`, and the requested version.
5. Call `validate_new_adapter(new_config, new_ref, is_update=uid in self.loras, old_ref=old_ref)`.
6. Call `_create_lora_adapter_from_tensors(ref, config, dict(named_tensors))`.
7. Call `memory_pool.stage(uid, version, new_adapter, self.lora_modules, self.embed_tokens_module, self.lm_head_module)`.
8. Store `PendingLoRAStage`; do not mutate live manager dictionaries.

Return `create_lora_update_result(success=False, error_message=str(error))` for validation or placement failures.

- [ ] **Step 5: Activate resident and nonresident adapters transactionally**

For a resident adapter, synchronize outstanding CUDA work, copy hidden-to-live, then commit CPU metadata. If live copy fails, call `load_lora_weight_to_buffer` with the previous adapter and preserve old dictionaries. If restoration also fails, log the adapter ID and serving slot with `logger.exception` and return an error containing `worker restart required`.

For a nonresident or first-time adapter, commit `configs`, `loras`, and `lora_refs` without allocating or evicting a serving slot. In both success paths, update pinned accounting by delta, discard the staged record, and return success.

- [ ] **Step 6: Run staging and native placement regression tests**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest -q \
  test/registered/unit/adapter_sync/test_lora_staging_backend.py \
  test/registered/unit/lora/test_lora_upsert.py \
  test/registered/unit/lora/test_mem_pool_ep_unit.py \
  test/registered/unit/lora/test_lora_manager_tied_lm_head.py \
  test/registered/unit/lora/test_lora_moe_inplace_unit.py
```

Expected: all tests pass; existing upsert and MoE/embedding placement remain green.

- [ ] **Step 7: Commit the native staging backend**

```bash
git add python/sglang/srt/adapter_sync/backends/lora.py \
  test/registered/unit/adapter_sync/test_lora_staging_backend.py
git commit -m "feat(lora): stage updates through native placement"
```

### Task 3: Add explicit startup selection and worker routing

**Files:**
- Modify: `python/sglang/srt/server_args.py:2890-2910, 9434-9465`
- Modify: `python/sglang/srt/model_executor/model_runner.py:75-90, 1265-1295`
- Modify: `python/sglang/srt/model_executor/model_runner_components/weight_updater.py:373-475`
- Modify: `python/sglang/srt/managers/tp_worker.py:180-215`
- Create: `test/registered/unit/lora/test_lora_staging_control.py`

**Interfaces:**
- Produces: `ServerArgs.enable_lora_staging: bool = False` / CLI `--enable-lora-staging`.
- Consumes: `StagedLoRAManager` from Task 2.
- Produces: native stage and activate routing for `load_format == "lora_adapter"`.
- Preserves: PEFT/OFT fallback for every other configuration.

- [ ] **Step 1: Write failing flag-selection and routing tests**

Create mock-based tests that assert:

```python
def test_staging_requires_native_lora(self):
    args = ServerArgs(model_path="Qwen/Qwen3-0.6B")
    args.enable_lora = False
    args.enable_lora_staging = True
    with self.assertRaisesRegex(ValueError, "requires --enable-lora"):
        args.check_lora_server_args()


def test_model_runner_selects_staged_manager(self):
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = MagicMock(enable_lora_staging=True)
    with patch(
        "sglang.srt.adapter_sync.backends.lora.StagedLoRAManager",
        new=sentinel.staged_manager,
    ):
        self.assertIs(
            runner._get_lora_manager_class(),
            sentinel.staged_manager,
        )


def test_activation_forwards_stable_id(self):
    worker = TpModelWorker.__new__(TpModelWorker)
    worker.model_runner = MagicMock()
    req = ActivateAdapterVersionReqInput(
        adapter_name="policy", adapter_id="id-a", adapter_version="8"
    )
    worker.activate_adapter_version(req)
    worker.model_runner.weight_updater.activate_adapter_version.assert_called_once_with(
        adapter_name="policy", adapter_id="id-a", adapter_version="8"
    )
```

Import `TpModelWorker` directly from `sglang.srt.managers.tp_worker`.

- [ ] **Step 2: Run the new control test and confirm the new flag/routing are absent**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/unit/lora/test_lora_staging_control.py
```

Expected: missing flag/selector failures and a missing `adapter_id` argument in activation.

- [ ] **Step 3: Add and validate `--enable-lora-staging`**

Place the new argument directly after `enable_lora`. In `check_lora_server_args`, fail before native LoRA initialization when staging is enabled without LoRA:

```python
if self.enable_lora_staging and not self.enable_lora:
    raise ValueError("--enable-lora-staging requires --enable-lora")
```

Do not auto-enable staging from PEFT flags and do not allocate staging memory when false.

- [ ] **Step 4: Select the staged manager at ModelRunner startup**

Add this exact selector to `ModelRunner`:

```python
def _get_lora_manager_class(self):
    if self.server_args.enable_lora_staging:
        from sglang.srt.adapter_sync.backends.lora import StagedLoRAManager

        return StagedLoRAManager
    return LoRAManager
```

In `init_lora_manager` replace only the constructor symbol:

```python
self.lora_manager = self._get_lora_manager_class()(
    base_model=self.model,
    base_hf_config=self.model_config.hf_config,
    max_loras_per_batch=get_lora().max_loras_per_batch,
    load_config=self.load_config,
    dtype=self.dtype,
    server_args=self.server_args,
    lora_backend=get_lora().lora_backend,
    tp_size=self.ps.tp_size,
    tp_rank=self.ps.tp_rank,
    max_lora_rank=get_lora().max_lora_rank,
    target_modules=get_lora().lora_target_modules,
    lora_paths=get_lora().lora_paths,
)
```

Keep CUDA-graph initialization after construction exactly where it is.

- [ ] **Step 5: Route native payloads before PEFT fallback**

In the model-runner `WeightUpdater.stage_adapter`, after NCCL receive and before `peft.stage_adapter`, branch only when the new flag is active and `load_format == "lora_adapter"`. Reconstruct a flattened payload with the existing `peft.reconstruct_oft_staging` helper, then call the native manager. Preserve OFT behavior on the fallback.

In activation, accept `adapter_id`, route to `model_runner.lora_manager.activate_adapter` when staging is enabled, and otherwise retain `peft.activate_adapter`. Treat the returned `LoRAUpdateOutput.success` and `error_message` as the `(bool, str)` boundary expected by the scheduler.

- [ ] **Step 6: Run flag/routing plus OFT integration regressions**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest -q \
  test/registered/unit/lora/test_lora_staging_control.py \
  test/registered/unit/peft/test_peft_config.py \
  test/registered/unit/adapter_sync/test_lora_staging_backend.py
```

Expected: all selected tests pass, and OFT staging tests remain unchanged.

- [ ] **Step 7: Commit startup and worker routing**

```bash
git add python/sglang/srt/server_args.py \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/model_executor/model_runner_components/weight_updater.py \
  python/sglang/srt/managers/tp_worker.py \
  test/registered/unit/lora/test_lora_staging_control.py
git commit -m "feat(lora): wire optional staged update routing"
```

### Task 4: Coordinate native stage and activation in the tokenizer manager

**Files:**
- Modify: `python/sglang/srt/managers/tokenizer_manager.py:650-680`
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py:520-615`
- Modify: `test/registered/unit/lora/test_lora_staging_control.py`

**Interfaces:**
- Produces: tokenizer-local `pending_lora_stage: Optional[LoRARef]`.
- Produces: tokenizer-local `failed_lora_activations: Dict[str, str]`.
- Consumes: `LoRARef.version`, `register_or_reuse(..., preserve_pinned=True)`, `register`, and `refresh`.
- Produces: `_is_native_lora_stage(obj) -> bool`, `_reserve_native_lora_stage(obj) -> LoRARef`, `_publish_native_lora_activation() -> None`, `_quarantine_native_lora_activation(name, message) -> None`, and `_assert_native_lora_available(lora_path) -> None`.
- Preserves: `register_peft_ref` and `bump_peft_version` for OFT/PEFT only.

- [ ] **Step 1: Add failing publication and failure-isolation tests**

Extend the control test with a tokenizer manager built through `__new__`, an `asyncio.Lock`, `LoRARegistry`, and `AsyncMock` communicators. Cover these exact outcomes:

```python
def test_stage_reuses_id_without_publishing_version(self):
    tm = _make_tm()
    old = LoRARef(
        lora_id="id-a",
        lora_name="policy",
        lora_path="__distributed__",
        pinned=True,
        version=3,
    )
    asyncio.run(tm.lora_registry.register(old))
    req = _stage_req(name="policy", version="4")

    success, _ = asyncio.run(tm.update_adapter_from_distributed(req))

    self.assertTrue(success)
    self.assertEqual(req.adapter_id, "id-a")
    self.assertEqual(tm.lora_registry.get_all_adapters()["policy"].version, 3)
    self.assertEqual(tm.pending_lora_stage.version, 4)
    self.assertTrue(tm.pending_lora_stage.pinned)


def test_activation_publishes_only_after_all_workers_succeed(self):
    tm = _make_tm()
    old = LoRARef(
        lora_id="id-a", lora_name="policy", lora_path="__distributed__", version=3
    )
    asyncio.run(tm.lora_registry.register(old))
    tm.pending_lora_stage = LoRARef(
        lora_id="id-a", lora_name="policy", lora_path="__distributed__", version=4
    )
    tm.activate_adapter_version_communicator = AsyncMock(
        return_value=[
            MagicMock(success=True, message="ok"),
            MagicMock(success=False, message="rank 1 failed"),
        ]
    )

    success, _ = asyncio.run(
        tm.activate_adapter_version(_activate_req("policy", "4"))
    )

    self.assertFalse(success)
    self.assertEqual(tm.lora_registry.get_all_adapters()["policy"].version, 3)
    self.assertIn("policy", tm.failed_lora_activations)
    with self.assertRaisesRegex(ValueError, "policy.*restart required"):
        tm._assert_native_lora_available("policy")
    tm._assert_native_lora_available(None)
    tm._assert_native_lora_available("unrelated")
```

Add named tests `test_first_stage_is_not_registered`, `test_same_stage_retry_reuses_pending_ref`, `test_conflicting_stage_reports_pending_identity`, `test_activation_forwards_exact_id`, `test_successful_activation_refreshes_existing_ref`, `test_successful_activation_registers_first_ref`, `test_quarantine_does_not_block_base_or_other_adapter`, and `test_multi_tokenizer_native_stage_is_rejected`. Each test asserts the named outcome and the unchanged old registry version before successful activation.

- [ ] **Step 2: Run the control tests and confirm current register-before-stage behavior fails**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/unit/lora/test_lora_staging_control.py
```

Expected: failures because the current method calls `register_peft_ref` during stage and never publishes a native `LoRARef.version` during activate.

- [ ] **Step 3: Reserve a native identity without registry publication**

Initialize `pending_lora_stage = None` and `failed_lora_activations = {}` beside `lora_registry`. Implement the helpers with these contracts:

```python
def _is_native_lora_stage(self, obj) -> bool:
    return (
        self.server_args.enable_lora_staging
        and obj.load_format == "lora_adapter"
    )

def _assert_native_lora_available(self, lora_path) -> None:
    names = [lora_path] if isinstance(lora_path, str) else (lora_path or [])
    for name in names:
        if name in self.failed_lora_activations:
            raise ValueError(
                f"LoRA adapter '{name}' is unavailable after a partial "
                "activation failure; restart required"
            )

def _quarantine_native_lora_activation(self, name: str, message: str) -> None:
    self.failed_lora_activations[name] = message
```

Call `_assert_native_lora_available(obj.lora_path)` at the start of `_resolve_lora_path`. Request admission already holds `model_update_lock.reader_lock`, while activation records quarantine under the writer lock, so admission cannot pass between the failed fan-out and quarantine publication.

Implement `_reserve_native_lora_stage` as an async helper:

1. Require `load_format == "lora_adapter"`, a nonempty `adapter_name`, and an integer `adapter_version`.
2. Reject `tokenizer_worker_num > 1` with the same rationale as native upsert.
3. Reject names already present in `failed_lora_activations` with `restart required`.
4. If pending matches name/version, reuse its ID.
5. If a different pending record exists, reject with its name, ID, and version.
6. Build `LoRARef(lora_name=obj.adapter_name, lora_path="__distributed__", pinned=False, reloadable=False, version=int(obj.adapter_version))`.
7. Call `register_or_reuse(candidate, upsert=True, preserve_pinned=True)` so an active adapter keeps both its stable ID and pinned state.
8. Save the resolved candidate only in `pending_lora_stage`, set `obj.adapter_id`, and return the candidate.

```python
async def _reserve_native_lora_stage(self, obj) -> LoRARef:
    if obj.load_format != "lora_adapter" or not obj.adapter_name:
        raise ValueError("native LoRA staging requires load_format=lora_adapter and adapter_name")
    try:
        version = int(obj.adapter_version)
    except (TypeError, ValueError) as exc:
        raise ValueError("native LoRA staging requires an integer adapter_version") from exc
    if self.server_args.tokenizer_worker_num > 1:
        raise ValueError("native LoRA staging requires tokenizer_worker_num == 1")
    if obj.adapter_name in self.failed_lora_activations:
        raise ValueError(
            f"LoRA adapter '{obj.adapter_name}' is quarantined; restart required"
        )

    pending = self.pending_lora_stage
    if pending is not None:
        if pending.lora_name == obj.adapter_name and pending.version == version:
            obj.adapter_id = pending.lora_id
            return pending
        raise ValueError(
            "staging slot already reserved for "
            f"name={pending.lora_name} id={pending.lora_id} version={pending.version}"
        )

    candidate, _ = await self.lora_registry.register_or_reuse(
        LoRARef(
            lora_name=obj.adapter_name,
            lora_path="__distributed__",
            pinned=False,
            reloadable=False,
            version=version,
        ),
        upsert=True,
        preserve_pinned=True,
    )
    self.pending_lora_stage = candidate
    obj.adapter_id = candidate.lora_id
    return candidate
```

Do not touch `lora_ref_cache` or `_registry` in this step.

- [ ] **Step 4: Use existing writer-lock draining but publish only after aggregate success**

For native activation, validate pending name/version, set `obj.adapter_id`, then use the existing `model_update_lock.writer_lock` fan-out. Merge the results and publish or quarantine while still inside that writer-lock block. The paused path performs the same merge and decision while `is_pause_cond` prevents unpause.

Implement `_publish_native_lora_activation` as:

```python
async def _publish_native_lora_activation(self) -> None:
    pending = self.pending_lora_stage
    registered = self.lora_registry.get_all_adapters().get(pending.lora_name)
    if registered is None:
        await self.lora_registry.register(pending)
    else:
        await self.lora_registry.refresh(pending)
    self.lora_ref_cache[pending.lora_name] = pending
    self.pending_lora_stage = None
```

On aggregate failure, call `_quarantine_native_lora_activation(pending.lora_name, message)` before releasing the writer lock. Retain the old registry version and pending tokenizer record, return a consistency failure naming the adapter/version, and include `restart required`. Do not claim the version is active. Successful activation clears any stale quarantine entry only defensively; a quarantined adapter cannot be retried in the same process.

- [ ] **Step 5: Preserve OFT and synchronous stage-then-activate behavior**

Keep the existing PEFT hook path for non-native requests. For native `double_buffer=False`, the scheduler stages and activates under the writer lock in one request. Publish the new native ref only when every returned `active_adapter_version` matches the requested version. Any failure or version disagreement calls the same quarantine helper before releasing the writer lock.

- [ ] **Step 6: Run tokenizer coordination and native regression tests**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest -q \
  test/registered/unit/lora/test_lora_staging_control.py \
  test/registered/unit/lora/test_lora_upsert.py \
  test/registered/unit/lora/test_lora_lease.py \
  test/registered/unit/managers/test_lora_update_result_merge.py
```

Expected: all selected tests pass; direct upsert still refreshes immediately and staged activation alone advances the version.

- [ ] **Step 7: Commit tokenizer coordination**

```bash
git add python/sglang/srt/managers/tokenizer_manager.py \
  python/sglang/srt/managers/tokenizer_control_mixin.py \
  test/registered/unit/lora/test_lora_staging_control.py
git commit -m "feat(lora): publish staged versions after activation"
```

### Task 5: Add an end-to-end staged-update regression

**Files:**
- Create: `test/registered/lora/test_lora_staged_update.py`
- Create: `test/registered/lora/test_lora_staged_update_tp.py`
- Reference: `test/registered/rl/test_update_weights_from_distributed.py:122-520`
- Reference: `test/registered/rl/test_lora_load_from_tensor.py:1-120`
- Reference: `test/registered/lora/test_lora_moe_tp_logprob_diff.py:20-70`

**Interfaces:**
- Consumes: `/init_weights_update_group`, `/update_adapter_from_distributed`, `/activate_adapter_version`, and `/generate`.
- Uses: base model `Qwen/Qwen3-0.6B` and adapter `charent/self_cognition_Alice` already used by registered LoRA RL tests.
- Produces: one registered two-GPU file for trainer+TP1 and one registered four-GPU file for trainer+TP2.

- [ ] **Step 1: Write the trainer+TP1 stage/activate test before running it**

Register the trainer+TP1 file exactly with:

```python
register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")
```

Use GPU 0 for the trainer process and launch the TP=1 server with `--base-gpu-id 1`. This avoids unsupported duplicate NCCL ranks on one device. Extract the rank-0 group setup from `init_process_hf` in `test_update_weights_from_distributed.py` and define:

```python
def _versioned_tensors(adapter, version: int):
    tensors = {
        name: tensor.detach().clone()
        for name, tensor in adapter.state_dict().items()
    }
    if version == 2:
        for name in sorted(tensors):
            if "lora_B" in name:
                tensors[name].add_(0.125)
    return tensors

def _stage_payload(name, version, tensors, adapter_config):
    return {
        "names": list(tensors),
        "dtypes": [str(t.dtype).removeprefix("torch.") for t in tensors.values()],
        "shapes": [list(t.shape) for t in tensors.values()],
        "group_name": "test_lora_stage_group",
        "weight_version": str(version),
        "adapter_version": str(version),
        "load_format": "lora_adapter",
        "adapter_config": adapter_config,
        "adapter_name": name,
        "double_buffer": True,
    }
```

The trainer broadcasts the tensors in the exact `names` order before posting the payload. Use two logical adapters built from the same v1 tensors: `policy-b` is staged and activated first, then `policy-a` is staged and activated. This guarantees that the multi-tenant assertion is always exercised rather than conditional.

`TestStagedLoRAUpdate.test_single_gpu` must:

1. Start SGLang with `--enable-lora --enable-lora-staging --max-lora-rank 64 --lora-target-modules all --max-loras-per-batch 2`.
2. Join the server and trainer to `test_lora_stage_group`.
3. Produce v1 and v2 with `_versioned_tensors`.
4. Stage and activate `policy-b` v1, then stage and activate `policy-a` v1.
5. Record greedy token IDs for `policy-a`, `policy-b`, and base.
6. Stage v2 and assert all three outputs still match their pre-stage values.
7. Activate v2 and assert adapter A changes while adapter B and base remain bitwise identical.
8. Restart a fresh server, stage and activate only `policy-a` v2, and assert its token IDs match the prior activated-v2 output exactly.
9. Repeat one shared prompt before and after activation to prove the versioned radix key misses old KV.

Use exact response checks:

```python
stage = requests.post(url + "/update_adapter_from_distributed", json=stage_payload)
stage.raise_for_status()
self.assertTrue(stage.json()["success"], stage.json())
self.assertEqual(stage.json()["staged_adapter_version"], version)

activate = requests.post(
    url + "/activate_adapter_version",
    json={
        "adapter_name": "policy",
        "adapter_version": version,
        "load_format": "lora_adapter",
    },
)
activate.raise_for_status()
self.assertTrue(activate.json()["success"], activate.json())
self.assertEqual(activate.json()["active_adapter_version"], version)
```

Add `test_decode_graph_on_and_off`, which calls the same scenario once with CUDA graphs enabled and once with `--disable-cuda-graph`. Add `test_hidden_slot_never_evicts` with `max_loras_per_batch=2` and both adapters resident; assert both names remain routable after staging `policy-a` v2 and before activation.

- [ ] **Step 2: Write separately registered TP=2 and MoE tests**

Register `test_lora_staged_update_tp.py` exactly with:

```python
register_cuda_ci(est_time=600, stage="base-b", runner_config="4-gpu-h100")
```

Import the shared broadcast/HTTP helpers from `test_lora_staged_update.py`. Use GPU 0 for the trainer and GPUs 1-2 for the TP=2 server. `TestStagedLoRAUpdateTP.test_tp2` launches `Qwen/Qwen3-0.6B` with `--base-gpu-id 1 --tp-size 2`, stages and activates the same `policy-a` versions on both ranks, and compares activated-v2 token IDs with a TP=1 reference run in the same four-GPU job.

`TestStagedLoRAUpdateTP.test_moe_sharded_placement` imports `MOE_BASE_MODEL_PATH` and `MOE_LORA_PATH` from `sglang.test.lora_utils`, launches with `--tp-size 2`, converts that adapter to the same distributed tensor payload, and proves stage invisibility plus activated output change. This test owns the MoE requirement; the dense TP=1 file does not claim MoE coverage.

- [ ] **Step 3: Run static registration and collection checks**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  scripts/lint/check_registered_tests.py
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest \
  --collect-only -q \
  test/registered/lora/test_lora_staged_update.py \
  test/registered/lora/test_lora_staged_update_tp.py
```

Expected: both files are registered for supported runner configurations (`2-gpu-large` and `4-gpu-h100`) and collection succeeds without launching a server.

- [ ] **Step 4: Commit the reusable end-to-end tests**

```bash
git add test/registered/lora/test_lora_staged_update.py \
  test/registered/lora/test_lora_staged_update_tp.py
git commit -m "test(lora): cover staged native adapter updates"
```

### Task 6: Qualify the implementation and close the development record

**Files:**
- Modify after evidence exists: `/Users/zqiu/Documents/GitHub/schematic/miles-imp/docs/reports/_src/2026-08-30-sglang-corrected-vs-oft-restructure.md`
- Regenerate after evidence exists: `/Users/zqiu/Documents/GitHub/schematic/miles-imp/docs/reports/2026-08-30-sglang-corrected-vs-oft-restructure.html`

**Interfaces:**
- Consumes: all commits and tests from Tasks 1-5.
- Produces: CPU evidence, Slurm GPU evidence, a final development report, and a branch ready for self-review.

- [ ] **Step 1: Run the complete focused CPU suite in one bounded command**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest -q \
  test/registered/unit/lora/test_lora_versioning.py \
  test/registered/unit/adapter_sync/test_lora_staging_backend.py \
  test/registered/unit/lora/test_lora_staging_control.py \
  test/registered/unit/lora/test_lora_upsert.py \
  test/registered/unit/lora/test_lora_lease.py \
  test/registered/unit/lora/test_mem_pool_ep_unit.py \
  test/registered/unit/lora/test_lora_manager_tied_lm_head.py \
  test/registered/unit/lora/test_lora_moe_inplace_unit.py \
  test/registered/unit/managers/test_lora_update_result_merge.py \
  test/registered/unit/managers/test_msgpack_ipc_roundtrip.py
```

Expected: every new staged-LoRA test and every test that passed in the recorded baseline passes. A full rerun may reproduce exactly the three pre-implementation failures recorded above; any additional failure or changed failure signature blocks qualification. If the three unrelated baseline defects have been fixed separately by this point, expect zero failures and zero errors.

- [ ] **Step 2: Run formatting and diff hygiene**

```bash
/home/zeju/.local/bin/uv run --isolated --no-project --with pre-commit \
  pre-commit run ruff --files \
  python/sglang/srt/lora/lora_registry.py \
  python/sglang/srt/adapter_sync/backends/lora.py \
  python/sglang/srt/managers/io_struct.py \
  python/sglang/srt/managers/tokenizer_manager.py \
  python/sglang/srt/managers/tokenizer_control_mixin.py \
  python/sglang/srt/managers/schedule_batch.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/managers/tp_worker.py \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/model_executor/model_runner_components/weight_updater.py \
  python/sglang/srt/server_args.py \
  test/registered/unit/lora/test_lora_versioning.py \
  test/registered/unit/adapter_sync/test_lora_staging_backend.py \
  test/registered/unit/lora/test_lora_staging_control.py \
  test/registered/lora/test_lora_staged_update.py \
  test/registered/lora/test_lora_staged_update_tp.py
git diff --check 1c15111fba954babe2b9742caefcbedb22ce306c...HEAD
```

Expected: both commands exit zero.

- [ ] **Step 3: Create the canonical Slurm run record before requesting GPUs**

Use execution ID `20260830-native-staged-lora-01a0532a` only after a read-only check proves the exact directory does not exist. If it exists, stop and choose a new collision-resistant suffix; never reuse or overwrite a prior run record. Then create:

```text
/data/home/zeju/.local/state/remote-cluster-runs/slurm/sglang/codex-port-staged-lora-native/20260830-native-staged-lora-01a0532a/qualification/
```

Record worktree, branch, exact commit, clean/dirty state, shared environment path, test command, requested GPU resources, and controller owner `codex` in `provenance.json`. Bind `stdout.log`, `stderr.log`, and `completion.status` before submission. Ask the user before submitting the multi-GPU portion.

- [ ] **Step 4: Run the registered trainer+TP1 test on a Slurm compute node**

After explicit approval for this multi-GPU run, request the established `viga`/`all` two-GPU shape (`1` node, `1` task, `24` CPUs, `160G` RAM, `2` GPUs, `02:00:00`) and execute:

```bash
srun --account=viga --partition=all --nodes=1 --ntasks=1 \
  --cpus-per-task=24 --mem=160G --gres=gpu:2 --time=02:00:00 \
  env PYTHONPATH="$PWD/python" \
  /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/lora/test_lora_staged_update.py \
  TestStagedLoRAUpdate.test_single_gpu \
  TestStagedLoRAUpdate.test_decode_graph_on_and_off \
  TestStagedLoRAUpdate.test_hidden_slot_never_evicts
```

Expected: the dedicated trainer rank broadcasts to the TP=1 server, stage is invisible, activation changes only adapter A, v2 matches a fresh v2 server, the versioned cache key prevents stale reuse, both graph modes pass, and the hidden slot never evicts either serving adapter.

- [ ] **Step 5: Run TP=2 and MoE cases**

After separate explicit approval, request a site-valid three-GPU shape on the `viga`/`all` 8-GPU nodes (`1` node, `1` task, `32` CPUs, `240G` RAM, `3` GPUs, `03:00:00`) and run:

```bash
srun --account=viga --partition=all --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --mem=240G --gres=gpu:3 --time=03:00:00 \
  env PYTHONPATH="$PWD/python" \
  /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/lora/test_lora_staged_update_tp.py \
  TestStagedLoRAUpdateTP.test_tp2 \
  TestStagedLoRAUpdateTP.test_moe_sharded_placement
```

Expected: one trainer rank plus two server ranks use distinct GPUs, every TP rank activates the identical version, and expert-sharded placement succeeds. CUDA-graph and slot-pressure cases run in the TP=1 file registered for the two-GPU runner.

- [ ] **Step 6: Snapshot terminal logs and verify the source identity**

Take one bounded snapshot into the matching Mac run-store suffix. Confirm `completion.status` exit code zero, the tested commit equals `git rev-parse HEAD`, and the worktree was clean for the terminal run. Do not report submission as a pass without this evidence.

- [ ] **Step 7: Update and finalize the living HTML report**

Change the report outcome from branch comparison to implemented port, add exact CPU/GPU commands and results, record any skipped hardware matrix item, set `status: final` only if all required gates passed, render, run the freshness check, and inspect light/dark/narrow layouts. Keep the report uncommitted unless the user asks to include it in its repository.

- [ ] **Step 8: Self-review the branch before PR preparation**

Inventory `1c15111fba954babe2b9742caefcbedb22ce306c...HEAD`, review every changed path, verify the staging slot never enters serving maps, verify no registry publication occurs on stage, and verify ordinary one-step upsert still follows its original code path. Fix every blocking finding and rerun the affected tests.

- [ ] **Step 9: Report the ready development branch**

Report the remote worktree, branch, commit series, test evidence, run-store paths, limitations, and the exact next safe action. Do not push, create a PR, merge, or delete the worktree without a new explicit request.

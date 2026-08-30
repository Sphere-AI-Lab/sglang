# Native Staged LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional two-phase stage/activate workflow to the corrected branch's native `--enable-lora` stack while preserving the existing one-step LoRA APIs and the existing OFT implementation.

**Architecture:** A staging-specific `LoRAManager` subclass builds adapters through the corrected branch's native tensor loader and places them in one hidden GPU slot owned by a staging-specific `LoRAMemoryPool` subclass. The tokenizer manager publishes a new `LoRARef.version` only after all scheduler workers activate successfully; request admission snapshots the ID and version together, and the radix key contains both.

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
- Publish a tokenizer-registry version only after every worker reports activation success.
- A recoverable resident-slot activation failure must restore the previous adapter with the native placement routine; a failed restore is a hard worker-restart error.
- GPU checks run under Slurm, never on the login node, and use the existing shared environment `/data/home/zeju/miles-orbit-dev/envs/candidate` with worktree source supplied through `PYTHONPATH`.
- Preserve all unrelated branch changes and do not delete remote files, jobs, worktrees, or sessions.

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
| `python/sglang/srt/managers/tokenizer_control_mixin.py` | Reserve staged native identity and publish it only after aggregate activation success. |
| `test/registered/unit/lora/test_lora_versioning.py` | Protect wire compatibility, atomic admission, propagation, and cache isolation. |
| `test/registered/unit/adapter_sync/test_lora_staging_backend.py` | Protect hidden-slot allocation, native placement, idempotency, conflict, activation, and rollback. |
| `test/registered/unit/lora/test_lora_staging_control.py` | Protect CLI selection, worker routing, and tokenizer-side publication rules. |
| `test/registered/lora/test_lora_staged_update.py` | Exercise real one-GPU and TP=2 stage/activate behavior, cache isolation, multi-tenancy, slot pressure, CUDA graphs, and MoE. |

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

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.schedule_batch import _extend_lora_extra_key
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()


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

Refactor the existing lookup/increment body into one private method used by both public methods. The new interface must return ID/version pairs without releasing `_registry_lock.writer_lock` between lookup and counter increment:

```python
async def acquire_with_version(self, lora_name):
    async with self._registry_lock.writer_lock:
        refs = self._lookup_refs_for_admission(lora_name)
        await self._increment_ref_counters(refs)
        if isinstance(lora_name, str):
            ref = refs[0]
            return ref.lora_id, ref.version
        return [ref.lora_id if ref is not None else None for ref in refs], [
            ref.version if ref is not None else None for ref in refs
        ]
```

Keep `acquire()` as a compatibility wrapper over the same locked implementation; it must not acquire a lease twice.

- [ ] **Step 4: Carry `lora_version` through every request shape**

Add scalar-or-list fields beside `lora_id` in `GenerateReqInput` and `EmbeddingReqInput`, normalize them when batching, copy the indexed value in each `__getitem__`, and add scalar fields to both tokenized request structs. In `_resolve_lora_path`, set both results and propagate both into `_sub_obj_cache`:

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

Pass `lora_version` through tokenizer construction, scheduler construction, and embedding construction exactly as `lora_id` is passed today.

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

Expected: all selected tests pass, including old `LoRARef` decoding and exact-once lease release.

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

Add manager tests that snapshot `configs`, `loras`, `lora_refs`, `num_pinned_loras`, and serving mappings before stage, then assert byte-for-byte/object-identity preservation until activate.

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
4. Build `LoRAConfig.from_dict` and a `LoRARef` with `reloadable=False` and the requested version.
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
    with patch("sglang.srt.adapter_sync.backends.lora.StagedLoRAManager") as cls:
        runner._construct_lora_manager = MagicMock(return_value=cls)
        self.assertIs(runner._lora_manager_class(), cls)


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

Use the concrete TP worker class name present in `tp_worker.py` when implementing the test import.

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

Factor construction so only the class changes:

```python
manager_cls = LoRAManager
if self.server_args.enable_lora_staging:
    from sglang.srt.adapter_sync.backends.lora import StagedLoRAManager

    manager_cls = StagedLoRAManager
self.lora_manager = manager_cls(
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
- Consumes: `LoRARef.version`, existing `register_or_reuse`, `register`, and `refresh`.
- Produces: `_is_native_lora_stage(obj)`, `_reserve_native_lora_stage(obj)`, and `_publish_native_lora_activation(obj)` helpers.
- Preserves: `register_peft_ref` and `bump_peft_version` for OFT/PEFT only.

- [ ] **Step 1: Add failing publication and failure-isolation tests**

Extend the control test with a tokenizer manager built through `__new__`, an `asyncio.Lock`, `LoRARegistry`, and `AsyncMock` communicators. Cover these exact outcomes:

```python
def test_stage_reuses_id_without_publishing_version(self):
    tm = _make_tm()
    old = LoRARef(
        lora_id="id-a", lora_name="policy", lora_path="__distributed__", version=3
    )
    asyncio.run(tm.lora_registry.register(old))
    req = _stage_req(name="policy", version="4")

    success, _ = asyncio.run(tm.update_adapter_from_distributed(req))

    self.assertTrue(success)
    self.assertEqual(req.adapter_id, "id-a")
    self.assertEqual(tm.lora_registry.get_all_adapters()["policy"].version, 3)
    self.assertEqual(tm.pending_lora_stage.version, 4)


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
```

Also cover first-time adapter invisibility, same-version retry, conflicting version/name rejection, exact ID forwarding, successful `refresh`, successful first-time `register`, and `tokenizer_worker_num > 1` rejection.

- [ ] **Step 2: Run the control tests and confirm current register-before-stage behavior fails**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/unit/lora/test_lora_staging_control.py
```

Expected: failures because the current method calls `register_peft_ref` during stage and never publishes a native `LoRARef.version` during activate.

- [ ] **Step 3: Reserve a native identity without registry publication**

Initialize `pending_lora_stage = None` beside `lora_registry`. For native staging:

1. Require `load_format == "lora_adapter"`, a nonempty `adapter_name`, and an integer `adapter_version`.
2. Reject `tokenizer_worker_num > 1` with the same rationale as native upsert.
3. If pending matches name/version, reuse its ID.
4. If a different pending record exists, reject with its name, ID, and version.
5. Call `register_or_reuse(candidate, upsert=True)` so an active adapter keeps its stable ID.
6. Save the candidate only in `pending_lora_stage` and set `obj.adapter_id`.

Do not touch `lora_ref_cache` or `_registry` in this step.

- [ ] **Step 4: Use existing writer-lock draining but publish only after aggregate success**

For native activation, validate pending name/version, set `obj.adapter_id`, then use the existing `model_update_lock.writer_lock` fan-out. After `FanOutCommunicator.merge_results` reports all success:

```python
pending = self.pending_lora_stage
registered = self.lora_registry.get_all_adapters().get(pending.lora_name)
if registered is None:
    await self.lora_registry.register(pending)
else:
    await self.lora_registry.refresh(pending)
self.lora_ref_cache[pending.lora_name] = pending
self.pending_lora_stage = None
```

On aggregate failure, retain the old registry version and report a consistency failure that names the adapter/version. Keep the pending tokenizer record for diagnostics; do not claim the version is active.

- [ ] **Step 5: Preserve OFT and synchronous stage-then-activate behavior**

Keep the existing PEFT hook path for non-native requests. For native `double_buffer=False`, the scheduler already stages and activates under the writer lock in one request; publish the new native ref only when every returned `active_adapter_version` matches the requested version.

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
- Reference: `test/registered/rl/test_update_weights_from_distributed.py:122-520`
- Reference: `test/registered/rl/test_lora_load_from_tensor.py:1-120`

**Interfaces:**
- Consumes: `/init_weights_update_group`, `/update_adapter_from_distributed`, `/activate_adapter_version`, and `/generate`.
- Uses: base model `Qwen/Qwen3-0.6B` and adapter `charent/self_cognition_Alice` already used by registered LoRA RL tests.
- Produces: a registered reusable GPU test, not a one-off script.

- [ ] **Step 1: Write the one-GPU stage/activate test before running it**

Register the file for `base-b-test-1-gpu-large`. Reuse the NCCL process-group setup pattern from `test_update_weights_from_distributed.py`. The test must:

1. Start SGLang with `--enable-lora --enable-lora-staging --max-lora-rank 64 --lora-target-modules all --max-loras-per-batch 2`.
2. Join the server and trainer to `test_lora_stage_group`.
3. Produce v1 and v2 from the same adapter, with v2 changing every `lora_B` tensor deterministically.
4. Stage v1 and activate v1.
5. Record greedy outputs for adapter A, adapter B when present, and base.
6. Stage v2 and assert all three outputs still match their pre-stage values.
7. Activate v2 and assert adapter A changes while adapter B and base remain bitwise identical.
8. Restart a fresh v2-only server and assert the activated v2 token IDs match it exactly.
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

- [ ] **Step 2: Run static registration and collection checks**

```bash
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  scripts/lint/check_registered_tests.py
env PYTHONPATH="$PWD/python" /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python -m pytest \
  --collect-only -q test/registered/lora/test_lora_staged_update.py
```

Expected: the test is registered and collection succeeds without launching a server.

- [ ] **Step 3: Commit the reusable end-to-end test**

```bash
git add test/registered/lora/test_lora_staged_update.py
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

Expected: zero failures and zero errors.

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
  test/registered/lora/test_lora_staged_update.py
git diff --check 1c15111fba954babe2b9742caefcbedb22ce306c...HEAD
```

Expected: both commands exit zero.

- [ ] **Step 3: Create the canonical Slurm run record before requesting GPUs**

Use execution ID `20260830-native-staged-lora-01a0532a` and create:

```text
/data/home/zeju/.local/state/remote-cluster-runs/slurm/sglang/codex-port-staged-lora-native/20260830-native-staged-lora-01a0532a/qualification/
```

Record worktree, branch, exact commit, clean/dirty state, shared environment path, test command, requested GPU resources, and controller owner `codex` in `provenance.json`. Bind `stdout.log`, `stderr.log`, and `completion.status` before submission. Ask the user before submitting the multi-GPU portion.

- [ ] **Step 4: Run the registered one-GPU test on a Slurm compute node**

Request the established `viga`/`all` one-GPU shape (`1` node, `1` task, `8` CPUs, `100G` RAM, `1` GPU, `01:00:00`) and execute:

```bash
srun --account=viga --partition=all --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=100G --gres=gpu:1 --time=01:00:00 \
  env PYTHONPATH="$PWD/python" \
  /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/lora/test_lora_staged_update.py \
  TestStagedLoRAUpdate.test_single_gpu
```

Expected: stage is invisible, activation changes only adapter A, v2 matches a fresh v2 server, and the versioned cache key prevents stale reuse.

- [ ] **Step 5: Run TP=2, CUDA-graph, slot-pressure, and MoE cases**

After explicit user approval for the multi-GPU run, request the established two-GPU shape (`1` node, `1` task, `24` CPUs, `160G` RAM, `2` GPUs, `01:30:00`) and run:

```bash
srun --account=viga --partition=all --nodes=1 --ntasks=1 \
  --cpus-per-task=24 --mem=160G --gres=gpu:2 --time=01:30:00 \
  env PYTHONPATH="$PWD/python" \
  /data/home/zeju/miles-orbit-dev/envs/candidate/bin/python \
  test/registered/lora/test_lora_staged_update.py \
  TestStagedLoRAUpdate.test_tp2 \
  TestStagedLoRAUpdate.test_decode_graph_on_and_off \
  TestStagedLoRAUpdate.test_hidden_slot_never_evicts \
  TestStagedLoRAUpdate.test_moe_sharded_placement
```

Expected: identical version on every TP rank, correct eager/graph output, no staging-slot eviction, and successful expert-sharded placement.

- [ ] **Step 6: Snapshot terminal logs and verify the source identity**

Take one bounded snapshot into the matching Mac run-store suffix. Confirm `completion.status` exit code zero, the tested commit equals `git rev-parse HEAD`, and the worktree was clean for the terminal run. Do not report submission as a pass without this evidence.

- [ ] **Step 7: Update and finalize the living HTML report**

Change the report outcome from branch comparison to implemented port, add exact CPU/GPU commands and results, record any skipped hardware matrix item, set `status: final` only if all required gates passed, render, run the freshness check, and inspect light/dark/narrow layouts. Keep the report uncommitted unless the user asks to include it in its repository.

- [ ] **Step 8: Self-review the branch before PR preparation**

Inventory `1c15111fba954babe2b9742caefcbedb22ce306c...HEAD`, review every changed path, verify the staging slot never enters serving maps, verify no registry publication occurs on stage, and verify ordinary one-step upsert still follows its original code path. Fix every blocking finding and rerun the affected tests.

- [ ] **Step 9: Report the ready development branch**

Report the remote worktree, branch, commit series, test evidence, run-store paths, limitations, and the exact next safe action. Do not push, create a PR, merge, or delete the worktree without a new explicit request.

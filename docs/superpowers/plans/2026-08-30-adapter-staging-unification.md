# Symmetric Adapter Staging (LoRA + OFT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give OFT a staged multi-tenant manager symmetric with the one LoRA just got (`StagedLoRAManager`), extract a shared `AdapterStagingBackend` interface so the tokenizer-layer code stops type-branching between methods, and retire both legacy `srt/peft/lora`/`srt/peft/oft` packages once both staged managers are GPU-validated.

**Architecture:** New `srt/oft/staged_manager.py` mirrors the relocated `srt/lora/staged_manager.py` pattern (moved there from the already-shipped `adapter_sync/backends/lora.py`) — one manager subclass that keeps full multi-tenant capability (routing/eviction) and adds a hidden-slot stage/activate transaction on top, not a separate single-active class. A new `AdapterStagingBackend` ABC + registry in `adapter_sync/tokenizer_backend.py` replaces the `if is_native_lora: ... else: ...` branching in `tokenizer_control_mixin.py` with two symmetric backend classes, dispatched by server config rather than per-request fields.

**Tech Stack:** Python, msgspec (io_struct request types), asyncio (tokenizer manager locks), PyTorch (GPU tensors, CUDA streams), pytest (`sglang.test.ci.ci_register.register_cpu_ci` for CI registration).

**Spec:** `/lustre/fast/fast/lechen/software/proj/docs/adapter-staging-unification/design-2026-08-30.md`

## Global Constraints

- Subclass, don't edit, existing manager/pool classes wherever the base class serves other consumers too (`OFTManager`, `OFTMemoryPool`, `AdapterMemPool` all have other callers — do not change their existing method signatures or behavior; only add new subclasses).
- Registries stay separate (`lora_registry` for LoRA, `peft_registry` for OFT) — do not attempt to unify them into one registry as part of this work (see spec's "Registries stay separate" rationale).
- No `getattr(obj, "x", default)` for fields the type guarantees — this codebase's `no-getattr-defensive` rule. Use direct attribute access or an explicit `None` default set at construction.
- Per-method staged manager/pool/tokenizer-backend code goes inside that method's own package (`srt/lora/staged_manager.py`, `srt/oft/staged_manager.py`), never `srt/peft/` and never a third `adapter_sync/backends/` location. `srt/adapter_sync/` holds only the method-agnostic `AdapterStagingBackend` ABC and the `_STAGING_BACKENDS` registry.
- Legacy deletion (Task 8a/8b) only proceeds after their respective GPU gates (Task 7a/7b) pass — do not delete `srt/peft/lora`/`srt/peft/oft` speculatively.

---

## Task 1: Extract `AdapterStagingBackend` + registry, relocate LoRA staging into `srt/lora/`

**Revised (2026-08-30): dispatch must be registry-driven (more
adapter methods are coming, and a hardcoded `if/elif` per method in
`tokenizer_control_mixin.py` means editing that shared file again for each
one), and per-method staged code moves into that method's own package
(`srt/lora/`, `srt/oft/`) instead of a third `adapter_sync/backends/`
location — see spec's Section 2/3 revisions.**

**Files:**
- Create: `python/sglang/srt/adapter_sync/tokenizer_backend.py`
- Create: `python/sglang/srt/lora/staged_manager.py` (moves the *content* of
  the already-shipped `python/sglang/srt/adapter_sync/backends/lora.py` —
  `StagedLoRAManager`, `StagedLoRAMemoryPool`, `PendingLoRAStage` — verbatim,
  plus the new `LoRAStagingBackend`)
- Delete: `python/sglang/srt/adapter_sync/backends/lora.py`,
  `python/sglang/srt/adapter_sync/backends/__init__.py` (empty once `lora.py`
  moves; confirm nothing else imports from `adapter_sync.backends` before
  deleting — `grep -rn "adapter_sync.backends" python/sglang/ test/`)
- Modify: `python/sglang/srt/model_executor/model_runner.py` (`_get_lora_manager_class`'s
  import line: `from sglang.srt.adapter_sync.backends.lora import StagedLoRAManager`
  → `from sglang.srt.lora.staged_manager import StagedLoRAManager`)
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py:529-769` (replace the `_reserve_native_lora_stage`/`_prepare_native_lora_activation`/`_publish_native_lora_activation`/`_finish_native_lora_activation`/`_quarantine_native_lora_activation`/`_assert_native_lora_available`/`_is_native_lora_stage` private methods and the `if is_native_lora` branches inside `update_adapter_from_distributed`/`activate_adapter_version` with a one-time registry-backed `_staging_backend_for`)
- Test: `test/registered/unit/adapter_sync/test_tokenizer_backend.py`
- Test: move `test/registered/unit/adapter_sync/test_lora_staging_backend.py`
  content to `test/registered/unit/lora/test_lora_staged_manager.py` (update
  its imports from `sglang.srt.adapter_sync.backends.lora` to
  `sglang.srt.lora.staged_manager`; the test bodies themselves don't change)

**Interfaces:**
- Produces: `AdapterStagingBackend` (ABC) with `async def reserve_stage(self, obj) -> None`, `def prepare_activation(self, obj) -> None`, `async def finish_activation(self, obj, results) -> Tuple[bool, str]`.
- Produces: `get_staging_backend(tm, obj) -> Optional[AdapterStagingBackend]` — the registry lookup, in `adapter_sync/tokenizer_backend.py`.
- Produces: `LoRAStagingBackend(AdapterStagingBackend)` in `srt/lora/staged_manager.py`, constructed as `LoRAStagingBackend(tm)` where `tm` is the `TokenizerManager` instance (needs `tm.lora_registry`, `tm.lora_ref_cache`, `tm.failed_lora_activations`, `tm.pending_lora_stage`, `tm.server_args`).
- Consumes (Task 4): `OFTStagingBackend` in `srt/oft/staged_manager.py` will implement the same ABC and register itself the same way.

- [ ] **Step 1: Write the failing test for the ABC contract**

```python
# test/registered/unit/adapter_sync/test_tokenizer_backend.py
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAdapterStagingBackendIsAbstract(unittest.TestCase):
    def test_cannot_instantiate_the_bare_interface(self):
        from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend

        with self.assertRaises(TypeError):
            AdapterStagingBackend()

    def test_lora_backend_implements_the_full_interface(self):
        from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend
        from sglang.srt.lora.staged_manager import LoRAStagingBackend

        self.assertTrue(issubclass(LoRAStagingBackend, AdapterStagingBackend))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_tokenizer_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sglang.srt.adapter_sync.tokenizer_backend'`
(this also fails to collect `test_lora_backend_implements_the_full_interface`
since `sglang.srt.lora.staged_manager` doesn't exist until Step 5 — both
failures are expected at this point, not a sign anything is wrong)

- [ ] **Step 3: Write `AdapterStagingBackend`**

```python
# python/sglang/srt/adapter_sync/tokenizer_backend.py
"""Tokenizer-layer staging interface shared by every staged adapter method.

Each PEFT method that supports staged (RL) weight updates implements this
ABC once, so TokenizerControlMixin.update_adapter_from_distributed /
activate_adapter_version can orchestrate the shared pause/lock/dispatch
sequence without branching on which method is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class AdapterStagingBackend(ABC):
    """One instance per TokenizerManager, selected by ``obj.load_format``."""

    @abstractmethod
    async def reserve_stage(self, obj) -> None:
        """Called before dispatch to the scheduler. Reserve/mint the adapter
        identity for this stage request, mutating ``obj.adapter_id`` if the
        method resolves identity from a name. Must be safe to call twice with
        the same (name, version) — the caller does not deduplicate."""

    @abstractmethod
    def prepare_activation(self, obj) -> None:
        """Called before dispatch, synchronously, for an activate request.
        Validate the pending stage matches ``obj``'s identity/version and
        raise ``ValueError`` if not. Mutates ``obj.adapter_id`` the same way
        ``reserve_stage`` does, for the activate wire path."""

    @abstractmethod
    async def finish_activation(self, obj, results: List) -> Tuple[bool, str]:
        """Called after every worker's stage-then-activate or activate RPC
        returns. ``results`` is the list of per-worker RPC outputs. Must
        publish the new identity into the method's registry only on
        all-worker success, and return ``(success, message)``."""


# Explicit registry, not decorator/import-side-effect magic: this codebase's
# packages are deliberately lazy-imported (see srt/oft/__init__.py's PEP 562
# pattern) to avoid circular imports and boot-time cost, so a module isn't
# guaranteed to have run its top-level code before dispatch needs it. Adding
# a third adapter method means adding one entry here -- never touching
# tokenizer_control_mixin.py again.
#
# Keyed by the server_args enable-flag, not obj.load_format: native LoRA and
# OFT are already mutually-exclusive server configurations (enable_lora vs.
# peft_method=="oft"), so at most one flag below is ever true for a given
# server process -- checking obj.load_format on top of that would be
# re-deriving information the server's own config already determines. Each
# backend's reserve_stage still validates obj.load_format internally as a
# sanity check; that's the right layer for it, not this lookup.
#
# is_enabled is a lambda doing DIRECT attribute access (sa.enable_lora_staging),
# not a string-keyed getattr(sa, flag_name, False) -- these fields are
# guaranteed real ServerArgs fields, so a getattr-with-default there would be
# exactly the no-getattr-defensive anti-pattern this codebase's rules forbid.
# getattr(module, class_name) below is a different, necessary case: this
# module defines AdapterStagingBackend, and srt/lora/staged_manager.py
# imports it to subclass -- an eager top-level import back from here would
# be a genuine circular import (staged_manager.py can't finish defining its
# class until AdapterStagingBackend already exists). The lazy
# importlib.import_module + getattr(module, class_name) is what breaks that
# cycle, not a style choice.
_STAGING_BACKENDS = [
    (lambda sa: sa.enable_lora_staging, "sglang.srt.lora.staged_manager", "LoRAStagingBackend"),
    (lambda sa: sa.oft_impl == "staged", "sglang.srt.oft.staged_manager", "OFTStagingBackend"),
]


def get_staging_backend(tm, obj):
    """Resolve the active staging backend for this server, or ``None`` if
    no staging method is enabled."""
    import importlib

    for is_enabled, module_path, class_name in _STAGING_BACKENDS:
        if is_enabled(tm.server_args):
            module = importlib.import_module(module_path)
            return getattr(module, class_name)(tm)
    return None
```

- [ ] **Step 4: Run test to verify the ABC checks pass**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_tokenizer_backend.py -v`
Expected: FAIL still, at `test_lora_backend_implements_the_full_interface` — `LoRAStagingBackend` doesn't exist yet. (`_STAGING_BACKENDS`'s OFT entry points at a module that doesn't exist until Task 4 — this is fine, `get_staging_backend` only imports it lazily when its `is_enabled` lambda returns True, which requires `oft_impl == "staged"`, not exercised by this task's tests.)

- [ ] **Step 5: Move the already-shipped LoRA staging code into `srt/lora/staged_manager.py`, add `LoRAStagingBackend`**

```bash
git mv python/sglang/srt/adapter_sync/backends/lora.py python/sglang/srt/lora/staged_manager.py
```

Then edit the moved file: no changes needed to `StagedLoRAMemoryPool`/
`StagedLoRAManager`/`PendingLoRAStage`'s bodies (the code that already
exists, read in full during planning, is unchanged by this move — only its
file location changes), but update its own imports (it previously imported
`sglang.srt.lora.lora_manager`/`lora_registry`/`mem_pool`/`lora.LoRAAdapter`/
`lora_config.LoRAConfig` from *outside* `srt/lora/`; now that the file lives
*inside* `srt/lora/`, these become sibling-module imports —
`from sglang.srt.lora.lora import LoRAAdapter` etc. stay as absolute imports,
unchanged, since this codebase uses absolute imports throughout, not
relative — confirm this by checking how `srt/lora/mem_pool.py` itself
imports its siblings before assuming no import lines need editing).

Append `LoRAStagingBackend`:

```python
from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend


class LoRAStagingBackend(AdapterStagingBackend):
    """Tokenizer-layer staging for native LoRA. Wraps TokenizerManager's
    lora_registry/lora_ref_cache/failed_lora_activations/pending_lora_stage
    state — same objects tokenizer_control_mixin.py used directly before
    this extraction; the fields did not move, only which class reads them."""

    def __init__(self, tm):
        self._tm = tm

    def _assert_available(self, lora_path) -> None:
        names = [lora_path] if isinstance(lora_path, str) else (lora_path or [])
        for name in names:
            if name in self._tm.failed_lora_activations:
                raise ValueError(
                    f"LoRA adapter '{name}' is unavailable after a partial "
                    "activation failure; restart required"
                )

    def _quarantine(self, name: str, message: str) -> None:
        self._tm.failed_lora_activations[name] = message

    async def reserve_stage(self, obj) -> None:
        # Reservation must be atomic across concurrent stage requests. The
        # communicator serializes dispatch, but reservation happens before it.
        async with self._tm.lora_update_lock:
            await self._reserve_locked(obj)

    async def _reserve_locked(self, obj) -> None:
        if obj.load_format != "lora_adapter" or not obj.adapter_name:
            raise ValueError(
                "native LoRA staging requires load_format=lora_adapter "
                "and adapter_name"
            )
        try:
            version = int(obj.adapter_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native LoRA staging requires an integer adapter_version"
            ) from exc
        if self._tm.server_args.tokenizer_worker_num > 1:
            raise ValueError("native LoRA staging requires tokenizer_worker_num == 1")
        if obj.adapter_name in self._tm.failed_lora_activations:
            raise ValueError(
                f"LoRA adapter '{obj.adapter_name}' is quarantined; restart required"
            )

        pending = self._tm.pending_lora_stage
        if pending is not None:
            if pending.lora_name == obj.adapter_name and pending.version == version:
                obj.adapter_id = pending.lora_id
                return
            raise ValueError(
                "staging slot already reserved for "
                f"name={pending.lora_name} id={pending.lora_id} "
                f"version={pending.version}"
            )

        candidate, _ = await self._tm.lora_registry.register_or_reuse(
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
        self._tm.pending_lora_stage = candidate
        obj.adapter_id = candidate.lora_id

    def prepare_activation(self, obj) -> None:
        try:
            version = int(obj.adapter_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native LoRA activation requires an integer adapter_version"
            ) from exc

        pending = self._tm.pending_lora_stage
        if pending is None or (pending.lora_name, pending.version) != (
            obj.adapter_name,
            version,
        ):
            if pending is None:
                detail = "no native LoRA stage is pending"
            else:
                detail = (
                    f"pending name={pending.lora_name} id={pending.lora_id} "
                    f"version={pending.version}"
                )
            raise ValueError(
                f"Cannot activate name={obj.adapter_name} version={version}; {detail}"
            )
        self._assert_available(obj.adapter_name)
        obj.adapter_id = pending.lora_id

    async def _publish(self) -> None:
        pending = self._tm.pending_lora_stage
        if pending is None:
            raise RuntimeError("No native LoRA stage is pending for publication")
        registered = self._tm.lora_registry.get_all_adapters().get(pending.lora_name)
        if registered is None:
            await self._tm.lora_registry.register(pending)
        else:
            await self._tm.lora_registry.refresh(pending)
        self._tm.lora_ref_cache[pending.lora_name] = pending
        self._tm.failed_lora_activations.pop(pending.lora_name, None)
        self._tm.pending_lora_stage = None

    async def finish_activation(self, obj, results):
        from sglang.srt.managers.communicator import FanOutCommunicator

        pending = self._tm.pending_lora_stage
        if pending is None:
            raise RuntimeError("No native LoRA stage is pending during activation")
        success, message = FanOutCommunicator.merge_results(results)
        expected_version = int(obj.adapter_version)

        def version_matches(result) -> bool:
            try:
                return int(result.active_adapter_version) == expected_version
            except (TypeError, ValueError):
                return False

        versions_match = bool(results) and all(version_matches(r) for r in results)
        if success and versions_match:
            await self._publish()
            return True, message

        active_versions = [getattr(r, "active_adapter_version", None) for r in results]
        failure = (
            "Native LoRA activation consistency failure for "
            f"adapter '{pending.lora_name}' version={pending.version}: "
            f"{message}; worker active versions={active_versions}; restart required"
        )
        self._quarantine(pending.lora_name, failure)
        return False, failure
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_tokenizer_backend.py -v`
Expected: PASS, both cases.

- [ ] **Step 7: Refactor `tokenizer_control_mixin.py`'s orchestrators to use the backend**

Replace lines 529-769 (the private helper methods plus the two orchestrator
functions) with:

```python
    def _staging_backend_for(self, obj):
        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        return get_staging_backend(self, obj)

    async def update_adapter_from_distributed(
        self: TokenizerManager,
        obj: UpdateAdapterFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Double-buffer PEFT STAGE over NCCL.

        double_buffer=True: LOCK-FREE stage into the reserved staging slot while
        generation continues (overlaps decode); no writer_lock. double_buffer=
        False: the synchronous distributed path stages then ACTIVATEs-in-place in
        the scheduler in one round-trip, so we hold model_update_lock.writer_lock
        (drain-to-idle, mirror update_weights_from_distributed) around it."""
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update adapter from distributed"

        from sglang.srt.peft import tokenizer_hooks as peft_tokenizer_hooks

        backend = self._staging_backend_for(obj)
        if backend is not None:
            await backend.reserve_stage(obj)
        else:
            # The existing PEFT path remains register-before-dispatch.
            await peft_tokenizer_hooks.register_peft_ref(self, obj)

        if obj.double_buffer:
            results = await self.update_adapter_from_distributed_communicator(obj)
            success, message = FanOutCommunicator.merge_results(results)
            if backend is not None:
                return success, message
        else:
            # Hold is_pause_cond while updating to prevent unpause from racing.
            async with self.is_pause_cond:
                is_paused = self.is_pause
                if is_paused:
                    results = await self.update_adapter_from_distributed_communicator(
                        obj
                    )
                    if backend is not None:
                        backend_result = await backend.finish_activation(obj, results)
            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    results = await self.update_adapter_from_distributed_communicator(
                        obj
                    )
                    if backend is not None:
                        backend_result = await backend.finish_activation(obj, results)
            if backend is not None:
                return backend_result
            success, message = FanOutCommunicator.merge_results(results)

        message += await peft_tokenizer_hooks.bump_peft_version(self, obj, success)
        return success, message

    async def activate_adapter_version(
        self: TokenizerManager,
        obj: ActivateAdapterVersionReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Double-buffer PEFT ACTIVATE (the drained atomic swap). The drain lives
        HERE: model_update_lock.writer_lock waits for all in-flight generation
        reader_locks to release (drain running_batch to empty) and blocks new
        admission -- exactly what update_weights_from_disk/from_distributed use.
        Only THEN is the activate control request sent to the scheduler (a simple
        staging->active flip, since the batch is already drained); releasing the
        lock on return resumes admission."""
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for activate adapter version"

        backend = self._staging_backend_for(obj)
        if backend is not None:
            backend.prepare_activation(obj)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.activate_adapter_version_communicator(obj)
                if backend is not None:
                    backend_result = await backend.finish_activation(obj, results)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.activate_adapter_version_communicator(obj)
                if backend is not None:
                    backend_result = await backend.finish_activation(obj, results)

        if backend is not None:
            return backend_result

        success, message = FanOutCommunicator.merge_results(results)
        return success, message
```

Note: `_staging_backend_for`/`get_staging_backend` already resolve
`"oft_adapter"` via `_BACKEND_REGISTRY` (Step 3) — that entry just can't
successfully import anything until Task 4 creates `srt/oft/staged_manager.py`.
No further edit to this file is needed once Task 4 lands; that's the entire
point of the registry. `prepare_activation` is intentionally synchronous
(matches the pre-refactor `_prepare_native_lora_activation`, which did no
awaiting).

- [ ] **Step 8: Run the full existing LoRA staging test suite to confirm no regression**

Run: `python3 -m pytest test/registered/unit/lora/test_lora_staged_manager.py test/registered/unit/lora/test_lora_staging_control.py test/registered/unit/lora/test_lora_versioning.py test/registered/unit/adapter_sync/ -v`
Expected: PASS, same pass count as before this task (this is a pure refactor — no test's expected behavior changes; `test_lora_staged_manager.py` is the moved/renamed `test_lora_staging_backend.py` from this task's Files section).

- [ ] **Step 9: Commit**

```bash
git add python/sglang/srt/adapter_sync/tokenizer_backend.py \
        python/sglang/srt/lora/staged_manager.py \
        python/sglang/srt/managers/tokenizer_control_mixin.py \
        python/sglang/srt/model_executor/model_runner.py \
        test/registered/unit/adapter_sync/test_tokenizer_backend.py \
        test/registered/unit/lora/test_lora_staged_manager.py
git rm python/sglang/srt/adapter_sync/backends/lora.py \
       python/sglang/srt/adapter_sync/backends/__init__.py \
       test/registered/unit/adapter_sync/test_lora_staging_backend.py
git commit -m "refactor(adapter_sync): extract AdapterStagingBackend, move LoRA staging into srt/lora/"
```

---

## Task 2: `StagedOFTMemoryPool` — hidden slot + per-uid stage/activate

**Files:**
- Create: `python/sglang/srt/oft/staged_manager.py`
- Test: `test/registered/unit/adapter_sync/test_oft_staging_backend.py`

**Interfaces:**
- Consumes: `sglang.srt.oft.mem_pool.OFTMemoryPool` (constructor signature — read `python/sglang/srt/oft/mem_pool.py:225-243` before writing this task's implementation; the plan's own reading of it, above, may drift from the file by the time this task executes).
- Consumes: `sglang.srt.oft.base.mem_pool.AdapterMemPool._groups` (dict of `name -> {key: tensor}`, each tensor's dim 0 is the adapter slot — confirmed via `python/sglang/srt/oft/base/mem_pool.py:80-113`).
- Produces: `StagedOFTMemoryPool(OFTMemoryPool)` with `stage(uid, version, named_tensors) -> None`, `activate(uid, version, destination) -> None`, `discard_stage(uid, version) -> None`, `staged_identity() -> Optional[Tuple[str, int]]`.

**Why this can't just call the inherited `AdapterMemPool.stage`/`activate`:**
`AdapterMemPool.activate()` (base class, still used by OFT's existing
single-active double-buffer path) blanket-copies every group's staging slot
into ONE pool-wide `active_idx`, tracked by a single `_staged_version`/
`_active_version` — there is no `uid` parameter at all. That's correct for
"exactly one adapter can ever be resident" but wrong once B1 multi-tenancy
means several different adapters are resident in different slots
simultaneously — activating one must not touch the others' slots. Do not
edit `AdapterMemPool` itself (other code depends on its current pool-wide
behavior for the non-multi-tenant single-active path) — override in the
subclass instead, mirroring `StagedLoRAMemoryPool`'s override-not-edit
pattern.

- [ ] **Step 1: Write the failing test for slot reservation**

```python
# test/registered/unit/adapter_sync/test_oft_staging_backend.py
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


TARGET_MODULE = "q_proj"
BLOCK_SIZE = 4


def _make_pool(max_ofts_per_batch=4):
    from sglang.srt.oft.staged_manager import StagedOFTMemoryPool

    base_hf_config = MagicMock()
    base_hf_config.num_hidden_layers = 1
    base_hf_config.hidden_size = 8
    pool = StagedOFTMemoryPool(
        base_hf_config=base_hf_config,
        max_ofts_per_batch=max_ofts_per_batch,
        dtype=torch.float32,
        tp_size=1,
        tp_rank=0,
        max_oft_block_size=BLOCK_SIZE,
        target_modules={TARGET_MODULE},
        base_model=MagicMock(),
        eviction_policy="lru",
        oft_added_tokens_size=0,
        oft_type="canonical_oft",
    )
    return pool


def _named_tensors_for_layer_0(fill_value: float):
    """Real _fill_slot payload (python/sglang/srt/oft/mem_pool.py:647-677):
    maps (target_module, layer_id) -> (r, block_size, slice_index, split_count).
    r is the compact per-block rotation-generator tensor _write_oft_r_block
    expects; shape/content correctness for the OFT math itself is covered by
    existing tests in test/registered/unit/oft/ -- these tests only need a
    tensor _write_oft_r_block will accept without raising, distinguishable by
    fill_value so slot-isolation assertions can tell slots apart."""
    r = torch.full((BLOCK_SIZE, BLOCK_SIZE), fill_value, dtype=torch.float32)
    return {(TARGET_MODULE, 0): (r, BLOCK_SIZE, None, 1)}


class TestOFTStagingSlotReservation(unittest.TestCase):
    def test_staging_slot_sits_outside_the_advertised_capacity(self):
        pool = _make_pool(max_ofts_per_batch=4)
        self.assertEqual(pool.max_ofts_per_batch, 4)
        self.assertEqual(pool.staging_idx, 4)

    def test_available_serving_slots_excludes_the_hidden_slot(self):
        pool = _make_pool(max_ofts_per_batch=4)
        self.assertEqual(pool.available_serving_slots(), 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sglang.srt.oft.staged_manager'`

- [ ] **Step 3: Write `StagedOFTMemoryPool`**

Before writing, read `python/sglang/srt/oft/mem_pool.py`'s `OFTMemoryPool.__init__`
and its `_declare_groups`/`init_buffers`-equivalent (name may differ —
`OFTMemoryPool` does not necessarily have a method literally named
`init_buffers` the way `LoRAMemoryPool` does; find the method that first
calls `register_buffer_group`/`_declare_groups` and allocates
`max_ofts_per_batch`-sized tensors) to confirm the exact override point.

```python
# python/sglang/srt/oft/staged_manager.py
"""Symmetric counterpart to backends/lora.py: OFT staging through one
hidden memory-pool slot, alongside B1's existing multi-tenant admission
and eviction (unaffected by this file)."""

import logging
from typing import Optional, Tuple

from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager

logger = logging.getLogger(__name__)


class StagedOFTMemoryPool(OFTMemoryPool):
    """OFT pool with one physical slot hidden from serving, and per-uid
    stage/activate (unlike the inherited AdapterMemPool.stage/activate,
    which are pool-wide single-slot and used only by the non-multi-tenant
    double-buffer path)."""

    def __init__(self, *args, **kwargs):
        self.staging_idx = None
        self._staged_uid = None
        self._staged_version = None
        self._active_versions = {}
        super().__init__(*args, **kwargs)
        # Allocate one extra physical slot on top of what OFTMemoryPool's
        # __init__ already registered via _declare_groups at
        # self.max_ofts_per_batch. Widen every group's tensors by one row.
        advertised = self.max_ofts_per_batch
        for name, keyed in self._groups.items():
            for key, tensor in keyed.items():
                widened = tensor.new_empty(advertised + 1, *tensor.shape[1:])
                widened[:advertised].copy_(tensor)
                keyed[key] = widened
        self.staging_idx = advertised

    def available_serving_slots(self) -> int:
        return self.max_ofts_per_batch

    def staged_identity(self) -> Optional[Tuple[str, int]]:
        if self._staged_uid is None:
            return None
        return self._staged_uid, self._staged_version

    def _require_staged_identity(self, uid: str, version: int) -> None:
        current = self.staged_identity()
        if current != (uid, version):
            detail = (
                "the staging slot is empty"
                if current is None
                else f"it holds uid={current[0]} version={current[1]}"
            )
            raise ValueError(
                f"No staged OFT adapter matches uid={uid} version={version}; {detail}."
            )

    def stage(self, uid: str, version: int, named_tensors) -> None:
        current = self.staged_identity()
        if current == (uid, version):
            return
        if current is not None:
            raise ValueError(
                f"Staging slot already holds uid={current[0]} version={current[1]}."
            )
        self._fill_slot(self.staging_idx, named_tensors)
        self._staged_uid = uid
        self._staged_version = version

    def activate(self, uid: str, version: int, destination: int) -> None:
        self._require_staged_identity(uid, version)
        if (
            destination < 0
            or destination >= self.max_ofts_per_batch
            or destination == self.staging_idx
        ):
            raise ValueError(
                f"OFT activation destination {destination} is not a serving slot."
            )
        for name, keyed in self._groups.items():
            for key in keyed:
                self.slot(name, key, destination).copy_(
                    self.slot(name, key, self.staging_idx)
                )
        self._active_versions[uid] = version
        self._staged_uid = None
        self._staged_version = None

    def discard_stage(self, uid: str, version: int) -> None:
        self._require_staged_identity(uid, version)
        self._staged_uid = None
        self._staged_version = None

    def active_version_for(self, uid: str) -> Optional[int]:
        return self._active_versions.get(uid)
```

Note on the widen-in-`__init__` approach: `OFTMemoryPool.__init__` (via its
superclass chain) already calls `register_buffer_group` for every module at
`self.max_ofts_per_batch` rows during `super().__init__(*args, **kwargs)`
above. Widening every group's tensors by one row *after* that call, rather
than passing `max_ofts_per_batch + 1` into `super().__init__` and shrinking
what's advertised (the `StagedLoRAMemoryPool` approach, via
`init_buffers`), is necessary because `OFTMemoryPool` doesn't expose a
single override point equivalent to LoRA's `init_buffers` — verify this
against the actual `OFTMemoryPool.__init__` body at implementation time,
and switch to the shrink-then-restore approach (matching
`StagedLoRAMemoryPool.init_buffers` exactly) if `OFTMemoryPool` does have
such a hook.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for stage/activate/discard**

```python
class TestOFTStagingTransaction(unittest.TestCase):
    def test_stage_then_activate_writes_only_the_destination_slot(self):
        pool = _make_pool(max_ofts_per_batch=4)
        slot_0_before = pool.slot(f"R:{TARGET_MODULE}", 0, 0).clone()

        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=9.0))
        pool.activate("adapter-a", 1, destination=2)

        self.assertTrue(
            (pool.slot(f"R:{TARGET_MODULE}", 0, 0) == slot_0_before).all(),
            "activating one uid must not touch slot 0",
        )
        self.assertTrue(
            (pool.slot(f"R:{TARGET_MODULE}", 0, 2) == 9.0).all(),
            "activate must copy the staged value into the destination slot",
        )
        self.assertEqual(pool.active_version_for("adapter-a"), 1)
        self.assertIsNone(pool.staged_identity())

    def test_activate_rejects_a_different_adapter_than_was_staged(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=9.0))
        with self.assertRaises(ValueError):
            pool.activate("adapter-b", 1, destination=2)

    def test_activate_rejects_the_staging_slot_as_a_destination(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=9.0))
        with self.assertRaises(ValueError):
            pool.activate("adapter-a", 1, destination=pool.staging_idx)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py::TestOFTStagingTransaction -v`
Expected: PASS on all three cases. If `_write_oft_r_block` rejects the
`_named_tensors_for_layer_0` fixture's tensor shape (e.g. it expects a
differently-shaped compact weight than a plain `(BLOCK_SIZE, BLOCK_SIZE)`
tensor), read `_write_oft_r_block` in `python/sglang/srt/oft/mem_pool.py`
(referenced from `_fill_slot` at line 670) and adjust the fixture's `r`
tensor shape to match — the assertions above (slot isolation, correct
destination) do not depend on the specific shape chosen.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/oft/staged_manager.py \
        test/registered/unit/adapter_sync/test_oft_staging_backend.py
git commit -m "feat(adapter_sync): StagedOFTMemoryPool with hidden slot + per-uid stage/activate"
```

---

## Task 3: `StagedOFTManager` — wire the pool into a multi-tenant-capable manager

**Files:**
- Modify: `python/sglang/srt/oft/staged_manager.py` (add `StagedOFTManager`)
- Test: `test/registered/unit/adapter_sync/test_oft_staging_backend.py` (extend)

**Interfaces:**
- Consumes: `StagedOFTMemoryPool` (Task 2).
- Consumes: `sglang.srt.oft.oft_manager.OFTManager` — its `init_memory_pool`-equivalent construction method and constructor kwargs (read `python/sglang/srt/oft/oft_manager.py:195-253` before writing; the plan's earlier reading of this file, captured above in this conversation's history, showed the constructor signature but not the exact pool-construction call site — confirm it's a method named similarly to LoRA's `init_memory_pool` before assuming the override point).
- Produces: `StagedOFTManager(OFTManager)` with `stage_adapter(named_tensors, config, name, version, adapter_id=None)` and `activate_adapter(name, version, adapter_id=None)`, both returning the same output type OFT's existing `AdapterManager.stage_adapter`/`activate_adapter` return (find this type in `python/sglang/srt/oft/base/manager.py:122-160` — likely a `(bool, str)` tuple based on that file's earlier-read signatures, but confirm, since `LoRAUpdateOutput` was LoRA's actual return type, not a raw tuple, and OFT's equivalent may differ).

- [ ] **Step 1: Write the failing test**

```python
class TestStagedOFTManagerConstruction(unittest.TestCase):
    def test_init_memory_pool_builds_a_staged_pool(self):
        # Build the minimal server_args/model stand-ins StagedOFTManager's
        # constructor chain needs -- follow the pattern already established
        # by test_lora_staging_backend.py's own manager-construction tests
        # for the equivalent LoRA case, adapted to OFTManager's actual
        # __init__ signature (python/sglang/srt/oft/oft_manager.py:195-253).
        ...
```

Do not write this step's full body speculatively — read
`test/registered/unit/adapter_sync/test_lora_staging_backend.py`'s
manager-construction test fixtures first (they already solve "how do you
construct a manager subclass without a real model") and adapt the same
fixture-building approach to `OFTManager`'s constructor, since the two
managers' constructors take different keyword arguments.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py::TestStagedOFTManagerConstruction -v`
Expected: FAIL — `StagedOFTManager` doesn't exist yet.

- [ ] **Step 3: Write `StagedOFTManager`**

```python
class PendingOFTStage:
    """CPU metadata retained until a staged OFT adapter is activated."""

    __slots__ = ("uid", "version", "named_tensors", "config", "name")

    def __init__(self, uid, version, named_tensors, config, name):
        self.uid = uid
        self.version = version
        self.named_tensors = named_tensors
        self.config = config
        self.name = name


class StagedOFTManager(OFTManager):
    """OFT manager with an explicit stage/activate transaction, alongside
    B1's existing multi-tenant admission and eviction (unaffected)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_oft_stage = None

    def stage_adapter(self, named_tensors, config, name, version, adapter_id=None):
        uid = adapter_id if adapter_id is not None else name
        try:
            version = int(version)
            pending = self._pending_oft_stage
            if pending is not None:
                if (pending.uid, pending.version) == (uid, version):
                    return True, "Succeeded to stage adapter online."
                raise ValueError(
                    f"An OFT stage is already pending for uid={pending.uid} "
                    f"version={pending.version}."
                )
            self.memory_pool.stage(uid, version, named_tensors)
            self._pending_oft_stage = PendingOFTStage(
                uid=uid,
                version=version,
                named_tensors=named_tensors,
                config=config,
                name=name,
            )
        except Exception as error:
            return False, str(error)
        return True, "Succeeded to stage adapter online."

    def activate_adapter(self, name, version, adapter_id=None):
        uid = adapter_id if adapter_id is not None else name
        try:
            version = int(version)
        except Exception as error:
            return False, str(error)

        pending = self._pending_oft_stage
        if pending is None or (pending.uid, pending.version) != (uid, version):
            detail = (
                "no OFT stage is pending"
                if pending is None
                else f"pending uid={pending.uid} version={pending.version}"
            )
            return False, f"Cannot activate uid={uid} version={version}; {detail}."

        destination = self.memory_pool.uid_to_buffer_id.get(uid)
        if destination is None:
            return False, f"No serving slot is reserved for adapter uid={uid}."
        try:
            self.memory_pool.activate(uid, version, destination)
        except Exception as activation_error:
            return False, str(activation_error)

        # REQUIRED, not optional: OFTManager.prepare_oft_batch reads
        # self.adapters[uid].block_size / self.configs[uid].block_size
        # (python/sglang/srt/oft/oft_manager.py:454-456) for every resident
        # uid on every batch. Activating a new uid without populating these
        # leaves it physically live in the GPU slot but invisible to the
        # manager's own bookkeeping -- the next prepare_oft_batch call for
        # this uid raises KeyError, or worse, silently reads a stale/absent
        # config. Before this line ships, read oft_manager.py:670-711 (the
        # existing streamed-adapter construction path,
        # `self.adapters[oft_ref.adapter_id] = oft_adapter`) and call
        # whatever helper it uses to build an OFTAdapter/OFTConfig from
        # pending.named_tensors/pending.config, then assign:
        #   self.configs[uid] = <constructed OFTConfig>
        #   self.adapters[uid] = <constructed OFTAdapter>
        # This is the direct analogue of StagedLoRAManager.activate_adapter's
        # `self.configs[uid] = pending.config; self.loras[uid] = pending.adapter`
        # (python/sglang/srt/adapter_sync/backends/lora.py:305-306) -- do not
        # skip it because OFT's pool-level activate() succeeded; the pool and
        # the manager track different, both-required state.

        self.memory_pool.discard_stage(uid, version)
        self._pending_oft_stage = None
        return True, "Succeeded to activate adapter version."
```

Note: unlike `StagedLoRAManager.activate_adapter`, this has no
restore-previous-adapter-on-failure branch, because OFT's `activate` (Task
2) copies directly from the staging slot to the destination slot in one
`.copy_()` per group/key — there is no window where the destination slot
is partially overwritten before failure (the copy either fully succeeds or
raises before any tensor is touched, since `_require_staged_identity`
validates before any `.copy_()` call). Confirm this reasoning holds against
the actual GPU behavior in Task 7b's gate — if a partial-copy failure mode
is discovered there, port `StagedLoRAManager`'s restore-on-failure pattern
back into this method.

- [ ] **Step 4: Write the test that guards the manager-bookkeeping gap flagged in Step 3's comment**

```python
class TestActivateUpdatesManagerBookkeeping(unittest.TestCase):
    def test_activate_populates_configs_and_adapters_for_the_new_uid(self):
        # Build a StagedOFTManager the same way Step 1's construction test
        # does, stage+activate "adapter-a", then assert:
        #   self.assertIn("adapter-a", manager.configs)
        #   self.assertIn("adapter-a", manager.adapters)
        # This must fail before Step 3's bookkeeping lines are written (the
        # dicts are only populated by the boot-time streamed-adapter path
        # today) and pass after. Do not skip this test to save time -- it is
        # the only thing that catches the exact silent-failure mode Step 3's
        # comment describes: a physically-active adapter invisible to
        # prepare_oft_batch's per-uid config/block-size lookups.
        ...
```

- [ ] **Step 5: Run test to verify it fails, then passes after Step 3's bookkeeping lines are added**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py -v`
Expected: FAIL until `StagedOFTManager.activate_adapter` populates
`self.configs[uid]`/`self.adapters[uid]` per Step 3's comment; PASS after.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/oft/staged_manager.py \
        test/registered/unit/adapter_sync/test_oft_staging_backend.py
git commit -m "feat(adapter_sync): StagedOFTManager, symmetric with StagedLoRAManager"
```

---

## Task 4: `OFTStagingBackend` — tokenizer-layer symmetry

**Files:**
- Modify: `python/sglang/srt/oft/staged_manager.py` (add `OFTStagingBackend`)
- Test: `test/registered/unit/adapter_sync/test_oft_staging_backend.py` (extend)

**Interfaces:**
- Consumes: `AdapterStagingBackend` (Task 1).
- Consumes: `sglang.srt.peft.tokenizer_hooks.register_peft_ref`, `bump_peft_version` (existing, `python/sglang/srt/peft/tokenizer_hooks.py:94-125`, read in full above).
- Produces: `OFTStagingBackend(AdapterStagingBackend)` constructed as `OFTStagingBackend(tm)`.

- [ ] **Step 1: Write the failing test**

```python
class TestOFTStagingBackendIsSymmetric(unittest.TestCase):
    def test_implements_the_shared_interface(self):
        from sglang.srt.oft.staged_manager import OFTStagingBackend
        from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend

        self.assertTrue(issubclass(OFTStagingBackend, AdapterStagingBackend))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py::TestOFTStagingBackendIsSymmetric -v`
Expected: FAIL — `OFTStagingBackend` doesn't exist yet.

- [ ] **Step 3: Write `OFTStagingBackend`**

```python
from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend


class OFTStagingBackend(AdapterStagingBackend):
    """Tokenizer-layer staging for OFT, wrapping the existing peft_tokenizer_hooks
    registry logic rather than reimplementing it -- OFT's tokenizer-side
    registration/version-bump behavior does not change with this refactor,
    only how it's selected."""

    def __init__(self, tm):
        self._tm = tm

    async def reserve_stage(self, obj) -> None:
        from sglang.srt.peft import tokenizer_hooks as peft_tokenizer_hooks

        await peft_tokenizer_hooks.register_peft_ref(self._tm, obj)

    def prepare_activation(self, obj) -> None:
        # OFT's existing activate path resolves identity from obj.adapter_id,
        # already set by reserve_stage on the prior stage call; no separate
        # pre-activation validation exists in the current peft_tokenizer_hooks
        # flow. Confirm no validation is being silently dropped here by
        # diffing this method's no-op against the pre-refactor call sites in
        # tokenizer_control_mixin.py's activate_adapter_version (Task 1's
        # diff) at implementation time.
        return

    async def finish_activation(self, obj, results):
        from sglang.srt.managers.communicator import FanOutCommunicator
        from sglang.srt.peft import tokenizer_hooks as peft_tokenizer_hooks

        success, message = FanOutCommunicator.merge_results(results)
        message += await peft_tokenizer_hooks.bump_peft_version(self._tm, obj, success)
        return success, message
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py -v`
Expected: PASS, all cases.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/staged_manager.py \
        test/registered/unit/adapter_sync/test_oft_staging_backend.py
git commit -m "feat(adapter_sync): OFTStagingBackend, symmetric with LoRAStagingBackend"
```

---

## Task 5: Extend `oft_impl` with a `"staged"` choice and wire manager construction

**Revised (2026-08-30): no new `enable_oft_staging` field.**
`peft/config.py` already has `oft_impl` (validated against
`OFT_IMPL_CHOICES`), documented as "which OFT implementation serves when
peft_method == 'oft'" — exactly the axis this task needs. A separate
boolean would let `oft_impl="peft"` + `enable_oft_staging=True` be set
simultaneously with no real meaning. Extend the existing enum to three
values instead: `"peft"` (legacy rollback), `"sibling"` (current default),
`"staged"` (this task's addition) — one field, no invalid combination
representable.

**No `tokenizer_control_mixin.py` edit in this task** — `_staging_backend_for`
(Task 1) and `_STAGING_BACKENDS` (Task 1, already lists the OFT entry as
`(lambda sa: sa.oft_impl == "staged", "sglang.srt.oft.staged_manager", "OFTStagingBackend")`)
were written registry-first specifically so this task doesn't need to touch
that file again. This task only needs: extending `OFT_IMPL_CHOICES`, and
manager construction.

**Files:**
- Modify: `python/sglang/srt/peft/config.py` (extend `OFT_IMPL_CHOICES` and `oft_impl`'s help text — find both via `grep -n "OFT_IMPL_CHOICES\|oft_impl" python/sglang/srt/peft/config.py`)
- Modify: `python/sglang/srt/model_executor/model_runner.py` (construction site, mirroring the existing `--enable-lora-staging` branch — read the exact current branch first, since Task 1-4 did not touch this file and its exact current shape wasn't re-verified in this planning pass)
- Test: `test/registered/unit/adapter_sync/test_tokenizer_backend.py` (extend)

**Interfaces:**
- Consumes: `StagedOFTManager` (Task 3), `OFTStagingBackend` (Task 4).

- [ ] **Step 1: Extend `OFT_IMPL_CHOICES` and `oft_impl`'s validation**

Read `python/sglang/srt/peft/config.py`'s current `OFT_IMPL_CHOICES`
definition and the `oft_impl` field's help text (both found via the grep
above) and add `"staged"` to the choices list, updating the help text to
name all three implementations. Also check
`validate_peft_args`'s existing `if server_args.oft_impl == "sibling":`
branch (around `peft/config.py:303`, read in full during planning — it
selects `OFTRef`'s implementation class for ref normalization) for whether
`"staged"` needs the same branch taken (it does — `StagedOFTManager` uses
the same `OFTRef` family as `sibling`, just with staging added; confirm
this by checking `StagedOFTManager`'s superclass chain, `OFTManager`, uses
the same `OFTRef` import as the plain sibling path).

- [ ] **Step 2: Write the failing test confirming the registry now resolves end-to-end**

```python
class TestStagingBackendSelection(unittest.TestCase):
    def test_selects_oft_backend_once_staged_impl_chosen(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = SimpleNamespace(
            server_args=SimpleNamespace(enable_lora_staging=False, oft_impl="staged")
        )
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsInstance(get_staging_backend(tm, obj), OFTStagingBackend)

    def test_selects_no_backend_for_the_plain_sibling_impl(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        tm = SimpleNamespace(
            server_args=SimpleNamespace(enable_lora_staging=False, oft_impl="sibling")
        )
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsNone(get_staging_backend(tm, obj))
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_tokenizer_backend.py::TestStagingBackendSelection -v`

Both cases in `TestStagingBackendSelection` should PASS immediately once
Step 1 lands `"staged"` in `OFT_IMPL_CHOICES` — Tasks 1-4 already built
everything `get_staging_backend` depends on (the registry entry, the
lambda, `srt/oft/staged_manager.py`); this step only needed `oft_impl` to
legally accept `"staged"` as a value in the first place. If either case
fails, the bug is in Task 1's registry or Task 4's `OFTStagingBackend`, not
in this task.

- [ ] **Step 4: Wire manager construction in `model_runner.py`**

Read the current construction branch first:

```bash
grep -n "oft_impl\|OFTManager\|_get_lora_manager_class" python/sglang/srt/model_executor/model_runner.py
```

Find wherever `model_runner.py` currently branches on `oft_impl` to choose
between `srt/peft/oft`'s manager and `srt/oft`'s `OFTManager` (the existing
two-way branch this task extends to three), and add the third arm: mirror
`_get_lora_manager_class`'s pattern (read in full during Task 1's Ask-1
audit) with an `_get_oft_manager_class` returning `StagedOFTManager` when
`server_args.oft_impl == "staged"`. This is an *addition* of a new
per-method helper / a third arm on an existing branch, not an edit to the
existing LoRA helper. The exact surrounding code (what constructs
`OFTManager` today, and how it currently reads `oft_impl`) must be
re-verified in this planning pass and this plan does not assume its exact
current shape.

- [ ] **Step 5: Run the full adapter_sync + lora + peft unit suite**

Run: `python3 -m pytest test/registered/unit/adapter_sync/ test/registered/unit/lora/ test/registered/unit/peft/ -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/server_args.py \
        python/sglang/srt/model_executor/model_runner.py \
        test/registered/unit/adapter_sync/test_tokenizer_backend.py
git commit -m "feat(oft): wire --enable-oft-staging through StagedOFTManager/OFTStagingBackend"
```

---

## Task 6: CPU-safe end-to-end unit coverage for OFT staging + multi-tenancy together

**Files:**
- Test: `test/registered/unit/adapter_sync/test_oft_staging_backend.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 2-5.

- [ ] **Step 1: Write the test that guards the actual bug this whole effort exists to fix**

```python
class TestStagingCoexistsWithMultiTenancy(unittest.TestCase):
    """Guards the exact gap found while designing this: AdapterMemPool.activate()
    is pool-wide (one _active_version for the whole pool); StagedOFTMemoryPool
    must NOT have that property, or admitting a second adapter while a first
    is being staged would corrupt the first's serving slot."""

    def test_two_resident_adapters_keep_independent_versions(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.uid_to_buffer_id["adapter-a"] = 0
        pool.uid_to_buffer_id["adapter-b"] = 1

        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=1.0))
        pool.activate("adapter-a", 1, destination=0)

        pool.stage("adapter-b", 5, _named_tensors_for_layer_0(fill_value=2.0))
        pool.activate("adapter-b", 5, destination=1)

        self.assertEqual(pool.active_version_for("adapter-a"), 1)
        self.assertEqual(pool.active_version_for("adapter-b"), 5)

    def test_activating_one_adapter_does_not_touch_a_second_resident_slot(self):
        pool = _make_pool(max_ofts_per_batch=4)
        pool.uid_to_buffer_id["adapter-a"] = 0
        pool.uid_to_buffer_id["adapter-b"] = 1
        slot_1_before = pool.slot(f"R:{TARGET_MODULE}", 0, 1).clone()

        pool.stage("adapter-a", 1, _named_tensors_for_layer_0(fill_value=1.0))
        pool.activate("adapter-a", 1, destination=0)

        self.assertTrue((pool.slot(f"R:{TARGET_MODULE}", 0, 1) == slot_1_before).all())
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python3 -m pytest test/registered/unit/adapter_sync/test_oft_staging_backend.py::TestStagingCoexistsWithMultiTenancy -v`
Expected: PASS if Task 2's `activate` override is correctly per-destination
(it should be, since it copies to `destination`, not to a pool-wide
`active_idx`) — this test's real job is to catch a regression if a future
edit reintroduces the pool-wide `AdapterMemPool.activate()` behavior by
accident. If it fails, the bug is in Task 2's `activate` override, not in
this test.

- [ ] **Step 3: Commit**

```bash
git add test/registered/unit/adapter_sync/test_oft_staging_backend.py
git commit -m "test(adapter_sync): guard OFT staging against the pool-wide-version regression"
```

---

## Task 7a: LoRA GPU validation gate — independent, does not wait on Tasks 2-6

**This track has no dependency on this plan's OFT work and can run as soon
as Task 1 lands (or even before it, against the native-staged-lora batch as
already shipped).** `StagedLoRAManager` already exists; nothing here is
blocked on `StagedOFTManager` existing. Splitting this out matters
concretely: `srt/peft/lora` is 2,495 LOC across 9 files that can be deleted
*now*, without waiting for the much larger `srt/peft/oft` (11,122 LOC, 32
files) to have its own replacement finished.

**Files:**
- The GPU test file already exists:
  `test/registered/lora/test_lora_staged_update.py` (shipped with the
  native-staged-lora batch). Check whether it's already been run to a
  recorded PASS (`docs/superpowers/plans/2026-08-30-native-staged-lora-implementation.md`'s
  own Steps 3-9 describe the intended run — its checkboxes were unchecked
  as of this plan's writing; confirm current status before assuming it's
  done or not done).

- [ ] **Step 1: Check whether this gate has already been run**

Read `docs/superpowers/plans/2026-08-30-native-staged-lora-implementation.md`'s
Steps 3-9 checkboxes and any run-store provenance it references
(`/data/home/*/.local/state/remote-cluster-runs/slurm/...`). If already
recorded PASS with a verified commit match, skip to Task 8a. If not, continue.

- [ ] **Step 2: Ask the user for explicit approval before requesting GPUs**,
      exactly as that plan's own Step 3 required.

- [ ] **Step 3: Run `test_lora_staged_update.py`/`test_lora_staged_update_tp.py`
      on the approved GPU allocation**, per that plan's Steps 4-5.

- [ ] **Step 4: Record results** (provenance, logs, exit codes, commit hash
      tested) the same way that plan's Step 6 did.

- [ ] **Step 5: Benchmark the single-active (capacity=1) case — this is what
      decides whether `srt/peft/lora` needs anything beyond deletion**

Configure `StagedLoRAManager` at `max_loras_per_batch=1` (one resident
serving slot + the hidden staging slot, i.e. the exact shape of the
single-active RL use case: one adapter identity, refreshed every step) and
measure per-step throughput against the historical `srt/peft/lora`
single-active-only baseline, same hardware/model/batch config for both.

**Revised expectation, based on prior internal experiments (not just
theory)**: a real regression here is the *likely* outcome, not an edge
case — past experience found a specialized single-adapter kernel measurably
faster than the multi-tenant kernel even at capacity=1, because the
multi-tenant kernel still does a per-request slot-index lookup for every
request, even when every request resolves to the same slot, whereas a
specialized kernel skips indexing and applies one dense rotation/GEMM
uniformly across the batch. Budget for needing the kernel-level fix below,
not just the bookkeeping one.

- If throughput is comparable (no material regression): nothing further to
  build — proceed straight to Task 8a.
- If there's a regression (the expected case per prior experience): fix it
  in two parts, both still inside `StagedLoRAManager`, not a separate class
  (a parallel LoRA weight-placement implementation is the exact drift risk
  `32a1e22907` already proved out):
  1. Skip `running_loras` admission-set bookkeeping entirely when
     `max_loras_per_batch == 1` (the Python-level part — cheap, and was
     always going to be needed regardless of the kernel finding).
  2. Add a kernel-level dispatch: when capacity==1, route to a specialized
     dense/uniform kernel instead of the general segmented multi-tenant
     one — an `if capacity == 1: ... else: ...` branch inside the existing
     forward/invoke path (the same seam pattern the `invoke=` parameter
     already provides for OFT's MoE runner), not a different weight-
     placement implementation or a different manager class.

Record the measured numbers (not just pass/fail) in this step's result —
they're the basis for either conclusion above, and for sizing how much of
the gap the kernel-level fix needs to close.

---

## Task 7b: OFT GPU validation gate — after Tasks 2-6

**Files:**
- Create: `test/registered/lora/test_oft_staged_update.py` (mirroring `test/registered/lora/test_lora_staged_update.py`'s structure)

This task is a template, not a checklist to execute unattended — mirror
`docs/superpowers/plans/2026-08-30-native-staged-lora-implementation.md`'s
own Steps 3-9 (Slurm run-record creation, explicit approval before each
multi-GPU submission, snapshot verification, self-review before considering
Task 8b) exactly, substituting OFT's staged manager/backend for LoRA's.

- [ ] **Step 1: Write GPU-requiring test cases** covering: single-GPU
      stage/activate round-trip producing bitwise-identical output to a
      fresh server booted directly at the new version; decode-graph on/off;
      the hidden slot never appearing in `available_serving_slots()`'s
      count during a live server's uptime; a second, unrelated resident
      adapter's output unaffected by another adapter's stage/activate
      cycle; TP>1 activation consistency (every rank reaches the same
      version or the update fails cleanly).

- [ ] **Step 2: Ask the user for explicit approval before requesting GPUs**,
      exactly as the native-staged-LoRA plan's own Step 3 required.

- [ ] **Step 3: Run on the approved GPU allocation.**

- [ ] **Step 4: Record results** (provenance, logs, exit codes) the same
      way the native-staged-LoRA implementation plan's Step 6 did.

- [ ] **Step 5: Benchmark the single-active (capacity=1) case — same
      question as Task 7a Step 5, for OFT — this is what decides whether
      `srt/peft/oft` needs anything beyond deletion**

Configure `StagedOFTManager` at the minimal capacity the single-active RL
use case needs (one resident serving slot + the hidden staging slot — check
`max_ofts_per_batch`'s actual minimum against Task 3's `StagedOFTMemoryPool`
sizing, since OFT's slot accounting may differ slightly from LoRA's) and
measure per-step throughput against the historical `srt/peft/oft`
single-active/double-buffer-only baseline, same hardware/model/batch config
for both.

**Same revised expectation as Task 7a Step 5, and for the same reason**:
prior internal experiments found a specialized single-adapter kernel
measurably faster than the multi-tenant kernel even at capacity=1 (skips
per-request slot-index lookup, applies one dense rotation over the whole
batch instead of a segmented one) — treat a regression here as the likely
outcome, not an edge case.

Same fix shape as Task 7a Step 5, both inside `StagedOFTManager` (not a
separate `SingleActiveOFTManager` class): (1) skip `running_ofts`
admission-set bookkeeping when capacity is minimal, and (2) a kernel-level
`if capacity == 1: ... else: ...` dispatch to a specialized dense/uniform
OFT rotation kernel instead of the general segmented multi-tenant one, via
the same `invoke=`-style seam OFT's MoE runner already uses elsewhere. If
no regression shows up: proceed straight to Task 8b. Record the measured
numbers, not just pass/fail.

---

## Task 8a: Delete `srt/peft/lora`

**Gated on Task 7a passing.** Independent of Task 8b — do not wait for
OFT's work to delete this. Do not start until Task 7a's GPU gate has actual
recorded PASS evidence, not just "the plan says it should work."

**Files:**
- Delete: `python/sglang/srt/peft/lora/` (entire directory)
- Modify: `python/sglang/srt/peft/config.py`, `python/sglang/srt/peft/integration.py`,
  `python/sglang/srt/peft/tokenizer_hooks.py`, `python/sglang/srt/server_args.py`
  — remove now-dead references to `--peft-method lora` legacy construction paths.
- Modify: `python/sglang/srt/model_executor/model_runner.py` — remove the
  legacy `srt/peft/lora`-backed manager construction branch.

- [ ] **Step 1: Grep for every remaining reference before deleting anything**

```bash
grep -rln "srt\.peft\.lora\|peft_method.*lora" python/sglang/ test/
```

Read every result. Do not delete the directory while a non-test file still
imports from it.

- [ ] **Step 2: Delete the directory and dead references found in Step 1**

- [ ] **Step 3: Run the full unit + adapter_sync + peft + lora test suites**

Run: `python3 -m pytest test/registered/unit/ -k "lora or peft or adapter_sync" -v`
Expected: PASS, or a clearly-explained set of removed test files whose only
purpose was exercising the deleted legacy path.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "cleanup: delete srt/peft/lora, superseded by srt/lora + StagedLoRAManager"
```

---

## Task 8b: Delete `srt/peft/oft`

**Gated on Task 7b passing.** Do not start until Task 7b's GPU gate has
actual recorded PASS evidence, not just "the plan says it should work."

**Files:**
- Delete: `python/sglang/srt/peft/oft/` (entire directory)
- Modify: `python/sglang/srt/peft/config.py`, `python/sglang/srt/peft/integration.py`,
  `python/sglang/srt/peft/tokenizer_hooks.py`, `python/sglang/srt/server_args.py`
  — remove now-dead references to `--peft-method oft` legacy construction
  paths and `--oft-impl peft`.
- Modify: `python/sglang/srt/model_executor/model_runner.py` — remove the
  legacy `srt/peft/oft`-backed manager construction branch.

- [ ] **Step 1: Grep for every remaining reference before deleting anything**

```bash
grep -rln "srt\.peft\.oft\|peft_method.*oft\|oft_impl.*peft" python/sglang/ test/
```

Read every result. Do not delete the directory while a non-test file still
imports from it.

- [ ] **Step 2: Delete the directory and dead references found in Step 1**

- [ ] **Step 3: Run the full unit + adapter_sync + peft + oft test suites**

Run: `python3 -m pytest test/registered/unit/ -k "oft or peft or adapter_sync" -v`
Expected: PASS, or a clearly-explained set of removed test files whose only
purpose was exercising the deleted legacy path.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "cleanup: delete srt/peft/oft, superseded by srt/oft + StagedOFTManager"
```

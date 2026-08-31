---
title: "Native OFT adapter RPC parity"
description: "Design for giving single-active (\"sibling\") OFT its own dedicated load/unload RPC surface, mirroring native LoRA's, instead of routing through the shared srt/peft mechanism."
---

# Native OFT adapter RPC parity

## Status

Approved design, branched off a review of Task 8b (`docs/superpowers/plans/2026-08-30-adapter-staging-unification.md`) that deleted `srt/peft/oft`. That review found `oft_impl == "sibling"`'s live async-RL weight-sync path is the only adapter-loading mechanism left, in either adapter family, still routed through the generic `srt/peft` package. This spec designs its replacement.

- Base branch: `feat/adapter-staging-unification`
- Repo: `/workspace/sglang-spherelab`

## Goal

Give the non-staged (`oft_impl == "sibling"`) `OFTManager` its own native adapter-loading RPC surface, at full functional parity with native LoRA's (`LoadLoRAAdapterFromTensorsReqInput`/`LoadLoRAAdapterFromDistributedReqInput`, tokenizer-side registry with refcounted acquire/release, LRU eviction, upsert-in-place refresh) — without depending on the shared `srt/peft` integration layer (`peft.maybe_load_adapter_format` → `srt/oft/streamed_weight_loader.py::load_streamed_oft_adapter`).

## Non-goals

- Touch `StagedOFTManager` or its `stage_adapter`/`activate_adapter_version` two-phase RPC — that mechanism already has its own dedicated, already-reviewed wire path and is out of scope.
- Retire the old shared `srt/peft` streamed-loader mechanism in this same change. That happens in a **follow-up task**, gated on this new path's own GPU evidence (see Sequencing).
- Change OFT's static, boot-time `--peft-paths` loading or its per-request adapter-path resolution (`resolve_peft_path`) — both keep using `tm.peft_registry`/`tm.peft_ref_cache` exactly as today, except for one targeted fix (see below).
- Introduce a brand-new registry class. `OFTRegistry`/`AdapterRegistry` already exist, are already live, and get extended in place rather than duplicated.
- Add vocabulary-extension support (`added_tokens_config`). OFT's rotation-based adapters never touch token embeddings.

## Existing behavior

Three OFT-adapter-tracking mechanisms exist today, at different layers:

1. **`srt/peft/tokenizer_hooks.py`** (tokenizer-manager side): `tm.peft_registry` (an `OFTRegistry` instance), `tm.peft_ref_cache` (a plain name→ref dict), and `tm.peft_update_lock` (an `asyncio.Lock`) are all constructed in `init_tokenizer_peft`. `peft_registry`/`peft_ref_cache` are genuinely live — `resolve_peft_path` calls `tm.peft_registry.acquire(path)` on every request carrying an adapter path. `peft_update_lock` has zero current callers.
2. **`srt/oft/base/registry.py`**: `AdapterRegistry` (base class of `OFTRegistry`) already implements refcounted `acquire`/`release`, async `wait_for_unload`, LRU eviction (`lru_adapter_name`), and an unconditional `replace()` — but `replace()` has zero current callers, and there is no two-phase "decide, then commit only on success" primitive matching LoRA's `register_or_reuse`/`refresh` pair, nor an atomic id+version snapshot matching LoRA's `acquire_with_version` (today's `resolve_peft_path` does `acquire()` then a separate `get_version_by_id()` call — two lock acquisitions, a narrow TOCTOU window).
3. **`srt/oft/oft_manager.py` / `srt/oft/streamed_weight_loader.py`** (scheduler/model-runner side): `OFTManager.refs` tracks GPU-resident adapters. The streamed-loader path (`_ensure_streaming_oft_adapter_slot`, reached via `weight_updater.py`'s `update_weights_from_tensor` → `srt/peft/integration.py::maybe_load_adapter_format` → `load_streamed_oft_adapter`) hard-rejects any second, differently-named resident adapter — a single-active restriction that exists at this layer only, not in the tokenizer-side registry, which already supports N adapters.

LoRA's equivalent mechanism (`srt/lora/lora_registry.py`'s `LoRARegistry`, `srt/managers/io_struct.py`'s `LoadLoRAAdapterFromTensorsReqInput`/`LoadLoRAAdapterFromDistributedReqInput`, handlers in `tokenizer_control_mixin.py`, dispatch in `scheduler.py`) has no dependency on `srt/peft` at all, and is itself mainline SGLang functionality this fork already tracks through upstream merges.

## Chosen approach

Extend the existing OFT-side infrastructure additively — new methods on `AdapterRegistry`, new wire types, new handler/dispatch methods mirroring LoRA's file-by-file — rather than building a parallel registry or bypassing what already works.

### `AdapterRegistry` extension (`srt/oft/base/registry.py`)

Add two new methods, generalizing LoRA's exact safety properties onto the shared base (used only by `OFTRegistry` today, but written generically per the base class's own "shared by the single-active peft methods" docstring):

- `resolve_or_reuse(ref: AdapterRef, upsert: bool = False, *, preserve_pinned: bool = False) -> Tuple[AdapterRef, bool]` — reader-lock, non-mutating. Mirrors `LoRARegistry.register_or_reuse` exactly: without `upsert`, returns `(ref, False)` unchanged; with `upsert` and an existing same-name entry, returns a copy carrying the existing `adapter_id` (and `pinned`, if `preserve_pinned`) with `reused=True`. Nothing is registered here — callers commit via `register()` (fresh) or the new `refresh()` (reused) only after the backend load succeeds, so a failed load never corrupts the registry.
- `refresh(ref: AdapterRef)` — writer-lock. Mirrors `LoRARegistry.refresh`: asserts the name is already registered under the same `adapter_id`, then overwrites and moves it to LRU-most-recent.
- `_acquire_refs(name)` + `acquire_with_version(name)` — mirrors LoRA's `_acquire_refs`/`acquire_with_version`: a single reader-lock snapshot of the matching `AdapterRef`(s), incrementing use counters and reading `adapter_version` directly off the same snapshot (no second lock acquisition). `resolve_peft_path` switches to this for its id+version resolution, closing the existing TOCTOU gap as a side effect.

`replace()` and the existing `acquire()` are left untouched (not refactored to share code with the new methods) — small duplication traded for zero risk to already-live call paths. New methods use generic names, matching the base class's own existing majority (6 of 9 methods have no OFT-flavored alias); no new aliases are added on `OFTRegistry` for them.

### Wire types (`srt/managers/io_struct.py`)

Mirrors LoRA's shape 1:1, including its single-shared-output-type simplification:

- `LoadOFTAdapterFromTensorsReqInput` / `LoadOFTAdapterFromDistributedReqInput` / `UnloadOFTAdapterReqInput` — same fields as their LoRA counterparts (`adapter_name`, `config_dict`, `serialized_named_tensors` or `names`/`dtypes`/`shapes`/`group_name`, `pinned`, `adapter_id`, `upsert`, `load_format`), each with a `to_ref()` building `OFTRef`. `added_tokens_config` is omitted (see Non-goals).
- One shared `OFTUpdateOutput{success, error_message, loaded_adapters}`, aliased as `LoadOFTAdapterFromTensorsReqOutput = LoadOFTAdapterFromDistributedReqOutput = UnloadOFTAdapterReqOutput = OFTUpdateOutput`, exactly matching `LoRAUpdateOutput`'s pattern.

### Tokenizer-manager handlers (`srt/managers/tokenizer_control_mixin.py`)

New async methods `load_oft_adapter_from_tensors`, `load_oft_adapter_from_distributed`, `unload_oft_adapter` — structurally identical to LoRA's equivalents, reusing infrastructure that already exists but is currently idle for this purpose: `tm.peft_registry` (calling the new `resolve_or_reuse`/`refresh`/`register`), `tm.peft_ref_cache`, and `tm.peft_update_lock` (currently zero callers — becomes this handler's critical-section lock, same role `lora_update_lock` plays for LoRA).

Gate: `server_args.peft_method == "oft" and server_args.oft_impl == "sibling"` — staged mode is untouched, keeps its own RPC pair. New `--max-loaded-ofts` flag drives LRU eviction identically to LoRA's `max_loaded_loras` check (evict via `lru_adapter_name(exclude_pinned=True)` + `unload_oft_adapter` when the registered count exceeds it).

### Scheduler dispatch (`srt/managers/scheduler.py`) + `tp_worker`

New dispatch entries and handler methods (`load_oft_adapter_from_tensors`, `load_oft_adapter_from_distributed`, `unload_oft_adapter`) mirroring LoRA's dispatch table entries 1:1, forwarding to `tp_worker`.

Per this repo's `modify-component-must-read.md` rule, edits to `Scheduler`/`TokenizerManager` go through the `large-class-style` skill.

### GPU-side admission (`srt/oft/oft_manager.py`)

New `OFTManager` methods (`load_adapter_from_tensors`, `load_adapter_from_distributed`) reusing the actual admission primitives `_ensure_streaming_oft_adapter_slot` already calls (`memory_pool.allocate_buffer_slot`, `register_streamed_adapter`, `unload_streamed_adapter`), invoked from this new entrypoint instead of through `srt/peft`. The single-active restriction in the old streamed-loader is **not** carried over: the new path allows dynamically-loaded adapters to become GPU-resident up to the existing `max_ofts_per_batch` capacity, sharing that same pool with statically-loaded (`--peft-paths`) adapters. Overflow there is handled by `OFTManager`'s existing per-batch LRU admission (`eviction_policy="lru"`, already implemented for `fetch_new_ofts`/`prepare_oft_batch`) — no new GPU-side eviction mechanism is needed. This is distinct from, and independent of, the tokenizer-side registry's own `--max-loaded-ofts` LRU eviction described above: one caps how many adapters the registry tracks at all (CPU-side bookkeeping), the other caps how many of those are simultaneously GPU-resident in a batch — the same two-tier split LoRA already keeps between `max_loaded_loras` and `max_loras_per_batch`.

### Engine + HTTP surface

`load_oft_adapter_from_tensors`/`_from_distributed`/`unload_oft_adapter` convenience methods on `Engine` (`srt/entrypoints/engine.py`) and matching `/load_oft_adapter_from_tensors`, `/load_oft_adapter_from_distributed`, `/unload_oft_adapter` HTTP routes (`srt/entrypoints/http_server.py`), mirroring LoRA's 1:1.

## Testing

- CPU unit tests for the three new `AdapterRegistry` methods (`resolve_or_reuse`, `refresh`, `acquire_with_version`), covering the same cases `lora_registry`'s own tests cover for its equivalents (fresh load, upsert-reuse, upsert-refresh, failed-load-leaves-registry-untouched).
- GPU integration test suite mirroring `test/registered/rl/test_lora_load_from_tensor.py`, covering: fresh load, upsert-refresh, LRU eviction past `--max-loaded-ofts`, and multi-adapter concurrent residency (the capability newly unlocked by relaxing the single-active restriction).

## Sequencing

This becomes its own implementation plan (via the `writing-plans` skill), separate from `2026-08-30-adapter-staging-unification`'s plan/ledger. Retiring the old `srt/peft` streamed-loader mechanism (`peft.maybe_load_adapter_format`'s `oft_adapter` branch, `srt/oft/streamed_weight_loader.py::load_streamed_oft_adapter`/`_ensure_streaming_oft_adapter_slot`) is that new plan's **final** task, gated on this new path's own GPU evidence passing review — the same prove-then-delete rhythm as Task 7b → Task 8b in the prior plan.

## Open questions / risks

- The GPU-side single-active-to-multi-tenant relaxation (loosening `_ensure_streaming_oft_adapter_slot`'s guard) is the one piece of this design that changes runtime behavior beyond additive new surface — it needs its own careful review during implementation, same as Task 8b's Finding A did.
- Whether any external caller (e.g. an RL training harness) depends on the *old* shared RPC's specific error strings or behavior is unknown from this repo alone; the retirement task should grep for external usage before deleting.

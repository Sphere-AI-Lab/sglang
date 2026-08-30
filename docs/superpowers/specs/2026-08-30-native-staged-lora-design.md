---
title: "Native staged LoRA on orbit-main-corrected"
description: "Design for two-phase versioned LoRA updates on the corrected branch's native serving stack."
---

# Native staged LoRA on orbit-main-corrected

## Status

Approved design for porting the staged-LoRA behavior from
oft-restructure-v0.5.18 onto orbit-main-corrected.

- Source behavior: 32af48d8df1e1999136a83f6993e25b43a9bfb24
- Target base: 1c15111fba954babe2b9742caefcbedb22ce306c
- Implementation branch: codex/port-staged-lora-native
- Runtime: Slurm, using the existing SGLang environment and scheduler workflow

## Goal

Provide a functional two-phase LoRA weight-update protocol on the corrected
branch's native --enable-lora stack:

1. Stage a version into a hidden GPU slot without changing served output.
2. Activate that version only after every worker staged successfully.
3. Keep radix-cache entries isolated by LoRA identity and active version.

The completed path must preserve the target branch's native LoRA registry,
leases, multi-adapter routing, upsert behavior, fused weight placement, and
eviction machinery.

## Non-goals

- Restore the fork-only --peft-method lora or --lora-impl sibling interface.
- Replace the existing OFT staging path.
- Rewrite the corrected branch's native one-step LoRA upsert API.
- Introduce dynamic REST load/unload features beyond the target's existing API.
- Change adapter math, kernels, or quantization behavior.
- Add a third public prepare/finalize phase. The external trainer contract remains
  stage then activate.
- Guarantee recovery from a fatal CUDA context error. Such an error makes the
  worker unhealthy and requires restart.

## Existing behavior

The target branch already supports native LoRA loading and one-step upsert.
That flow builds a new LoRAAdapter, synchronizes outstanding CUDA work, and
rewrites a resident serving slot in place. It restores the prior adapter when a
recoverable placement error occurs.

The target also contains the shared adapter_sync package at commit 59f9dd2c5,
but its staged native-LoRA backend is not constructed by production code. The
nine later commits on oft-restructure-v0.5.18 add wiring and fixes, but assume
an older native LoRA implementation. Their intent must be ported, not their
text copied wholesale.

## Chosen approach

Extend the native LoRA stack through subclasses selected at server startup.
This keeps upstream-facing LoRA files minimally changed while allowing staging
to reuse the target's current construction and placement routines.

### Server selection

Add --enable-lora-staging as a native LoRA option. It is valid only with
--enable-lora. When false, ModelRunner constructs the existing LoRAManager.
When true, it constructs StagedLoRAManager.

The flag makes the extra GPU allocation explicit. Existing native LoRA users
pay no staging-slot memory cost and retain their current behavior.

### StagedLoRAMemoryPool

StagedLoRAMemoryPool subclasses the target branch's LoRAMemoryPool.

- It allocates one additional physical adapter slot.
- It continues to advertise the configured serving capacity.
- The final physical slot is staging_idx and is never entered in serving-slot
  routing, eviction, or batch metadata.
- It records at most one outstanding staged identity and version.
- It uses native load_lora_weight_to_buffer for every placement operation.
  This preserves fused qkv and gate/up layouts, rank padding, TP slicing,
  embedding and lm_head handling, and MoE variants.

The pool exposes these method-level contracts:

- stage(version, uid, adapter): fill only staging_idx after validating that the
  outstanding staged record is compatible.
- activate(version, uid, destination): copy staging_idx into a resident serving
  slot when one exists.
- discard_stage(version, uid): clear staged metadata after a failed or cancelled
  transaction without changing served state.
- staged_identity(): return the outstanding uid and version for diagnostics and
  idempotency checks.

### StagedLoRAManager

StagedLoRAManager subclasses the target branch's LoRAManager.

Stage:

1. Resolve the tokenizer-supplied lora_id. Adapter name is descriptive and must
   not replace the stable ID.
2. Build LoRAConfig and LoRAAdapter through the target's
   _create_lora_adapter_from_tensors path.
3. Validate target modules, ranks, added-token metadata, and compatibility with
   an existing adapter.
4. Preserve the existing adapter's pinned status when the stable uid is reused;
   first-time streamed adapters are unpinned.
5. Place the adapter into staging_idx with the native pool loader.
6. Retain the staged adapter, configuration, reference candidate, uid, and
   version in manager-local pending state.
7. Return success without changing configs, loras, lora_refs, the tokenizer
   registry, serving routing, or active versions.

Activate:

1. Verify uid and version match the outstanding staged record.
2. Verify staging_idx is distinct from every advertised serving slot.
3. If the adapter is resident, drain current work through the existing
   scheduler writer path and copy staging_idx into its current serving slot.
4. If the adapter is not resident, do not allocate or evict a serving slot.
   Commit the staged CPU-side adapter; native paging loads it on the next
   request.
5. Commit configs, loras, lora_refs, pinned accounting, and active version only
   after the serving update succeeds.
6. Clear staged state and return the activated version.

For a first-time streamed adapter, activation registers it only after every
worker succeeds. No request can name an adapter that merely staged.

## Native registry and version identity

Extend LoRARef with a trailing version integer whose default is zero. LoRARef is
array-like, so the trailing default preserves decoding compatibility with
older payloads.

The tokenizer registry owns the externally visible active version:

- Stage reserves or reuses a stable lora_id but does not publish a new version.
- Activate sends the same uid and requested version to every scheduler worker.
- Only an all-success response registers or refreshes LoRARef with the new
  version.
- A failed aggregate activation leaves the registry at the old version and
  returns failure.
- If workers disagree during activation, the tokenizer records the affected
  adapter in failed_lora_activations before releasing model_update_lock. Future
  admission for that adapter fails with a restart-required error. Base-model and
  unrelated-adapter admission remain available.

Request admission must snapshot lora_id and version under the same registry
lock that increments the adapter lease counter. This prevents an activation
from changing the version between identity lookup and request admission.

## Radix-cache isolation

Native LoRA requests extend the radix-cache key with:

    |lora:<lora_id>:v<version>

The base model retains its existing key. Activating a new version therefore
misses prefixes cached by an older version without flushing unrelated base or
adapter prefixes.

The existing one-step upsert keeps its current cache-flush behavior in this
scope. Migrating one-step upsert to versioned keys can be a separate change.

## Routing

The model-runner WeightUpdater handles native staged LoRA before falling back to
the existing PEFT integration:

- With --enable-lora-staging, LoRA payloads call
  model_runner.lora_manager.stage_adapter and activate_adapter.
- OFT and rollback PEFT implementations continue through peft.stage_adapter and
  peft.activate_adapter.
- A native LoRA server without --enable-lora-staging rejects the two-phase
  endpoint with a precise configuration error.

No native LoRA request is routed through peft_method.

## State invariants

- Staging never changes the output of a request admitted before activation.
- There is one hidden staging slot and at most one outstanding staged payload.
- Repeating the same uid and version is idempotent.
- A different uid or version cannot silently overwrite an outstanding stage.
- staging_idx is outside advertised capacity and never appears in
  uid_to_buffer_id or buffer_id_to_uid.
- Activation requires an exact uid and version match.
- Registry version publication occurs only after all workers report success.
- Adapter B and base-model routing remain unchanged while adapter A stages or
  activates.
- A version change never relies on a whole-cache flush for correctness.

## Failure handling

- Configuration, tensor-name, shape, rank, fusion, or target-module errors fail
  during stage and leave active state untouched.
- A stage conflict reports the outstanding uid and version.
- Activation performs identity, version, slot, and shape preflight before any
  live copy.
- A recoverable live-slot copy failure restores the previous adapter through
  native placement, keeps the previous manager metadata, and returns failure.
- If restoration fails, log the affected adapter and slot as corrupted and
  return a hard failure requiring worker restart.
- A multi-worker stage failure prevents activation.
- A multi-worker activation failure is reported as a consistency failure. The
  implementation must avoid publishing the new tokenizer-registry version.
  Recoverable local failures restore their prior slot before responding. Because
  a worker that already succeeded cannot be assumed to roll back, the tokenizer
  must quarantine the affected adapter before admission resumes. The quarantine
  is fail-closed and lasts until process restart.

## Verification

### CPU and environment-independent tests

- Hidden slot allocation and advertised-capacity isolation.
- Native fused projection, rank padding, embedding, lm_head, and MoE placement.
- Stage does not modify active manager state or serving buffers.
- Same uid/version stage retry is idempotent.
- Conflicting staged identity or version is rejected.
- Activation requires the exact uid and version.
- Resident activation changes only the selected serving slot.
- Nonresident activation commits metadata without evicting another adapter.
- Failed placement restores prior manager and buffer state.
- Registry publication occurs only after aggregate success.
- Aggregate activation failure quarantines only the affected adapter before the
  writer lock is released; base and unrelated adapters remain admissible.
- Existing pinned metadata survives a staged refresh.
- Request admission snapshots id and version atomically.
- Radix keys distinguish versions and preserve base keys.
- Existing native LoRA upsert, registry, lease, eviction, and multi-LoRA tests
  remain green.

### GPU qualification

- TP=1 serving: allocate a second physical GPU for the NCCL trainer rank, serve
  v1, stage v2, prove output remains v1, activate v2, and compare bitwise
  against a fresh v2 server.
- Cache negative control: withholding version reuses a stale prefix; including
  version produces a miss and correct output.
- TP=2: allocate a distinct trainer GPU plus two server GPUs; every server rank
  stages and activates the same version and output matches the single-version
  reference.
- Multi-tenant: updating adapter A leaves adapter B and base output unchanged.
- Slot pressure: eviction never selects the hidden staging slot.
- Decode CUDA graphs enabled and disabled: activation is applied and does not
  silently no-op.
- MoE: run the existing supported LoRA MoE fixture with expert sharding active.

GPU qualification runs under Slurm and records source commit, environment,
allocation, logs, completion status, and artifacts in the canonical remote run
store.

## Migration

Orbit launch configuration changes from the fork-only PEFT LoRA selection to:

    --enable-lora
    --enable-lora-staging

Existing adapter stage and activate HTTP payloads retain their names, IDs, and
versions. OFT launch configuration and endpoints remain unchanged.

## Implementation boundaries

The implementation should be organized into reviewable units:

1. Native registry version snapshot and radix-cache identity.
2. Staged native LoRA pool and manager with unit tests.
3. ModelRunner and WeightUpdater routing plus CLI validation.
4. Regression suite and GPU qualification.

The nine source commits are evidence for required failure modes, not a required
commit structure. The new branch should produce a concise series aligned with
the corrected target's current interfaces.

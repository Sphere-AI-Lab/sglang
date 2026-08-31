# Retire Legacy `srt/peft` Design

## Status

Approved in design review on 2026-08-31.

The reviewed Sphere reference is `a89f20362e2f6dd6ba135caa558b4fabdff9812c`
on `orbit-main-corrected`. Implementation must record and use the final source
SHA because that branch may receive additional commits before the port begins.

The Impossible implementation worktree starts from `sglang-miles` at
`f3863a3f12c4e4db92374158bb1b843ed7cf6ed9`. The Radix/Sphere comparison base
is pinned to `5a0132a7623b0da58f98540f1edcca6cf154c72c` and intentionally excludes the
two later Radix-only commits.

## Problem

Sphere SGLang contains a canonical `srt/oft` implementation and native staged
LoRA under `srt/lora`, but it also carries a second, legacy implementation tree
under `srt/peft`:

- `srt/peft/oft` is selectable through `--oft-impl peft` as an OFT rollback
  backend.
- `srt/peft/lora` is selectable through `--peft-method lora` as the old
  single-active LoRA backend.
- `srt/peft/base` supports those legacy implementations.
- The top-level `srt/peft` modules still provide live OFT configuration,
  integration, request types, and tokenizer hooks.

Keeping both implementations approximately doubles the adapter-serving code,
allows the backends to drift, and leaves tests and runtime seams split across
legacy and canonical import paths. Direct deletion is unsafe because the
top-level control plane and several runtime branches remain active.

## Goals

1. Make `srt/oft` the only OFT implementation.
2. Make native staged LoRA under `srt/lora` the only LoRA implementation.
3. Remove `srt/peft/oft`, `srt/peft/lora`, `srt/peft/base`, and finally the
   complete `srt/peft` package.
4. Move the surviving OFT control plane into `srt/oft` without changing its
   supported behavior.
5. Remove `--oft-impl peft` and remove `lora` as a valid value of
   `--peft-method`.
6. Preserve `--peft-method oft` for OFT enablement and preserve the native
   staged-LoRA interface, including `--enable-lora`.
7. Prove functional, numerical, distributed, lifecycle, synchronization, and
   performance equivalence before accepting the deletion.

## Non-goals

- Preserve the legacy implementation-selection flags as aliases.
- Preserve imports from `sglang.srt.peft`.
- Add another generic adapter framework or compatibility layer.
- Validate DeepSeek MLA/MoE or Kimi K2.5 in this migration. Their special
  paths are deferred until they are required in production.
- Require bitwise equality from a kernel that is demonstrably nondeterministic.
- Incorporate the two post-`5a0132a7` Radix-only commits.

## Compatibility Contract

The migration preserves supported user-visible functionality, not obsolete
implementation choices.

The following must remain equivalent:

- Base-model serving with no adapter.
- Canonical OFT startup loading, dynamic loading, unloading, switching, mixed
  batching, streaming, and weight synchronization.
- Native staged-LoRA startup loading, dynamic loading, unloading, switching,
  mixed batching, streaming, and weight synchronization.
- Public request/response schemas other than the intentionally removed legacy
  selectors.
- Adapter identity, version ordering, stale-update rejection, error responses,
  and rollback after a failed load or update.
- Prefill, decode, CUDA-graph, eager, tensor-parallel, and expert-parallel
  behavior in the approved test scope.

The following are intentional incompatibilities:

- `--oft-impl peft` is rejected because no alternate OFT backend remains.
- `--peft-method lora` is rejected; native staged LoRA is enabled through its
  native interface.
- Imports from `sglang.srt.peft` fail after the final deletion.

## Architecture

### Canonical OFT

All OFT-specific configuration, integration hooks, request types, tokenizer
operations, registries, managers, streamed loading, kernels, and model hooks
live under `sglang.srt.oft`. Runtime consumers import canonical OFT symbols
directly. No runtime dispatch selects between OFT implementations.

### Native staged LoRA

Native LoRA continues to live under `sglang.srt.lora` and uses
`sglang.srt.adapter_sync` for the shared staged-weight synchronization
mechanism. It must not route through the former generic PEFT integration layer.

### Shared synchronization

`sglang.srt.adapter_sync` remains a narrowly scoped shared service for staged
adapter weight transfer, versioning, and synchronization. OFT and LoRA may
consume it, but OFT-specific configuration and request behavior do not move
into this package.

### Configuration surface

`--peft-method oft` remains accepted to avoid unrelated OFT interface churn.
Its choices no longer include `lora`. `--oft-impl` is removed rather than
retained with a single meaningless choice. Native staged LoRA continues to use
its native arguments.

## Migration Sequence

Each stage must be independently reviewable and leave the branch runnable.
Promotion stops at the first failed gate.

1. Build the differential oracle harness and prove that it detects an
   intentionally introduced mismatch against unchanged code.
2. Freeze oracle bundles from the final pre-removal source SHA.
3. Add canonical OFT control-plane entry points under `srt/oft` and switch all
   consumers while legacy implementations remain available.
4. Remove OFT implementation dispatch, `--oft-impl peft`, fallback branches,
   and `srt/peft/oft`.
5. Remove `--peft-method lora`, the legacy single-active LoRA routing,
   `srt/peft/lora`, and `srt/peft/base`.
6. Remove the remaining `srt/peft` package after every live control-plane
   consumer has moved to `srt/oft` or native LoRA.
7. Update tests and documentation and add static guards that reject renewed
   legacy imports or flags.
8. Run the complete candidate matrix and performance gate against the frozen
   oracle bundles.

## Differential Oracle

The oracle is generated from executable behavior, not manually interpreted
logs. Every bundle records:

- Git SHA, dirty state, environment identity, package versions, CUDA/driver
  versions, GPU model and UUID, and parallelism settings.
- Cryptographic hashes for model checkpoints, adapters, tokenizer inputs, and
  test manifests.
- Complete server arguments and relevant environment variables.
- Fixed prompts, tokenized inputs, random seeds, request ordering, batching,
  and warm-up procedure.
- Generated token IDs and text, per-token logprobs, selected raw logits when
  exposed, output shapes and dtypes, and adapter state transitions.
- Startup time, request latency, throughput, and peak allocated/reserved GPU
  memory.
- Structured status, stdout, stderr, and server logs.

For each model/precision cell, the unchanged source records:

1. No-adapter control.
2. Legacy OFT.
3. Canonical OFT.
4. Legacy single-active LoRA.
5. Native staged LoRA.

The post-removal candidate records:

1. No-adapter control.
2. Canonical OFT.
3. Native staged LoRA.

Two comparisons are mandatory:

- The canonical backend before removal versus the same canonical backend after
  removal proves that cleanup did not alter canonical behavior.
- Legacy versus canonical common behavior proves that the replacement covers
  the functionality being retired.

Oracle bundles are immutable and addressed by manifest hash. A comparison
failure identifies the matrix cell, scenario, request, token position,
quantity, observed values, and allowed tolerance.

## Test Matrix

The mandatory end-to-end matrix contains six cells:

| Model | Architecture | Precision | Required GPU |
| --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 | Dense | BF16 | H100 |
| Qwen3-4B-Instruct-2507 | Dense | FP8 | H100 |
| Qwen3-4B-Instruct-2507 | Dense | NVFP4 | B200 |
| Qwen3-30B-A3B | MoE | BF16 | H100 |
| Qwen3-30B-A3B | MoE | FP8 | H100 |
| Qwen3-30B-A3B | MoE | NVFP4 | B200 |

Every cell runs no-adapter, OFT, and native staged-LoRA candidate modes. The
pre-removal characterization additionally runs the matching legacy OFT and
legacy LoRA modes.

Every adapter mode covers:

- Adapter supplied at server startup.
- Dynamic load, inference, unload, and post-unload base-model restoration.
- Adapter A to B to A switching.
- Base, A, and B requests mixed within one batch.
- Concurrent streaming and non-streaming requests.
- Short and long prefill followed by decode.
- CUDA graphs enabled and disabled.
- Successful staged weight synchronization with increasing versions.
- Duplicate and stale update rejection.
- Invalid adapter ID, incompatible adapter metadata, interrupted/failed load,
  and transactional rollback.
- Restart using the same adapter manifest.

Qwen3 MoE must also cover tensor and expert parallelism. Single-GPU success is
not sufficient for the MoE cells.

## Test Gates

### Gate 1: Static and CPU

- No source, test, or documentation import references `sglang.srt.peft` after
  final deletion.
- No runtime branches or argument help text reference `oft_impl`,
  `--oft-impl`, or `--peft-method lora`.
- Removed selectors fail with deliberate, tested argument errors rather than
  later import errors.
- Registry, serialization, version ordering, lifecycle state machines, and
  request validation pass without a GPU.
- Existing OFT, LoRA, adapter-sync, and server-argument tests pass after their
  imports are migrated.

### Gate 2: GPU kernels and components

- OFT rotation, projection, fused kernels, streamed loading, sharding, and MoE
  dispatch pass in BF16, FP8, and NVFP4 where applicable.
- Native LoRA dense and MoE layers, batching, segmented GEMM, and weight
  loading pass in all applicable precisions.
- CUDA-graph capture, replay, invalidation, and eager fallback pass.
- Deterministic component comparisons prefer exact equality and emit complete
  diagnostics when tolerance comparison is required.

### Gate 3: End-to-end differential

All six matrix cells and all required lifecycle scenarios pass the frozen
oracle comparisons. Removing an adapter restores the exact no-adapter output
under deterministic settings.

### Gate 4: Distributed, stress, and performance

- Qwen3 MoE passes its required TP/EP configurations.
- Repeated load/switch/unload and synchronization cycles do not leak adapter
  state or GPU memory.
- Sustained mixed-adapter concurrency remains correct.
- Three measured repetitions run after warm-up.
- Median throughput does not regress by more than 5%.
- Peak GPU memory does not increase by more than 5%.
- Latency is recorded and investigated when it moves by more than 5%; a known
  workload-level tradeoff must be documented before acceptance.

## Numerical Equivalence

Determinism is enabled wherever supported. Prompts, tokenized inputs, seeds,
request order, batch shapes, kernel settings, and hardware class are fixed.

Comparison proceeds in this order:

1. Exact token IDs, shapes, dtypes, state transitions, and errors.
2. Exact logits/logprobs when deterministic execution provides them.
3. Dtype-specific absolute and relative tolerances only for a kernel shown to
   remain nondeterministic under the fixed configuration.

Tolerance values must be established from repeated unchanged-baseline runs,
checked into the manifest by named precision/kernel path, and never widened in
response to a candidate failure without a separate review. Token divergence is
always a failure even when individual logits are within tolerance.

## Parallel Compute Strategy

All GPU work runs through HTCondor on `mpi3`; login nodes are used only for
control and inspection. The approved bid is 100, and multiple concurrent jobs
are authorized.

The orchestrator fans out by model, precision, adapter mode, revision, and
scenario group. Functional baseline/candidate and legacy/canonical pairs may
run concurrently on separate GPUs of the same model. Performance comparisons
use matched placements and repeat with placement swapped when needed to remove
GPU-to-GPU bias.

H100 allocations cover BF16 and FP8. B200 allocations are mandatory for
NVFP4. Jobs request the smallest sufficient GPU group so fragmented B200
capacity can be used. Full-node H100 allocations split into independent server
instances when isolation and performance validity are preserved.

Each shard writes a durable, unique run directory containing provenance,
manifest, event log, stdout, stderr, server logs, result bundle, and completion
status. A barrier waits for all shards in a gate before promotion. One failure
does not cancel unrelated running shards; the aggregate report includes every
observed mismatch.

## Failure Handling and Rollback

- A static or CPU failure blocks GPU submission for the affected revision.
- An oracle-harness self-test failure invalidates all bundles it produced.
- A numerical mismatch is reproduced once under the identical manifest before
  classification; tolerances are not adjusted automatically.
- A failed adapter operation must leave the previous adapter state usable and
  must not corrupt the base model.
- A stage that fails its gate is reverted as a stage or repaired before later
  deletion work proceeds.
- Pre-removal oracle bundles and the final source SHA remain available for
  diagnosis until the PR is merged.

## Acceptance Criteria

The migration is ready for PR review only when:

1. The complete `sglang/srt/peft` package is absent.
2. The removed selectors and legacy imports have explicit negative tests.
3. All static, CPU, GPU component, six-cell end-to-end, distributed, stress,
   and performance gates pass.
4. Canonical pre-removal and post-removal behavior satisfies the equivalence
   contract.
5. Canonical common behavior satisfies the legacy-versus-canonical comparison.
6. Deterministic comparisons are exact wherever the unchanged baseline is
   exact; every tolerance exception is named and justified.
7. Throughput and peak-memory regressions remain within 5%.
8. The PR documents the intentionally removed interfaces and the explicitly
   deferred DeepSeek and Kimi coverage.


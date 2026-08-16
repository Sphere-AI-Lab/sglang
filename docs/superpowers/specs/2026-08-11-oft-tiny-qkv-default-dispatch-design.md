# OFT Tiny-QKV Default Dispatch Design

**Date:** 2026-08-11
**Repository:** SGLang
**Scope:** Default dense OFT QKV dispatch for tiny rotation pools

## Context

SGLang now supports power-of-two OFT block sizes from 4 through 1024 in the
fused dense rotate-project kernel. The optimized BS4/8 kernel is correct, but
a production-shaped H100 comparison found opposite performance for the two
dense merged-projection paths. The table reports the geometric mean of
`fused latency / unfused latency`; values below 1 favor fusion.

| Production path | Pool block width | Eager | CUDA graph |
|---|---:|---:|---:|
| QKV | 4 | 1.169 | 4.968 |
| QKV | 8 | 0.962 | 4.045 |
| Gate/up | 4 | 0.460 | 0.443 |
| Gate/up | 8 | 0.469 | 0.467 |

The serving default enables CUDA graphs for supported generation decode
batches. Consequently, tiny QKV pools pay a large penalty on the common
decode path when they select the fused kernel. Gate/up must retain its fused
path because its production fusion is structurally different and consistently
faster than its unfused fallback.

## Goals

- Make the existing unfused dense QKV path the default when the OFT rotation
  pool's static block width is 4 or 8.
- Keep dense QKV fusion unchanged for static block widths of 16 and larger.
- Keep dense gate/up fusion unchanged at every supported block width.
- Preserve output semantics, adapter loading, public APIs, CUDA-graph capture,
  and direct access to the tiny fused kernel for tests and benchmarking.

## Non-goals

- Removing the fused BS4/8 kernel or its direct correctness coverage.
- Selecting a path from the active adapter's runtime block-size scalar.
- Adding a server flag, environment variable, GPU-architecture check, or
  request-scoped policy.
- Changing grouped-MoE, non-BF16, quantized, segmented, or mixed-adapter paths.
- Reworking the gate/up fused path.

## Chosen Design

`split_dense_merged_projection` already owns the dense QKV and gate/up fast-path
selection. Add the static predicate `R.shape[-1] >= 16` only to its QKV fused
condition. Do not add the predicate to the shared eligibility condition and do
not change the gate/up condition.

For a four-dimensional serving buffer, dispatch becomes:

```text
QKV and R.shape[-1] in {4, 8}  -> existing run_qkv_oft fallback + projections
QKV and R.shape[-1] >= 16      -> existing fused eligibility and fallback rules
gate/up at every supported BS  -> existing fused eligibility and fallback rules
```

The threshold is intentionally based on `R.shape[-1]`, which is the pool's
capture-static maximum OFT block width. It is not based on runtime `bsv`. If a
pool is allocated with width 16 or larger but currently serves a BS4/8 adapter,
the pool remains eligible for QKV fusion. This preserves one Python-selected
topology between CUDA-graph capture and replay.

No backend method, launcher signature, tensor layout, transport, or public
configuration changes. A tiny QKV miss records the existing
`fallback_ineligible` outcome and continues through the established unfused
code. The direct fused kernel remains callable by its existing tests and
benchmarks.

## Correctness and CUDA-Graph Behavior

The selected paths already implement the same OFT transform and projection.
The completed comparison covered 160 unique BF16 cases across BS4/8, active
rotation and identity, bias present and absent, QKV and gate/up shapes. Every
case passed the current per-element rule: absolute difference at most `2e-3`
or an exact/adjacent BF16 value.

Using a static tensor-shape predicate is CUDA-graph safe:

- A tiny pool captures the unfused QKV topology during the identity sentinel.
- Active replay updates existing backend metadata without changing that
  topology.
- A non-tiny pool continues to capture the fused QKV topology.
- Gate/up capture and replay are unchanged.

Runtime or request-scoped block-size dispatch is excluded because Python
cannot change a captured graph's nodes during replay.

## Verification

Add CPU dispatch tests around `split_dense_merged_projection` that assert
consumer-visible outputs from distinct deterministic backend paths:

- QKV with static widths 4 and 8 produces the existing unfused-path result.
- QKV at the boundary width 16 produces the fused-path result.
- Gate/up with static widths 4 and 8 still produces the fused-path result.

The tests must exercise the real dispatch function and infer the selected path
from its output, rather than asserting mock call counts or source text.

Then run the existing OFT fused/unfused parity and CUDA-graph tests on GPU. The
new default must not weaken or remove direct BS4/8 fused-kernel coverage.

## Risk and Rollback

The performance evidence is from one H100 environment. A universal static
policy avoids architecture-specific configuration and favors the default
CUDA-graph decode path, but eager QKV at BS8 may be about 4 percent slower.
This is accepted because the graph path was about 75 percent faster with the
unfused selection and is the normal serving decode mode.

Rollback is one dispatch predicate: removing the QKV-only
`R.shape[-1] >= 16` condition restores the previous default without changing
kernel support or serialized adapter data.

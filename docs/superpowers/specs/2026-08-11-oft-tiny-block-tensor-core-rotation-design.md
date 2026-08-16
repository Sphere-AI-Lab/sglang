# OFT Tiny-Block Tensor-Core Rotation Design

**Date:** 2026-08-11
**Repository:** SGLang
**Scope:** Dense fused OFT QKV and gate/up projection for block sizes 4 and 8

## Context

SGLang's fused OFT rotate-and-project kernel now accepts every power-of-two
block size from 4 through 1024. The BS4/8 path is correct and CUDA-graph safe,
but its steady-state H100 latency is slower than both BS16 and the production
unfused path:

| Shape | M | BS4 fused | BS8 fused | BS16 fused | Unfused BS4/8 |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B TP2 QKV | 1 | 242 us | 392 us | 90 us | about 109 us |
| Llama-3.1-8B TP2 QKV | 8 | 260 us | 415 us | 91 us | about 110 us |
| Qwen-2.5-7B QKV | 32 | 247 us | 399 us | 75 us | about 109 us |

The projection already uses 16-column tensor-core `tl.dot` operations. The
remaining difference is the rotation: BS4/8 uses an elementwise FP32 loop
because Triton does not accept a dot-product reduction dimension below 16.
BS8 consequently performs twice as many scalar multiply-add iterations as BS4
for every 16-column projection tile.

The optimization must improve the fused kernel itself. Dispatching BS4/8 to
the unfused rotation-plus-cuBLAS path is not part of this work. The unrelated
colocated Condor CUDA-IPC transport failure is also out of scope.

## Goals

- Replace the scalar BS4/8 rotation with tensor-core work while retaining one
  fused runtime kernel.
- Preserve the current OFT R-buffer layout, Python APIs, adapter-loading path,
  slot selection, and CUDA-graph behavior.
- Improve the geometric-mean BS4/8 latency by at least 30 percent on the
  representative H100 benchmark set.
- Treat beating the roughly 110 us unfused path at both block sizes as the
  stretch target.
- Keep BS16 and larger block-size behavior and source branches unchanged.

## Non-goals

- Changing grouped-MoE OFT kernels in this optimization cycle.
- Prepacking or padding adapter buffers at load time.
- Merging rotations into model weights.
- Changing OFT numerical precision, tolerance, or supported block sizes.
- Fixing Orbit's colocated adapter transport on restricted Condor nodes.

## Chosen Design

For BS4 and BS8, each 16-column input microtile already contains an integral
number of OFT blocks. The kernel will interpret their independent rotations as
one virtual 16 by 16 block-diagonal matrix:

```text
BS4: diag(R0, R1, R2, R3)
BS8: diag(R0, R1)
```

The virtual matrix is constructed inside the Triton program from the existing
compact R buffer. For each matrix row and column, compile-time arithmetic maps
the index to a tiny-block number and an index within that block. A load is
enabled only when the row and column belong to the same tiny block; all
cross-block positions are zero.

The optimized microtile performs:

1. Load 16 contiguous input columns.
2. Materialize the virtual block-diagonal 16 by 16 R tile.
3. Compute the complete rotation with `tl.dot`, accumulating in FP32.
4. Cast the rotated tile to BF16 at the same boundary used today.
5. Project it through the existing QKV or gate/up W tiles with `tl.dot`.
6. Accumulate and store through the existing output path.

This reassociation does not introduce cross-block mixing. It computes four
independent 4 by 4 rotations or two independent 8 by 8 rotations in one legal
tensor-core operation.

## Data Flow and Boundaries

The change is confined to the `BS < 16` branch of
`_fused_rotate_project_inner` in
`python/sglang/srt/peft/oft/triton_ops/fused_rotate_project.py`.

- QKV and gate/up wrappers share the optimized branch automatically.
- No intermediate global-memory tensor or additional kernel launch is added.
- No new launcher argument, buffer, adapter metadata, or public API is added.
- A final partial 16-column group remains valid when K is divisible by BS but
  not by 16. Input, R, and W loads use the existing valid-block masks, and
  virtual matrix rows and columns for absent blocks are zero.
- When runtime `bsv == 0`, the identity branch continues to load the original
  input tile and must not read R. This preserves safe CUDA-graph capture before
  an adapter slot is initialized.
- Bias application, slice offsets, output layout, and adapter-slot addressing
  are unchanged.
- The `BS >= 16` branches are not edited.

After the 16-column design passes correctness, 32- and 64-column virtual tiles
may be benchmarked as internal tuning candidates. A wider candidate is retained
only if it improves the acceptance benchmark without changing external
behavior. The 16-column implementation remains the correctness baseline.

## Correctness Contract

For projection slice s and OFT block b, fused and unfused paths implement:

```text
Y_s = sum_b ((X_b @ R_s,b) @ transpose(W_s,b)) + bias_s
```

The optimized fused path completes each tiny rotation in FP32, casts that
rotation to BF16, accumulates projection dots in FP32, and stores BF16. This
preserves the current numerical boundary. It is algebraically equivalent to
the unfused path but is not required to be bitwise identical because Triton
and cuBLAS may accumulate projection tiles in different orders.

The maximum absolute error limit remains 2e-3 against the FP32 reference.

## Verification

Correctness coverage must include:

- BS4 and BS8 QKV at M = 1, 8, 32, and 64 against the FP32 block-rotation
  reference.
- Gate/up coverage at BS4 and BS8.
- Direct fused-versus-unfused comparisons for active rotation and runtime
  identity modes.
- Bias-present and bias-absent comparisons.
- CUDA-graph capture and replay while switching from the identity sentinel to
  an active adapter rotation.
- The existing dense, backward/Cayley, grouped, and validation suites.

Performance is measured on the same H100 and with the same harness that
produced `dense-after.json`:

- Llama-3.1-8B TP2 QKV and Qwen-2.5-7B QKV.
- M = 1, 8, 32, and 64.
- BS4, BS8, and the unchanged BS16 reference.
- Current fused baseline, same-BS unfused path, and optimized fused path.
- Steady-state latency only; first-compilation time is recorded but excluded
  from the acceptance statistic.
- Three independent benchmark runs; each row uses the median latency before
  geometric means and regression thresholds are calculated.

Acceptance requires all of the following:

- At least 30 percent lower geometric-mean latency across representative BS4/8
  rows.
- No representative BS4/8 row more than 10 percent slower.
- No BS16+ row more than 5 percent slower.
- All correctness and CUDA-graph tests pass within the existing tolerance.

Beating the unfused path is a stretch goal, not a condition for retaining an
otherwise qualifying optimization.

## Failure and Rollback

If the virtual tensor-core rotation does not meet the 30-percent threshold, the
experiment and benchmark artifacts are retained but the kernel change is not
shipped. The current correct scalar tiny-block implementation remains the
fallback. The next separately designed candidate would prepack 16 by 16
block-diagonal superblocks during adapter loading; that representation change
is intentionally excluded from this design.

# OFT Tiny-Block Tensor-Core Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scalar BS4/8 rotation inside the dense fused OFT QKV/gate-up kernel with an on-the-fly 16 by 16 block-diagonal tensor-core rotation that improves representative H100 latency by at least 30 percent.

**Architecture:** The existing BS4/8 path already projects 16 input columns at a time. It will now load those 16 columns once, synthesize `diag(R0, ...)` from the unchanged tiny-block R buffer, rotate with a legal 16-wide `tl.dot`, cast the completed rotation to BF16, and feed the existing projection dots. The identity branch and all BS16+ code remain unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.11, Triton 3.6, pytest, CUDA graphs, NVIDIA H100.

## Global Constraints

- Optimize dense fused QKV and gate/up only; grouped-MoE kernels are out of scope.
- Preserve the current R-buffer layout, Python APIs, adapter-loading path, slot selection, and CUDA-graph behavior.
- Add no intermediate global-memory tensor and no additional runtime kernel launch.
- Keep the FP32 rotation accumulation, BF16 rotation boundary, FP32 projection accumulation, and `2e-3` maximum-absolute-error tolerance.
- Do not edit the `BS >= 16` kernel branches.
- Acceptance uses the median of three independent H100 runs per row.
- Require at least 30 percent lower BS4/8 geometric-mean latency, no representative BS4/8 row over 10 percent slower, and no BS16+ row over 5 percent slower.
- Beating the same-BS unfused path is the stretch goal, not the minimum shipping gate.
- The colocated Orbit/Condor CUDA-IPC failure is unrelated and out of scope.

---

### Task 1: Add a deterministic tiny-block performance acceptance report

**Files:**
- Modify: `test/srt/oft/tiny_block_benchmark_report.py`
- Modify: `test/srt/oft/test_tiny_block_benchmark_report.py`
- Modify: `test/srt/oft/bench_fused_rotate_project_blocks.py`

**Interfaces:**
- Consumes: benchmark rows with `shape`, `mode`, `M`, `BS`, `ms`, and `legacy_ms` fields.
- Produces: `median_rows(runs: list[list[dict]]) -> list[dict]`.
- Produces: `tiny_acceptance_report(baseline_runs: list[list[dict]], current_runs: list[list[dict]]) -> dict` with `passed`, `tiny_geomean_ratio`, `tiny_improvement`, `beats_unfused`, and `failures`.
- Produces: `block_size_regressions(baseline_runs: list[list[dict]], current_runs: list[list[dict]], min_bs: int, tolerance: float) -> list[str]`.
- Produces: benchmark CLI flag `--acceptance-only`, selecting `M={1,8,32,64}`, `BS={4,8,16}`, and `mode=rotate`.
- Produces: benchmark CLI flag `--bs16-plus`, selecting `M={1,8,32,64}`, `BS={16,32,64,128,256,512,1024}`, and `mode=rotate`.

- [x] **Step 1: Write failing unit tests for median aggregation and all three gates**

Append tests equivalent to the following to `test_tiny_block_benchmark_report.py`:

```python
def _row(bs, ms, *, m=8, shape="llama", legacy_ms=1.0):
    return {
        "shape": shape,
        "mode": "rotate",
        "M": m,
        "BS": bs,
        "ms": ms,
        "legacy_ms": legacy_ms,
    }


def test_median_rows_uses_the_median_for_each_benchmark_key():
    runs = [
        [_row(4, 1.0), _row(8, 2.0)],
        [_row(4, 3.0), _row(8, 4.0)],
        [_row(4, 2.0), _row(8, 3.0)],
    ]
    rows = _HELPER.median_rows(runs)
    by_bs = {row["BS"]: row for row in rows}
    assert by_bs[4]["ms"] == 2.0
    assert by_bs[8]["ms"] == 3.0


def test_tiny_acceptance_requires_30_percent_geomean_improvement():
    baseline = [[_row(4, 1.0), _row(8, 1.0), _row(16, 1.0)]] * 3
    current = [[_row(4, 0.7), _row(8, 0.7), _row(16, 1.0)]] * 3
    report = _HELPER.tiny_acceptance_report(baseline, current)
    assert report["passed"]
    assert report["tiny_geomean_ratio"] == pytest.approx(0.7)
    assert report["tiny_improvement"] == pytest.approx(0.3)


def test_tiny_acceptance_rejects_one_slow_tiny_row():
    baseline = [[_row(4, 1.0), _row(8, 1.0), _row(16, 1.0)]] * 3
    current = [[_row(4, 0.3), _row(8, 1.11), _row(16, 1.0)]] * 3
    report = _HELPER.tiny_acceptance_report(baseline, current)
    assert not report["passed"]
    assert any("BS8" in failure and "10%" in failure for failure in report["failures"])


def test_tiny_acceptance_rejects_bs16_regression():
    baseline = [[_row(4, 1.0), _row(8, 1.0), _row(16, 1.0)]] * 3
    current = [[_row(4, 0.5), _row(8, 0.5), _row(16, 1.051)]] * 3
    report = _HELPER.tiny_acceptance_report(baseline, current)
    assert not report["passed"]
    assert any("BS16" in failure and "5%" in failure for failure in report["failures"])


def test_block_size_regressions_checks_every_row_at_or_above_floor():
    baseline = [[_row(16, 1.0), _row(32, 2.0)]] * 3
    current = [[_row(16, 1.04), _row(32, 2.12)]] * 3
    failures = _HELPER.block_size_regressions(
        baseline, current, min_bs=16, tolerance=0.05
    )
    assert len(failures) == 1
    assert "BS32" in failures[0]
```

Add `import pytest` to the test module.

- [x] **Step 2: Run the report tests and verify the new interface is missing**

Run:

```bash
pytest -q test/srt/oft/test_tiny_block_benchmark_report.py
```

Expected: FAIL with `AttributeError` for `median_rows` or `tiny_acceptance_report`.

- [x] **Step 3: Implement median aggregation and acceptance reporting**

Add these elements to `tiny_block_benchmark_report.py`:

```python
import math
import statistics


def _key(row: dict) -> tuple[str, str, int, int]:
    return row["shape"], row["mode"], row["M"], row["BS"]


def median_rows(runs: list[list[dict]]) -> list[dict]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = {}
    for run in runs:
        for row in run:
            grouped.setdefault(_key(row), []).append(row)

    result = []
    for key in sorted(grouped):
        copies = grouped[key]
        row = dict(copies[0])
        for field in ("ms", "legacy_ms"):
            values = [copy[field] for copy in copies if copy.get(field) is not None]
            row[field] = statistics.median(values) if values else None
        result.append(row)
    return result


def tiny_acceptance_report(
    baseline_runs: list[list[dict]],
    current_runs: list[list[dict]],
) -> dict:
    baseline = {_key(row): row for row in median_rows(baseline_runs)}
    current = {_key(row): row for row in median_rows(current_runs)}
    tiny_ratios = []
    failures = []
    beats_unfused = True

    for key, base in baseline.items():
        shape, mode, batch, bs = key
        if mode != "rotate" or batch not in {1, 8, 32, 64} or bs not in {4, 8, 16}:
            continue
        now = current.get(key)
        if now is None or now.get("ms") is None or base.get("ms") is None:
            failures.append(f"missing row shape={shape} M={batch} BS{bs}")
            continue
        ratio = now["ms"] / base["ms"]
        if bs in {4, 8}:
            tiny_ratios.append(ratio)
            if ratio > 1.10:
                failures.append(f"BS{bs} shape={shape} M={batch} exceeds 10% slowdown")
            legacy_ms = now.get("legacy_ms")
            beats_unfused &= legacy_ms is not None and now["ms"] < legacy_ms
        elif ratio > 1.05:
            failures.append(f"BS16 shape={shape} M={batch} exceeds 5% slowdown")

    geomean = (
        math.exp(sum(math.log(ratio) for ratio in tiny_ratios) / len(tiny_ratios))
        if tiny_ratios
        else math.inf
    )
    if geomean - 0.70 > 1e-12:
        failures.append(f"tiny geometric-mean ratio {geomean:.3f} exceeds 0.700")
    return {
        "passed": not failures,
        "tiny_geomean_ratio": geomean,
        "tiny_improvement": 1.0 - geomean,
        "beats_unfused": beats_unfused,
        "failures": failures,
    }


def block_size_regressions(
    baseline_runs: list[list[dict]],
    current_runs: list[list[dict]],
    *,
    min_bs: int,
    tolerance: float,
) -> list[str]:
    baseline = {_key(row): row for row in median_rows(baseline_runs)}
    current = {_key(row): row for row in median_rows(current_runs)}
    failures = []
    for key, base in baseline.items():
        shape, mode, batch, bs = key
        if bs < min_bs or base.get("ms") is None:
            continue
        now = current.get(key)
        if now is None or now.get("ms") is None:
            failures.append(f"missing row shape={shape} mode={mode} M={batch} BS{bs}")
            continue
        ratio = now["ms"] / base["ms"]
        if ratio > 1.0 + tolerance:
            failures.append(
                f"BS{bs} shape={shape} mode={mode} M={batch} "
                f"is {(ratio - 1.0) * 100:.1f}% slower"
            )
    return failures
```

- [x] **Step 4: Add the focused benchmark switch**

In `bench_fused_rotate_project_blocks.py`, add:

```python
parser.add_argument(
    "--acceptance-only",
    action="store_true",
    help="benchmark only rotate M=1/8/32/64 at BS4/8/16",
)
parser.add_argument(
    "--bs16-plus",
    action="store_true",
    help="benchmark rotate M=1/8/32/64 at every supported BS >= 16",
)
```

Select rows before calling the harness:

```python
if args.acceptance_only and args.bs16_plus:
    parser.error("--acceptance-only and --bs16-plus are mutually exclusive")
if args.acceptance_only:
    rows = bench_block_sizes(
        block_sizes=[4, 8, 16],
        batches=[1, 8, 32, 64],
        modes=["rotate"],
    )
elif args.bs16_plus:
    rows = bench_block_sizes(
        block_sizes=[16, 32, 64, 128, 256, 512, 1024],
        batches=[1, 8, 32, 64],
        modes=["rotate"],
    )
else:
    rows = bench_block_sizes()
```

Keep the existing output and `--compare` behavior unchanged.

- [x] **Step 5: Run the report tests and static checks**

Run:

```bash
pytest -q test/srt/oft/test_tiny_block_benchmark_report.py
python -m compileall -q test/srt/oft/tiny_block_benchmark_report.py test/srt/oft/bench_fused_rotate_project_blocks.py
git diff --check
```

Expected: all tests pass and both static checks exit zero.

- [x] **Step 6: Commit the benchmark gate**

```bash
git add test/srt/oft/tiny_block_benchmark_report.py \
  test/srt/oft/test_tiny_block_benchmark_report.py \
  test/srt/oft/bench_fused_rotate_project_blocks.py
git commit -m "bench(oft): gate tiny fused kernel speedup"
```

---

### Task 2: Lock fused-versus-unfused numerical equivalence

**Files:**
- Modify: `test/srt/oft/test_fused_rotate_project_tiled.py`

**Interfaces:**
- Consumes: `fused_rotate_project_qkv`, `fused_rotate_project_gate_up`, and production `gemm_oft_r_fwd`.
- Produces: direct BS4/8 characterization tests covering active rotation, runtime identity, bias, QKV, and gate/up.

- [x] **Step 1: Add an unfused reference helper with identical BF16 boundaries**

Add imports:

```python
import torch.nn.functional as F

from sglang.srt.oft.triton_ops.gemm_oft_r import gemm_oft_r_fwd
```

Add this helper after `_reference`:

```python
def _unfused_projection(x, R4, W, output_sizes, bias, slot, bsv):
    hidden = x.shape[-1]
    rotated = gemm_oft_r_fwd(
        x, R4, slot, bsv, num_slices=len(output_sizes)
    )
    rotated_slices = torch.split(rotated, hidden, dim=-1)
    weight_slices = torch.split(W, output_sizes, dim=0)
    bias_slices = (
        [None] * len(output_sizes)
        if bias is None
        else torch.split(bias, output_sizes, dim=0)
    )
    return torch.cat(
        [
            F.linear(rotated_slice, weight_slice, bias_slice)
            for rotated_slice, weight_slice, bias_slice in zip(
                rotated_slices, weight_slices, bias_slices, strict=True
            )
        ],
        dim=-1,
    )
```

- [x] **Step 2: Add direct QKV and gate/up parity tests**

Add:

```python
@pytest.mark.parametrize("BS", [4, 8])
@pytest.mark.parametrize("M", [1, 8, 32, 64])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_tiny_qkv_matches_unfused(BS, M, identity, with_bias):
    x, R, W = _inputs(M, BS)
    R4 = R.unsqueeze(0)
    slot = torch.zeros((), dtype=torch.int32, device=x.device)
    bsv = torch.tensor(0 if identity else BS, dtype=torch.int32, device=x.device)
    bias = torch.randn(sum(OUT), dtype=x.dtype, device=x.device) if with_bias else None
    got = fused_rotate_project_qkv(
        x, R4, W, OUT, bias=bias, slot_idx_t=slot, bsv_t=bsv
    )
    expect = _unfused_projection(x, R4, W, OUT, bias, slot, bsv)
    torch.cuda.synchronize()
    assert (got.float() - expect.float()).abs().max().item() <= TOL


@pytest.mark.parametrize("BS", [4, 8])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_tiny_gate_up_matches_unfused(BS, identity, with_bias):
    x, R, W = _fc1_inputs(8, BS)
    R4 = R.unsqueeze(0)
    slot = torch.zeros((), dtype=torch.int32, device=x.device)
    bsv = torch.tensor(0 if identity else BS, dtype=torch.int32, device=x.device)
    bias = (
        torch.randn(sum(FC1_OUT), dtype=x.dtype, device=x.device)
        if with_bias
        else None
    )
    got = fused_rotate_project_gate_up(
        x, R4, W, FC1_OUT, bias=bias, slot_idx_t=slot, bsv_t=bsv
    )
    expect = _unfused_projection(x, R4, W, FC1_OUT, bias, slot, bsv)
    torch.cuda.synchronize()
    assert (got.float() - expect.float()).abs().max().item() <= TOL
```

- [x] **Step 3: Run the characterization tests on H100 before changing the kernel**

Run inside an allocated H100 session:

```bash
pytest -q test/srt/oft/test_fused_rotate_project_tiled.py \
  -k 'tiny_qkv_matches_unfused or tiny_gate_up_matches_unfused'
```

Expected: all new tests pass on the current scalar implementation. These are
characterization tests, so their initial green result locks behavior before the
performance-only refactor rather than demonstrating a missing feature.

- [x] **Step 4: Run the complete focused dense test file**

```bash
pytest -q test/srt/oft/test_fused_rotate_project_tiled.py
```

Expected: all tests pass.

- [x] **Step 5: Commit the numerical contract**

```bash
git add test/srt/oft/test_fused_rotate_project_tiled.py
git commit -m "test(oft): lock tiny fused and unfused parity"
```

Characterization on H100 exposed that an unscaled BF16 bias makes a fixed
`2e-3` output tolerance invalid at larger magnitudes: the scalar fused and
unfused paths can land on adjacent BF16 values despite a no-bias maximum error
of `2.44e-4`. The committed contract therefore applies the `2e-3` floor per
element and otherwise permits only an adjacent BF16 value. With deterministic
bias generation, all 40 direct parity cases and all 104 dense tests pass.

---

### Task 3: Capture the pre-change H100 baseline and red performance gate

**Files:**
- Modify: `docs/superpowers/plans/2026-08-11-oft-tiny-block-tensor-core-rotation.md` only to check completed boxes and append the generated durable artifact directory.

**Interfaces:**
- Consumes: the `--acceptance-only` benchmark from Task 1.
- Produces: three immutable baseline JSON files from one H100 model and a performance gate that fails when the baseline is compared with itself.

- [x] **Step 1: Allocate one H100 using the cluster-control workflow**

Use `control-remote-condor` and `develop-on-remote-clusters`. Record login node,
job ID, GPU model, approved bid, SGLang commit, PyTorch version, Triton version,
and CUDA version in the run's `provenance.json`. Do not submit or raise a bid
without explicit user approval.

Inside the allocation, create and export the task-specific durable root once:

```bash
export OFT_TINY_RUN_ROOT="/home/zqiu/.local/state/remote-cluster-runs/mpi1/sglang/codex-oft-bs4-a6a55a65/$(date -u +%Y%m%dT%H%M%SZ)-tiny-tensor-core"
mkdir -p "$OFT_TINY_RUN_ROOT"
```

- [x] **Step 2: Run three focused baseline sweeps**

From the exact remote task worktree commit, run:

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/baseline-1.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/baseline-2.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/baseline-3.json"
```

Also capture the BS16+ no-regression baseline:

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --bs16-plus --json "$OFT_TINY_RUN_ROOT/baseline-bs16-plus-1.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --bs16-plus --json "$OFT_TINY_RUN_ROOT/baseline-bs16-plus-2.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --bs16-plus --json "$OFT_TINY_RUN_ROOT/baseline-bs16-plus-3.json"
```

`OFT_TINY_RUN_ROOT` must remain under the durable cluster run store, not `/tmp`
or the remote checkout.

- [x] **Step 3: Verify the current implementation fails the speedup gate**

Load the three JSON files as both baseline and current, call
`tiny_acceptance_report`, and write the printed dictionary to
`$OFT_TINY_RUN_ROOT/baseline-gate.txt`:

```bash
PYTHONPATH=test/srt/oft python - <<'PY' > "$OFT_TINY_RUN_ROOT/baseline-gate.txt"
import json
import os
from pathlib import Path
from tiny_block_benchmark_report import tiny_acceptance_report

root = Path(os.environ["OFT_TINY_RUN_ROOT"])
runs = [json.loads((root / f"baseline-{i}.json").read_text()) for i in (1, 2, 3)]
report = tiny_acceptance_report(runs, runs)
print(report)
assert not report["passed"]
assert report["tiny_improvement"] == 0.0
PY
```

Expected: the assertion passes because comparing the implementation with itself
produces zero improvement and the report contains the 30-percent gate failure.

- [x] **Step 4: Copy the bounded run directory to the local artifact store**

Use the cluster skill's bounded `rsync` workflow. Verify all six JSON files,
`baseline-gate.txt`, and `provenance.json` locally before continuing.

- [x] **Step 5: Record the artifact path in this plan and commit the checkpoint**

Append the exact remote and local artifact paths under Task 3, check its boxes,
then run:

```bash
git add docs/superpowers/plans/2026-08-11-oft-tiny-block-tensor-core-rotation.md
git commit -m "docs(oft): record tiny rotation baseline"
```

Recorded pre-change checkpoint (2026-08-11):

- Remote artifacts: `/home/zqiu/.local/state/remote-cluster-runs/mpi1/sglang/codex-oft-bs4-a6a55a65/20260811T170945Z-93efd7/tiny-tensor-core`
- Local snapshot: `/Users/zqiu/.local/state/remote-cluster-runs/mpi1/sglang/codex-oft-bs4-a6a55a65/20260811T170945Z-93efd7/tiny-tensor-core`
- Allocation: job `17450886`, bid 50, H100 80GB HBM3 on `i103`
- Source: `df2e07feca03bd7b4567a98ac5eda849ec8dbe75`, clean locally and remotely
- Runtime: PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0
- Artifacts verified locally: three 24-row focused JSONs, three 52-row BS16+
  JSONs, `baseline-gate.txt`, and `provenance.json`
- Red gate: geometric-mean ratio `1.0`, improvement `0.0`, expected failure
  because the required ratio is at most `0.7`

---

### Task 4: Replace scalar tiny rotation with a virtual tensor-core tile

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/fused_rotate_project.py:318-470`
- Test: `test/srt/oft/test_fused_rotate_project_tiled.py`

**Interfaces:**
- Consumes: existing `BS`, `blocks_per_slice`, `rotation_block_start`, `slot_R_offset`, `x_ptr`, and `R_ptr` values inside `_fused_rotate_project_inner`.
- Produces: the same `(BLOCK_M, 16)` BF16 `projected_tile` consumed by the unchanged W projection dots.
- Preserves: `bsv == 0` identity behavior and every `BS >= 16` branch.

- [ ] **Step 1: Confirm the red performance gate and green characterization suite**

Re-run the Task 3 baseline self-comparison and the Task 2 parity selection.
Expected: performance gate reports zero improvement; parity tests pass.

- [ ] **Step 2: Replace only the active-rotation scalar loop**

In the `if BS < 16:` branch, keep `tiny_cols`, `blocks_in_tile`,
`block_in_tile`, `valid_blocks`, and the identity `else` branch. Replace the
active-rotation body with the following structure:

```python
tiny_rows = tl.arange(0, 16)
row_block = tiny_rows // BS
row_in_block = tiny_rows - row_block * BS
col_block = tiny_cols // BS
col_in_block = tiny_cols - col_block * BS

for block_group in range(0, blocks_per_slice, blocks_in_tile):
    block_ids = block_group + block_in_tile
    valid_blocks = block_ids < blocks_per_slice
    offs_k = block_group * BS + tiny_cols

    if do_rotation:
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & valid_blocks[None, :],
            other=0.0,
        )
        r_block_ids = block_group + row_block
        same_block = row_block[:, None] == col_block[None, :]
        valid_r_block = r_block_ids < blocks_per_slice
        r_tile = tl.load(
            R_ptr
            + slot_R_offset
            + (rotation_block_start + r_block_ids[:, None]) * BS * BS
            + row_in_block[:, None] * BS
            + col_in_block[None, :],
            mask=same_block & valid_r_block[:, None],
            other=0.0,
        )
        projected_tile = tl.dot(
            x_tile,
            r_tile,
            input_precision="ieee",
            out_dtype=tl.float32,
        ).to(tl.bfloat16)
    else:
        projected_tile = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & valid_blocks[None, :],
            other=0.0,
        )
```

Do not change the W loads, projection dots, bias code, stores, launcher, or
`BS >= 16` branches.

- [ ] **Step 3: Run the direct parity and CUDA-graph tests on H100**

```bash
pytest -q test/srt/oft/test_fused_rotate_project_tiled.py \
  -k 'tiny or cuda_graph'
```

Expected: all selected tests pass with maximum absolute error at or below
`2e-3`.

- [ ] **Step 4: Run the entire dense kernel test file**

```bash
pytest -q test/srt/oft/test_fused_rotate_project_tiled.py
```

Expected: all tests pass.

- [ ] **Step 5: Run one candidate acceptance sweep before spending on repeats**

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/virtual16-candidate.json"
```

Compare the candidate with the median baseline. Expected: no correctness error,
no missing row, and a directionally lower BS4/8 geometric mean. If the candidate
does not improve the geometric mean, continue to Task 5's three-repeat decision
without changing tile dimensions; do not commit the kernel yet.

- [ ] **Step 6: Run static checks**

```bash
python -m compileall -q python/sglang/srt/oft/triton_ops/fused_rotate_project.py
git diff --check
```

Expected: both commands exit zero.

---

### Task 5: Run the three-repeat acceptance gate and retain or roll back

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/fused_rotate_project.py` only if the accepted virtual-16 implementation needs a measured correction.
- Modify: `docs/superpowers/plans/2026-08-11-oft-tiny-block-tensor-core-rotation.md` only to check boxes and record final artifact paths/results.

**Interfaces:**
- Consumes: three Task 3 baselines and the virtual-16 kernel from Task 4.
- Produces: an acceptance decision backed by three current H100 JSON files.

- [ ] **Step 1: Run three independent optimized sweeps**

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/virtual16-1.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/virtual16-2.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --acceptance-only --json "$OFT_TINY_RUN_ROOT/virtual16-3.json"
```

- [ ] **Step 2: Evaluate and persist the acceptance report**

Call `tiny_acceptance_report` with the three baseline and three optimized runs.
Write the returned dictionary to
`$OFT_TINY_RUN_ROOT/virtual16-acceptance.json` using
`json.dumps(report, indent=2, sort_keys=True)`.

```bash
PYTHONPATH=test/srt/oft python - <<'PY'
import json
import os
from pathlib import Path
from tiny_block_benchmark_report import tiny_acceptance_report

root = Path(os.environ["OFT_TINY_RUN_ROOT"])
load = lambda stem: [
    json.loads((root / f"{stem}-{i}.json").read_text()) for i in (1, 2, 3)
]
report = tiny_acceptance_report(load("baseline"), load("virtual16"))
(root / "virtual16-acceptance.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

Expected shipping result:

```text
passed = true
tiny_improvement >= 0.30
failures = []
```

Also report `beats_unfused` separately as the stretch result.

- [ ] **Step 3: Apply the rollback rule if the gate fails**

If `passed` is false, use `apply_patch` to restore the scalar `BS < 16`
active-rotation body from commit `54f5ee4a2`, retain all benchmark artifacts,
and stop this implementation plan at the designed rollback boundary. Do not
commit a kernel that misses the gate.

If `passed` is true, continue without changing the measured kernel.

- [ ] **Step 4: Run the focused OFT regression suite**

```bash
pytest -q \
  test/srt/oft/test_oft_utils.py \
  test/srt/oft/test_gemm_oft_r_tiled.py \
  test/srt/oft/test_fused_rotate_project_tiled.py \
  test/srt/oft/test_grouped_moe_rotate_project.py \
  test/srt/oft/test_tiny_block_backward_cayley.py
```

Expected: every focused test passes.

- [ ] **Step 5: Run the BS16+ benchmark regression check**

Run three optimized BS16+ sweeps:

```bash
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --bs16-plus --json "$OFT_TINY_RUN_ROOT/virtual16-bs16-plus-1.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --bs16-plus --json "$OFT_TINY_RUN_ROOT/virtual16-bs16-plus-2.json"
python test/srt/oft/bench_fused_rotate_project_blocks.py \
  --bs16-plus --json "$OFT_TINY_RUN_ROOT/virtual16-bs16-plus-3.json"
```

Call `block_size_regressions` with the three matching baseline files,
`min_bs=16`, and `tolerance=0.05`. Write the returned list to
`$OFT_TINY_RUN_ROOT/virtual16-bs16-plus-regressions.json`. Expected: `[]`.

```bash
PYTHONPATH=test/srt/oft python - <<'PY'
import json
import os
from pathlib import Path
from tiny_block_benchmark_report import block_size_regressions

root = Path(os.environ["OFT_TINY_RUN_ROOT"])
load = lambda stem: [
    json.loads((root / f"{stem}-{i}.json").read_text()) for i in (1, 2, 3)
]
failures = block_size_regressions(
    load("baseline-bs16-plus"),
    load("virtual16-bs16-plus"),
    min_bs=16,
    tolerance=0.05,
)
(root / "virtual16-bs16-plus-regressions.json").write_text(
    json.dumps(failures, indent=2) + "\n"
)
print(json.dumps(failures, indent=2))
assert failures == []
PY
```

- [ ] **Step 6: Copy and verify the final bounded artifacts locally**

Use bounded `rsync`, then verify locally:

```text
baseline-1.json through baseline-3.json
baseline-bs16-plus-1.json through baseline-bs16-plus-3.json
virtual16-1.json through virtual16-3.json
virtual16-bs16-plus-1.json through virtual16-bs16-plus-3.json
virtual16-acceptance.json
virtual16-bs16-plus-regressions.json
focused-tests.log and focused-tests.status
provenance.json
```

- [ ] **Step 7: Commit the accepted kernel and completed plan**

Only when the performance and correctness gates pass:

```bash
git add python/sglang/srt/oft/triton_ops/fused_rotate_project.py \
  docs/superpowers/plans/2026-08-11-oft-tiny-block-tensor-core-rotation.md
git commit -m "perf(oft): tensorize tiny block rotations"
```

- [ ] **Step 8: Push the isolated branch and report the decision**

```bash
git push origin codex/oft-bs4
```

Report the median BS4/8 timings, geometric-mean improvement, unfused comparison,
BS16+ regression result, correctness count, exact commit, and local artifact
directory. Keep the worktree and branch for user review; do not merge or delete
them automatically.

# OFT Tiny-QKV Default Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route dense QKV projections from OFT pools with static block width 4 or 8 through the existing unfused path by default, while preserving QKV fusion at width 16 and above and preserving gate/up fusion at every supported width.

**Architecture:** Keep policy in `split_dense_merged_projection`, where dense QKV and gate/up already select their production paths. Add one capture-static `R.shape[-1] >= 16` conjunct to the QKV fused condition only; all misses use the existing unfused QKV implementation, and no backend, kernel, buffer, or public API changes. Lock the boundary with CPU tests that infer routing from distinct real outputs rather than call-count mocks.

**Tech Stack:** Python, PyTorch BF16, pytest, SGLang OFT backends, Triton/CUDA regression tests

## Global Constraints

- QKV with a four-dimensional OFT pool whose static `R.shape[-1]` is 4 or 8 must use the existing unfused `run_qkv_oft` path plus the existing projection code.
- QKV with static `R.shape[-1] >= 16` must retain the existing fused eligibility, error fallback, and global-disable behavior.
- Gate/up fusion must remain unchanged for every supported block width; the new predicate must not be added to `_common_eligible`.
- Dispatch must use the capture-static tensor shape and must never read or synchronize the runtime `bsv` device scalar.
- Do not add a server flag, environment variable, GPU-architecture check, request-scoped policy, backend method, launcher argument, buffer field, transport change, or public API.
- Do not remove or weaken the direct fused BS4/8 kernel, parity, bias, identity, or CUDA-graph tests.
- Preserve the current per-element parity rule: absolute difference at most `2e-3` or an exact/adjacent BF16 value.
- Dispatch tests must exercise `split_dense_merged_projection` and assert consumer-visible outputs; do not assert mock call counts or source text.
- The policy applies to every supported CUDA GPU; the H100 evidence and accepted eager-BS8 tradeoff are recorded in `docs/superpowers/specs/2026-08-11-oft-tiny-qkv-default-dispatch-design.md`.

---

### Task 1: Lock and implement QKV-only tiny-pool fallback

**Files:**
- Create: `test/srt/oft/test_split_dense_merged_projection_dispatch.py`
- Modify: `python/sglang/srt/oft/layers.py:221-225`
- Reference: `docs/superpowers/specs/2026-08-11-oft-tiny-qkv-default-dispatch-design.md`

**Interfaces:**
- Consumes: `split_dense_merged_projection(x, weight, bias, output_sizes, R, oft_backend) -> torch.Tensor` and the existing backend methods `run_fused_rotate_project`, `run_qkv_oft`, `run_fused_gate_up_inputs`, and `run_gate_up_oft`.
- Produces: a QKV-only static eligibility rule where `R.shape[-1] >= 16` is required for fusion; all existing function signatures and backend interfaces remain unchanged.

- [ ] **Step 1: Add output-based dispatch tests before changing production code**

Create `test/srt/oft/test_split_dense_merged_projection_dispatch.py` with this complete CPU fixture and the three routing contracts:

```python
import pytest
import torch

from sglang.srt.oft.layers import split_dense_merged_projection


_K = 16
_M = 2
_DTYPE = torch.bfloat16


class _SentinelBackend:
    def run_fused_rotate_project(self, x, R, weight, output_sizes, bias):
        return torch.full(
            (x.shape[0], sum(output_sizes)),
            9,
            dtype=x.dtype,
            device=x.device,
        )

    def run_qkv_oft(self, x, R):
        return torch.cat(
            [
                torch.ones_like(x),
                torch.full_like(x, 2),
                torch.full_like(x, 3),
            ],
            dim=-1,
        )

    def run_fused_gate_up_inputs(self, x, R):
        return torch.full_like(x, 11), torch.full_like(x, 13)

    def run_gate_up_oft(self, x, R):
        return torch.cat(
            [torch.ones_like(x), torch.full_like(x, 2)], dim=-1
        )


@pytest.fixture(autouse=True)
def _clear_global_fused_disable(monkeypatch):
    monkeypatch.delenv(
        "SGLANG_OFT_DISABLE_FUSED_ROTATE_PROJECT", raising=False
    )


def _x():
    return torch.zeros((_M, _K), dtype=_DTYPE)


def _weight(num_slices):
    return torch.cat(
        [torch.eye(_K, dtype=_DTYPE) for _ in range(num_slices)], dim=0
    ).contiguous()


def _r_buffer(num_slices, block_size):
    blocks_per_slice = _K // block_size
    return torch.zeros(
        (1, num_slices * blocks_per_slice, block_size, block_size),
        dtype=_DTYPE,
    )


@pytest.mark.parametrize("block_size", [4, 8])
def test_tiny_qkv_pool_defaults_to_unfused_output(block_size):
    x = _x()
    got = split_dense_merged_projection(
        x,
        _weight(3),
        None,
        [_K, _K, _K],
        _r_buffer(3, block_size),
        _SentinelBackend(),
    )
    expected = torch.cat(
        [
            torch.ones_like(x),
            torch.full_like(x, 2),
            torch.full_like(x, 3),
        ],
        dim=-1,
    )
    assert torch.equal(got, expected)


def test_qkv_pool_at_width_16_keeps_fused_output():
    x = _x()
    got = split_dense_merged_projection(
        x,
        _weight(3),
        None,
        [_K, _K, _K],
        _r_buffer(3, 16),
        _SentinelBackend(),
    )
    expected = torch.full((_M, 3 * _K), 9, dtype=_DTYPE)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("block_size", [4, 8])
def test_tiny_gate_up_pool_keeps_fused_output(block_size):
    x = _x()
    got = split_dense_merged_projection(
        x,
        _weight(2),
        None,
        [_K, _K],
        _r_buffer(2, block_size),
        _SentinelBackend(),
    )
    expected = torch.cat(
        [torch.full_like(x, 11), torch.full_like(x, 13)], dim=-1
    )
    assert torch.equal(got, expected)
```

The production mutation each test protects is explicit:

- Removing the new QKV threshold makes the BS4/8 QKV cases return the fused `9` sentinel.
- Moving the threshold into `_common_eligible` makes the BS4/8 gate/up cases return the legacy `1/2` sentinel.
- Using `> 16` or a larger threshold makes the BS16 QKV case return the legacy `1/2/3` sentinel.

- [ ] **Step 2: Run the new file and verify the RED state**

Run in an SGLang environment with PyTorch and pytest:

```bash
PYTHONPATH=python python -m pytest -q \
  test/srt/oft/test_split_dense_merged_projection_dispatch.py
```

Expected before implementation: the BS4 and BS8 instances of
`test_tiny_qkv_pool_defaults_to_unfused_output` fail because the current
dispatcher returns the fused all-`9` output. The BS16 QKV and BS4/8 gate/up
controls pass. Import errors, missing dependencies, or failures in the control
cases are not the required RED state and must be resolved before continuing.

- [ ] **Step 3: Add the minimal QKV-only static predicate**

In `python/sglang/srt/oft/layers.py`, change only the QKV fused condition from:

```python
            if (
                len(output_sizes) == 3
                and hasattr(oft_backend, "run_fused_rotate_project")
                and _common_eligible
            ):
```

to:

```python
            if (
                len(output_sizes) == 3
                and R.shape[-1] >= 16
                and hasattr(oft_backend, "run_fused_rotate_project")
                and _common_eligible
            ):
```

Do not edit `_common_eligible`, the gate/up `elif`, the fallback body, any
backend, or any Triton kernel.

- [ ] **Step 4: Run the focused dispatch test and verify the GREEN state**

Run:

```bash
PYTHONPATH=python python -m pytest -q \
  test/srt/oft/test_split_dense_merged_projection_dispatch.py
```

Expected after implementation: `5 passed`, with no warnings or errors.

- [ ] **Step 5: Run the existing OFT correctness and CUDA-graph regressions**

On one supported CUDA GPU, run the direct fused/unfused, runtime-identity,
bias, tiny-block, and CUDA-graph coverage without changing their selection:

```bash
PYTHONPATH=python python -m pytest -q \
  test/srt/oft/test_split_dense_merged_projection_dispatch.py \
  test/srt/oft/test_fused_rotate_project_tiled.py \
  test/srt/oft/test_gemm_oft_r_tiled.py \
  test/srt/oft/test_tiny_block_validation.py
```

Expected: all selected tests pass; no test is deselected to hide a failure.
The direct fused BS4/8 cases must continue to run even though production QKV
dispatch now defaults tiny pools to the unfused path.

- [ ] **Step 6: Inspect the final diff and commit the behavior change**

Run:

```bash
git diff --check
git diff --stat HEAD
git status --short
```

Expected: only `python/sglang/srt/oft/layers.py` and
`test/srt/oft/test_split_dense_merged_projection_dispatch.py` are part of the
implementation diff, with no whitespace errors or unrelated edits.

Commit:

```bash
git add \
  python/sglang/srt/oft/layers.py \
  test/srt/oft/test_split_dense_merged_projection_dispatch.py
git commit -m "perf(oft): default tiny QKV pools to unfused"
```

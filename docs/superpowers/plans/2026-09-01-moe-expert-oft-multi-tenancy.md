# MoE Expert-OFT Multi-Tenancy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let two or more concurrently-resident OFT adapters that both target MoE expert weights apply their own rotation correctly, per token, in the same batch — matching what `FusedMoEWithLoRA` already does for LoRA — without changing the behavior or performance of today's single/no-adapter path.

**Architecture:** Reuse the per-token `weight_indices` list `OFTManager.prepare_oft_batch` already builds (today only consumed by the dense path) to derive a per-token slot-index tensor for MoE layers, and push it onto each `FusedMoE` module as a new "live-read" attribute — the same pattern `moe.w13_oft_r` etc. already use. When at most one non-identity slot is present in the batch (the common case), MoE forward takes today's exact code path unchanged. When two or more are present, it takes a new path: reads the pool's already-allocated multi-slot buffer directly (instead of the single active-slot view) and calls a new Triton kernel variant that selects each token's own slot's R-block instead of one shared R for the whole batch.

**Tech Stack:** Python (PyTorch), Triton. sglang fork (`Sphere-AI-Lab/sglang`), `python/sglang/srt/oft/`.

**Spec:** `docs/superpowers/specs/2026-09-01-moe-expert-oft-multi-tenancy-design.md`

## Global Constraints

- No new CLI flag. Capacity is bounded by the existing `--max-ofts-per-batch`, which already sizes every buffer group's leading (slot) dimension via `register_buffer_group` (`oft/base/mem_pool.py:80-87`) — mirrors LoRA's reuse of `--max-loras-per-batch` for its own MoE buffers.
- The single/no-adapter fast path (today's exact code) must be byte-identical in output and must not regress in performance. Every task that touches shared code must preserve this.
- `canonical_oft` (split per-sub-projection: separate `w1_oft_r`/`w3_oft_r`) is the priority layout; the legacy fused `oft_type="oft"` (`w13_oft_r`) must not regress its *existing single-adapter* behavior, but multi-tenant correctness for the legacy layout specifically is out of scope for this plan.
- A token whose resident adapter has no MoE-target weights loaded for a given layer must fall back to identity rotation for that token — not an error.
- Do not touch the dense (non-MoE) OFT path, LoRA's own MoE path, or the staged/double-buffer OFT manager (`StagedOFTManager`, which correctly relies on exactly-one-thing-active-at-a-time and is out of scope here).

---

## File Structure

- **`python/sglang/srt/oft/oft_manager.py`** (existing, modified): `OFTManager.prepare_oft_batch` gains the per-batch MoE multi-tenancy decision and per-token slot-index tensor; a new helper method pushes it onto the resident `FusedMoE` modules. `OFTManager._init_identity_expert_oft_for_cuda_graph` gains one more attribute binding per module (the full multi-slot buffer reference).
- **`python/sglang/srt/oft/oft_moe_runners.py`** (existing, modified): `make_oft_invoke`'s `invoke` closure branches to a new sibling of `_oft_prerotate` when multi-tenancy is active. New function `_oft_prerotate_multi_tenant` added, same file (mirrors `_oft_prerotate`'s existing shape/pattern).
- **`python/sglang/srt/oft/triton_ops/block_rotate.py`** (existing, modified): new Triton kernel `_oft_block_rotate_kernel_multi_slot` and its Python entry point `apply_oft_rotation_triton_multi_slot`, alongside the existing single-slot kernel (left completely unchanged).
- **`test/registered/unit/oft/test_oft_moe_multi_tenancy.py`** (new): unit tests for the per-batch decision and slot-index tensor construction (Task 1), and the wiring/attribute-pushing step (Task 2), using lightweight test doubles — no GPU required.
- **`test/registered/kernels/ops/test_oft_moe_rotation_multi_slot.py`** (new): the Triton kernel correctness oracle (Task 4) — requires GPU.
- **`test/registered/rl/test_oft_moe_multi_tenant_e2e.py`** (new): end-to-end GPU test with a real MoE model and two concurrently-resident MoE-target OFT adapters (Task 5) — requires GPU.

---

## Task 1: Per-batch MoE multi-tenancy decision and slot-index tensor

**Files:**
- Modify: `python/sglang/srt/oft/oft_manager.py:700-736` (`OFTManager.prepare_oft_batch`)
- Test: `test/registered/unit/oft/test_oft_moe_multi_tenancy.py`

**Interfaces:**
- Consumes: nothing new — reuses `weight_indices: list[int]` already computed at `oft_manager.py:710-728` (one entry per token in the forward batch; `0` means "identity/base slot, no real adapter for this token"; any other value is a real resident adapter's buffer slot index, from `self.memory_pool.get_buffer_id(uid)`).
- Produces: `OFTManager._moe_multi_tenant_slot_ids: Optional[torch.Tensor]` — `None` when at most one distinct non-zero slot is present in `weight_indices` (the fast-path case); otherwise a `torch.long` tensor of shape `(len(weight_indices),)` holding `weight_indices` verbatim as a tensor, on `forward_batch.input_ids.device`. Task 2 reads this attribute by name.

**Design note (read before implementing):** the multi-tenant decision is computed **once, globally, from the whole batch's `weight_indices`**, not per MoE layer. This means a batch with 2 resident adapters takes the multi-tenant MoE path for every MoE layer, even a layer where (unusually) only one of those two adapters actually has MoE-target weights loaded. This is a deliberate simplification: computing an exact per-layer count would need per-layer bookkeeping this plan doesn't otherwise need, and the global check is still fully correct (never wrong output), only occasionally not maximally fast for that one layer — which the spec's non-goals explicitly allow (multi-tenant path performance is not required in this iteration). The single/no-adapter fast path itself is unaffected: with ≤1 distinct real adapter resident in the whole batch, every MoE layer always takes the fast path.

- [ ] **Step 1: Write the failing test**

Create `test/registered/unit/oft/test_oft_moe_multi_tenancy.py`:

```python
"""Unit tests for OFTManager's per-batch MoE multi-tenancy decision.

No GPU required: constructs weight_indices directly and calls the method
under test via a lightweight OFTManager stand-in, following this codebase's
established pattern of binding the real production method onto a minimal
double (see test_oft_native_admission.py) rather than reimplementing the
logic.
"""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.oft.oft_manager import OFTManager


class TestMoeMultiTenancyDecision(unittest.TestCase):
    def _make_tm(self):
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    def test_single_resident_adapter_yields_none(self):
        tm = self._make_tm()
        # All tokens map to the same real slot (2) or the base slot (0).
        weight_indices = [2, 2, 0, 2, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNone(result)

    def test_no_resident_adapter_yields_none(self):
        tm = self._make_tm()
        weight_indices = [0, 0, 0]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNone(result)

    def test_two_resident_adapters_yields_tensor(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 0, 1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertTrue(
            torch.equal(result, torch.tensor([1, 2, 0, 1, 2], dtype=torch.long))
        )

    def test_three_resident_adapters_yields_tensor(self):
        tm = self._make_tm()
        weight_indices = [1, 2, 3]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=torch.device("cpu")
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v`
Expected: FAIL — `AttributeError: type object 'OFTManager' has no attribute '_compute_moe_multi_tenant_slot_ids'`

- [ ] **Step 3: Write minimal implementation**

In `python/sglang/srt/oft/oft_manager.py`, add this method to `OFTManager` (near `prepare_oft_batch`, e.g. directly above it):

```python
def _compute_moe_multi_tenant_slot_ids(
    self, weight_indices: list, device: "torch.device"
) -> "Optional[torch.Tensor]":
    """Decide whether this batch needs the multi-tenant MoE OFT path.

    ``weight_indices`` is the same per-token buffer-slot list
    ``prepare_oft_batch`` already computes for the dense path (0 = identity/
    base slot, no real adapter for that token). Returns None when at most
    one distinct non-zero slot is present in the whole batch (the common
    case: MoE forward takes today's unchanged single-slot path); otherwise
    returns a torch.long tensor of the same length holding weight_indices
    verbatim, for the multi-tenant MoE kernel to index per token.

    This check is global (whole batch, not per MoE layer) -- see this
    plan's Task 1 design note for why that is a deliberate, still-correct
    simplification.
    """
    distinct_real_slots = {idx for idx in weight_indices if idx != 0}
    if len(distinct_real_slots) <= 1:
        return None
    return torch.tensor(weight_indices, dtype=torch.long, device=device)
```

Then update `prepare_oft_batch` (`oft_manager.py:700-736`) to call it and stash the result, right after the existing `weight_indices`/`oft_block_sizes` loop and before the `self.oft_backend.prepare_oft_batch(...)` call:

```python
        self._moe_multi_tenant_slot_ids = self._compute_moe_multi_tenant_slot_ids(
            weight_indices, device=forward_batch.input_ids.device
        )
```

And initialize `self._moe_multi_tenant_slot_ids = None` in `OFTManager.__init__` alongside the other per-batch state (find the `__init__` method and add it next to similar attributes — do not guess the exact line without reading `__init__` first, since this file has changed several times this session).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/oft_manager.py test/registered/unit/oft/test_oft_moe_multi_tenancy.py
git commit -m "feat(oft): compute per-batch MoE multi-tenancy decision from existing weight_indices"
```

---

## Task 2: Push the decision onto FusedMoE modules

**Files:**
- Modify: `python/sglang/srt/oft/oft_manager.py` (`OFTManager.prepare_oft_batch`, `OFTManager._init_identity_expert_oft_for_cuda_graph`)
- Test: `test/registered/unit/oft/test_oft_moe_multi_tenancy.py` (extend from Task 1)

**Interfaces:**
- Consumes: `self._moe_multi_tenant_slot_ids` (Task 1); `self._find_fused_moe_modules()` (existing method, already used by `_init_identity_expert_oft_for_cuda_graph` — returns `Dict[layer_id, moe_module]`); `self.memory_pool._groups` (existing, `oft/base/mem_pool.py:82`, `Dict[str, Dict[key, torch.Tensor]]` where each tensor has shape `(max_adapters_per_batch, *per_key_shape)`).
- Produces: on every MoE module returned by `_find_fused_moe_modules()`, two new "live-read" attributes (read fresh by Task 3 on every forward call, exactly like the existing `moe.w13_oft_r` pattern):
  - `moe._oft_moe_multi_tenant_slot_ids: Optional[torch.Tensor]` — set every batch (mirrors `self._moe_multi_tenant_slot_ids`).
  - `moe._oft_w13_oft_r_all_slots`, `moe._oft_w1_oft_r_all_slots`, `moe._oft_w3_oft_r_all_slots`, `moe._oft_w2_oft_r_all_slots: Optional[torch.Tensor]` — set **once**, at the same time as the existing single-slot bindings in `_init_identity_expert_oft_for_cuda_graph` (these are stable buffer objects that get mutated in place, never reallocated, so binding once is correct and matches the existing pattern for `moe.w13_oft_r` etc.). Each is `self.memory_pool._groups[group_name][layer_id]` directly (the full `(max_adapters_per_batch, ...)` tensor, not a single-slot view), or `None` if that group was never registered (matching whichever of `w13_oft_r`/`w1_oft_r`/`w3_oft_r`/`w2_oft_r` is actually declared, per `_declare_expert_groups`'s existing split-vs-fused gating).

- [ ] **Step 1: Write the failing test**

Append to `test/registered/unit/oft/test_oft_moe_multi_tenancy.py`:

```python
class TestPushSlotIdsOntoMoeModules(unittest.TestCase):
    def _make_tm_with_one_moe_module(self):
        moe = SimpleNamespace()
        tm = SimpleNamespace()
        tm._moe_multi_tenant_slot_ids = None
        tm._find_fused_moe_modules = lambda: {0: moe}
        tm._push_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._push_moe_multi_tenant_slot_ids, tm
        )
        return tm, moe

    def test_pushes_none_when_no_multi_tenancy(self):
        tm, moe = self._make_tm_with_one_moe_module()
        tm._moe_multi_tenant_slot_ids = None
        tm._push_moe_multi_tenant_slot_ids()
        self.assertIsNone(moe._oft_moe_multi_tenant_slot_ids)

    def test_pushes_tensor_when_multi_tenant(self):
        tm, moe = self._make_tm_with_one_moe_module()
        slot_ids = torch.tensor([1, 2, 1], dtype=torch.long)
        tm._moe_multi_tenant_slot_ids = slot_ids
        tm._push_moe_multi_tenant_slot_ids()
        self.assertIs(moe._oft_moe_multi_tenant_slot_ids, slot_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v`
Expected: FAIL — `AttributeError: type object 'OFTManager' has no attribute '_push_moe_multi_tenant_slot_ids'`

- [ ] **Step 3: Write minimal implementation**

Add to `OFTManager` in `oft_manager.py` (near `_compute_moe_multi_tenant_slot_ids` from Task 1):

```python
def _push_moe_multi_tenant_slot_ids(self) -> None:
    """Make this batch's MoE multi-tenancy decision visible to every
    resident FusedMoE module, read live by oft_moe_runners.make_oft_invoke
    on every kernel-invoker call -- same self-gating pattern as
    moe.w13_oft_r etc."""
    for moe in self._find_fused_moe_modules().values():
        moe._oft_moe_multi_tenant_slot_ids = self._moe_multi_tenant_slot_ids
```

Call it from `prepare_oft_batch`, immediately after the `self._moe_multi_tenant_slot_ids = ...` line added in Task 1:

```python
        self._push_moe_multi_tenant_slot_ids()
```

Then, in `_init_identity_expert_oft_for_cuda_graph` (`oft_manager.py:1202` onward), add the one-time full-buffer bindings. Read the current body first (it was re-read during planning; the exact insertion points are the same places the existing single-view bindings happen — inside the `for layer_id, moe in self._find_fused_moe_modules().items():` loop, once per branch). For each of the four groups, bind the full tensor if the group was registered, else `None`:

```python
                moe._oft_w1_oft_r_all_slots = self.memory_pool._groups.get(
                    "w1_oft_r", {}
                ).get(layer_id)
                moe._oft_w3_oft_r_all_slots = self.memory_pool._groups.get(
                    "w3_oft_r", {}
                ).get(layer_id)
```

for the split (`w13_is_split`) branch, and:

```python
                moe._oft_w13_oft_r_all_slots = self.memory_pool._groups.get(
                    "w13_oft_r", {}
                ).get(layer_id)
```

for the legacy fused branch, and (unconditionally, alongside the existing `w2_oft_r` binding):

```python
            moe._oft_w2_oft_r_all_slots = self.memory_pool._groups.get(
                "w2_oft_r", {}
            ).get(layer_id)
```

Place each line directly next to its existing single-slot counterpart (`moe.w1_oft_r = self.memory_pool.active_view(...)` etc.) so a future reader sees both bindings together. Also add `self._moe_multi_tenant_slot_ids = None` to `OFTManager.__init__` if not already added in Task 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v`
Expected: PASS, all 6 tests (4 from Task 1 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/oft_manager.py test/registered/unit/oft/test_oft_moe_multi_tenancy.py
git commit -m "feat(oft): push MoE multi-tenancy slot ids and full-buffer refs onto FusedMoE modules"
```

---

## Task 3: Branch to the multi-tenant kernel invoker in oft_moe_runners.py

**Files:**
- Modify: `python/sglang/srt/oft/oft_moe_runners.py`
- Test: `test/registered/unit/oft/test_oft_moe_multi_tenancy.py` (extend)

**Interfaces:**
- Consumes: `layer._oft_moe_multi_tenant_slot_ids`, `layer._oft_w13_oft_r_all_slots` / `_oft_w1_oft_r_all_slots` / `_oft_w3_oft_r_all_slots` / `_oft_w2_oft_r_all_slots` (Task 2). `apply_oft_rotation_triton_multi_slot(A, oft_r_all_slots, slot_ids, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, block_m=64) -> torch.Tensor` (Task 4 — not yet implemented; this task's own tests mock it, per Step 1 below, so Task 3 does not depend on Task 4 being done first).
- Produces: `_oft_prerotate_multi_tenant(A, oft_r_all_slots, slot_ids, C, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, num_experts, block_size_m)` — same signature shape as the existing `_oft_prerotate`, with two extra positional args (`oft_r_all_slots`, `slot_ids`) inserted after `oft_r`'s position. Returns the same 7-tuple `_oft_prerotate` returns.

- [ ] **Step 1: Write the failing test**

Read `_oft_prerotate`'s exact current signature and return shape from `oft_moe_runners.py` before writing this test (it was read during planning; confirm line numbers are still current, since this file may have shifted). Append to `test/registered/unit/oft/test_oft_moe_multi_tenancy.py`:

```python
from unittest.mock import patch


class TestMakeOftInvokeMultiTenantBranch(unittest.TestCase):
    def test_invoke_uses_multi_tenant_path_when_slot_ids_present(self):
        from sglang.srt.oft import oft_moe_runners

        layer = SimpleNamespace(
            w13_weight=object(),
            w2_weight=object(),
            w13_oft_r=torch.zeros(2, 1, 4, 4),
            w1_oft_r=None,
            w3_oft_r=None,
            w2_oft_r=None,
            _oft_moe_multi_tenant_slot_ids=torch.tensor([1, 2], dtype=torch.long),
            _oft_w13_oft_r_all_slots=torch.zeros(3, 2, 1, 4, 4),
        )
        real_invoke = unittest.mock.Mock()
        invoke = oft_moe_runners.make_oft_invoke(layer, real_invoke)

        A = torch.zeros(2, 8)
        with patch.object(
            oft_moe_runners, "_oft_prerotate_multi_tenant"
        ) as mock_multi, patch.object(oft_moe_runners, "_oft_prerotate") as mock_single:
            mock_multi.return_value = (A, A, None, None, None, None, None)
            invoke(
                A, layer.w13_weight, None, A, None, None, None,
                None, None, None, None, None, None, 1,
                {"BLOCK_SIZE_M": 32},
            )
            mock_multi.assert_called_once()
            mock_single.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v`
Expected: FAIL — either an `AttributeError` on `_oft_prerotate_multi_tenant` not existing on the module, or `mock_multi.assert_called_once()` failing because today's code always calls `_oft_prerotate` regardless of the new attributes (since `make_oft_invoke` doesn't look for them yet).

- [ ] **Step 3: Write minimal implementation**

In `oft_moe_runners.py`, add the new function (placed directly after `_oft_prerotate`'s existing definition — read that function's exact current body first so this mirrors its structure and its `_QUANT_KWARGS`/fp8-dequant handling exactly; do not diverge on those parts):

```python
def _oft_prerotate_multi_tenant(
    A,
    oft_r_all_slots,
    slot_ids,
    C,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    top_k,
    num_experts,
    block_size_m,
):
    """Multi-tenant sibling of _oft_prerotate: selects each token's own
    adapter's R-block (via slot_ids) instead of applying one shared R to
    the whole batch. Only exercised when >=2 distinct adapters carrying
    MoE OFT weights are resident in the current forward batch (see
    OFTManager._compute_moe_multi_tenant_slot_ids); the single-adapter
    fast path (_oft_prerotate) is completely unaffected by this function's
    existence.
    """
    from sglang.srt.oft.triton_ops import apply_oft_rotation_triton_multi_slot

    A_rot = apply_oft_rotation_triton_multi_slot(
        A,
        oft_r_all_slots,
        slot_ids,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_m=block_size_m,
    )
    C = C.reshape(-1, 1, C.shape[-1])
    topk_weights = topk_weights.reshape(-1, 1)
    topk_ids = topk_ids.reshape(-1, 1)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, num_experts
    )
    return (
        A_rot,
        C,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
```

(This mirrors `_oft_prerotate`'s existing body verbatim except for the multi-slot kernel call — read `_oft_prerotate`'s current body and use the exact same `moe_align_block_size` import/call shape; do not invent a different reshape convention.)

Then modify `make_oft_invoke`'s `invoke` closure. Find the two call sites of `_oft_prerotate` — one inside `_run_gate_up_split` (for the split w1/w3 case) and one inline in `invoke` itself (for the fused w13 / w2 case) — and branch on `getattr(layer, "_oft_moe_multi_tenant_slot_ids", None)` at each. For the inline case (fused w13 / w2), replace:

```python
        a, c, tw, ti, sti, ei, ntpp = _oft_prerotate(
            A,
            oft_r,
            C,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            B.shape[0],
            config["BLOCK_SIZE_M"],
        )
```

with:

```python
        slot_ids = getattr(layer, "_oft_moe_multi_tenant_slot_ids", None)
        if slot_ids is not None:
            oft_r_all_slots = (
                layer._oft_w13_oft_r_all_slots
                if is_gate_up
                else layer._oft_w2_oft_r_all_slots
            )
            a, c, tw, ti, sti, ei, ntpp = _oft_prerotate_multi_tenant(
                A,
                oft_r_all_slots,
                slot_ids,
                C,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                B.shape[0],
                config["BLOCK_SIZE_M"],
            )
        else:
            a, c, tw, ti, sti, ei, ntpp = _oft_prerotate(
                A,
                oft_r,
                C,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                B.shape[0],
                config["BLOCK_SIZE_M"],
            )
```

Apply the equivalent branch inside `_run_gate_up_split`'s per-half loop (using `layer._oft_w1_oft_r_all_slots` / `layer._oft_w3_oft_r_all_slots` matched to `oft_r is w1_oft_r` vs `w3_oft_r`), reusing `layer._oft_moe_multi_tenant_slot_ids` the same way. Read `_run_gate_up_split`'s current body first (this plan's earlier investigation read it in full; confirm nothing shifted) before editing it, since it loops over `(half_slice, oft_r)` pairs and the branch needs to pick the matching `_all_slots` attribute per iteration.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v`
Expected: PASS, all tests including the new one. Also re-run the full suite to catch any regression in the fast path's own tests: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit -k "oft or peft" -v` (repo has a stale unrelated editable sglang install at `/sgl-workspace/sglang` that shadows this repo unless `PYTHONPATH=python` is set).

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/oft_moe_runners.py test/registered/unit/oft/test_oft_moe_multi_tenancy.py
git commit -m "feat(oft): branch MoE kernel invoker to the multi-tenant path when 2+ adapters resident"
```

---

## Task 4: Multi-slot Triton rotation kernel

**Files:**
- Modify: `python/sglang/srt/oft/triton_ops/block_rotate.py`
- Test: `test/registered/kernels/ops/test_oft_moe_rotation_multi_slot.py` (new, GPU required)

**Interfaces:**
- Consumes: nothing new from earlier tasks (this is the leaf kernel; Task 3 already calls it by the name defined here).
- Produces: `apply_oft_rotation_triton_multi_slot(A: torch.Tensor, oft_r_all_slots: torch.Tensor, slot_ids: torch.Tensor, topk_ids: torch.Tensor, sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor, top_k: int, block_m: int = 64) -> torch.Tensor` — same return contract as the existing `apply_oft_rotation_triton`: `(M * top_k, K)`, one rotated row per token-expert pair. `oft_r_all_slots` shape: `(S, E, num_blocks, bs, bs)` (S = `max_adapters_per_batch`, the pool's slot dimension). `slot_ids` shape: `(M,)`, `torch.long`, one entry per **original** token (same indexing convention as `weight_indices`/`topk_ids`'s first dimension before top-k expansion).

**Algorithm (implement exactly this — it generalizes the existing kernel's `OFT_BLOCK_SIZE < 16` elementwise fallback branch to a per-row-selectable R, since a shared `tl.dot` matmul across a `BLOCK_M` tile cannot use a different R for different rows; this trades the `tl.dot`/tensor-core path for elementwise accumulation on the multi-tenant path only — acceptable per this plan's constraint that only the single-adapter fast path's performance must not regress):**

For each of the `BLOCK_M` rows in a block (all sharing the same `expert`, per `moe_align_block_size`, but each row potentially a *different* adapter slot):
1. Recover `orig_ids = sorted_ids // top_k` exactly as today.
2. Gather each row's own slot: `slot = tl.load(slot_ids_ptr + orig_ids, mask=token_mask, other=0)`.
3. For each `k` in `range(OFT_BLOCK_SIZE)` (same loop structure as the existing fallback branch): load `a_col` per row exactly as today; load `r_row` **per row** now — `R_ptr + slot[:, None] * stride_rs + expert * stride_re + pid_blk * stride_rb + k * stride_ri + out_cols[None, :] * stride_rj`, i.e. a `(BLOCK_M, OFT_BLOCK_SIZE)` gather instead of the existing fallback's single shared `(OFT_BLOCK_SIZE,)` row; accumulate `rot_accum += a_col[:, None] * r_row` (same elementwise-broadcast accumulation as today, just with `r_row` now 2D).
4. Store exactly as today.

- [ ] **Step 1: Write the failing test**

Read the existing kernel test conventions first: search `test/registered/kernels/ops/` for any existing OFT rotation kernel test (if one exists, follow its exact GPU-test harness/decorator conventions; if none exists, follow the nearest MoE kernel test's conventions in that same directory, e.g. `test_moe_topk_softmax.py`). Create `test/registered/kernels/ops/test_oft_moe_rotation_multi_slot.py`:

```python
"""GPU correctness oracle for the multi-slot MoE OFT rotation kernel.

Proves per-token slot selection is correct by comparing the multi-slot
kernel's output, split by each token's own slot, against running the
EXISTING single-slot kernel with only that one slot's R matrix -- the
strongest available correctness oracle, since it reuses already-trusted
kernel code as the ground truth rather than a hand-rolled reference.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.oft.triton_ops import (
    apply_oft_rotation_triton,
    apply_oft_rotation_triton_multi_slot,
)


class TestMultiSlotRotationMatchesSingleSlotPerAdapter(unittest.TestCase):
    def test_two_adapters_each_match_isolated_single_slot_run(self):
        if not torch.cuda.is_available():
            self.skipTest("requires GPU")
        torch.manual_seed(0)
        device = "cuda"
        num_tokens, hidden, num_experts, top_k, bs = 8, 32, 4, 1, 16
        num_blocks = hidden // bs
        num_slots = 3

        A = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, 64, num_experts
        )

        oft_r_all_slots = torch.randn(
            num_slots, num_experts, num_blocks, bs, bs, device=device, dtype=torch.bfloat16
        )
        # Half the tokens use slot 1, half use slot 2 (no slot-0/identity tokens
        # in this test -- identity fallback is covered by the wiring tests).
        slot_ids = torch.tensor(
            [1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long, device=device
        )

        multi_out = apply_oft_rotation_triton_multi_slot(
            A, oft_r_all_slots, slot_ids, topk_ids, sorted_token_ids,
            expert_ids, num_tokens_post_padded, top_k,
        )

        for slot in (1, 2):
            single_out = apply_oft_rotation_triton(
                A, oft_r_all_slots[slot], topk_ids, sorted_token_ids,
                expert_ids, num_tokens_post_padded, top_k,
            )
            token_mask = slot_ids == slot
            # top_k=1 here, so row i of the *_expanded output corresponds
            # directly to original token i.
            torch.testing.assert_close(
                multi_out[token_mask], single_out[token_mask], atol=1e-2, rtol=1e-2
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/kernels/ops/test_oft_moe_rotation_multi_slot.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_oft_rotation_triton_multi_slot'`

- [ ] **Step 3: Write minimal implementation**

In `python/sglang/srt/oft/triton_ops/block_rotate.py`, add the new kernel and entry point below the existing ones (leave `_oft_block_rotate_kernel` and `apply_oft_rotation_triton` completely untouched):

```python
@triton.jit
def _oft_block_rotate_kernel_multi_slot(
    # Input: (M, K)
    A_ptr,
    stride_am,
    stride_ak,
    # Output: (M_expanded, K)
    A_rot_ptr,
    stride_arm,
    stride_ark,
    # R matrices: (S, E, num_blocks, bs, bs) -- one extra leading slot axis
    # versus the single-slot kernel.
    R_ptr,
    stride_rs,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    # Per-original-token slot assignment: (M,)
    slot_ids_ptr,
    # Token routing
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    # Dimensions
    top_k: tl.constexpr,
    K: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Per-token-slot-selectable sibling of _oft_block_rotate_kernel.

    Each BLOCK_M-sized group of sorted tokens shares one expert (per
    moe_align_block_size) but may span MULTIPLE adapter slots -- so, unlike
    the single-slot kernel, this cannot batch the rotation as one tl.dot
    matmul per block (that requires one shared R for the whole tile).
    Instead it gathers each row's own slot's R row per k and accumulates
    elementwise -- the same algorithm as the single-slot kernel's own
    OFT_BLOCK_SIZE < 16 fallback branch, generalized to a per-row R.
    """
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(sorted_token_ids_ptr + offs_m)
    sorted_ids = sorted_ids.to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert < 0:
        return

    orig_ids = sorted_ids // top_k
    slot = tl.load(slot_ids_ptr + orig_ids, mask=token_mask, other=0).to(tl.int64)

    k_base = pid_blk * OFT_BLOCK_SIZE
    out_cols = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
    rot_accum = tl.zeros((BLOCK_M, OFT_BLOCK_SIZE), dtype=tl.float32)

    for k in range(OFT_BLOCK_SIZE):
        a_col = tl.load(
            A_ptr + orig_ids * stride_am + (k_base + k) * stride_ak,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        r_row = tl.load(
            R_ptr
            + slot[:, None] * stride_rs
            + expert * stride_re
            + pid_blk * stride_rb
            + k * stride_ri
            + out_cols[None, :] * stride_rj,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        rot_accum += a_col[:, None] * r_row

    out_k_offs = (k_base + tl.arange(0, OFT_BLOCK_SIZE)).to(tl.int64)
    out_ptrs = A_rot_ptr + sorted_ids[:, None] * stride_arm + out_k_offs[None, :] * stride_ark
    tl.store(out_ptrs, rot_accum.to(A_rot_ptr.dtype.element_ty), mask=token_mask[:, None])


def apply_oft_rotation_triton_multi_slot(
    A: torch.Tensor,               # (M, K)
    oft_r_all_slots: torch.Tensor, # (S, E, num_blocks, bs, bs)
    slot_ids: torch.Tensor,        # (M,) long, one entry per original token
    topk_ids: torch.Tensor,        # (M, top_k)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int = 64,
) -> torch.Tensor:
    """Multi-slot sibling of apply_oft_rotation_triton: selects each
    token's own adapter's R-block via slot_ids instead of applying one
    shared R to every token. See _oft_block_rotate_kernel_multi_slot's
    docstring for why this can't reuse the single-slot kernel's tl.dot path.
    """
    M, K = A.shape
    if oft_r_all_slots.dim() != 5:
        raise ValueError(
            "oft_r_all_slots must be 5D (slots, experts, blocks, bs, bs), "
            f"got {tuple(oft_r_all_slots.shape)}"
        )
    bs = oft_r_all_slots.shape[-1]
    from sglang.srt.oft.utils import validate_oft_block_size

    validate_oft_block_size(bs)
    if tuple(oft_r_all_slots.shape[-2:]) != (bs, bs):
        raise ValueError(
            f"OFT blocks must be square, got {tuple(oft_r_all_slots.shape[-2:])}"
        )
    if K % bs != 0:
        raise ValueError(f"OFT hidden size {K} must be divisible by block size {bs}")
    num_blocks = K // bs
    if oft_r_all_slots.shape[2] != num_blocks:
        raise ValueError(
            f"oft_r_all_slots has {oft_r_all_slots.shape[2]} blocks, "
            f"expected {num_blocks} for K={K}, BS={bs}"
        )
    if slot_ids.shape[0] != M:
        raise ValueError(
            f"slot_ids has {slot_ids.shape[0]} entries, expected {M} (one per token)"
        )

    A_rot = torch.empty(M * top_k, K, device=A.device, dtype=A.dtype)

    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, block_m), num_blocks)

    _oft_block_rotate_kernel_multi_slot[grid](
        A, A.stride(0), A.stride(1),
        A_rot, A_rot.stride(0), A_rot.stride(1),
        oft_r_all_slots,
        oft_r_all_slots.stride(0), oft_r_all_slots.stride(1),
        oft_r_all_slots.stride(2), oft_r_all_slots.stride(3), oft_r_all_slots.stride(4),
        slot_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_ids.numel(),
        top_k=top_k,
        K=K,
        OFT_BLOCK_SIZE=bs,
        BLOCK_M=block_m,
    )

    return A_rot
```

Then export it from `python/sglang/srt/oft/triton_ops/__init__.py` alongside the existing `apply_oft_rotation_triton` export (read that file's current export list first and add the new name in the same style/location).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/kernels/ops/test_oft_moe_rotation_multi_slot.py -v`
Expected: PASS (requires GPU).

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/triton_ops/block_rotate.py python/sglang/srt/oft/triton_ops/__init__.py test/registered/kernels/ops/test_oft_moe_rotation_multi_slot.py
git commit -m "feat(oft): add per-token-slot-selectable MoE OFT rotation kernel"
```

---

## Task 5: End-to-end wiring verification and fast-path regression guard

**Files:**
- Test: `test/registered/rl/test_oft_moe_multi_tenant_e2e.py` (new, GPU required)

**Interfaces:**
- Consumes: everything from Tasks 1-4 — no new production code in this task, verification only.

- [ ] **Step 1: Write the failing test**

Find an existing GPU test that loads a small MoE model with expert-target OFT adapters via the native RPC (search `test/registered/rl/test_oft_load_from_tensor.py` for its model/server-launch fixtures and the pattern used to load an OFT adapter whose config targets `gate_up_proj`/`down_proj` on a MoE model — reuse that exact fixture/launch style, do not invent a new one). Create `test/registered/rl/test_oft_moe_multi_tenant_e2e.py` following that file's launch conventions, with two test methods:

```python
"""End-to-end GPU test: MoE expert-OFT multi-tenancy (this plan's Task 5).

1. test_fast_path_unchanged_with_one_moe_adapter: regression guard -- a
   single MoE-target OFT adapter resident must produce identical output to
   the pre-existing (fast) code path.
2. test_two_moe_adapters_apply_correctly: the actual fix -- two
   concurrently-resident MoE-target OFT adapters, each must produce output
   matching what that adapter alone would produce, not a shared/clobbered
   result (the original bug this plan fixes).

Follows test_oft_load_from_tensor.py's server-launch and adapter-loading
conventions (small Qwen3 MoE base model, native load_oft_adapter_from_tensors
RPC, --peft-method oft --oft-impl sibling, target_modules including
gate_up_proj/down_proj so the adapter carries real expert weights).
"""

# Implementer: port this file's server launch, adapter-tensor construction,
# and generate() helpers directly from test_oft_load_from_tensor.py's own
# fixtures (read that file's setUpClass/tearDownClass and its
# _load_oft_adapter_from_tensors helper before writing this test body) --
# do not reinvent them. The two test bodies below are the new logic this
# task actually adds.

import unittest


class TestMoeMultiTenantEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Port test_oft_load_from_tensor.py's setUpClass: launch a server
        # for a small Qwen3-MoE (or equivalent already-used-in-this-repo MoE
        # base model) with --enable-oft --peft-method oft --oft-impl sibling
        # --max-ofts-per-batch >= 3 --max-oft-block-size <matching the test
        # adapters below>.
        raise NotImplementedError("port from test_oft_load_from_tensor.py")

    @classmethod
    def tearDownClass(cls):
        raise NotImplementedError("port from test_oft_load_from_tensor.py")

    def test_fast_path_unchanged_with_one_moe_adapter(self):
        """Load one OFT adapter targeting gate_up_proj/down_proj (real,
        non-identity expert rotation weights). Generate with it, and assert
        the output matches a known-good baseline captured before this
        plan's changes landed (or, if no stored baseline exists yet,
        capture one now on the pre-Task-1 commit and commit it alongside
        this test) -- proving the single-adapter path is unaffected.
        """
        raise NotImplementedError

    def test_two_moe_adapters_apply_correctly(self):
        """Load two OFT adapters, both targeting gate_up_proj/down_proj with
        DIFFERENT (not identity, not equal to each other) expert rotation
        weights. In one batch, generate from each adapter (e.g. two
        concurrent requests, one per adapter, or --parallel style
        interleaving matching how test_upsert_refresh or similar existing
        multi-adapter tests in test_oft_load_from_tensor.py drive concurrent
        requests). Assert each request's output matches what THAT adapter
        alone would produce (load only that one adapter, generate the same
        prompt, compare) -- proving neither adapter's rotation was silently
        dropped or the other's clobbered.
        """
        raise NotImplementedError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/rl/test_oft_moe_multi_tenant_e2e.py -v`
Expected: FAIL with `NotImplementedError` on all methods (this is the scaffold; the implementer fills in the two test bodies and the fixture methods by porting `test_oft_load_from_tensor.py`'s real helpers, then reruns).

- [ ] **Step 3: Write minimal implementation**

Port `test_oft_load_from_tensor.py`'s `setUpClass`/`tearDownClass` and its tensor-adapter-construction helper into this file's `setUpClass`/`tearDownClass`, adjusted to use a MoE base model and adapter `target_modules` that include `gate_up_proj`/`down_proj` (read that file's exact helper signatures first; do not guess argument names). Implement `test_fast_path_unchanged_with_one_moe_adapter` by loading one such adapter, generating a fixed prompt, and asserting the output text is deterministic and non-trivial (differs from a bare-base-model generation on the same prompt, proving the rotation actually applied) — this is the fast-path regression guard; if this plan's earlier tasks broke the single-adapter path, this test catches it even without a pre-recorded baseline, since a real, working rotation should differ from the unrotated base model.

Implement `test_two_moe_adapters_apply_correctly` by: (a) loading adapter A alone, generating prompt P, recording output A_alone; (b) unloading, loading adapter B alone, generating the same prompt P, recording output B_alone (sanity: assert `A_alone != B_alone`, otherwise the test adapters aren't actually different and prove nothing); (c) loading both A and B concurrently resident, issuing two requests in the same batch — one naming A, one naming B, on the same prompt P — and asserting the A-named request's output equals `A_alone` and the B-named request's output equals `B_alone`. This is the actual multi-tenancy proof: before this plan, both concurrent requests would silently receive whichever adapter's weights happened to be live on the shared buffer last, so this assertion would fail on the pre-fix code and pass after Tasks 1-4 land.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/rl/test_oft_moe_multi_tenant_e2e.py -v`
Expected: PASS (requires GPU; requires Tasks 1-4 complete).

- [ ] **Step 5: Commit**

```bash
git add test/registered/rl/test_oft_moe_multi_tenant_e2e.py
git commit -m "test(oft): end-to-end proof of MoE expert-OFT multi-tenancy and fast-path regression guard"
```

---

## Self-Review

**Spec coverage:**
- "Per-batch weight_indices-equivalent for MoE tokens, reusing the same uid→buffer_id mapping" → Task 1 (reuses `weight_indices` from `oft_manager.py:710-728` directly, no new bookkeeping).
- "Fast path unchanged, multi-tenant path new" → Task 3's branch (`if slot_ids is not None`) + Task 5's `test_fast_path_unchanged_with_one_moe_adapter`.
- "No new capacity knob, reuse --max-ofts-per-batch" → confirmed in Global Constraints and Task 2's interface (`max_adapters_per_batch` is the existing pool dimension, verified from `oft/base/mem_pool.py:80-87` during planning).
- "canonical_oft priority, legacy oft must not regress single-adapter behavior" → Global Constraints; Task 3's branch touches both `w13_oft_r` (legacy) and `w1_oft_r`/`w3_oft_r` (split) call sites identically, and Task 5's fast-path test covers whichever layout the test model's adapters use.
- "Fallback to identity for tokens whose adapter has no MoE weights" → covered implicitly: `weight_indices[i] = 0` already means "identity/base slot" for such tokens in Task 1's reused list, and slot 0 of any registered buffer group is always identity-initialized by `_declare_expert_groups`'s existing identity-fill code — no new code needed, but this should be called out explicitly to a future reader, which Task 1's design note does.
- "New kernel variant selecting per-expert R-blocks using slot index" → Task 4.
- Testing plan (unit tests for decision logic, GPU test with 2 adapters producing correct per-adapter output, fast-path regression test) → Tasks 1/2 (unit), Task 4 (kernel oracle), Task 5 (e2e).

**Placeholder scan:** Task 5's test bodies are intentionally scaffolded with `NotImplementedError` in Step 1 (the failing-test step, which is supposed to fail) and then given concrete port-and-assert instructions in Step 3 — this is the plan's required TDD shape (write failing test, then implement), not an unfilled placeholder; the *instructions* for what Step 3 must produce are concrete and unambiguous (which file to port from, what to assert, why). No other step contains TBD/TODO/"handle appropriately" language.

**Type/signature consistency across tasks:**
- `_compute_moe_multi_tenant_slot_ids` (Task 1) returns `Optional[torch.Tensor]` (long dtype) — `_push_moe_multi_tenant_slot_ids` (Task 2) consumes exactly that via `self._moe_multi_tenant_slot_ids`, no transformation. Consistent.
- Task 2 produces `moe._oft_moe_multi_tenant_slot_ids` and four `_oft_w{13,1,3,2}_oft_r_all_slots` attributes — Task 3's `invoke` closure reads exactly those five names, no others. Consistent.
- `_oft_prerotate_multi_tenant`'s signature (Task 3) matches `_oft_prerotate`'s existing signature with `oft_r_all_slots, slot_ids` inserted after the `oft_r`-equivalent position, and both call sites in Task 3's Step 3 pass arguments in that same order. Consistent.
- `apply_oft_rotation_triton_multi_slot`'s signature (Task 4) matches exactly what `_oft_prerotate_multi_tenant` (Task 3) calls it with (`A, oft_r_all_slots, slot_ids, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, block_m=block_size_m`). Consistent.

No gaps found; no fixes needed beyond what's already reflected above.

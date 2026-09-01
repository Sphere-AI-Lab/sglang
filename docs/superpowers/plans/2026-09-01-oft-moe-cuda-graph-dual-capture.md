# OFT MoE CUDA-Graph Dual-Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CUDA-graph-replayed decode correctly handle 2+ concurrently-resident MoE-target OFT adapters, by porting this codebase's existing LoRA/DSA dual-capture pattern (`decode_cuda_graph_runner.py`) to add OFT as a fourth capture-variant axis, plus a persistent per-token routing buffer mirroring LoRA's own `moe_cg_buffers`.

**Architecture:** A process-global `_capture_oft_variant` flag (mirroring `_capture_lora_variant`/`_capture_dsa_variant` in `capture_mode.py`) lets `OFTManager._compute_moe_multi_tenant_slot_ids` force the multi-tenant code path during a dedicated capture pass, regardless of the capture-time dummy batch's own adapter count. `DecodeCudaGraphRunner` gains an `oft_variants` capture axis (Cartesian-multiplied with the existing `lora_variants`/`dsa_variants` axes, opt-in only when the server actually needs it) and a live `_resolve_oft_variant` used at replay-eligibility time to pick the matching captured graph. A new persistent buffer on `OFTManager`, allocated once in `init_cuda_graph_batch_info` and refreshed in place every real batch, replaces the current per-call fresh-tensor allocation that broke CUDA-graph pointer stability.

**Tech Stack:** Python (PyTorch), CUDA graphs. sglang fork (`Sphere-AI-Lab/sglang`), spanning `python/sglang/srt/model_executor/runner_utils/`, `python/sglang/srt/model_executor/runner/`, `python/sglang/srt/oft/`.

**Spec:** `docs/superpowers/specs/2026-09-01-oft-moe-cuda-graph-dual-capture-design.md`

## Global Constraints

- Zero additional capture time, GPU memory, or replay overhead for servers not using MoE-target OFT with 2+ potential resident adapters — mirrors `record_nolora_graph`/`dsa_dual_graph`'s existing opt-in discipline exactly. No new CLI flag; gate on existing OFT server-args (MoE targeting + effective capacity > 1).
- No change to LoRA's or DSA's existing variant axes or their capture/replay behavior.
- The legacy fused `oft_type="oft"` layout only needs to be *safe* (no crash) under this feature, not correct — matches the base MoE-multi-tenancy plan's own non-goal.
- `--enable-dp-attention` is explicitly OUT of scope for this first cut (see Task 4) — guarded with a clear assertion/log, not silently wrong. This resolves the spec's open item on DP-attention.
- The persistent buffer lives on `OFTManager` (not `OFTMemoryPool`) — this resolves the spec's other open item, matching where the comparable existing CUDA-graph-static config (`max_bs_in_cuda_graph`, set via `OFTManager.init_cuda_graph_batch_info`) already lives.

---

## File Structure

- **`python/sglang/srt/model_executor/runner_utils/capture_mode.py`** (existing, modified): new `_capture_oft_variant` / `get_capture_oft_variant()` / `_set_capture_oft_variant()` triplet, mirroring the existing `_capture_lora_variant` pattern exactly.
- **`python/sglang/srt/model_executor/runner/shape_key.py`** (existing, modified): `ShapeKey` gains an `oft_variant: Optional[str] = None` field.
- **`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`** (existing, modified): `_resolve_oft_variant`, `_make_graph_key`'s new parameter, the `oft_variants` capture-loop axis, `capture_one_shape`'s new parameter, and the 3 replay/eligibility call sites that build a `graph_key`.
- **`python/sglang/srt/oft/oft_manager.py`** (existing, modified): new persistent buffer state in `__init__`/`init_cuda_graph_batch_info`, and `_compute_moe_multi_tenant_slot_ids`'s three behavior additions (capture-time forcing, in-place buffer write, DP-attention guard); `prepare_oft_batch`'s call site passes `use_cuda_graph`; the stale "KNOWN LIMITATION" docstring is corrected.
- **`test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py`** (new): unit tests for the flag pair, `_resolve_oft_variant`, and the persistent-buffer/capture-forcing logic — no GPU required.
- **`test/registered/rl/test_oft_moe_multi_tenant_cuda_graph_e2e.py`** (new, GPU required): real end-to-end proof (2 adapters, CUDA graphs enabled, concurrent requests, correct per-adapter output) plus a single-adapter regression guard.

---

## Task 1: `_capture_oft_variant` flag pair and `ShapeKey.oft_variant`

**Files:**
- Modify: `python/sglang/srt/model_executor/runner_utils/capture_mode.py`
- Modify: `python/sglang/srt/model_executor/runner/shape_key.py`
- Test: `test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `get_capture_oft_variant() -> Optional[str]`, `_set_capture_oft_variant(variant: Optional[str]) -> None` (module-level functions in `capture_mode.py`, read/set the module-global `_capture_oft_variant`). `ShapeKey.oft_variant: Optional[str] = None` (new dataclass field, default `None`, so every existing `ShapeKey(...)` construction anywhere in the codebase that doesn't pass it keeps working unchanged).

- [ ] **Step 1: Write the failing test**

Create `test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py`:

```python
"""Unit tests for the OFT MoE CUDA-graph dual-capture flag pair and ShapeKey field.

No GPU required.
"""

import unittest

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_oft_variant,
    get_capture_oft_variant,
)


class TestCaptureOftVariantFlag(unittest.TestCase):
    def tearDown(self):
        # Module-global state -- reset so tests don't leak into each other.
        _set_capture_oft_variant(None)

    def test_defaults_to_none(self):
        self.assertIsNone(get_capture_oft_variant())

    def test_set_and_get_round_trip(self):
        _set_capture_oft_variant("oft_multi")
        self.assertEqual(get_capture_oft_variant(), "oft_multi")
        _set_capture_oft_variant("oft_single")
        self.assertEqual(get_capture_oft_variant(), "oft_single")
        _set_capture_oft_variant(None)
        self.assertIsNone(get_capture_oft_variant())


class TestShapeKeyOftVariantField(unittest.TestCase):
    def test_default_none_and_backward_compatible_construction(self):
        # Existing call sites across the codebase construct ShapeKey without
        # oft_variant -- this must keep working unchanged.
        key = ShapeKey(size=4, stream_idx=None, variant_label="lora", dsa_variant="dense")
        self.assertIsNone(key.oft_variant)

    def test_oft_variant_is_part_of_equality_and_hash(self):
        # ShapeKey is a frozen dataclass used as a dict key elsewhere in the
        # runner -- two keys differing only in oft_variant must compare
        # unequal and hash differently, or captured graphs would collide.
        a = ShapeKey(size=4, oft_variant="oft_single")
        b = ShapeKey(size=4, oft_variant="oft_multi")
        self.assertNotEqual(a, b)
        self.assertNotEqual(hash(a), hash(b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py -v`
Expected: FAIL — `ImportError: cannot import name '_set_capture_oft_variant'` (and `ShapeKey(...)` calls will fail with `TypeError: __init__() got an unexpected keyword argument 'oft_variant'` once that import is stubbed out, but the import failure is what you'll see first).

- [ ] **Step 3: Write minimal implementation**

In `python/sglang/srt/model_executor/runner_utils/capture_mode.py`, add directly after the existing `_capture_dsa_variant` declaration (~line 43):

```python
# When capturing dual OFT MoE-expert graphs (single-slot / multi-slot), tracks
# which variant is being captured. Read by OFTManager's capture-time branch to
# force the multi-slot routing path regardless of the capture-time dummy
# batch's own adapter count.
# None = not dual-capturing; the single-slot path is captured (today's
# behavior, and the only behavior when a server has no MoE-target OFT
# adapters needing more than one resident slot).
_capture_oft_variant: Optional[str] = None
```

and after the existing `_set_capture_dsa_variant` function (~line 81):

```python
def get_capture_oft_variant() -> Optional[str]:
    """Return the OFT MoE-expert variant being captured ("oft_single"/
    "oft_multi"), or None when dual-variant capture is not active."""
    return _capture_oft_variant


def _set_capture_oft_variant(variant: Optional[str]) -> None:
    global _capture_oft_variant
    _capture_oft_variant = variant
```

In `python/sglang/srt/model_executor/runner/shape_key.py`, add the field to `ShapeKey` (after `dsa_variant`) and extend its docstring:

```python
    dsa_variant: DSA decode dual-graph variant ("dense" / "sparse"), or None
        when DSA dual-graph capture is not enabled. Composes with variant_label
        so LoRA and DSA variants can be captured independently.
    oft_variant: OFT MoE-expert dual-graph variant ("oft_single" / "oft_multi"),
        or None when OFT dual-graph capture is not enabled. Composes
        independently with variant_label and dsa_variant.
    """

    size: int
    stream_idx: Optional[int] = None
    variant_label: Optional[str] = None
    dsa_variant: Optional[str] = None
    oft_variant: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/model_executor/runner_utils/capture_mode.py python/sglang/srt/model_executor/runner/shape_key.py test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py
git commit -m "feat(oft): add capture_oft_variant flag and ShapeKey.oft_variant field"
```

---

## Task 2: `_resolve_oft_variant` and replay-side graph-key threading

**Files:**
- Modify: `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
- Test: `test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py` (extend)

**Interfaces:**
- Consumes: `ShapeKey.oft_variant` (Task 1). `forward_batch.adapter_ids: Optional[List[str]]` (existing field, `[req.adapter_id for req in batch.reqs]` — one entry per request).
- Produces: `DecodeCudaGraphRunner._resolve_oft_variant(forward_batch) -> Optional[str]`, used by this task's own 3 call sites. Later tasks (Task 3) don't need anything new from this task beyond the fact that `_make_graph_key` now accepts `oft_variant`.

**Design note:** `_resolve_oft_variant` deliberately duplicates the tiny "how many distinct real adapters" check `OFTManager._compute_moe_multi_tenant_slot_ids` already has, rather than reaching into `OFTManager` — this mirrors `_resolve_lora_variant`, which reads `forward_batch.lora_ids` directly without reaching into `LoRAManager`. Keep the two checks logically identical (same criterion: >1 distinct non-`None`/non-zero adapter means "multi"), but they are separate small functions in separate modules by design, matching the established pattern.

This task does NOT make dual-capture actually happen yet (that's Task 3) — after this task, `_resolve_oft_variant` always returns `None` in practice, because nothing has enabled dual-capture or captured an `"oft_multi"`-keyed graph yet. This task is independently testable (the resolve logic and graph-key threading are correct in isolation) without needing Task 3 to exist first.

- [ ] **Step 1: Write the failing test**

Read `_resolve_lora_variant`'s exact current body in `decode_cuda_graph_runner.py` before writing this test (it was read during planning; confirm nothing shifted). Append to `test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class TestResolveOftVariant(unittest.TestCase):
    def _make_runner(self, record_oft_variant_graph):
        runner = MagicMock(spec=DecodeCudaGraphRunner)
        runner.record_oft_variant_graph = record_oft_variant_graph
        runner._resolve_oft_variant = DecodeCudaGraphRunner._resolve_oft_variant.__get__(
            runner
        )
        return runner

    def test_returns_none_when_dual_capture_not_enabled(self):
        runner = self._make_runner(record_oft_variant_graph=False)
        forward_batch = SimpleNamespace(adapter_ids=["a", "b", None])
        self.assertIsNone(runner._resolve_oft_variant(forward_batch))

    def test_returns_oft_single_for_at_most_one_distinct_adapter(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=["a", "a", None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_single")

    def test_returns_oft_single_when_no_adapters_at_all(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=[None, None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_single")

    def test_returns_oft_multi_for_two_or_more_distinct_adapters(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=["a", "b", None])
        self.assertEqual(runner._resolve_oft_variant(forward_batch), "oft_multi")

    def test_returns_none_when_adapter_ids_is_none(self):
        runner = self._make_runner(record_oft_variant_graph=True)
        forward_batch = SimpleNamespace(adapter_ids=None)
        self.assertIsNone(runner._resolve_oft_variant(forward_batch))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py -v`
Expected: FAIL — `AttributeError: type object 'DecodeCudaGraphRunner' has no attribute '_resolve_oft_variant'`.

- [ ] **Step 3: Write minimal implementation**

In `decode_cuda_graph_runner.py`, add directly after `_resolve_lora_variant` (~line 593):

```python
    def _resolve_oft_variant(self, forward_batch: ForwardBatch) -> Optional[str]:
        """Host dispatch: pick which pre-captured OFT MoE-expert decode graph
        to replay, from the batch's real per-request adapter identity.

        Mirrors _resolve_lora_variant's shape deliberately: reads
        forward_batch.adapter_ids directly rather than reaching into
        OFTManager, matching how _resolve_lora_variant reads
        forward_batch.lora_ids directly rather than reaching into
        LoRAManager. Returns None when OFT dual-graph capture is not enabled
        for this server (the common case: zero extra cost)."""
        if not getattr(self, "record_oft_variant_graph", False):
            return None
        if forward_batch.adapter_ids is None:
            return None
        distinct_real = {uid for uid in forward_batch.adapter_ids if uid is not None}
        return "oft_multi" if len(distinct_real) > 1 else "oft_single"
```

Update `_make_graph_key` (~line 553) to accept and thread the new parameter:

```python
    def _make_graph_key(
        self, size, stream_idx=None, variant_label=None, dsa_variant=None, oft_variant=None
    ):
        return ShapeKey(
            size=size,
            stream_idx=stream_idx,
            variant_label=variant_label,
            dsa_variant=dsa_variant,
            oft_variant=oft_variant,
        )
```

Update the 3 call sites that build a live (replay-time) `graph_key`/`_replay_graph_key`, threading `oft_variant=self._resolve_oft_variant(forward_batch)` alongside the existing `variant_label`/`dsa_variant` arguments:

1. `can_run_graph` (~line 693-697):
   ```python
        graph_key = self._make_graph_key(
            cuda_graph_bs,
            stream_idx=get_current_stream_idx() if self.enable_pdmux else None,
            variant_label=self._resolve_lora_variant(forward_batch),
            dsa_variant=self._resolve_dsa_variant(forward_batch),
            oft_variant=self._resolve_oft_variant(forward_batch),
        )
   ```
   (Read this call site's current exact form first — confirm whether `dsa_variant` is already passed here or was previously omitted; the planning-time read of this specific call showed only `variant_label` being passed, unlike the other two replay sites which pass both `variant_label` and `dsa_variant`. Investigate why before assuming this is an oversight to fix or a deliberate omission to match — if deliberate, add `oft_variant` following the same reasoning; if you determine it's an inconsistency unrelated to this task, do not fix it as a drive-by, just add `oft_variant` consistently with whatever this call site already does for `dsa_variant`.)

2. and 3. The two `self._replay_graph_key = self._make_graph_key(graph_size_key, stream_idx, variant_label, dsa_variant)` call sites (~line 1332 and ~1432, both preceded by `variant_label = self._resolve_lora_variant(forward_batch)` / `dsa_variant = self._resolve_dsa_variant(forward_batch)`): add a third line `oft_variant = self._resolve_oft_variant(forward_batch)` and thread it as the 5th positional (or keyword) argument to `_make_graph_key`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py -v`
Expected: PASS, all tests from Task 1 and Task 2.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py
git commit -m "feat(oft): resolve and thread the OFT MoE variant through replay-time graph keys"
```

---

## Task 3: Capture-loop `oft_variants` axis and the dual-capture gating condition

**Files:**
- Modify: `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
- Test: `test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py` (extend)

**Interfaces:**
- Consumes: `_set_capture_oft_variant` (Task 1), `capture_one_shape`'s existing `variant_label`/`dsa_variant` parameter pattern.
- Produces: `self.record_oft_variant_graph: bool` (new instance attribute, set in `__init__`) — Task 2's `_resolve_oft_variant` already reads this via `getattr`. `capture_one_shape(..., oft_variant: Optional[str] = None)` — no later task calls this directly, but Task 4's implementer should know this parameter now exists when tracing how `_compute_moe_multi_tenant_slot_ids` gets invoked during capture.

**Design note — finding the gating precedent:** Before writing `record_oft_variant_graph`'s condition, find and read where `self.dsa_dual_graph` is computed in `DecodeCudaGraphRunner.__init__` (~lines 255-280 as read during planning: `self.dsa_dual_graph = False` followed by a real-config check, `is_hip() and is_deepseek_dsa(hf_config)`, before setting it `True`) — this is your concrete model. Also locate where `record_nolora_graph` gets computed (grep for it across `__init__`'s full body; the planning-time search found it only referenced via `getattr(self, "record_nolora_graph", False)` and explicit `= False` overrides in draft/speculative runner subclasses, meaning the *base* `DecodeCudaGraphRunner.__init__` must set it via some condition not yet located — find that condition and use it as a second data point, since it's the more directly analogous "adapter-driven" axis, not just a hardware-driven one like DSA).

Set `self.record_oft_variant_graph` based on real OFT server config: the server has OFT enabled, targets MoE expert modules, and has effective capacity for more than one resident MoE-target adapter (mirror however `record_nolora_graph` determines "could this server ever need >1 adapter" for LoRA — likely something checking `max_loras_per_batch > 1` or equivalent; find and mirror the OFT equivalent, likely `max_ofts_per_batch`/`--max-loaded-ofts` reachable from `model_runner.server_args`). If you cannot cleanly determine whether MoE targeting is active from here (`server_args` alone may not say which modules are targeted), it is acceptable and honest to gate more conservatively — e.g. on "OFT enabled AND effective adapter capacity > 1" without also requiring "targets MoE" — since the false-positive cost (capturing an unused oft_multi graph on a non-MoE-OFT server) is graceful/self-gating, not a correctness bug, but note in your report whichever condition you used and why.

- [ ] **Step 1: Write the failing test**

Read `_capture_one_stream`'s full current body first (already read once during planning — confirm it's unchanged) and `capture_one_shape`'s current signature. Append to the test file:

```python
class TestOftVariantsCaptureAxis(unittest.TestCase):
    def test_single_no_op_variant_when_dual_capture_disabled(self):
        runner = MagicMock(spec=DecodeCudaGraphRunner)
        runner.record_oft_variant_graph = False
        # Mirrors how lora_variants / dsa_variants degrade to a single no-op
        # entry when their own dual-capture flag is off.
        oft_variants = (
            [("oft_multi", True), ("oft_single", False)]
            if getattr(runner, "record_oft_variant_graph", False)
            else [(None, None)]
        )
        self.assertEqual(oft_variants, [(None, None)])

    def test_two_variants_when_dual_capture_enabled(self):
        runner = MagicMock(spec=DecodeCudaGraphRunner)
        runner.record_oft_variant_graph = True
        oft_variants = (
            [("oft_multi", True), ("oft_single", False)]
            if getattr(runner, "record_oft_variant_graph", False)
            else [(None, None)]
        )
        self.assertEqual(oft_variants, [("oft_multi", True), ("oft_single", False)])
```

(This test pins the exact list-construction expression you'll paste into `_capture_one_stream` — it's a guard against a future edit changing the tuple order or values, not a test of the whole capture loop, which needs a real GPU to exercise meaningfully and is covered by Task 5's e2e test instead.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py -v`
Expected: These two specific tests currently pass trivially (the expression is inlined in the test itself, not yet reading real production code) — this is intentionally a pin/guard test, not a RED/GREEN TDD pair against not-yet-written production code. Skip the RED step for this specific pair; proceed directly to Step 3, then re-run in Step 4 to confirm the *real* implementation you add produces list values identical to what this test already pins.

- [ ] **Step 3: Write minimal implementation**

In `DecodeCudaGraphRunner.__init__`, near where `self.dsa_dual_graph`/`record_nolora_graph` are set (find the exact spot per the Design note above), add:

```python
        # OFT MoE-expert dual-graph capture: single-slot (today's fast path)
        # and multi-slot (per-token routing) variants, selected at replay by
        # how many distinct adapters are actually resident. Opt-in: a server
        # without OFT, or with no more than one possible resident MoE-target
        # adapter, captures only the single-slot variant it already did
        # before this feature existed -- zero extra cost.
        self.record_oft_variant_graph = <resolved condition -- see Design note>
```

In `_capture_one_stream` (~line 1104), add the new axis alongside `lora_variants`/`dsa_variants`:

```python
        oft_variants = (
            [("oft_multi", True), ("oft_single", False)]
            if getattr(self, "record_oft_variant_graph", False)
            else [(None, None)]
        )
```

Nest it into the existing loop (innermost, alongside `dsa_variants`, since OFT's axis is independent of both LoRA's and DSA's — order doesn't matter for correctness, only for capture-time memory-sharing locality, which is not a concern for this first cut):

```python
            for variant_label, _variant_has_lora in lora_variants:
                _set_capture_lora_variant(variant_label)
                for dsa_variant in dsa_variants:
                    _set_capture_dsa_variant(dsa_variant)
                    for oft_variant, _oft_variant_has_multi in oft_variants:
                        _set_capture_oft_variant(oft_variant)
                        with torch_compile_decoration.patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs,
                            num_tokens=bs * self.captured_req_width,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            if dsa_variant is None:
                                self.capture_one_shape(
                                    bs, forward, stream_idx, variant_label,
                                    oft_variant=oft_variant,
                                )
                            else:
                                self.capture_one_shape(
                                    bs, forward, stream_idx, variant_label, dsa_variant,
                                    oft_variant=oft_variant,
                                )
        _set_capture_dsa_variant(None)
        _set_capture_oft_variant(None)
```

(Follow the existing `dsa_variant is None` branch's exact reasoning for why `capture_one_shape` is called two different ways — read the comment above it in the current file about draft-runner subclass overrides not expecting the extra positional arg. Passing `oft_variant` as a KEYWORD argument with a default of `None` on `capture_one_shape`, rather than a new positional argument, avoids that same subclass-override-signature problem entirely — this is the safer choice; do not add it positionally.)

Update `capture_one_shape`'s signature to accept it and thread it into its own `_make_graph_key` call (~line 1151 and ~1257):

```python
    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
        dsa_variant: Optional[str] = None,
        oft_variant: Optional[str] = None,
    ):
```

```python
                shape_key = self._make_graph_key(
                    self._capture_graph_size(bs=bs, num_tokens=num_tokens),
                    stream_idx,
                    variant_label,
                    dsa_variant,
                    oft_variant=oft_variant,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py -v`
Expected: PASS, all tests from Tasks 1-3.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py test/registered/unit/oft/test_oft_moe_cuda_graph_dual_capture.py
git commit -m "feat(oft): capture single-slot and multi-slot OFT MoE variant graphs"
```

---

## Task 4: Persistent slot_ids buffer and OFT-side capture-time forcing

**Files:**
- Modify: `python/sglang/srt/oft/oft_manager.py`
- Test: extend `test/registered/unit/oft/test_oft_moe_multi_tenancy.py` (the base plan's existing test file for `_compute_moe_multi_tenant_slot_ids`)

**Interfaces:**
- Consumes: `get_capture_oft_variant()` (Task 1). `self.max_bs_in_cuda_graph` (existing, set by `init_cuda_graph_batch_info`).
- Produces: `self._moe_cg_slot_ids_buffer: Optional[torch.Tensor]` (new `OFTManager` attribute; `None` until `init_cuda_graph_batch_info` runs, then a persistent `(max_bs_in_cuda_graph * num_tokens_per_bs,)` `torch.long` tensor). `_compute_moe_multi_tenant_slot_ids`'s signature gains a `use_cuda_graph: bool` parameter.

**Design note:** This is the task with the most correctness risk in this plan — take real care, mirroring `LoRABackend._add_moe_lora_info`'s in-place-write discipline exactly (read that function's body again before writing this task's code, per what was read during planning: it reads `self.moe_cg_buffers[key]` when `batch_info.use_cuda_graph`, else builds a fresh eager tensor, and always WRITES into whichever tensor it resolved to, in place, never reassigning the buffer object itself).

- [ ] **Step 1: Write the failing test**

Read `_compute_moe_multi_tenant_slot_ids`'s full current docstring and body (`oft_manager.py` ~lines 699-795) and `OFTManager.__init__`/`init_cuda_graph_batch_info` (~lines 192-282) before writing this test — this task changes both. Append to `test/registered/unit/oft/test_oft_moe_multi_tenancy.py` (follow that file's existing `SimpleNamespace` + `MethodType` pattern for binding the real method under test):

```python
class TestPersistentSlotIdsBufferAndCaptureForcing(unittest.TestCase):
    def _make_tm(self, max_bs_in_cuda_graph=None, buffer=None):
        tm = SimpleNamespace()
        tm.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        tm._moe_cg_slot_ids_buffer = buffer
        tm._compute_moe_multi_tenant_slot_ids = MethodType(
            OFTManager._compute_moe_multi_tenant_slot_ids, tm
        )
        return tm

    def test_capture_forcing_builds_tensor_even_with_le_one_distinct_slot(self):
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            _set_capture_oft_variant,
        )
        self.addCleanup(_set_capture_oft_variant, None)
        _set_capture_oft_variant("oft_multi")

        buffer = torch.full((8,), -1, dtype=torch.long)
        tm = self._make_tm(max_bs_in_cuda_graph=4, buffer=buffer)
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.long),
            forward_mode=SimpleNamespace(is_extend=lambda: False, is_cuda_graph=lambda: True),
            batch_size=2,
            spec_info=None,
            extend_seq_lens=None,
        )
        # Only ONE distinct real slot -- would normally early-return None.
        weight_indices = [1, 1]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, forward_batch, use_cuda_graph=True
        )
        self.assertIsNotNone(result)  # forced, not None, because capturing "oft_multi"

    def test_use_cuda_graph_writes_into_persistent_buffer_in_place(self):
        buffer = torch.full((8,), -1, dtype=torch.long)
        buffer_id_before = id(buffer)
        tm = self._make_tm(max_bs_in_cuda_graph=4, buffer=buffer)
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.long),
            forward_mode=SimpleNamespace(is_extend=lambda: False, is_cuda_graph=lambda: True),
            batch_size=2,
            spec_info=None,
            extend_seq_lens=None,
        )
        weight_indices = [1, 2]  # two distinct real slots -- genuinely multi
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, forward_batch, use_cuda_graph=True
        )
        self.assertIsNotNone(result)
        # The RETURNED tensor must be a view/slice of the SAME buffer object,
        # not a fresh allocation -- this is the whole point of the fix.
        self.assertEqual(result.data_ptr(), buffer.data_ptr())
        self.assertTrue(torch.equal(result, torch.tensor([1, 2], dtype=torch.long)))

    def test_eager_path_unaffected_still_returns_fresh_tensor(self):
        tm = self._make_tm(max_bs_in_cuda_graph=4, buffer=torch.full((8,), -1, dtype=torch.long))
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.long),
            forward_mode=SimpleNamespace(is_extend=lambda: False, is_cuda_graph=lambda: False),
            batch_size=2,
            spec_info=None,
            extend_seq_lens=None,
        )
        weight_indices = [1, 2]
        result = tm._compute_moe_multi_tenant_slot_ids(
            weight_indices, forward_batch, use_cuda_graph=False
        )
        self.assertIsNotNone(result)
        self.assertNotEqual(result.data_ptr(), tm._moe_cg_slot_ids_buffer.data_ptr())
```

(These construct `forward_batch` as a minimal `SimpleNamespace` rather than a real `ForwardBatch` — read `generate_sequence_lengths`'s actual signature/requirements first (already partially understood from the existing docstring: "reads forward_mode, batch_size, spec_info and the extend_seq_lens pair") and adjust the fakes above if they don't satisfy it; the exact fake shape matters more than what's sketched here, since this is the one function this task changes the internals of.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v -k TestPersistentSlotIdsBufferAndCaptureForcing`
Expected: FAIL — `TypeError: _compute_moe_multi_tenant_slot_ids() got an unexpected keyword argument 'use_cuda_graph'` (and `AttributeError` on `tm._moe_cg_slot_ids_buffer` not existing as a real concept the function reads yet).

- [ ] **Step 3: Write minimal implementation**

In `OFTManager.__init__`, next to the existing `self._moe_multi_tenant_slot_ids: Optional[torch.Tensor] = None` line:

```python
        # Persistent, pre-allocated per-token slot-id buffer for CUDA-graph
        # replay of the multi-tenant MoE OFT path (allocated in
        # init_cuda_graph_batch_info once max_bs_in_cuda_graph is known).
        # Fixes the pointer-instability bug documented on
        # _compute_moe_multi_tenant_slot_ids before this plan: a fresh
        # tensor allocated on every eager call cannot be read correctly by a
        # captured graph, since the graph's kernel launch holds a pointer to
        # whatever address existed at capture time.
        self._moe_cg_slot_ids_buffer: Optional[torch.Tensor] = None
```

In `init_cuda_graph_batch_info`, allocate it once the max sizes are known:

```python
    def init_cuda_graph_batch_info(
        self, max_bs_in_cuda_graph: int, num_tokens_per_bs: int
    ):
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        self._moe_cg_slot_ids_buffer = torch.zeros(
            max_bs_in_cuda_graph * num_tokens_per_bs,
            dtype=torch.long,
            device=self.device,
        )
        self.oft_backend.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=max_bs_in_cuda_graph,
            num_tokens_per_bs=num_tokens_per_bs,
        )
```

(Confirm `num_tokens_per_bs`'s real meaning by reading its only other use just above — the plan assumes it is the max per-request token count a captured decode graph could see, e.g. speculative-decode draft width; if that assumption is wrong, size the buffer correctly instead and say so in your report.)

Rewrite `_compute_moe_multi_tenant_slot_ids` — same docstring intent, three new behaviors (capture-time forcing, in-place buffer write, DP-attention guard):

```python
    def _compute_moe_multi_tenant_slot_ids(
        self, weight_indices: list, forward_batch: ForwardBatch, use_cuda_graph: bool
    ) -> "Optional[torch.Tensor]":
        """Decide whether this batch needs the multi-tenant MoE OFT path, and
        if so build its PER-TOKEN adapter-slot tensor.

        [... keep the existing docstring's explanation of weight_indices'
        per-request granularity and the per-token expansion via
        generate_sequence_lengths verbatim; DELETE the "KNOWN LIMITATION"
        paragraph -- superseded by this task's fix -- replacing it with a
        short note that CUDA-graph replay is now correct via
        get_capture_oft_variant()-forced capture plus a persistent buffer,
        see decode_cuda_graph_runner.py's oft_variants capture axis.]

        use_cuda_graph: when True, the returned (non-None) tensor is a VIEW
        into self._moe_cg_slot_ids_buffer, written in place -- required for
        CUDA-graph replay pointer stability (mirrors
        LoRABackend._add_moe_lora_info's moe_cg_buffers discipline). When
        False (eager), behaves exactly as before: a fresh tensor.

        --enable-dp-attention is NOT supported by the CUDA-graph path yet:
        [resolve per Global Constraints -- assert/guard clearly, do not
        silently mis-size the buffer].
        """
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_capture_oft_variant,
        )

        capturing_multi = get_capture_oft_variant() == "oft_multi"
        distinct_real_slots = {idx for idx in weight_indices if idx != 0}
        if len(distinct_real_slots) <= 1 and not capturing_multi:
            return None

        device = forward_batch.input_ids.device
        tokens_per_request = generate_sequence_lengths(forward_batch, device=device)
        if tokens_per_request.shape[0] != len(weight_indices):
            raise RuntimeError(
                "Multi-tenant MoE OFT slot expansion needs one token count per "
                f"request, got {tokens_per_request.shape[0]} counts for "
                f"{len(weight_indices)} requests "
                f"(forward_mode={forward_batch.forward_mode})"
            )
        per_request_slots = torch.tensor(weight_indices, dtype=torch.long, device=device)
        expanded = per_request_slots.repeat_interleave(tokens_per_request.to(torch.long))

        if not use_cuda_graph:
            return expanded

        assert self._moe_cg_slot_ids_buffer is not None, (
            "use_cuda_graph=True but init_cuda_graph_batch_info was never called"
        )
        num_tokens = expanded.shape[0]
        # --enable-dp-attention gathers MoE tokens across DP ranks, which can
        # exceed this rank's own local per-token buffer sizing -- explicitly
        # unsupported for the CUDA-graph path in this first cut (see plan's
        # Global Constraints). [Resolve the exact guard expression against
        # real server_args access from this class -- assert/raise clearly if
        # DP-attention is enabled and this path is reached, rather than
        # silently truncating or overrunning the buffer below.]
        if num_tokens > self._moe_cg_slot_ids_buffer.shape[0]:
            raise RuntimeError(
                f"Multi-tenant MoE OFT CUDA-graph buffer holds "
                f"{self._moe_cg_slot_ids_buffer.shape[0]} token slots, but this "
                f"batch needs {num_tokens} -- this should have been caught by "
                "the runner's own use_cuda_graph eligibility check before "
                "reaching here; verify that check accounts for DP-attention "
                "gathering if this fires."
            )
        self._moe_cg_slot_ids_buffer[:num_tokens].copy_(expanded)
        return self._moe_cg_slot_ids_buffer[:num_tokens]
```

Update `prepare_oft_batch`'s call site to pass the already-computed `use_cuda_graph` flag:

```python
        self._moe_multi_tenant_slot_ids = self._compute_moe_multi_tenant_slot_ids(
            weight_indices, forward_batch, use_cuda_graph=use_cuda_graph
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/unit/oft/test_oft_moe_multi_tenancy.py -v` (full file, to also confirm no regression in the base plan's own existing tests for this function, which will need their call sites updated for the new required `use_cuda_graph` parameter — update them rather than leaving them broken).
Expected: PASS, all tests including the 3 new ones.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/oft/oft_manager.py test/registered/unit/oft/test_oft_moe_multi_tenancy.py
git commit -m "feat(oft): persistent CUDA-graph slot_ids buffer and capture-time forcing"
```

---

## Task 5: End-to-end proof, regression guard, and known-limitation doc cleanup

**Files:**
- Test: `test/registered/rl/test_oft_moe_multi_tenant_cuda_graph_e2e.py` (new, GPU required)
- Modify: `python/sglang/srt/oft/oft_manager.py` (docstring only — remove/correct the now-stale "KNOWN LIMITATION" text if Task 4 didn't already fully rewrite it)

**Interfaces:**
- Consumes: everything from Tasks 1-4 — no new production code in this task besides the docstring cleanup.

- [ ] **Step 1: Write the failing test**

Port `test_oft_moe_multi_tenant_e2e.py`'s (the base plan's own Task 5, if it landed — check whether that file exists yet; if not, port directly from `test_oft_load_from_tensor.py`'s server-launch/adapter-loading fixtures the same way that task did) server-launch and two-adapter-loading helpers, but launch the server WITHOUT `--disable-cuda-graph` this time (the base plan's e2e test deliberately disabled CUDA graphs to avoid the bug this plan fixes — this test's whole point is to enable them). Create `test/registered/rl/test_oft_moe_multi_tenant_cuda_graph_e2e.py`:

```python
"""End-to-end GPU test: MoE expert-OFT multi-tenancy under CUDA-graph decode
(this plan's Task 5).

1. test_single_adapter_decode_graph_unaffected: regression guard -- a single
   MoE-target OFT adapter resident, CUDA graphs enabled, must produce
   identical output to before this plan (the fast path must be untouched).
2. test_two_adapters_correct_under_cuda_graph_replay: the actual fix -- two
   concurrently-resident MoE-target OFT adapters, CUDA graphs enabled
   (NOT disabled, unlike the base plan's own e2e test), decode replay must
   apply each adapter's own rotation correctly -- this is exactly the
   scenario that silently produced wrong output before this plan.

Follows test_oft_load_from_tensor.py's server-launch and adapter-loading
conventions, same base model and target_modules choices as the base plan's
own Task 5 e2e test, but WITHOUT --disable-cuda-graph.
"""

# Implementer: port setUpClass/tearDownClass and adapter-tensor-construction
# helpers directly from the base MoE-multi-tenancy plan's own Task 5 test
# file (test_oft_moe_multi_tenant_e2e.py) if it exists, or from
# test_oft_load_from_tensor.py otherwise -- do not reinvent them. The two
# test bodies below are the new logic this task adds; port them from the
# base plan's own Task 5 assertions, changing only the CUDA-graph server arg.

import unittest


class TestMoeMultiTenantCudaGraphEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise NotImplementedError("port setUpClass, WITHOUT --disable-cuda-graph")

    @classmethod
    def tearDownClass(cls):
        raise NotImplementedError("port from the base plan's own Task 5 test")

    def test_single_adapter_decode_graph_unaffected(self):
        """Load one MoE-target OFT adapter, generate enough tokens to exercise
        decode-phase CUDA graph replay, assert output is deterministic and
        non-trivial (differs from base-model generation) -- proving the
        existing single-slot fast path still works correctly and was not
        slowed or broken by this plan's capture-loop changes."""
        raise NotImplementedError

    def test_two_adapters_correct_under_cuda_graph_replay(self):
        """Port the base plan's own Task 5 two-adapter assertion
        (test_two_moe_adapters_apply_correctly) verbatim in spirit, but run
        it with CUDA graphs enabled: load adapter A alone, generate enough
        tokens for decode-graph replay, record output; load adapter B alone,
        same prompt, record output (sanity: A != B); load both concurrently,
        issue one request per adapter in the same batch/step, and assert
        each matches its own isolated output. Before this plan, this
        assertion would silently fail under CUDA graphs (both requests would
        see whichever adapter's rotation the single-slot graph happened to
        have captured); it should now pass."""
        raise NotImplementedError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/rl/test_oft_moe_multi_tenant_cuda_graph_e2e.py -v`
Expected: FAIL with `NotImplementedError` on all methods (scaffold; fill in per Step 3).

- [ ] **Step 3: Write minimal implementation**

Port the fixtures and both test bodies per the docstrings above. For `test_two_adapters_correct_under_cuda_graph_replay`, generate ENOUGH tokens per request (e.g. 20+) so the request definitely transitions from prefill/extend into decode-phase CUDA-graph replay — a test that only prefills would not exercise the bug this plan fixes at all, since OFT already disables prefill graphs entirely (unrelated, pre-existing).

Also in this step: read `_compute_moe_multi_tenant_slot_ids`'s current docstring in `oft_manager.py` (should already be mostly rewritten by Task 4) and confirm it no longer claims CUDA-graph replay is unsafe — if any stale wording remains, correct it now.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo-root> && PYTHONPATH=python python3 -m pytest test/registered/rl/test_oft_moe_multi_tenant_cuda_graph_e2e.py -v`
Expected: PASS (requires GPU; requires Tasks 1-4 complete).

- [ ] **Step 5: Commit**

```bash
git add test/registered/rl/test_oft_moe_multi_tenant_cuda_graph_e2e.py python/sglang/srt/oft/oft_manager.py
git commit -m "test(oft): end-to-end proof of MoE expert-OFT multi-tenancy under CUDA-graph decode"
```

---

## Self-Review

**Spec coverage:**
- "Fourth capture-variant axis, mirroring LoRA/DSA" → Tasks 1 (flag+key field), 3 (capture loop).
- "Persistent per-token routing buffer, mirroring moe_cg_buffers" → Task 4.
- "Capture-time forcing via get_capture_oft_variant()" → Task 4.
- "Opt-in gating, zero cost for non-OFT/single-adapter servers" → Task 3's `record_oft_variant_graph` condition + Task 2's `getattr(..., False)` default.
- "Resolve which class owns the buffer" → resolved in Global Constraints (`OFTManager`, matching `max_bs_in_cuda_graph`'s existing home) and Task 4.
- "Resolve DP-attention" → resolved in Global Constraints (explicitly excluded, guarded not silent) and Task 4's guard.
- "Error handling: batch exceeding buffer capacity" → Task 4's `RuntimeError` guard (mirrors LoRA's `use_cuda_graph = False` demotion in spirit, though as an assertion here since the runner-level eligibility check — mirroring LoRA's `prepare_lora_batch` demotion — is a real gap this plan's own scope doesn't cover; flagged inline in Task 4 as something the implementer should verify is unreachable given Task 2/3's own eligibility gating, or extend if not).
- "Correct legacy-fused layout must not crash (safety, not correctness)" → not separately tasked; carried by Task 4's forcing logic reusing the exact same `_compute_moe_multi_tenant_slot_ids` code path Task 3/4 of the base plan already made safe for this case (no new legacy-fused-specific risk introduced here, since this task doesn't touch the read-side kernel dispatch at all).
- "Update stale KNOWN LIMITATION doc" → Task 4 (rewrite) + Task 5 (final confirmation pass).
- Testing plan (unit tests for resolve/flag logic, GPU e2e with 2 adapters + CUDA graphs, single-adapter regression guard) → Tasks 1-3 (unit), Task 5 (e2e + regression).

**Placeholder scan:** Task 5's Step 1 scaffolds with `NotImplementedError` (the required TDD shape, same as the base plan's own Task 5) with concrete port-and-assert instructions for Step 3 — not an unfilled placeholder. Several Task 3/4 code blocks contain an explicit bracketed instruction (e.g. "[resolve per Global Constraints...]") rather than literal code, for the two items the spec explicitly left open for plan-time/implementation-time resolution (the exact `record_oft_variant_graph` gating expression, and the exact DP-attention guard expression) — these are the plan's own genuinely open items (documented as such in the spec), not vague hand-waves; each comes with a concrete instruction for what to go find and how to verify it, and a fallback default (conservative gating; hard RuntimeError guard) if the exact ideal condition can't be pinned down. This is consistent with how the base plan's own Task 3/4 handled similarly-scoped ambiguities.

**Type/signature consistency across tasks:**
- `get_capture_oft_variant()`/`_set_capture_oft_variant()` (Task 1) — used identically by Task 2's `_resolve_oft_variant` reasoning-by-analogy (not directly, since resolve reads `forward_batch` not the capture flag) and Task 4's `_compute_moe_multi_tenant_slot_ids` (which DOES call `get_capture_oft_variant()` directly). Consistent.
- `ShapeKey.oft_variant` (Task 1) — `_make_graph_key`'s new parameter (Task 2) and `capture_one_shape`'s new parameter (Task 3) both thread a value of the same type (`Optional[str]`, one of `"oft_single"`/`"oft_multi"`/`None`) into it. Consistent.
- `_compute_moe_multi_tenant_slot_ids`'s new `use_cuda_graph: bool` parameter (Task 4) — the only caller is `prepare_oft_batch`, updated in the same task. The base plan's own existing tests for this function (in `test_oft_moe_multi_tenancy.py`) call it with 2 positional args today; Task 4 explicitly requires updating those call sites to pass the new required parameter, called out in Task 4's Step 4.

No gaps found beyond the two explicitly-flagged open items, both of which carry a concrete resolution path and a safe fallback.

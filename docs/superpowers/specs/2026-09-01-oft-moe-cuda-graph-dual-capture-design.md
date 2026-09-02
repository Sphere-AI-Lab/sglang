# OFT MoE Expert Multi-Tenancy Under CUDA-Graph Decode — Dual-Capture Design

## Problem

The MoE expert-OFT multi-tenancy plan (`docs/superpowers/specs/2026-09-01-moe-expert-oft-multi-tenancy-design.md`, `docs/superpowers/plans/2026-09-01-moe-expert-oft-multi-tenancy.md`) built correct per-token adapter routing for the eager execution path, but deliberately left a known, disclosed gap: CUDA-graph-*replayed* decode can silently apply one adapter's rotation to every token once 2+ MoE-target OFT adapters are resident. Root cause, established during that plan's Task 4:

- Decode graph **capture** runs with `adapter_ids = [None] * bs` (`decode_cuda_graph_runner.py`'s dummy-batch construction), so at capture time the multi-tenancy decision naturally resolves to "single/none," and the single-slot kernel is what gets recorded into the graph.
- CUDA-graph **replay** never re-executes the surrounding Python branching logic — only the pre-recorded kernel launches run — so a graph captured this way always replays the single-slot kernel, regardless of how many adapters are actually resident at replay time.
- Even if the *right* kernel were forced into the graph, `OFTManager.prepare_oft_batch` currently allocates the per-token `slot_ids` tensor fresh every call (`torch.tensor(weight_indices, ...)`), outside the capture region — the captured kernel's pointer to that tensor would go stale the moment a new tensor is allocated on a later call, so replay would still read wrong data.

This is a genuine correctness gap in the plan's primary scenario (dynamically-loaded, concurrently-resident adapters) whenever CUDA graphs are enabled for decode — a common, often-default performance path in real serving. The prior plan scoped a real fix out as too large and documented it as a known limitation instead. This spec is that follow-up, explicitly requested by the user rather than left deferred.

## Precedent: this codebase already has the extensibility point this needs

`DecodeCudaGraphRunner` already captures graphs along multiple independent variant axes — LoRA (`lora`/`nolora`) and DSA (`dense`/`sparse`) — each following the same three-part pattern:

1. A process-global flag pair in `model_executor/runner_utils/capture_mode.py` (e.g. `_capture_lora_variant` / `get_capture_lora_variant()` / `_set_capture_lora_variant()`) that model/runtime code reads *during capture* to force the correct branch into the graph.
2. A `_resolve_<axis>_variant(forward_batch)` method on `DecodeCudaGraphRunner`, evaluated live (never captured) at replay-eligibility time (`can_run_graph`) and at the actual replay call sites, reading the real current batch state to select which pre-captured graph to use.
3. A named field on `ShapeKey` (`variant_label` for LoRA, `dsa_variant` for DSA) so multiple axes combine into one cache key without colliding, and a Cartesian loop in `_capture_one_stream` that captures one graph per combination of axis values.

Both existing axes are **opt-in**: `lora_variants`/`dsa_variants` default to a single no-op variant (`[(None, None)]` / `[None]`) unless the server configuration actually requires dual capture (`record_nolora_graph`, `dsa_dual_graph`), so a server that doesn't use the feature pays zero extra capture time or GPU memory for it.

Separately, LoRA's own MoE routing already solved the pointer-stability problem this needs: `LoRABackend.moe_cg_buffers` (`lora/backend/base_backend.py`) pre-allocates a fixed-size `token_lora_mapping` tensor once, at graph-build time; every real batch (`_add_moe_lora_info`) writes the current batch's routing data into that *same* buffer in place, rather than allocating a new tensor — so the captured kernel's pointer stays valid across replays and only the buffer's contents change.

## Goals

- CUDA-graph-replayed decode produces correct per-adapter output when 2+ MoE-target OFT adapters are resident, matching what eager execution already does.
- Zero additional capture time, GPU memory, or replay overhead for servers not using OFT, or using OFT with at most one resident MoE-target adapter at a time (today's already-working, already-fast case) — mirroring the existing axes' opt-in discipline exactly.
- No change to LoRA's or DSA's existing variant axes; this is an independent fourth axis.

## Non-goals

- The legacy fused `oft_type="oft"` layout's multi-tenant correctness — out of scope, matching the base multi-tenancy plan's own non-goal. This axis only needs to be *safe* (no crash) for that layout under dual-capture, not correct.
- Optimizing the multi-tenant kernel's own performance — already an accepted tradeoff from the base plan.
- `--enable-dp-attention` is a real interaction (LoRA's own `moe_cg_buffers` sizing accounts for DP-gathered token counts, `get_gathered_moe_num_tokens`) — this design follows the same pattern, but is not required to support every DP configuration on day one if a narrower, clearly-guarded first cut is safer; see Testing.

## Architecture

Four coordinated additions, one per precedent-established extension point, plus the persistent buffer:

1. **`capture_mode.py`**: add `_capture_oft_variant: Optional[str]`, `get_capture_oft_variant()`, `_set_capture_oft_variant(variant)` — byte-for-byte mirroring the LoRA/DSA pair.
2. **`decode_cuda_graph_runner.py`**:
   - `_resolve_oft_variant(forward_batch)`: mirrors `_resolve_lora_variant`. Returns `None` when dual-capture isn't enabled for this server (see gating below); otherwise inspects `forward_batch.adapter_ids` directly (the same field OFT's own `weight_indices` construction already reads) and returns `"oft_multi"` when 2+ distinct non-`None` values are present, else `"oft_single"`. This mirrors the *same* global, whole-batch simplification the base plan's Task 1 already made for `_compute_moe_multi_tenant_slot_ids` — reusing the identical distinct-count check, not inventing a second one.
   - `ShapeKey` gains an `oft_variant` field; `_make_graph_key` accepts and threads it through, alongside `variant_label`/`dsa_variant`.
   - `_capture_one_stream`'s nested loop gains an `oft_variants` axis (`[("oft_multi", True), ("oft_single", False)]` when dual-capture is enabled for this server, else `[(None, None)]`), calling `_set_capture_oft_variant(variant_label)` around each `capture_one_shape` call, Cartesian-multiplied with the existing `lora_variants`/`dsa_variants` loops exactly as those two already are with each other.
   - Every existing replay/eligibility call site that builds a `graph_key` (`can_run_graph`, and the replay call sites already passing `variant_label=self._resolve_lora_variant(...)`) also passes `oft_variant=self._resolve_oft_variant(forward_batch)`.
3. **Persistent buffer**: a new `moe_cg_buffers`-equivalent for OFT's `slot_ids`, owned by `OFTMemoryPool` or `OFTManager` (whichever already owns comparably-scoped CUDA-graph-static state — check `OFTMemoryPool`'s existing `max_bs_in_cuda_graph`-conditioned allocations for the established home). Pre-allocated once, sized to the maximum token count a captured "oft_multi" graph could see (mirroring `moe_cg_buffers`'s `max_bs * dp_size` sizing and its `top_k`-aware padding math). `OFTManager.prepare_oft_batch`, when `use_cuda_graph` is true, writes the current batch's per-token slot assignment into this buffer in place (`buffer.copy_(...)` or an equivalent in-place op) instead of allocating `torch.tensor(weight_indices, ...)` fresh; `_compute_moe_multi_tenant_slot_ids` returns a *view* of this persistent buffer rather than a new tensor whenever `use_cuda_graph` is true.
4. **OFT capture-time forcing**: `_compute_moe_multi_tenant_slot_ids`, when `get_capture_oft_variant() == "oft_multi"`, skips its normal early-return (which would otherwise see the capture-time dummy batch's `≤1` distinct adapters and return `None`) and always builds/returns the per-token tensor — so the "oft_multi" graph variant's capture pass genuinely records the multi-tenant kernel path, using the persistent buffer from (3). When `get_capture_oft_variant() == "oft_single"` (or not in dual-capture mode at all), behavior is completely unchanged from today.

**Gating (opt-in):** dual-capture for OFT is enabled only when the server is actually configured for MoE-target OFT with room for more than one resident adapter (e.g. `--enable-oft` targeting MoE modules, and `--max-loaded-ofts` — or the equivalent effective capacity — greater than 1). A server without OFT, or with OFT limited to a single resident adapter, takes the existing `[(None, None)]` no-op path and pays nothing extra, exactly like `record_nolora_graph`/`dsa_dual_graph` today.

## Data flow

- **Capture** (once, at server startup or graph-warm-up): for each `(bs, lora_variant, dsa_variant, oft_variant)` combination the enabled axes produce, `_set_capture_oft_variant(oft_variant)` is set, then `capture_one_shape` runs a dummy forward pass; when `oft_variant == "oft_multi"`, `_compute_moe_multi_tenant_slot_ids` is forced to build the general per-token tensor (from the persistent buffer, populated with capture-time dummy values), so the multi-slot kernel's launch is what gets recorded for that graph.
- **Replay eligibility / selection** (every real forward step): `_resolve_oft_variant(forward_batch)` reads the real, current `forward_batch.adapter_ids` and returns the matching label; `can_run_graph`/the replay call sites use it (combined with the other axes' labels) to select which previously-captured graph to replay.
- **Replay** (every real forward step, when using the "oft_multi" graph): `OFTManager.prepare_oft_batch` writes the current batch's real slot assignment into the persistent buffer in place; the captured kernel launch, replayed against that same buffer's (now-updated) contents, rotates each token with its actual currently-resident adapter.

## Error handling

- A batch whose real per-token count would exceed the persistent buffer's capacity (mirroring LoRA's own `moe_cg_buffers`-vs-DP-gathered-count check) must **not** attempt to replay the "oft_multi" graph — fall back to eager for that one forward step, exactly matching `LoRAManager.prepare_lora_batch`'s existing `use_cuda_graph = False` demotion for the analogous LoRA case.
- The legacy-fused (`oft_type="oft"`) layout: forcing `oft_multi` capture must not crash for a deployment that has a legacy-fused adapter loaded at boot (recall Task 3's `_oft_w13_oft_r_all_slots`-unbound guard) — verify the capture-time forced path degrades to the same clear, already-implemented `RuntimeError` rather than a new failure mode, or is excluded from dual-capture entirely if the interaction can't be made safe cheaply.

## Testing

- Unit test for `_resolve_oft_variant`'s decision logic (mirrors Task 1's own `_compute_moe_multi_tenant_slot_ids` tests, at the `DecodeCudaGraphRunner` layer instead).
- A real GPU test: enable decode CUDA graphs, dynamically load 2 MoE-target OFT adapters, issue concurrent decode requests naming each, and assert each gets its own adapter's output — the same proof Task 5 already does for eager mode, now also passing under CUDA graphs. This directly supersedes the "known limitation" documentation the base plan added; once this lands, that documentation should be corrected or removed.
- A regression test confirming the single-adapter (or zero-adapter) decode-CUDA-graph path is unaffected in output and capture time/memory when this feature is compiled in but not triggered (the opt-in gating's whole point).

## Open items for the implementation plan

- Exact home for the new persistent buffer (`OFTMemoryPool` vs `OFTManager`) — an implementation detail to resolve by following the established pattern for comparable existing CUDA-graph-static state in this codebase.
- Whether `--enable-dp-attention` needs full support in the first cut or can be explicitly excluded/guarded (see Non-goals) — a plan-time and possibly a first-task-in-the-plan decision, not a design ambiguity.

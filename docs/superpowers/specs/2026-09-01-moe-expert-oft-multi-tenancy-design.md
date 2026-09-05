# MoE Expert-OFT Multi-Tenancy — Design

## Problem

`FusedMoEWithOFT.forward` (`python/sglang/srt/oft/layers.py`) applies expert OFT
rotation by calling `self._oft_runner.run(dispatch_output, self._quant_info)`.
The runner's kernel invoker (`_oft_prerotate` in `oft/oft_moe_runners.py`) closes
over a single tensor per layer — whatever is currently bound to
`moe.w13_oft_r`/`w1_oft_r`/`w3_oft_r`/`w2_oft_r` — and applies it to every token
in the batch uniformly. There is no per-token mechanism to select a different
adapter's rotation for a different token.

The underlying storage is **not** the bottleneck: `oft/mem_pool.py`'s
`_declare_expert_groups` already registers these buffers as ordinary multi-slot
pool groups (`register_buffer_group`, indexed via `self.slot(name, layer_id,
slot_idx)`), identical in kind to the dense path's per-adapter buffer slots.
`apply_streamed_expert_oft` can already target an explicit `slot_idx` on write
(used today by the double-buffer stage/activate path). The gap is entirely on
the **read side**: `FusedMoEWithOFT.forward` always reads one fixed Python
attribute — whichever slot's view `OFTManager` last bound to `moe.w13_oft_r` —
so writing a second adapter's rotation into a different slot makes that
adapter's rotation silently never apply (a correctness bug, not a crash),
rather than making both adapters correctly resident. This is a known,
documented limitation (`apply_streamed_expert_oft`'s docstring, `oft_manager.py`)
that was deliberately scoped out of the native OFT adapter RPC branch
(`feat/oft-native-adapter-rpc`, merged into `orbit-main-corrected`
2026-09-01) as its own follow-up.

**Goal:** let two or more concurrently-resident OFT adapters that both target
MoE expert weights be applied correctly, per token, in the same batch —
matching what `FusedMoEWithLoRA` already does for LoRA — without regressing
the performance or behavior of the common case (zero or one resident
MoE-target adapter).

## Non-goals

- Changing the dense (non-MoE) OFT path — it already has per-token routing
  (`prepare_oft_batch`'s `weight_indices` → `run_oft_r_sgemm`) and is unaffected.
- A new capacity knob. Per-adapter MoE buffer slots are governed by the same
  `--max-ofts-per-batch` cap that already sizes every other per-adapter buffer
  group in the pool — mirroring how LoRA's MoE path reuses `--max-loras-per-batch`
  with no separate knob. No new flag.
- Optimizing the multi-tenant kernel path's performance beyond correctness.
  The single/no-adapter fast path must not regress; the multi-tenant path
  only needs to be correct in this iteration.
- The legacy fused `oft_type="oft"` variant is not a priority. `canonical_oft`
  (split per-sub-projection) is "orbit's only trained variant" per
  `peft/config.py`'s own comments — the new mechanism must not break the
  legacy fused variant's *existing* single-adapter behavior, but multi-tenant
  correctness for the legacy variant specifically is not a requirement.
- Rebuilding storage/allocation. The pool's multi-slot buffer groups already
  exist; this plan only changes what the read side does with them.

## Architecture

Two additions, layered on top of the existing pool without changing its
storage:

1. **A per-batch `weight_indices`-equivalent for MoE tokens**, built once per
   forward batch (no GPU sync), reusing the *same* `uid → buffer_id` mapping
   `prepare_oft_batch` already computes for the dense path. Every token that
   currently has a `weight_indices` entry for the dense path gets the same
   slot index for the MoE path — there is exactly one buffer-slot assignment
   per adapter per batch, shared across dense and MoE application, not two
   independent ones. This mirrors `FusedMoEWithLoRA`'s `token_lora_mapping`,
   which is likewise derived from the same per-batch adapter/slot bookkeeping
   the rest of the LoRA path already has.

2. **A second kernel path**, added alongside the existing one, not replacing
   it:
   - **Fast path (unchanged):** when the current batch's resident adapters
     include at most one non-identity adapter carrying MoE-target OFT
     weights, `FusedMoEWithOFT.forward` takes exactly today's code path —
     `_oft_prerotate` closing over the single active-slot tensor. Zero
     performance or behavior change for the common case. This mirrors
     `FusedMoEWithLoRA.forward`'s existing `self.lora_backend.batch_info is
     None` early-out.
   - **Multi-tenant path (new):** when more than one resident adapter in the
     batch carries MoE-target weights, `forward` instead passes the pool's
     multi-slot buffer group directly (not a single bound view) plus the
     per-token slot-index tensor from (1) into a new rotation kernel variant
     that indexes per token instead of applying one rotation to the whole
     batch.

Deciding which path to take is a single cheap check per forward call (how many
distinct uids with populated MoE buffers are in `cur_uids`), computed from
data the pool already tracks — no new state.

## Components

- **`oft/mem_pool.py`**: no storage changes. Add a query the runner can use to
  determine, for a given `cur_uids`, how many of them have real (non-identity)
  MoE-target buffers populated for a given layer, and to hand back the
  per-token slot-index tensor built from the existing `uid_to_buffer_id`
  mapping. This is bookkeeping, not new allocation.
- **`oft/oft_manager.py`**: `apply_streamed_expert_oft` already supports
  writing to an explicit `slot_idx` — unchanged. `OFTManager` gains the
  per-batch decision-and-mapping step (reusing `prepare_oft_batch`'s existing
  per-token slot computation rather than duplicating it) and passes the
  result into `FusedMoEWithOFT.forward` the same way `LoRAManager` already
  makes `token_lora_mapping` available to `FusedMoEWithLoRA`.
- **`oft/layers.py`** (`FusedMoEWithOFT.forward`): branches on the
  fast-path/multi-tenant-path decision; on the fast path, behaves exactly as
  today; on the multi-tenant path, passes the buffer group + per-token
  mapping into the runner instead of a single quant_info-only call.
- **`oft/oft_moe_runners.py`**: `_oft_prerotate` stays as-is for the fast
  path. A new sibling function (same "kernel invoker substitution" pattern,
  same verbatim-numerics discipline the existing one follows) accepts the
  per-token mapping and the multi-slot buffer instead of one closed-over
  tensor.
- **`oft/triton_ops/` kernel(s)** (the actual Triton kernel(s) underlying
  `apply_oft_rotation_triton`, e.g. `grouped_moe_rotate_project.py`): the new
  multi-tenant path needs a kernel variant that selects the right per-expert
  R-block using the token's assigned slot index, not just its expert id. This
  is the one genuinely new piece of kernel code in this plan; everything else
  is wiring over existing storage and existing per-batch bookkeeping.

## Data flow

1. Scheduler assembles a batch; `prepare_oft_batch` (unchanged) resolves each
   resident adapter to a buffer slot exactly as it does today for the dense
   path.
2. `OFTManager` computes, from that same mapping, how many distinct
   MoE-target-carrying uids are in the batch and (if more than one) the
   per-token slot-index tensor.
3. `FusedMoEWithOFT.forward` reads that decision. Fast path: identical to
   today. Multi-tenant path: passes the pool's buffer group reference + the
   per-token slot-index tensor into the new runner path.
4. The new kernel variant applies each token's own adapter's rotation using
   its assigned slot, instead of one rotation for the whole batch.

## Error handling

- A token whose resident adapter has **no** MoE-target weights loaded (e.g.
  an adapter that only targets dense/attention layers) must fall back to the
  identity rotation for that token in the multi-tenant path — matching how
  the dense path already treats adapters without weights for a given target
  module. This is not an error case, just a per-token no-op.
- If the multi-tenant path's slot-index tensor references a slot the pool
  reports as not actually populated for this layer (a real bug, not a
  legitimate no-adapter case), raise a clear error rather than silently
  reading uninitialized buffer contents — consistent with how the rest of
  this codebase's recent fixes (this session's C1 fix round) converted
  silent-corruption-risk paths into clear failures.
- The legacy fused `oft_type="oft"` variant: the fast path continues to work
  for it unchanged. Whether the new multi-tenant kernel variant supports it
  is not required by this plan (see Non-goals); if unsupported, resident
  legacy-fused adapters should be excluded from the multi-tenant candidate
  count with a clear log message, not silently miscomputed.

## Testing

- Unit tests for the new per-batch decision logic (0/1/2+ MoE-target
  adapters resident → correct fast-path-vs-multi-tenant choice; correct
  per-token slot-index tensor construction from a given `uid_to_buffer_id`
  mapping) — binding the real production methods the way this session's
  other OFT test suites already do (`MethodType` onto lightweight doubles),
  not reimplementing the logic.
- A real-GPU regression test with two concurrently-resident MoE-target OFT
  adapters, asserting each adapter's output actually differs and matches
  what that adapter alone would produce in isolation (the true multi-tenancy
  proof) — check whether LoRA has an equivalent existing MoE multi-tenancy
  test to mirror the validation shape from.
- A real-GPU regression test confirming the fast path's output and
  performance are unchanged from before this change, for the common
  0/1-adapter case.

## Open items for the implementation plan

- Exact tensor shapes/dtypes for the new kernel variant's per-token slot
  index argument, and how `MERGED_OFT_PROJ_GROUPS`/split-vs-fused layout
  interacts with it for `canonical_oft`.
- Whether the "how many MoE-target uids resident" check belongs on
  `OFTManager` or `OFTMemoryPool` — an implementation detail, not a design
  question.

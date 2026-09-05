from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Literal, Sequence, Tuple

import torch

from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorMetadata

logger = logging.getLogger(__name__)

# Match FusedMoE expert OFT R names emitted by Megatron-Bridge HF export, e.g.
#   model.layers.3.mlp.experts.17.gate_proj.oft_R
# These bypass the dense `_resolve_oft_tensor_plan` path (which writes to a
# per-layer `R_buffer["gate_up_proj"|"down_proj"][layer_id]` slot with no
# expert dimension) and are dispatched per-FusedMoE via
# `oft_manager.apply_streamed_expert_oft`. Canonical grouped expert FC1 emits
# independent gate/up rotations, so up_proj is preserved here; oft_manager
# disambiguates split (gate != up) from legacy shared-R (only gate streamed)
# state and routes to w1_oft_r/w3_oft_r vs w13_oft_r accordingly.
_EXPERT_OFT_RE = re.compile(
    r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.oft_R"
)
_DSV4_EXPERT_OFT_RE = re.compile(
    r"(?:mlp|ffn)\.experts\.(\d+)\.(w1|w2|w3)\.oft_R"
)
_DSV4_TO_FUSED_EXPERT_OFT_PROJ = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}

type FlattenedOFTTensorPayload = tuple[
    Literal["flattened_oft_payload"],
    bytes,
    List[FlattenedTensorMetadata],
    List[Tuple[str, int]],
]


def _partition_expert_oft_tensors(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    *,
    tp_rank: int | None = None,
) -> Tuple[
    Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    List[Tuple[str, torch.Tensor]],
]:
    """Split incoming OFT tensors into expert vs. dense buckets.

    Returns (fused_expert_layer_dict, dsv4_expert_layer_dict, dense_named_tensors).
    Expert layer dicts: layer_id -> global_expert_id -> {proj.oft_R: tensor}.
    Dense entries (attention, non-MoE MLP, embeddings) pass through unchanged
    so the existing dense `_resolve_oft_tensor_plan` path handles them — this
    is also why MLA names are routed correctly without any MLA-specific code.
    """
    from sglang.srt.layers.utils import get_layer_id
    from sglang.srt.oft._streamed_audit import record_expert_partition

    fused_expert_layer_dict: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    dsv4_expert_layer_dict: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    dense: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in named_tensors:
        m = _EXPERT_OFT_RE.search(name)
        family = "fused"
        if m is None:
            m = _DSV4_EXPERT_OFT_RE.search(name)
            family = "dsv4"
        if m is None:
            dense.append((name, tensor))
            continue
        proj = m.group(2)
        layer_id = get_layer_id(name)
        if layer_id is None:
            dense.append((name, tensor))
            continue
        expert_id = int(m.group(1))
        expert_layer_dict = (
            fused_expert_layer_dict
            if family == "fused"
            else dsv4_expert_layer_dict
        )
        layer = expert_layer_dict.setdefault(layer_id, {})
        ew = layer.setdefault(expert_id, {})
        ew[f"{proj}.oft_R"] = tensor
        record_expert_partition(layer_id, expert_id, proj, tensor, tp_rank=tp_rank)
    return fused_expert_layer_dict, dsv4_expert_layer_dict, dense


def _convert_dsv4_expert_chunk_to_fused(
    dsv4_expert_chunk: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """Map Bridge DSV4-style grouped expert OFT keys to FusedMoE keys.

    Kimi K2.5 checkpoints are served by the DeepseekV2/Kimi FusedMoE model in
    this SGLang tree, while Megatron-Bridge exports grouped expert OFT using
    DSV4-style names: ``ffn.experts.<id>.w{1,2,3}.oft_R``.  When no DeepSeekV4
    MoE modules exist, those tensors need to feed the FusedMoE writer instead.
    """
    fused: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    for layer_id, layer_chunk in dsv4_expert_chunk.items():
        fused_layer = fused.setdefault(layer_id, {})
        for expert_id, expert_weights in layer_chunk.items():
            fused_expert = fused_layer.setdefault(expert_id, {})
            for name, tensor in expert_weights.items():
                if not name.endswith(".oft_R"):
                    continue
                proj = name[: -len(".oft_R")]
                fused_proj = _DSV4_TO_FUSED_EXPERT_OFT_PROJ.get(proj)
                if fused_proj is None:
                    continue
                fused_expert[f"{fused_proj}.oft_R"] = tensor
    return fused


def _merge_expert_oft_chunks(
    dst: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    src: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
) -> None:
    for layer_id, layer_chunk in src.items():
        dst_layer = dst.setdefault(layer_id, {})
        for expert_id, expert_weights in layer_chunk.items():
            dst_expert = dst_layer.setdefault(expert_id, {})
            dst_expert.update(expert_weights)


# Live full-block tensors inside one _flush_oft_group_chunk call: the expanded
# skew matrix, cayley_neumann's R accumulator, Q_squared, the rolling Q_power,
# and the packed_r result. The chunk limit has to be measured against THIS,
# not against the compact payload: compact storage is the upper triangle
# (~half a full matrix), so accounting in compact bytes under-counted the
# true scratch by ~10x. At the old accounting a "512 MB" chunk of b1024
# compact weights expanded to ~4-5 GB of scratch -- and with the row-parallel
# groups not chunked at all, one OFT adapter load transiently consumed 7-9 GB.
# Colocated with a paused ~20 GB KV arena whose resume runs IMMEDIATELY after
# the update, that put free memory at the resume threshold and the arena
# came back up only by luck (measured: resume at 23.7 GB free succeeded on
# one rollout and OOM'd on the next).
_CAYLEY_LIVE_FULL_TENSORS = 5


def _resolve_oft_batch_chunk_limit_bytes() -> int:
    """Ceiling on the estimated per-chunk scratch, in bytes. 0 disables chunking.

    SGLANG_OFT_BATCH_CHUNK_MB bounds the WORKING SET of one precompute_oft_r
    call -- expanded skew-symmetric blocks and the Cayley intermediates -- as
    estimated by _oft_working_set_bytes. It does not bound the compact payload.
    """
    raw = os.getenv("SGLANG_OFT_BATCH_CHUNK_MB", "512").strip()
    try:
        value_mb = float(raw)
    except ValueError:
        value_mb = 512.0
    if value_mb <= 0:
        return 0
    return int(value_mb * (1 << 20))


def _oft_working_set_bytes(compact_weight: torch.Tensor, block_size: int) -> int:
    """Scratch one compact tensor costs inside precompute_oft_r.

    compact_weight is (num_blocks, n_elements); every full-block intermediate
    is (num_blocks, block_size, block_size) in the same dtype.
    """
    num_blocks = compact_weight.shape[0]
    full_block_bytes = block_size * block_size * compact_weight.element_size()
    return num_blocks * full_block_bytes * _CAYLEY_LIVE_FULL_TENSORS


def _flush_oft_group_chunk(
    memory_pool,
    buffer_id: int,
    block_size: int,
    target_device,
    group_items,
) -> None:
    from sglang.srt.oft._streamed_audit import record_dense_write
    from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

    normalized_items = []
    for item in group_items:
        if len(item) == 3:
            layer_id, fused_target, compact_weight = item
            normalized_items.append((layer_id, fused_target, compact_weight, None, 1))
        else:
            normalized_items.append(item)

    packed_weight = torch.cat(
        [
            compact_weight
            if compact_weight.device == target_device
            else compact_weight.to(target_device)
            for _, _, compact_weight, _, _ in normalized_items
        ],
        dim=0,
    )
    packed_r = precompute_oft_r(packed_weight, block_size)

    offset = 0
    for layer_id, fused_target, compact_weight, slice_index, split_count in normalized_items:
        next_offset = offset + compact_weight.shape[0]
        memory_pool._write_precomputed_oft_r(
            buffer_id,
            fused_target,
            layer_id,
            packed_r[offset:next_offset],
            block_size,
            slice_index=slice_index,
            split_count=split_count,
        )
        record_dense_write(
            fused_target,
            layer_id,
            packed_r[offset:next_offset],
            tp_rank=memory_pool.tp_rank,
        )
        offset = next_offset


def _resolve_streamed_oft_tensor_groups(
    oft_manager,
    named_tensors: List[Tuple[str, torch.Tensor]],
    block_size: int,
):
    """Resolve+validate a streamed OFT payload against the current model,
    WITHOUT touching any buffer slot (no buffer_id needed, nothing mutated).

    Split out of the combined write so the native multi-tenant admission
    path (OFTManager.load_adapter_from_tensors) can run this BEFORE
    evicting a resident adapter to make room: realistic failures (bad
    tensor names, target modules the current model doesn't have, a
    DSV4-style expert payload with no FusedMoE to absorb it) are caught
    here, while whatever adapter would otherwise be evicted is still
    intact. Returns (plan, error_message); plan is None iff error_message
    is set. plan is (fused_expert_chunk, non_row_groups, row_parallel_groups,
    direct_writes) for _commit_streamed_oft_tensor_groups.
    """
    from sglang.srt.layers.utils import get_layer_id

    memory_pool = oft_manager.memory_pool
    oft_modules = oft_manager.oft_modules

    # MoE expert OFT R cannot share the dense per-layer R_buffer slots
    # (those have no expert dimension and would silently overwrite each
    # other across experts). Peel expert names off the front and dispatch
    # them as a layer-grouped batch to the FusedMoE-aware writer; what
    # remains is dense attention / non-MoE MLP / embeddings, which the
    # existing `_resolve_oft_tensor_plan` path handles correctly (incl.
    # MLA q_a/q_b/kv_a/kv_b — those are dense names).
    fused_expert_chunk, dsv4_expert_chunk, named_tensors = (
        _partition_expert_oft_tensors(
            named_tensors,
            tp_rank=memory_pool.tp_rank,
        )
    )
    if dsv4_expert_chunk:
        has_dsv4_moe = False  # fork DeepSeekV4 model dropped (Task 2b)
        has_fused_moe = bool(oft_manager._find_fused_moe_modules())
        if has_fused_moe and not has_dsv4_moe:
            converted = _convert_dsv4_expert_chunk_to_fused(dsv4_expert_chunk)
            _merge_expert_oft_chunks(fused_expert_chunk, converted)
            dsv4_expert_chunk = {}
            if os.getenv("ORBIT_LOG_WEIGHT_SYNC", "").strip().lower() not in {
                "",
                "0",
                "false",
                "no",
            }:
                logger.info(
                    "Routed DSV4-style streamed expert OFT payload into "
                    "FusedMoE writer: layers=%s tensors=%s",
                    sorted(converted.keys()),
                    sum(
                        len(expert_weights)
                        for layer_chunk in converted.values()
                        for expert_weights in layer_chunk.values()
                    ),
                )
        if dsv4_expert_chunk:
            # Fork DeepSeekV4 model was dropped (Task 2b); DSV4-named
            # expert OFT adapters are converted onto FusedMoE above.
            # Reaching here means no FusedMoE was available to absorb
            # them -- a validation failure, not a write failure, so
            # detect it here rather than after any mutation.
            return None, (
                "DSV4-style expert OFT adapter has no FusedMoE target "
                "(fork DeepSeekV4 model support was removed)"
            )

    # CanonicalOFT: pre-stack per-slice q_proj/k_proj/v_proj (and gate/up)
    # tensors into a single fused ``qkv_proj.oft_R`` (and ``gate_up_proj.oft_R``)
    # so the existing dense dispatch path writes one stacked tensor per fused
    # buffer. Legacy shared-R names (single q_proj with bit-identical k/v)
    # pass through unchanged — they are handled by the duplicate-skip logic
    # downstream.
    from sglang.srt.oft.mem_pool import normalize_merged_oft_weights

    named_tensors_dict = dict(named_tensors)
    if len(named_tensors_dict) == len(named_tensors):
        named_tensors = list(
            normalize_merged_oft_weights(
                named_tensors_dict,
                available_fused_targets=set(memory_pool.R_buffer),
            ).items()
        )

    non_row_groups = {}
    row_parallel_groups = {}
    direct_writes = []
    unresolved_names = []

    for name, tensor in named_tensors:
        layer_id = get_layer_id(name)
        if layer_id is not None:
            try:
                fused_target, slice_module, is_row_parallel, slice_index, split_count = (
                    memory_pool._resolve_oft_tensor_plan(name, oft_modules, layer_id)
                )
            except (KeyError, ValueError, IndexError) as exc:
                unresolved_names.append(f"{name} ({exc})")
                continue

            target_device = memory_pool.R_buffer[fused_target][layer_id].device
            compact_weight = tensor

            if is_row_parallel:
                compact_weight = memory_pool._slice_oft_compact_weight(
                    compact_weight,
                    slice_module,
                )
                group_key = (
                    target_device,
                    fused_target,
                    tuple(compact_weight.shape[1:]),
                    compact_weight.dtype,
                )
                row_parallel_groups.setdefault(group_key, []).append(
                    (layer_id, fused_target, compact_weight, slice_index, split_count)
                )
            else:
                group_key = (
                    target_device,
                    fused_target,
                    tuple(compact_weight.shape[1:]),
                    compact_weight.dtype,
                )
                non_row_groups.setdefault(group_key, []).append(
                    (layer_id, fused_target, compact_weight, slice_index, split_count)
                )
        elif "embed_tokens" in name or "lm_head" in name:
            # Deferred, not written here: this function must not mutate any
            # buffer slot (see docstring) -- _commit_streamed_oft_tensor_groups
            # performs the actual write once a buffer_id exists.
            direct_writes.append((name, tensor))
        elif ".oft_" in name or name.endswith(".oft_R"):
            unresolved_names.append(name)

    if unresolved_names:
        shown = ", ".join(unresolved_names[:8])
        more = (
            ""
            if len(unresolved_names) <= 8
            else f", ... (+{len(unresolved_names) - 8} more)"
        )
        return None, f"Unresolved OFT tensor names: {shown}{more}"

    return (fused_expert_chunk, non_row_groups, row_parallel_groups, direct_writes), ""


def _commit_streamed_oft_tensor_groups(
    oft_manager,
    named_tensors: List[Tuple[str, torch.Tensor]],
    plan,
    buffer_id: int,
    block_size: int,
    oft_name: str,
    oft_id: str | None,
) -> tuple[bool, str]:
    """Write an already-resolved plan (see _resolve_streamed_oft_tensor_groups)
    into buffer_id's R_buffer slots. named_tensors is the ORIGINAL raw
    payload, needed only for the diagnostic ORBIT_LOG_WEIGHT_SYNC summary
    below -- the actual write uses `plan`, not `named_tensors`.
    """
    fused_expert_chunk, non_row_groups, row_parallel_groups, direct_writes = plan
    memory_pool = oft_manager.memory_pool
    oft_modules = oft_manager.oft_modules

    if os.getenv("ORBIT_LOG_WEIGHT_SYNC", "").strip().lower() not in {"", "0", "false", "no"}:
        samples = []
        max_abs = 0.0
        total_nonzero = 0
        for name, tensor in named_tensors:
            if not torch.is_tensor(tensor):
                continue
            detached = tensor.detach()
            cur_max = float(detached.float().abs().max().item()) if detached.numel() else 0.0
            cur_mean = float(detached.float().abs().mean().item()) if detached.numel() else 0.0
            cur_nonzero = int((detached != 0).sum().item())
            max_abs = max(max_abs, cur_max)
            total_nonzero += cur_nonzero
            if len(samples) < 8:
                samples.append(
                    f"{name}:shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"max={cur_max:.3e} mean={cur_mean:.3e} nonzero={cur_nonzero}"
                )
        logger.info(
            "OFT streamed payload adapter=%s oft_id=%s buffer_id=%s "
            "tensor_count=%s max_abs=%.6e total_nonzero=%s samples=%s",
            oft_name,
            oft_id,
            buffer_id,
            len(named_tensors),
            max_abs,
            total_nonzero,
            samples,
        )

    for name, tensor in direct_writes:
        memory_pool.load_oft_weight_direct(
            buffer_id, name, tensor, block_size, oft_modules, 0
        )

    if fused_expert_chunk:
        oft_manager.apply_streamed_expert_oft(
            fused_expert_chunk, block_size, slot_idx=buffer_id
        )

    batch_chunk_limit_bytes = _resolve_oft_batch_chunk_limit_bytes()
    # Row-parallel groups go through the SAME chunked flush as everything else.
    # They used to be flushed whole, which made the limit meaningless exactly
    # where it mattered most: fc2/down_proj is row-parallel and carries the
    # most blocks of any module in the model, so the largest group was the one
    # group the limiter never saw.
    for group_key, group_items in list(non_row_groups.items()) + list(
        row_parallel_groups.items()
    ):
        _flush_oft_group_in_chunks(
            memory_pool,
            buffer_id,
            block_size,
            group_key[0],
            group_items,
            batch_chunk_limit_bytes,
        )

    # Return the Cayley scratch to the driver, not just to this process's
    # allocator cache. The very next thing the training loop does after an
    # adapter update is resume the paused KV-cache arena, and torch_memory_saver
    # re-creates it with raw cuMemCreate -- which the caching allocator's
    # retained high-water mark can starve. empty_cache() releases only unused
    # cached blocks, so nothing live is touched.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True, "Success"


def _flush_oft_group_in_chunks(
    memory_pool,
    buffer_id: int,
    block_size: int,
    target_device,
    group_items,
    batch_chunk_limit_bytes: int,
) -> None:
    """Flush one fused-target group in working-set-bounded chunks."""
    chunk_items = []
    chunk_bytes = 0
    for item in group_items:
        working_bytes = _oft_working_set_bytes(item[2], block_size)
        if (
            batch_chunk_limit_bytes > 0
            and chunk_items
            and chunk_bytes + working_bytes > batch_chunk_limit_bytes
        ):
            _flush_oft_group_chunk(
                memory_pool,
                buffer_id,
                block_size,
                target_device,
                chunk_items,
            )
            chunk_items = []
            chunk_bytes = 0

        chunk_items.append(item)
        chunk_bytes += working_bytes

    if chunk_items:
        _flush_oft_group_chunk(
            memory_pool,
            buffer_id,
            block_size,
            target_device,
            chunk_items,
        )

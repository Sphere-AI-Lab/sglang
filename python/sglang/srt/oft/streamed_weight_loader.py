from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Literal, Sequence, Tuple

import torch

from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

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


def get_tensor_alias_key(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        storage.data_ptr(),
    )


def dedupe_named_tensors_by_storage(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, int]]]:
    unique_named_tensors: list[tuple[str, torch.Tensor]] = []
    entries: list[tuple[str, int]] = []
    key_to_index: dict[tuple, int] = {}

    for name, tensor in named_tensors:
        alias_key = get_tensor_alias_key(tensor)
        unique_index = key_to_index.get(alias_key)
        if unique_index is None:
            unique_index = len(unique_named_tensors)
            key_to_index[alias_key] = unique_index
            unique_named_tensors.append((name, tensor))
        entries.append((name, unique_index))

    return unique_named_tensors, entries


def serialize_flattened_oft_payload(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
) -> bytes:
    unique_named_tensors, entries = dedupe_named_tensors_by_storage(named_tensors)
    flattened_bucket = FlattenedTensorBucket(named_tensors=list(unique_named_tensors))
    payload: FlattenedOFTTensorPayload = (
        "flattened_oft_payload",
        MultiprocessingSerializer.serialize(
            flattened_bucket.get_flattened_tensor().detach()
        ),
        flattened_bucket.get_metadata(),
        entries,
    )
    return MultiprocessingSerializer.serialize(payload)


def normalize_oft_weight_payload(
    payload: FlattenedOFTTensorPayload,
    *,
    device,
) -> list[tuple[str, torch.Tensor]]:
    """Deserialize a flattened OFT payload to a list of (name, tensor).

    OFT sync (orbit, verl) always sends the flattened bucket form built by
    `serialize_flattened_oft_payload`. The legacy non-flattened path was
    never exercised in production and has been removed.
    """
    assert (
        isinstance(payload, tuple)
        and len(payload) == 4
        and payload[0] == "flattened_oft_payload"
    ), "OFT update_weights_from_tensor expects a FlattenedOFTTensorPayload"
    _, serialized_flattened_tensor, metadata, entries = payload
    flattened_tensor = MultiprocessingSerializer.deserialize(
        serialized_flattened_tensor
    ).to(device)
    bucket = FlattenedTensorBucket(
        flattened_tensor=flattened_tensor,
        metadata=metadata,
    )
    unique_named_tensors = bucket.reconstruct_tensors()
    unique_tensors = [tensor for _, tensor in unique_named_tensors]
    return [
        (name, unique_tensors[unique_index])
        for name, unique_index in entries
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


def _resolve_oft_batch_chunk_limit_bytes() -> int:
    raw = os.getenv("SGLANG_OFT_BATCH_CHUNK_MB", "512").strip()
    try:
        value_mb = float(raw)
    except ValueError:
        value_mb = 512.0
    if value_mb <= 0:
        return 0
    return int(value_mb * (1 << 20))


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
            tp_rank=getattr(memory_pool, "tp_rank", None),
        )
        offset = next_offset


def _ensure_streaming_oft_adapter_slot(
    model_runner,
    adapter_config: dict,
    adapter_name: str,
    adapter_id: str | None,
) -> tuple[int, int]:
    if (
        hasattr(model_runner, "_oft_streaming_buffer_id")
        and model_runner._oft_streaming_name == adapter_name
    ):
        return (
            model_runner._oft_streaming_buffer_id,
            model_runner._oft_streaming_block_size,
        )

    existing_id = None
    for oft_id, ref in list(model_runner.oft_manager.oft_refs.items()):
        if ref.oft_name == adapter_name:
            existing_id = oft_id
            break
    if existing_id is not None:
        old_ref = model_runner.oft_manager.oft_refs[existing_id]
        model_runner.oft_manager.unload_streamed_adapter(old_ref)

    buffer_id = model_runner.oft_manager.memory_pool.allocate_buffer_slot()
    model_runner.oft_manager.memory_pool.reset_buffer_slot_to_identity(buffer_id)

    oft_ref = OFTRef(
        oft_id=adapter_id,
        oft_name=adapter_name,
        oft_path=adapter_name,
        pinned=False,
    )
    result = model_runner.oft_manager.register_streamed_adapter(
        oft_ref, buffer_id, adapter_config
    )
    if not result.success:
        raise RuntimeError(
            f"Failed to register OFT adapter: {result.error_message}"
        )

    block_size = adapter_config.get("oft_block_size", 32)
    model_runner._oft_streaming_buffer_id = buffer_id
    model_runner._oft_streaming_name = adapter_name
    model_runner._oft_streaming_block_size = block_size
    return buffer_id, block_size


def load_streamed_oft_adapter(
    model_runner,
    named_tensors: List[Tuple[str, torch.Tensor]],
    adapter_config: dict,
    adapter_name: str,
    adapter_id: str | None = None,
) -> tuple[bool, str]:
    from sglang.srt.layers.utils import get_layer_id

    assert adapter_config is not None, "adapter_config is required for oft_adapter"
    assert adapter_name is not None, "adapter_name is required for oft_adapter"

    try:
        buffer_id, block_size = _ensure_streaming_oft_adapter_slot(
            model_runner,
            adapter_config,
            adapter_name,
            adapter_id,
        )
    except RuntimeError as exc:
        return False, str(exc)

    memory_pool = model_runner.oft_manager.memory_pool
    oft_modules = model_runner.oft_manager.oft_modules
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
            "OFT streamed payload adapter=%s adapter_id=%s buffer_id=%s "
            "tensor_count=%s max_abs=%.6e total_nonzero=%s samples=%s",
            adapter_name,
            adapter_id,
            buffer_id,
            len(named_tensors),
            max_abs,
            total_nonzero,
            samples,
        )

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
            tp_rank=getattr(memory_pool, "tp_rank", None),
        )
    )
    if dsv4_expert_chunk:
        oft_manager = model_runner.oft_manager
        has_dsv4_moe = bool(oft_manager._find_dsv4_moe_modules())
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
            memory_pool.load_oft_weight_direct(
                buffer_id, name, tensor, block_size, oft_modules, 0
            )
        elif ".oft_" in name or name.endswith(".oft_R"):
            unresolved_names.append(name)

    if unresolved_names:
        shown = ", ".join(unresolved_names[:8])
        more = (
            ""
            if len(unresolved_names) <= 8
            else f", ... (+{len(unresolved_names) - 8} more)"
        )
        return False, f"Unresolved OFT tensor names: {shown}{more}"

    if fused_expert_chunk:
        model_runner.oft_manager.apply_streamed_expert_oft(
            fused_expert_chunk, block_size
        )
    if dsv4_expert_chunk:
        model_runner.oft_manager.apply_streamed_dsv4_expert_oft(
            dsv4_expert_chunk, block_size
        )

    batch_chunk_limit_bytes = _resolve_oft_batch_chunk_limit_bytes()
    for group_key, group_items in non_row_groups.items():
        target_device = group_key[0]
        chunk_items = []
        chunk_bytes = 0

        for item in group_items:
            compact_weight = item[2]
            compact_bytes = compact_weight.numel() * compact_weight.element_size()
            if (
                batch_chunk_limit_bytes > 0
                and chunk_items
                and chunk_bytes + compact_bytes > batch_chunk_limit_bytes
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
            chunk_bytes += compact_bytes

        if chunk_items:
            _flush_oft_group_chunk(
                memory_pool,
                buffer_id,
                block_size,
                target_device,
                chunk_items,
            )

    for group_key, group_items in row_parallel_groups.items():
        _flush_oft_group_chunk(
            memory_pool,
            buffer_id,
            block_size,
            group_key[0],
            group_items,
        )

    return True, "Success"

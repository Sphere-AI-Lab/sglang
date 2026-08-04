"""LoRA streamed weight-sync (CUDA IPC single-slot), mirroring peft/oft.

The wire payload is structurally identical to peft/oft's FlattenedOFTTensorPayload
(only the tag string differs), so orbit's IPC sender reuses its OFT serializer with
a "lora" tag. This module is the RECEIVER contract.
"""
import logging
import re
from typing import List, Literal, Sequence, Tuple

import torch

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.peft.oft.streamed_weight_loader import (
    dedupe_named_tensors_by_storage,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

FlattenedLoRATensorPayload = Tuple[Literal["flattened_lora_payload"], bytes, list, list]

logger = logging.getLogger(__name__)

_EXPERT_LORA_RE = re.compile(
    r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)"
)


def _partition_expert_lora_tensors(named_tensors):
    """Split (name, tensor) into (expert_layer_dict, dense_named_tensors).
    expert_layer_dict: {layer_id: {expert_id: {"{proj}.lora_{A|B}": tensor}}}.
    """
    expert_layer_dict = {}
    dense = []
    for name, tensor in named_tensors:
        m = _EXPERT_LORA_RE.search(name)
        layer_id = get_layer_id(name) if m is not None else None
        if m is None or layer_id is None:
            dense.append((name, tensor))
            continue
        expert_id = int(m.group(1))
        key = f"{m.group(2)}.lora_{m.group(3)}"
        expert_layer_dict.setdefault(layer_id, {}).setdefault(expert_id, {})[key] = tensor
    return expert_layer_dict, dense


def serialize_flattened_lora_payload(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
) -> bytes:
    unique_named_tensors, entries = dedupe_named_tensors_by_storage(named_tensors)
    flattened_bucket = FlattenedTensorBucket(named_tensors=list(unique_named_tensors))
    payload: FlattenedLoRATensorPayload = (
        "flattened_lora_payload",
        MultiprocessingSerializer.serialize(
            flattened_bucket.get_flattened_tensor().detach()
        ),
        flattened_bucket.get_metadata(),
        entries,
    )
    return MultiprocessingSerializer.serialize(payload)


def normalize_lora_weight_payload(
    payload: FlattenedLoRATensorPayload,
    *,
    device,
) -> List[Tuple[str, torch.Tensor]]:
    assert (
        isinstance(payload, tuple)
        and len(payload) == 4
        and payload[0] == "flattened_lora_payload"
    ), "LoRA update_weights_from_tensor expects a FlattenedLoRATensorPayload"
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


def _scaling_from_config(adapter_config, model_runner) -> float:
    if adapter_config and "lora_alpha" in adapter_config and "r" in adapter_config:
        return float(adapter_config["lora_alpha"]) / float(adapter_config["r"])
    active = model_runner.peft_lora_manager._active_adapter()
    assert active is not None, (
        "streamed LoRA sync needs an initial adapter (or an adapter_config with "
        "lora_alpha/r) to determine scaling"
    )
    return active.scaling


def load_streamed_lora_adapter(
    model_runner,
    named_tensors: List[Tuple[str, torch.Tensor]],
    adapter_config: dict,
    adapter_name: str,
    adapter_id: str = None,
) -> Tuple[bool, str]:
    """Single-active CUDA IPC streamed LoRA update: copy_ new A/B into the
    already-wrapped layers. Requires an initial adapter established at launch
    (MVP; wrap-only populate-by-sync is a documented fast-follow).

    For the expert path: see the full-expert-set scatter invariant documented
    on `LoRAManager.apply_streamed_expert_lora` -- a streamed expert payload
    must carry the full expert set the initial adapter established, or absent
    experts silently retain their prior (stale) values."""
    assert adapter_name is not None, "adapter_name is required for lora_adapter"
    try:
        scaling = _scaling_from_config(adapter_config, model_runner)
        expert_layer_dict, dense = _partition_expert_lora_tensors(named_tensors)
        model_runner.peft_lora_manager.apply_streamed_update(dense, scaling)
        if expert_layer_dict:
            model_runner.peft_lora_manager.apply_streamed_expert_lora(
                expert_layer_dict, scaling
            )
        return True, ""
    except Exception as exc:  # converts any failure into the (bool, str) result
        # the RPC contract expects; the caller (model_runner.update_weights_
        # from_tensor) does NOT wrap this call in its own try/except.
        logger.exception("streamed LoRA update failed for adapter=%s", adapter_name)
        return False, str(exc)

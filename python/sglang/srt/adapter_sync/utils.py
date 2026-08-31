"""Helpers the shared adapter core needs, kept here so ``srt/adapter_sync``
does not import any single method's package."""

from typing import Set

import torch

from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:
    """Return the entry of ``target_modules`` that matches ``full_module_name``.

    Copied from ``srt/oft/utils.py`` (WS2-1): the shared core must not depend on
    ``srt/oft``, or a LoRA-only deployment would drag the OFT package in.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )


def reconstruct_adapter_staging(
    staging: list[tuple[str, torch.Tensor]],
    payload_metadata: dict,
) -> list[tuple[str, torch.Tensor]]:
    """Reconstruct a flattened adapter payload received over NCCL."""
    assert len(staging) == 1 and staging[0][0] == "__flattened__", (
        "adapter payload_metadata expects exactly one '__flattened__' tensor, "
        f"got names={[name for name, _ in staging]}"
    )
    flattened_tensor = staging[0][1]
    metadata = [
        FlattenedTensorMetadata(
            name=item["name"],
            shape=torch.Size(item["shape"]),
            dtype=getattr(torch, item["dtype"]),
            start_idx=item["start_idx"],
            end_idx=item["end_idx"],
            numel=item["numel"],
        )
        for item in payload_metadata["metadata"]
    ]
    bucket = FlattenedTensorBucket(
        flattened_tensor=flattened_tensor,
        metadata=metadata,
    )
    unique_tensors = [tensor for _, tensor in bucket.reconstruct_tensors()]
    return [
        (name, unique_tensors[int(index)])
        for name, index in payload_metadata["extra"]["entries"]
    ]

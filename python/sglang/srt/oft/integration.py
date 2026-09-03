"""OFT integration façade — the thin seam ``model_runner.py`` calls through
for the OFT-manager lifecycle (init), plus ``reconstruct_oft_staging``, a
wire-format helper shared by native-LoRA-staged, OFT-staged, and OFT-sibling
NCCL adapter payloads (see model_runner_components/weight_updater.py).

``model_runner.py`` imports this module as ``oft_integration`` and makes a
thin guarded call-out (``oft_integration.maybe_init_oft_manager(self,
server_args)``) so upstream rebases stay cheap -- ``model_runner.py`` is a
frozen orchestration-only file (see the ``large-class-style`` skill), so the
manager-construction logic that LoRA's own (older, non-conformant)
``init_lora_manager`` inlines directly cannot live there.

Every OTHER caller whose LoRA equivalent lives inline (not behind a separate
module) does the same for OFT instead of routing through here, matching
LoRA's own code shape exactly: schedule_batch.py's ``_extend_oft_extra_key``
(next to ``_extend_lora_extra_key``), scheduler.py's
``Scheduler._can_schedule_oft_req`` (next to ``_can_schedule_lora_req``),
forward_batch_info.py's inline ``ForwardBatch.init_new`` block (next to the
``enable_lora`` block), the cuda-graph runners' direct
``model_runner.oft_manager`` calls (next to their ``lora_manager`` calls,
with ``DecodeCudaGraphRunner._prepare_oft_replay_batch`` the one exception --
its multi-step temporary-swap logic has no LoRA equivalent, since LoRA's
``lora_ids`` restores generically via ``buffer_registry.fill_from`` while
``adapter_ids`` isn't a registered buffer field), and
model_runner_components/weight_updater.py's direct
``model_runner.oft_manager.stage_adapter``/``activate_adapter`` calls (next
to the native-LoRA and OFT-staged branches).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

__all__ = [
    "maybe_init_oft_manager",
    "reconstruct_oft_staging",
]


def maybe_init_oft_manager(
    model_runner: "ModelRunner", server_args: "ServerArgs"
) -> None:
    """Single OFT init seam ``model_runner.py`` calls: build the adapter
    manager when ``server_args.enable_oft`` is set, building an OFTManager on
    ``model_runner.oft_manager``. No-op otherwise.
    """
    if server_args.enable_oft:
        _init_oft_manager(model_runner, server_args)


def _get_oft_manager_class(server_args: "ServerArgs"):
    """Resolve the OFTManager class for ``server_args.oft_impl``, mirroring
    ``ModelRunner._get_lora_manager_class``'s pattern for OFT's two choices.

    Imported lazily: OFTManager pulls in vocab_parallel_embedding ->
    communicator -> forward_batch_info, which forms an import cycle when a
    module in that chain imports this façade at module scope (e.g.
    forward_batch_info). Deferring keeps ``oft.integration`` light to import.
    """
    if server_args.oft_impl == "staged":
        from sglang.srt.oft.staged_manager import StagedOFTManager

        return StagedOFTManager
    from sglang.srt.oft.oft_manager import OFTManager

    return OFTManager


def _init_oft_manager(model_runner: "ModelRunner", server_args: "ServerArgs") -> None:
    """Body of the former ``ModelRunner.init_oft_manager``."""
    OFTManager = _get_oft_manager_class(server_args)

    # Runtime witness: every boot names the stack that actually serves (sibling
    # vs staged), so "which implementation ran" is in the log, not inferred.
    logger.info(
        "OFT implementation: %s (oft_impl=%s)",
        OFTManager.__module__,
        server_args.oft_impl,
    )

    model_runner.oft_manager = OFTManager(
        base_model=model_runner.model,
        base_hf_config=model_runner.model_config.hf_config,
        max_ofts_per_batch=model_runner.server_args.max_ofts_per_batch,
        load_config=model_runner.load_config,
        dtype=model_runner.dtype,
        server_args=model_runner.server_args,
        oft_backend=model_runner.server_args.oft_backend,
        tp_size=model_runner.ps.tp_size,
        tp_rank=model_runner.ps.tp_rank,
        max_oft_block_size=model_runner.server_args.max_oft_block_size,
        target_modules=model_runner.server_args.oft_target_modules,
        memory_saver_adapter=model_runner.memory_saver_adapter,
        memory_saver_cpu_backup=model_runner.server_args.enable_weights_cpu_backup,
    )


def reconstruct_oft_staging(
    staging: list[tuple[str, torch.Tensor]],
    payload_metadata: dict,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a flattened NCCL OFT tensor back to per-weight tensors.

    Wire format (canonical — orbit NcclBackend always pre-serializes via
    _flatten_meta_to_json before calling update_adapter_from_distributed):
      payload_metadata = {
          "metadata": [
              {"name": str, "shape": [int, ...], "dtype": str,
               "start_idx": int, "end_idx": int, "numel": int},
              ...
          ],
          "extra": {"entries": [[name_str, unique_index_int], ...]},
      }
    All values are plain JSON primitives; torch.dtype / torch.Size are
    never present on the wire.
    """
    assert len(staging) == 1 and staging[0][0] == "__flattened__", (
        "OFT oft_adapter with payload_metadata expects exactly one "
        f"'__flattened__' tensor, got names={[n for n,_ in staging]}"
    )
    flattened_tensor = staging[0][1]

    raw_metadata = payload_metadata["metadata"]
    entries = payload_metadata["extra"]["entries"]  # list[[str, int]]

    # Rebuild FlattenedTensorMetadata from JSON-deserialized dicts.
    # dtype is a plain string (e.g. "float32"), shape is a list of ints.
    reconstructed_meta = [
        FlattenedTensorMetadata(
            name=m["name"],
            shape=torch.Size(m["shape"]),
            dtype=getattr(torch, m["dtype"]),
            start_idx=m["start_idx"],
            end_idx=m["end_idx"],
            numel=m["numel"],
        )
        for m in raw_metadata
    ]

    bucket = FlattenedTensorBucket(
        flattened_tensor=flattened_tensor,
        metadata=reconstructed_meta,
    )
    unique_named_tensors = bucket.reconstruct_tensors()
    unique_tensors = [t for _, t in unique_named_tensors]
    # entries is list of [name, unique_index] (JSON arrays become lists).
    return [(name, unique_tensors[int(idx)]) for name, idx in entries]

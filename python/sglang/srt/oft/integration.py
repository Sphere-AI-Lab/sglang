"""OFT integration façade — the thin seam ``model_runner.py`` calls through.

Owns the OFT-manager lifecycle (init/load/unload). Task 6 imports this module
as ``oft`` and keeps the model runner's call-outs thin, while this provider
owns the canonical adapter lifecycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.streamed_weight_loader import (
    FlattenedOFTTensorPayload,
)
from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.managers.forward_batch_info import ForwardBatch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

__all__ = [
    "FlattenedOFTTensorPayload",
    "OFTRef",
    "NOT_HANDLED",
    "maybe_init_oft_manager",
    "reconstruct_oft_staging",
    "maybe_load_adapter",
    "maybe_load_adapter_from_tensors",
    "maybe_load_adapter_from_distributed",
    "maybe_unload_adapter",
    "maybe_dummy_ids",
    "maybe_prepare_oft_batch",
    "maybe_admit_request",
    "maybe_extend_extra_key",
    "maybe_init_cuda_graph_batch_info",
    "maybe_prepare_replay_batch",
    "maybe_apply_forward",
    "stage_adapter",
    "activate_adapter",
]

# Sentinel returned when a weight update is not an OFT adapter payload.
NOT_HANDLED = object()


def maybe_init_oft_manager(
    model_runner: "ModelRunner", server_args: "ServerArgs"
) -> None:
    """Build the canonical staged-capable OFT manager when OFT is enabled."""
    if server_args.peft_method == "oft":
        _init_oft_manager(model_runner, server_args)


def _init_oft_manager(model_runner: "ModelRunner", server_args: "ServerArgs") -> None:
    """Body of the former ``ModelRunner.init_oft_manager``."""
    # Imported lazily to avoid the model-runner/forward-batch import cycle.
    from sglang.srt.oft.staged_manager import StagedOFTManager

    logger.info("OFT implementation: %s", StagedOFTManager.__module__)
    model_runner.oft_manager = StagedOFTManager(
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
        target_modules=model_runner.server_args.peft_target_modules,
        adapter_paths=model_runner.server_args.peft_paths,
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


def stage_adapter(
    model_runner: "ModelRunner",
    load_format,
    tensors,
    adapter_config: Optional[dict],
    adapter_name: Optional[str],
    adapter_id: Optional[str],
    version,
    *,
    payload_metadata: Optional[dict] = None,
    double_buffer: bool = True,
):
    """Canonical OFT STAGING fill. Reconstructs the NCCL wire payload, then
    calls the resolved manager's ``stage_adapter``. Returns ``NOT_HANDLED``
    for non-OFT payloads.

    ``double_buffer=False`` (distributed sync without ``--adapter-double-
    buffer``) is rejected for OFT: with DB-off sizing the pool inherits the
    ``AdapterMemPool`` base defaults (active_idx=0, staging_idx=1), and
    ``_acquire_buffer_slot`` gives the base-identity placeholder slot 0
    (==active_idx) and the live per-token adapter gather slot 1
    (==staging_idx). ``stage()`` correctly fills slot 1 (the adapter's own
    gather slot), but ``activate()`` unconditionally copies every group's
    staging_idx->active_idx (slot1->slot0) -- CLOBBERING the base-identity
    slot with adapter data instead of updating the adapter in place (the
    runtime already reads the adapter's weights straight from slot 1; there
    is no correct use for copying into slot 0).
    """
    if payload_metadata is not None:
        tensors = reconstruct_oft_staging(tensors, payload_metadata)

    if load_format == "oft_adapter":
        if not double_buffer:
            raise ValueError(
                "distributed non-double-buffer OFT adapter sync via "
                "stage/activate is not supported; enable "
                "--adapter-double-buffer (double-buffer) or use the "
                "IPC/colocate weight-sync."
            )
        return model_runner.oft_manager.stage_adapter(
            tensors, adapter_config, adapter_name, version, adapter_id=adapter_id
        )

    return NOT_HANDLED


def activate_adapter(
    model_runner: "ModelRunner", adapter_name: str, version, adapter_id=None
):
    """Activate the staged OFT identity, or return ``NOT_HANDLED`` if disabled."""
    if model_runner.server_args.peft_method == "oft":
        return model_runner.oft_manager.activate_adapter(
            adapter_name, version, adapter_id=adapter_id
        )
    return NOT_HANDLED


def maybe_load_adapter(model_runner: "ModelRunner", oft_ref: "OFTRef"):
    """Body of the former ``ModelRunner.load_oft_adapter``."""
    logger.info(
        f"OFT adapter loading starts: {oft_ref}. "
        f"avail mem={get_available_gpu_memory(model_runner.device, model_runner.gpu_id):.2f} GB"
    )

    result = model_runner.oft_manager.load_oft_adapter(oft_ref)

    logger.info(
        f"OFT adapter loading completes: {oft_ref}. "
        f"avail mem={get_available_gpu_memory(model_runner.device, model_runner.gpu_id):.2f} GB"
    )

    return result


def maybe_load_adapter_from_tensors(
    model_runner: "ModelRunner",
    oft_ref: "OFTRef",
    tensors,
    config_dict,
    *,
    upsert: bool = False,
):
    """Body of the former ``ModelRunner.load_oft_adapter_from_tensors``."""
    logger.info(f"OFT adapter loading from tensors starts: {oft_ref}.")
    result = model_runner.oft_manager.load_adapter_from_tensors(
        oft_ref, tensors, config_dict, upsert=upsert
    )
    logger.info(f"OFT adapter loading from tensors completes: {oft_ref}.")
    return result


def maybe_load_adapter_from_distributed(
    model_runner: "ModelRunner",
    oft_ref: "OFTRef",
    names,
    dtypes,
    shapes,
    config_dict,
    group_name,
    *,
    upsert: bool = False,
):
    """Load native OFT tensors received by the model runner's updater."""
    logger.info(f"OFT adapter loading from distributed starts: {oft_ref}.")
    result = model_runner.oft_manager.load_adapter_from_distributed(
        oft_ref,
        names,
        dtypes,
        shapes,
        config_dict,
        group_name,
        model_runner.weight_updater,
        upsert=upsert,
    )
    logger.info(f"OFT adapter loading from distributed completes: {oft_ref}.")
    return result


def maybe_unload_adapter(model_runner: "ModelRunner", oft_ref: "OFTRef"):
    """Body of the former ``ModelRunner.unload_oft_adapter``."""
    logger.info(
        f"OFT adapter unloading starts: {oft_ref}. "
        f"avail mem={get_available_gpu_memory(model_runner.device, model_runner.gpu_id):.2f} GB"
    )

    result = model_runner.oft_manager.unload_oft_adapter(oft_ref)

    logger.info(
        f"OFT adapter unloading completes: {oft_ref}. "
        f"avail mem={get_available_gpu_memory(model_runner.device, model_runner.gpu_id):.2f} GB"
    )

    return result


def maybe_dummy_ids(server_args: "ServerArgs", batch_size: int):
    """Returns ``[None] * batch_size`` if OFT is enabled, else ``None``."""
    if server_args.peft_method == "oft":
        return [None] * batch_size
    return None


def maybe_prepare_oft_batch(
    model_runner: "ModelRunner", forward_batch: "ForwardBatch"
) -> None:
    """Prepare OFT batch metadata for graph capture or eager execution."""
    if (
        model_runner.server_args.peft_method == "oft"
        and forward_batch.adapter_ids is not None
    ):
        model_runner.oft_manager.prepare_oft_batch(forward_batch)


def maybe_admit_request(scheduler: "Scheduler", req: "Req", running_ofts) -> bool:
    """Body of the former inline OFT admission check in ``Scheduler``.

    Returns True to admit the request, False if the caller should ``continue``.
    """
    if scheduler.oft_drainer and not scheduler.oft_drainer.can_schedule(req):
        return False

    new_oft_set = {req.adapter_id} | running_ofts
    return scheduler.tp_worker.model_runner.oft_manager.validate_oft_batch(
        new_oft_set
    )


def maybe_extend_extra_key(extra_key, adapter_id, adapter_version) -> str:
    """Body of the former inline OFT extra_key extension in ``Req.__init__``."""
    if adapter_id is not None:
        extra_key = (extra_key or "") + f"|oft:{adapter_id}:v{0 if adapter_version is None else adapter_version}"
    return extra_key


def maybe_init_cuda_graph_batch_info(
    model_runner: "ModelRunner", max_bs: int, num_tokens_per_bs: int
) -> None:
    """Body of the former inline OFT cuda-graph batch-info init in ``CudaGraphRunner``."""
    if model_runner.server_args.peft_method == "oft":
        model_runner.oft_manager.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=max_bs, num_tokens_per_bs=num_tokens_per_bs
        )


def maybe_prepare_replay_batch(
    model_runner: "ModelRunner", forward_batch: "ForwardBatch", bs: int, raw_bs: int
) -> None:
    """Body of the former inline OFT replay-batch prep in ``CudaGraphRunner``."""
    if model_runner.server_args.peft_method == "oft" and forward_batch.adapter_ids is not None:
        original_batch_size = forward_batch.batch_size
        original_oft_ids = forward_batch.adapter_ids
        forward_batch.batch_size = bs
        forward_batch.adapter_ids = original_oft_ids + [None] * (bs - raw_bs)
        model_runner.oft_manager.prepare_oft_batch(forward_batch)
        forward_batch.batch_size = original_batch_size
        forward_batch.adapter_ids = original_oft_ids


def maybe_apply_forward(model_runner: "ModelRunner", forward_batch: "ForwardBatch") -> None:
    """Body of the former inline OFT apply in ``ForwardBatch.init_new``."""
    if model_runner.server_args.peft_method == "oft":
        model_runner.oft_manager.fetch_new_ofts(set(forward_batch.adapter_ids))
        model_runner.oft_manager.prepare_oft_batch(forward_batch)

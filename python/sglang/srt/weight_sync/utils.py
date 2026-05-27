from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.oft.streamed_weight_loader import serialize_flattened_oft_payload
from sglang.srt.utils import MultiprocessingSerializer


async def update_weights(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
):
    """
    Update weights for the inference engine.
    This function is designed to be stateless, so that the caller process could keep the stateful engine.
    Example Use Case:
        - Multiple Producer Process will call this function in a SPMD style

    Args:
        engine: The inference engine created by the caller process.
        params_batch: A list of (name, tensor) tuples. We batched the tensors to avoid the overhead of cpu call.
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp"
        device_mesh: The device mesh.
        load_format: The format of the weights.
    """
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    # [
    #   (name0, ipc_tensor0_tp0),
    #   (name1, ipc_tensor1_tp0),
    # ]
    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(
                _preprocess_tensor_for_update_weights(tensor.detach())
            ),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    # [
    #   [ (name0, ipc_tensor0_tp0), (name1, ipc_tensor1_tp0) ],
    #   [ (name0, ipc_tensor0_tp1), (name1, ipc_tensor1_tp1) ],
    # ]
    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        # Use zip(*) to "transpose" the data structure.
        # After transpose, the data structure is like:
        # [
        #   ( (name0, ipc_tensor0_tp0), (name0, ipc_tensor0_tp1) ),
        #   ( (name1, ipc_tensor1_tp0), (name1, ipc_tensor1_tp1) ),
        # ]
        logical_tensors = zip(*gathered_serialized_batches, strict=True)

        named_tensors = [
            # [
            #   (name0, LocalSerializedTensor(values=[ipc_tensor0_tp0, ipc_tensor0_tp1])),
            #   (name1, LocalSerializedTensor(values=[ipc_tensor1_tp0, ipc_tensor1_tp1])),
            # ]
            (
                tensor_group[0][0],
                LocalSerializedTensor(
                    values=[rank_part[1] for rank_part in tensor_group]
                ),
            )
            for tensor_group in logical_tensors
        ]

        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(infer_tp_size)
            ],
            load_format=load_format,
        )

        return await engine.update_weights_from_tensor(update_weights_request)


async def update_oft_weights(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    adapter_config: dict,
    adapter_name: str,
):
    """
    Update OFT adapter weights using the same CUDA IPC gather mechanism as full fine-tuning.

    Each bucket of tensors is gathered via NCCL, then written directly into the
    pre-allocated GPU R_buffer on the sglang side. No CPU storage, no accumulation,
    no separate finalize step — identical flow to full fine-tuning.

    Args:
        engine: The inference engine.
        params_batch: OFT adapter tensors as (name, tensor) tuples for this bucket.
        device_mesh_key: Typically "infer_tp".
        device_mesh: The device mesh.
        adapter_config: OFT config dict (from peft OFTConfig).
        adapter_name: Name for the OFT adapter in sglang.
    """
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    serialized_payload = serialize_flattened_oft_payload(
        [
            (name, _preprocess_tensor_for_update_weights(tensor.detach()))
            for name, tensor in params_batch
        ]
    )

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    dist.gather_object(
        obj=serialized_payload,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=gathered_serialized_batches,
            load_format="oft_adapter",
            adapter_config=adapter_config,
            adapter_name=adapter_name,
        )

        return await engine.update_weights_from_tensor(update_weights_request)


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """
    Preprocess the tensor for update weights.
    Example Use Case:
        - FSDP: we gather tensor by calling full_tensor in _preprocess_tensor_for_update_weights
        - Megatron: we do nothing here, assuming it is gathered when feed into this func

    Args:
        tensor: The tensor to be preprocessed.

    Returns:
        The full tensor if it is a DTensor, otherwise the original tensor.
    """
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor

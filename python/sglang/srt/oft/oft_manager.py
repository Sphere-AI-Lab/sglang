# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.oft.base.manager import AdapterManager
from sglang.srt.oft.backend.base_backend import BaseOFTBackend
from sglang.srt.oft.backend.oft_registry import get_backend_from_name
from sglang.srt.oft.layers import BaseLayerWithOFT, get_oft_layer
from sglang.srt.oft.oft import OFTAdapter
from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.mem_pool import EMPTY_SLOT, OFTMemoryPool
from sglang.srt.oft.utils import (
    get_normalized_target_modules,
    validate_oft_block_size,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import AutoConfig

if TYPE_CHECKING:
    from sglang.srt.oft.io_types import OFTUpdateOutput

logger = logging.getLogger(__name__)


_MISSING_CONFIG_ATTR = object()


def _orbit_log_weight_sync_enabled() -> bool:
    return os.getenv("ORBIT_LOG_WEIGHT_SYNC", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }


def _expert_oft_delta_summary(buffer: Optional[torch.Tensor], block_size: int):
    if buffer is None or buffer.numel() == 0:
        return 0, 0.0
    eye = torch.eye(block_size, device=buffer.device, dtype=torch.float32)
    delta = (buffer.detach().float() - eye.view(1, 1, block_size, block_size)).abs()
    per_expert = delta.amax(dim=(1, 2, 3))
    return int((per_expert > 0).sum().item()), float(per_expert.max().item())


def _get_hf_config_attr(base_hf_config: AutoConfig, attr_name: str):
    value = getattr(base_hf_config, attr_name, _MISSING_CONFIG_ATTR)
    if value is not _MISSING_CONFIG_ATTR:
        return value

    text_config = getattr(base_hf_config, "text_config", None)
    if text_config is not None:
        value = getattr(text_config, attr_name, _MISSING_CONFIG_ATTR)
        if value is not _MISSING_CONFIG_ATTR:
            return value

    raise AttributeError(
        f"{type(base_hf_config).__name__} has no attribute {attr_name!r} "
        "on the top-level config or text_config"
    )


def validate_model_oft_target_modules(
    base_model: torch.nn.Module,
    target_modules: Iterable[str],
    *,
    source: str,
) -> None:
    """Let a model reject unsupported OFT target module suffixes early."""
    validate_fn = getattr(base_model, "validate_oft_target_modules", None)
    if validate_fn is None:
        return

    normalized_target_modules = set(target_modules)
    try:
        validate_fn(normalized_target_modules)
    except Exception as exc:
        raise ValueError(
            f"Model rejected OFT target modules from {source}: {exc}"
        ) from exc


def _first_expert_oft_tensor(ew_dict, name: str):
    for ew in ew_dict.values():
        tensor = ew.get(name)
        if tensor is not None:
            return tensor
    return None


def _fill_expert_oft_identity(buffer: torch.Tensor) -> None:
    buffer.zero_()
    if buffer.numel() == 0:
        return
    block_size = buffer.shape[-1]
    eye = torch.eye(block_size, dtype=buffer.dtype, device=buffer.device)
    buffer[...] = eye


def _expert_oft_buffer_desc(buffer: Optional[torch.Tensor]) -> str:
    if buffer is None:
        return "None"
    return (
        f"shape={tuple(buffer.shape)}, dtype={buffer.dtype}, "
        f"device={buffer.device}, data_ptr={buffer.data_ptr()}"
    )


_FUSED_MOE_DEVICE_TENSOR_ATTRS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_qweight",
    "w2_qweight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    "w13_scales",
    "w2_scales",
)


def _get_fused_moe_weight_device(moe) -> torch.device:
    for attr in _FUSED_MOE_DEVICE_TENSOR_ATTRS:
        tensor = getattr(moe, attr, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.device

    if isinstance(moe, torch.nn.Module):
        for tensor in moe.parameters(recurse=False):
            return tensor.device
        for tensor in moe.buffers(recurse=False):
            return tensor.device

    raise AttributeError(
        f"Cannot infer expert weight device for {type(moe).__name__}; "
        "expected one of "
        f"{', '.join(_FUSED_MOE_DEVICE_TENSOR_ATTRS)} or a direct module "
        "parameter/buffer."
    )


def _raise_streamed_expert_oft_buffer_mismatch(
    *,
    layer_id: int,
    projection: str,
    current_buffer: Optional[torch.Tensor],
    incoming_shape: tuple,
    incoming_dtype: torch.dtype,
    incoming_device: torch.device,
) -> None:
    message = (
        "Streamed expert OFT update would replace a CUDA-graph-captured "
        "R buffer; refusing to disable CUDA Graph silently. "
        f"layer_id={layer_id}, projection={projection}, "
        f"captured_buffer={_expert_oft_buffer_desc(current_buffer)}, "
        f"incoming_shape={incoming_shape}, incoming_dtype={incoming_dtype}, "
        f"incoming_device={incoming_device}. "
        "This is unexpected for Orbit streamed OFT with fixed "
        "block_size/block_share/dtype. Check --oft-target-modules, "
        "OFT block size/block_share, TP/EP layout, and Megatron Bridge "
        "export dtype."
    )
    logger.error(message)
    raise RuntimeError(message)


class OFTManager(AdapterManager):
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_ofts_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        server_args: ServerArgs,
        oft_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
        max_oft_block_size: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        adapter_paths: Optional[List[OFTRef]] = None,
        memory_saver_adapter=None,
        memory_saver_cpu_backup: bool = False,
    ):
        self.base_model: torch.nn.Module = base_model
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_ofts_per_batch: int = max_ofts_per_batch
        self.max_adapters_per_batch = max_ofts_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.oft_r_dtype: torch.dtype = self._resolve_oft_r_dtype(
            dtype, server_args.oft_dtype
        )
        # Single global split(canonical)-vs-fused signal (OFTArgs.oft_type):
        # drives the MoE expert gate/up group layout the pool registers
        # (see OFTMemoryPool._declare_expert_groups) and the CUDA-graph
        # pre-alloc layout below (_init_identity_expert_oft_for_cuda_graph).
        self.oft_type: str = server_args.oft_type
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.oft_added_tokens_size: Optional[int] = None
        self.memory_saver_adapter = memory_saver_adapter
        self.memory_saver_cpu_backup = memory_saver_cpu_backup

        # Every resident adapter fits the pool by construction (boot capacity
        # check in init_state), so eviction never fires and the policy is not
        # a knob until the B2 pool-overflow work.
        self.eviction_policy = "lru"

        # OFT backend for running orthogonal transform kernels
        logger.info(f"Using {oft_backend} as backend of OFT kernels.")
        backend_type = get_backend_from_name(oft_backend)
        self.oft_backend: BaseOFTBackend = backend_type(
            max_ofts_per_batch=max_ofts_per_batch,
            device=self.device,
            server_args=server_args,
        )

        # Initialize mutable internal state of the OFTManager.
        self.init_state(
            max_oft_block_size=max_oft_block_size,
            target_modules=target_modules,
            adapter_paths=adapter_paths,
        )

    @staticmethod
    def _resolve_oft_r_dtype(
        model_dtype: torch.dtype, oft_dtype: Optional[str]
    ) -> torch.dtype:
        raw = "model" if oft_dtype is None else str(oft_dtype).strip().lower()
        if raw in ("", "model", "model_dtype", "auto"):
            return model_dtype
        if raw in ("fp32", "float32"):
            return torch.float32
        if raw in ("bf16", "bfloat16"):
            return torch.bfloat16
        if raw in ("fp16", "float16", "half"):
            return torch.float16
        raise ValueError(
            f"Unsupported --oft-dtype={raw!r}; expected fp32, model, bf16, or fp16."
        )

    def init_cuda_graph_batch_info(
        self, max_bs_in_cuda_graph: int, num_tokens_per_bs: int
    ):
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        self.oft_backend.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=max_bs_in_cuda_graph,
            num_tokens_per_bs=num_tokens_per_bs,
        )

    def load_oft_adapter(self, oft_ref: OFTRef) -> "OFTUpdateOutput":
        return self.load_adapter(oft_ref)

    def unload_oft_adapter(self, oft_ref: OFTRef) -> "OFTUpdateOutput":
        return self.unload_adapter(oft_ref)

    def create_oft_update_result(
        self, success: bool, error_message: str = ""
    ) -> "OFTUpdateOutput":
        return self._make_update_result(success, error_message)

    def _update_output_cls(self):
        from sglang.srt.oft.io_types import OFTUpdateOutput

        return OFTUpdateOutput

    def _build_config(self, path):
        return OFTConfig(path)

    def _load_weights(self, ref):
        self.load_oft_weights(ref)

    def _clear_expert_on_unload(self, adapter):
        if adapter is not None and any(
            hasattr(layer, "expert_weights") and layer.expert_weights
            for layer in adapter.layers
        ):
            self._clear_expert_oft()

    def _unload_streamed_adapter(self, ref):
        return self.unload_streamed_adapter(ref)

    def validate_new_adapter(self, oft_config: OFTConfig, oft_ref: OFTRef):
        """
        Validate if an adapter can be loaded into the current OFT memory pool and generate error if it is incompatible.
        """
        if oft_config.oft_added_tokens_size > 0:
            raise ValueError(
                f"OFT serving currently doesn't support adapters that add tokens to the vocabulary"
            )

        # Check if this OFT adapter is already loaded
        for existing_oft_ref in self.refs.values():
            if oft_ref.adapter_name == existing_oft_ref.adapter_name:
                raise ValueError(
                    f"Failed to load OFT adapter {oft_ref.adapter_name} because it is already loaded"
                )

            if oft_ref.adapter_path == existing_oft_ref.adapter_path:
                logger.warning(
                    f"{oft_ref.adapter_path} is already loaded with name: {existing_oft_ref.adapter_name}, "
                    f"but another copy is being loaded with name: {oft_ref.adapter_name}"
                )

        if isinstance(oft_config.target_modules, list):
            validate_model_oft_target_modules(
                self.base_model,
                get_normalized_target_modules(oft_config.target_modules),
                source=f"adapter '{oft_ref.adapter_name}' PEFT config",
            )

        # Check if the OFT adapter shape is compatible with the current OFT memory pool configuration.
        memory_pool = getattr(self, "memory_pool", None)
        incompatible = memory_pool and not memory_pool.can_support(oft_config)
        if incompatible:
            raise ValueError(
                f"OFT adapter {oft_ref.adapter_name} with block_size {oft_config.block_size} is incompatible with the current "
                "OFT memory pool configuration. Please ensure that the OFT adapter's block_size is within the configured "
                "`--max-oft-block-size` and that the target modules are included in `--oft-target-modules`."
            )

        # Ensure pinned OFT adapters does not exceed maximal limit or cause starvation.
        if oft_ref.pinned and self.num_pinned >= self.max_ofts_per_batch - 1:
            raise ValueError(
                f"Failed to load OFT adapter {oft_ref.adapter_name} as a pinned adapter. It is not allowed to pin all slots "
                "in the OFT memory pool to avoid starvation for unpinned adapters and base models. Please increase your "
                "`--max-ofts-per-batch` or load it as unpinned OFT adapters."
            )

    def register_streamed_adapter(
        self,
        oft_ref: OFTRef,
        buffer_id: int,
        config_dict: dict,
    ) -> "OFTUpdateOutput":
        """Register a pre-loaded buffer slot as an OFT adapter.

        Used by the direct-to-GPU weight update path where tensors are written
        directly into the R_buffer without going through an OFTAdapter object.
        The buffer slot must already contain the precomputed R matrices.
        """
        try:
            # Guards against double-counting if this is ever called twice for
            # the same ref -- mirrors unload_streamed_adapter's symmetric
            # was_registered guard.
            was_already_registered = oft_ref.adapter_id in self.refs
            config = OFTConfig.from_dict(config_dict)
            self.configs[oft_ref.adapter_id] = config
            self.refs[oft_ref.adapter_id] = oft_ref
            # Register buffer slot mapping so inference can find this adapter
            self.memory_pool.uid_to_buffer_id[oft_ref.adapter_id] = buffer_id
            self.memory_pool.buffer_id_to_uid[buffer_id] = oft_ref.adapter_id
            if not was_already_registered:
                # Keeps num_pinned accurate for pinned adapters loaded over
                # the wire -- this used to never be counted here, silently
                # under-counting num_pinned and making validate_new_adapter's
                # anti-starvation guard (below) and validate_batch's
                # mem_pool_vacancy arithmetic (base/manager.py) wrong for
                # pinned wire-loaded adapters. Mirrors AdapterManager.
                # load_adapter's `self.num_pinned += int(ref.pinned)`.
                self.num_pinned += int(oft_ref.pinned)
        except Exception as e:
            return self.create_oft_update_result(
                success=False,
                error_message=str(e),
            )
        return self.create_oft_update_result(success=True)

    def unload_streamed_adapter(self, oft_ref: OFTRef) -> "OFTUpdateOutput":
        """Unload an adapter that was registered via register_streamed_adapter.

        Unlike unload_oft_adapter, this does not try to access self.adapters
        since streamed adapters have no OFTAdapter object.
        """
        try:
            was_registered = oft_ref.adapter_id in self.refs
            buffer_id = self.memory_pool.uid_to_buffer_id.get(oft_ref.adapter_id)
            if buffer_id is not None:
                # Restore the base-model passthrough before making this slot
                # available for reuse. A zero OFT matrix is not identity and
                # would silently corrupt a base request routed through it.
                self.memory_pool.reset_buffer_slot_to_identity(buffer_id)
                # The non-staged streamed MoE path binds expert rotations
                # directly on each FusedMoE module rather than in this slot.
                # Clear those global bindings before base traffic resumes.
                self._clear_expert_oft()
            if oft_ref.adapter_id in self.configs:
                del self.configs[oft_ref.adapter_id]
            if oft_ref.adapter_id in self.refs:
                del self.refs[oft_ref.adapter_id]
            if was_registered:
                self.num_pinned -= int(oft_ref.pinned)
            active_versions = getattr(self.memory_pool, "_active_versions", None)
            if active_versions is not None:
                active_versions.pop(oft_ref.adapter_id, None)
            # Clean up buffer slot mapping
            if buffer_id is not None:
                del self.memory_pool.uid_to_buffer_id[oft_ref.adapter_id]
                self.memory_pool.buffer_id_to_uid[buffer_id] = EMPTY_SLOT
                self.memory_pool.eviction_policy.remove(oft_ref.adapter_id)
        except Exception as e:
            return self.create_oft_update_result(
                success=False,
                error_message=str(e),
            )
        return self.create_oft_update_result(success=True)

    def _unload_streamed_adapter_if_not_disk_backed(
        self, oft_ref: OFTRef, *, context: str
    ) -> "OFTUpdateOutput":
        """Fully unload ``oft_ref`` via ``unload_streamed_adapter``, unless it
        has a CPU-side ``OFTAdapter`` entry in ``self.adapters`` -- i.e. it
        was loaded from disk via ``--peft-paths``, not over the wire.

        ``self.refs``/``self.configs`` are shared by both disk-backed and
        wire-loaded (streamed) adapters, but only wire-loaded ones lack a
        ``self.adapters`` entry. Calling ``unload_streamed_adapter`` on a
        disk-backed adapter would delete its ``configs``/``refs`` entries
        while leaving ``self.adapters[uid]`` behind and ``num_pinned``
        un-decremented -- a half-unloaded state that later confuses
        ``AdapterManager.unload_adapter``'s disk-vs-streamed dispatch (it
        would find ``adapter_id in self.adapters`` still true but
        ``configs``/``refs`` already gone). A disk-backed adapter simply
        losing its GPU buffer slot (already done by the caller, for the
        eviction case, via ``allocate_buffer_slot_with_eviction``) is fine on
        its own: it's exactly the same state as any adapter that has never
        yet been admitted into a batch, and it will be paged back into a
        fresh slot the next time a batch needs it.
        """
        if oft_ref.adapter_id in self.adapters:
            logger.info(
                "Freeing the GPU buffer slot of disk-backed OFT adapter "
                "'%s' (%s) without unloading it -- it remains loaded and "
                "will be re-admitted into a fresh slot when next "
                "referenced.",
                oft_ref.adapter_name,
                context,
            )
            return self._make_update_result(success=True)
        return self.unload_streamed_adapter(oft_ref)

    def load_adapter_from_tensors(
        self, ref: OFTRef, named_tensors, config_dict: dict, *, upsert: bool = False
    ) -> "OFTUpdateOutput":
        """Native-RPC admission path: like _ensure_streaming_oft_adapter_slot,
        but multi-tenant (no single-active restriction) since this serves
        the new native load_oft_adapter_from_tensors RPC, not the legacy
        srt/peft streamed path. Capacity is still bounded by
        max_ofts_per_batch via the memory pool's own admission.

        Validates the payload (_resolve_streamed_oft_tensor_groups) BEFORE
        evicting anything: unlike the legacy single-active path (which only
        ever replaces the SAME-named adapter), this path can evict a
        DIFFERENT, unrelated resident adapter to make room, and that
        adapter has no CPU-side backing to reload from if evicted for
        nothing. Resolving first means a malformed payload (bad tensor
        names, unsupported target modules, an unabsorbable DSV4 expert
        chunk -- the realistic failure modes) is caught while any would-be
        eviction victim is still intact. The remaining, much narrower risk
        (a failure inside the actual GPU write/precompute after validation
        already passed -- e.g. an OOM) is not eliminated: there is no spare
        buffer slot to stage into first, so once eviction has happened,
        that adapter is gone regardless of how the subsequent commit turns
        out. That residual case is called out explicitly in the returned
        error message rather than left silent -- and without the cleanup
        call in the commit-failure branch below, the NEW adapter's own
        `ref` would also have been left looking valid and resident (its
        buffer/config registration already committed to self.refs/
        self.configs/memory_pool.uid_to_buffer_id before the write is
        attempted) despite its buffer's contents being partially written or
        undefined -- a silently wrong-serving-results failure mode, worse
        than the disclosed evicted-adapter one. `unload_streamed_adapter(ref)`
        undoes that registration on commit failure so the phantom ref is
        never left resident.

        KNOWN LIMITATION: multi-tenant serving is only correct for
        dense-target-module adapters. If two concurrently-resident adapters
        both carry MoE/expert OFT weights, they silently share state
        (`apply_streamed_expert_oft` writes onto module-level
        `moe.w13_oft_r`/`w1_oft_r`/`w3_oft_r`/`w2_oft_r`, not a per-adapter
        slot) because `FusedMoEWithOFT.forward` has no per-token
        adapter-routing mechanism, unlike the dense path (`prepare_oft_batch`'s
        `weight_indices`) or LoRA's MoE path (`token_lora_mapping`). See
        `apply_streamed_expert_oft`'s docstring for why `slot_idx` alone
        doesn't fix this."""
        from sglang.srt.oft.streamed_weight_loader import (
            _commit_streamed_oft_tensor_groups,
            _resolve_streamed_oft_tensor_groups,
        )

        # The from-tensors RPC entry point deserializes a raw
        # Dict[str, torch.Tensor] (TpModelWorker._deserialize_own_rank
        # preserves whatever type the client serialized); the
        # from-distributed entry point (which delegates into this method)
        # already hands over List[Tuple[str, torch.Tensor]] from
        # WeightUpdater.receive_weights_from_distributed. The resolve/commit
        # helpers (and _partition_expert_oft_tensors inside them) require
        # the list-of-tuples form and iterate it directly -- normalize once
        # here so both callers land in the same shape.
        if isinstance(named_tensors, dict):
            named_tensors = list(named_tensors.items())

        try:
            block_size = config_dict.get("oft_block_size", 32)
            max_block_size = self.memory_pool.max_oft_block_size
            if block_size != max_block_size:
                return self._make_update_result(
                    success=False,
                    error_message=(
                        f"OFT adapter '{ref.adapter_name}' has block_size="
                        f"{block_size}, but the server pool is allocated for "
                        f"--max-oft-block-size={max_block_size}; smaller or "
                        "mixed block sizes are unsupported."
                    ),
                )

            plan, resolve_error = _resolve_streamed_oft_tensor_groups(
                self, named_tensors, block_size
            )
            if plan is None:
                return self._make_update_result(
                    success=False, error_message=resolve_error
                )

            existing_id = None
            for ref_id, existing_ref in list(self.refs.items()):
                if existing_ref.adapter_name == ref.adapter_name:
                    existing_id = ref_id
                    break
            if existing_id is not None:
                if not upsert:
                    return self._make_update_result(
                        success=False,
                        error_message=(
                            f"OFT adapter '{ref.adapter_name}' is already "
                            "loaded; pass upsert=True to refresh it in place."
                        ),
                    )
                if existing_id in self.adapters:
                    return self._make_update_result(
                        success=False,
                        error_message=(
                            f"OFT adapter '{ref.adapter_name}' is currently loaded "
                            "from disk (--peft-paths) and cannot be converted to a "
                            "wire-loaded adapter via upsert. Use a different adapter "
                            "name for the wire-loaded adapter."
                        ),
                    )
                self._unload_streamed_adapter_if_not_disk_backed(
                    self.refs[existing_id], context="upsert"
                )

            buffer_id, evicted_uid = self.memory_pool.allocate_buffer_slot_with_eviction(
                self.refs
            )
            evicted_name = (
                self.refs[evicted_uid].adapter_name if evicted_uid is not None else None
            )
            if evicted_uid is not None:
                evict_result = self._unload_streamed_adapter_if_not_disk_backed(
                    self.refs[evicted_uid], context="eviction"
                )
                if not evict_result.success:
                    return evict_result
            self.memory_pool.reset_buffer_slot_to_identity(buffer_id)
            result = self.register_streamed_adapter(ref, buffer_id, config_dict)
            if not result.success:
                return result
            # Mark resident immediately (not just on first generate()): this
            # adapter must be a real eviction_policy.select_victim candidate
            # from the moment it's admitted, or a later admission-time
            # eviction attempt could pick among only just-loaded,
            # never-served adapters and find nothing tracked to select.
            self.memory_pool.eviction_policy.mark_used(ref.adapter_id)

            success, error_message = _commit_streamed_oft_tensor_groups(
                self,
                named_tensors,
                plan,
                buffer_id,
                block_size,
                ref.adapter_name,
                ref.adapter_id,
            )
            if not success:
                # Undo the registration/mark_used above: without this, refs/
                # configs/uid_to_buffer_id would still list `ref` as valid
                # and resident, pointing at a buffer whose contents are now
                # partially written or undefined -- silently wrong serving
                # results, not just a clear failure. Check the result like
                # the evicted-adapter cleanup above does: this is a
                # best-effort cleanup inside an already-failing path, so we
                # still return the original commit error either way, but a
                # cleanup failure must be visible (logged), not swallowed --
                # that would defeat the exact guarantee this call exists for.
                cleanup_result = self.unload_streamed_adapter(ref)
                if not cleanup_result.success:
                    logger.error(
                        "Failed to clean up OFT adapter '%s' after a failed "
                        "commit: %s",
                        ref.adapter_name,
                        cleanup_result.error_message,
                    )
                if evicted_name is not None:
                    error_message = (
                        f"adapter '{evicted_name}' was evicted to make room, "
                        f"and the new load also failed: {error_message}"
                    )
                return self._make_update_result(
                    success=False, error_message=error_message
                )
        except Exception as e:
            return self._make_update_result(success=False, error_message=str(e))
        return self._make_update_result(success=True)

    def load_adapter_from_distributed(
        self,
        ref: OFTRef,
        names,
        dtypes,
        shapes,
        config_dict: dict,
        group_name: str,
        weight_updater,
        *,
        upsert: bool = False,
    ) -> "OFTUpdateOutput":
        """Receives the adapter's tensors over the process group via the
        model runner's WeightUpdater, then delegates to
        load_adapter_from_tensors for admission."""
        try:
            tensors = weight_updater.receive_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        except Exception as e:
            return self._make_update_result(
                success=False,
                error_message=f"Failed to receive OFT adapter weights: {e}.",
            )
        return self.load_adapter_from_tensors(
            ref, tensors, config_dict, upsert=upsert
        )

    def validate_oft_batch(self, adapter_ids: set[Optional[str]]) -> bool:
        return self.validate_batch(adapter_ids)

    def fetch_new_ofts(
        self, new_ofts: set[Optional[str]], running_ofts: set[Optional[str]] = set()
    ):
        return self.fetch_new_adapters(new_ofts, running_ofts)

    def _prepare_mem_pool_batch(self, cur_uids):
        self.memory_pool.prepare_oft_batch(
            cur_uids=cur_uids,
            oft_adapters=self.adapters,
            oft_modules=self.adapter_modules,
            oft_refs=self.refs.copy(),
            oft_embed_tokens_module=self.embed_tokens_module,
            oft_lm_head_module=self.lm_head_module,
        )

    def prepare_oft_batch(self, forward_batch: ForwardBatch):
        # set up batch info shared by all oft modules
        bs = forward_batch.batch_size

        use_cuda_graph = (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        )

        weight_indices = [0] * len(forward_batch.adapter_ids)
        oft_block_sizes = [0] * self.max_ofts_per_batch
        for i, uid in enumerate(forward_batch.adapter_ids):
            # Mirrors upstream LoRAManager.prepare_lora_batch: a uid with no
            # resident slot keeps weight_indices[i] = 0 rather than raising.
            # Real requests are always resident (fetch_new_ofts runs first);
            # the CUDA-graph replay path pads adapter_ids with None WITHOUT a
            # fetch, so an evicted base slot lands here. Those padded rows are
            # discarded, so slot 0's contents are immaterial for them.
            if uid not in self.memory_pool.uid_to_buffer_id:
                continue
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                if uid in self.adapters:
                    oft_block_sizes[weight_indices[i]] = self.adapters[uid].block_size
                elif uid in self.configs:
                    oft_block_sizes[weight_indices[i]] = self.configs[uid].block_size
                else:
                    raise KeyError(f"OFT adapter {uid} not found in ofts or configs")
        # Do in-place updates when CUDA graph is enabled and the batch forward mode
        # could use CUDA graph.
        self.oft_backend.prepare_oft_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            oft_block_sizes=oft_block_sizes,
            use_cuda_graph=use_cuda_graph,
        )

    def update_oft_info(self):
        return self.update_info()

    def _set_module_info(self, module, target_module, layer_id):
        module.set_oft_info(
            self.memory_pool.get_tensor(target_module=target_module, layer_id=layer_id),
        )

    def _update_embedding_info(self):
        if self.embed_tokens_module is not None:
            self.embed_tokens_module.set_oft_info(
                self.memory_pool.get_embedding_tensor("added_tokens"),
                self.memory_pool.get_embedding_tensor("embed_tokens"),
            )
        if self.lm_head_module is not None:
            self.lm_head_module.set_oft_info(
                self.memory_pool.get_embedding_tensor("lm_head"),
            )

    def init_state(
        self,
        max_oft_block_size: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        adapter_paths: Optional[List[OFTRef]] = None,
    ):
        """
        Initialize the internal (mutable) state of the OFTManager.

        When `adapter_paths` is provided and not empty, it might be used for inferring OFT shape info such as
        the target modules and max_oft_block_size.
        """

        assert adapter_paths or (
            max_oft_block_size is not None and target_modules is not None
        ), "When no initial --oft-paths is provided, you need to specify both --max-oft-block-size and --oft-target-modules for OFT initialization."

        self.init_oft_adapters(adapter_paths)
        self.init_oft_shapes(
            max_oft_block_size=max_oft_block_size,
            target_modules=target_modules,
        )
        self.init_oft_modules()
        # Replace each expert-OFT-target FusedMoE with a FusedMoEWithOFT wrapper
        # (own peft_enabled runner) so OFT rides a dedicated non-fused runner like
        # LoRA. Runs after init_oft_modules (dense) and before anything that walks
        # the MoE modules, and invalidates the finder cache so later callers see
        # through the wrapper to base_layer.
        n_expert_wrapped = self._install_moe_oft_wrappers()
        # Expert OFT has no adapter slot dimension (kernels index by expert
        # only), so more than one resident adapter is unrepresentable on the
        # expert path. Dense targets are unaffected.
        if n_expert_wrapped and len(self.refs) > 1:
            raise ValueError(
                f"Multi-adapter OFT serving is unsupported on MoE expert "
                f"targets: {len(self.refs)} adapters are loaded but expert OFT "
                "buffers are single-adapter. Serve one adapter, or remove "
                "expert projections from --peft-target-modules."
            )
        self.init_memory_pool()
        self.update_oft_info()
        self._init_identity_expert_oft_for_cuda_graph()
        wrapped_module_count = sum(
            len(layer_modules) for layer_modules in self.adapter_modules
        )
        wrapped_layer_count = sum(
            1 for layer_modules in self.adapter_modules if layer_modules
        )
        loaded_adapter_names = sorted(
            str(oft_ref.adapter_name) for oft_ref in self.refs.values()
        )
        logger.info(
            "event=oft_manager_initialized target_modules=%s "
            "max_oft_block_size=%s backend=%s wrapped_modules=%d "
            "wrapped_layers=%d loaded_adapters=%s max_ofts_per_batch=%s "
            "base_identity_slot=%s",
            sorted(self.target_modules),
            self.max_oft_block_size,
            type(self.oft_backend).__name__,
            wrapped_module_count,
            wrapped_layer_count,
            loaded_adapter_names,
            self.max_ofts_per_batch,
            None in self.memory_pool.uid_to_buffer_id,
        )

    def init_oft_adapters(self, adapter_paths: Optional[List[OFTRef]] = None):
        return self.init_adapters(adapter_paths)

    def init_oft_shapes(
        self,
        max_oft_block_size: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
    ):
        """Infer OFT target modules and max_oft_block_size from loaded adapters if not provided."""

        self.target_modules = (
            get_normalized_target_modules(target_modules) if target_modules else set()
        )
        if self.target_modules:
            validate_model_oft_target_modules(
                self.base_model,
                self.target_modules,
                source="server --oft-target-modules",
            )

        for adapter_id, config in self.configs.items():
            # Handle PEFT shorthand strings like "all-linear" or "all".
            # These cannot be resolved to concrete module names without
            # inspecting the base model, so we require the user to specify
            # --oft-target-modules explicitly when such shorthands are used.
            if isinstance(config.target_modules, str):
                if config.target_modules in ("all-linear", "all"):
                    if target_modules is not None:
                        # CLI --oft-target-modules already provided; skip
                        # per-adapter inference for this adapter.
                        continue
                    else:
                        adapter_name = self.refs[adapter_id].adapter_name
                        raise ValueError(
                            f"OFT adapter '{adapter_name}' uses "
                            f"target_modules='{config.target_modules}' which cannot "
                            "be resolved automatically. Please explicitly specify "
                            "--oft-target-modules during server startup. You can "
                            "specify 'all' to enable all supported module types."
                        )
                else:
                    raise ValueError(
                        f"SGLang does not recognize target_modules="
                        f"'{config.target_modules}'. Please use a list of module "
                        "name suffixes in the adapter's PEFT config, or explicitly "
                        "specify --oft-target-modules during server startup."
                    )

            if not isinstance(config.target_modules, list):
                raise ValueError(
                    f"SGLang currently only supports inferring OFT target modules when a list of "
                    "suffixes is provided in `target_modules` field of PEFT config. Please explicitly "
                    "specify `--oft-target-modules` during server startup. You can specify `all` to "
                    "enable all support modules types. "
                )

            adapter_target_modules = get_normalized_target_modules(
                config.target_modules
            )
            adapter_name = self.refs[adapter_id].adapter_name
            validate_model_oft_target_modules(
                self.base_model,
                adapter_target_modules,
                source=f"adapter '{adapter_name}' PEFT config",
            )

            if target_modules is not None:
                # When `--oft-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    raise ValueError(
                        f"OFT adapter '{adapter_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --oft-target-modules {sorted(self.target_modules)}. "
                        f"Please update --oft-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
            else:
                # Otherwise, infer target_modules from adapter configs.
                self.target_modules.update(adapter_target_modules)

        if max_oft_block_size is not None:
            self.max_oft_block_size = validate_oft_block_size(max_oft_block_size)
        else:
            self.max_oft_block_size = max(
                [x.block_size for x in self.configs.values()],
                default=0,
            )

        # One geometry per server: pool tiles are allocated once at
        # (max_oft_block_size x max_oft_block_size) and every adapter's R must
        # match them exactly -- block size is geometry, not a capacity to pad.
        block_sizes = {x.block_size for x in self.configs.values()}
        if block_sizes and block_sizes != {self.max_oft_block_size}:
            raise ValueError(
                f"All OFT adapters must use the server's max block size "
                f"(--max-oft-block-size={self.max_oft_block_size}); got adapter "
                f"block size(s) {sorted(block_sizes)}. Smaller or mixed block "
                f"sizes are unsupported."
            )

        # Auto-infer self.oft_added_tokens_size from loaded OFT configs
        if self.oft_added_tokens_size is None:
            inferred_extra_vocab_size = next(
                (
                    x.oft_added_tokens_size
                    for x in self.configs.values()
                    if x.oft_added_tokens_size > 0
                ),
                0,
            )
            if inferred_extra_vocab_size > 0:
                logger.info(
                    f"self.oft_added_tokens_size={inferred_extra_vocab_size} from OFT adapters."
                )
            self.oft_added_tokens_size = inferred_extra_vocab_size

    def load_oft_weights(self, oft_ref: OFTRef):
        """
        Load the weights of an OFT adapter to CPU memory.
        """
        oft_adapter = OFTAdapter(
            oft_ref.adapter_id,
            self.configs[oft_ref.adapter_id],
            self.base_hf_config,
            self.load_config,
            self.oft_backend,
        )
        oft_adapter.initialize_weights()

        self.adapters[oft_ref.adapter_id] = oft_adapter

        # Set expert OFT weights on FusedMoE layers if present
        if any(
            hasattr(layer, "expert_weights") and layer.expert_weights
            for layer in oft_adapter.layers
        ):
            self._set_expert_oft(oft_adapter)

    def load_oft_weights_from_tensors(
        self, oft_ref: OFTRef, tensors: Dict[str, torch.Tensor]
    ):
        """
        Load the weights of an OFT adapter from tensors to CPU memory.
        """
        oft_adapter = OFTAdapter(
            oft_ref.adapter_id,
            self.configs[oft_ref.adapter_id],
            self.base_hf_config,
            self.load_config,
            self.oft_backend,
        )
        oft_adapter.initialize_weights_from_tensors(tensors)
        self.adapters[oft_ref.adapter_id] = oft_adapter

        # Set expert OFT weights on FusedMoE layers if present
        if any(
            hasattr(layer, "expert_weights") and layer.expert_weights
            for layer in oft_adapter.layers
        ):
            self._set_expert_oft(oft_adapter)

    def init_memory_pool(self):
        """(Re)initialize the OFT memory pool based on the current configurations."""
        external_target_modules = set()
        getter = getattr(self.base_model, "get_oft_external_target_modules", None)
        if getter is not None:
            external_target_modules = set(getter())
        self.memory_pool = OFTMemoryPool(
            base_hf_config=self.base_hf_config,
            max_ofts_per_batch=self.max_ofts_per_batch,
            dtype=self.oft_r_dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_oft_block_size=self.max_oft_block_size,
            target_modules=self.target_modules,
            base_model=self.base_model,
            oft_type=self.oft_type,
            oft_modules=self.adapter_modules,
            external_target_modules=external_target_modules,
            eviction_policy=self.eviction_policy,
            oft_added_tokens_size=self.oft_added_tokens_size,
            memory_saver_adapter=self.memory_saver_adapter,
            memory_saver_cpu_backup=self.memory_saver_cpu_backup,
            double_buffer=False,
        )
        logger.info(
            "Using %s for OFT R buffers (model dtype %s).",
            self.oft_r_dtype,
            self.dtype,
        )

        # Initializing memory pool with base model
        self.fetch_new_ofts({None})

    def set_oft_module(self, module_name, module):
        return self.set_adapter_module(module_name, module)

    def _get_adapter_layer(self, module):
        return get_oft_layer(module, self.oft_backend)

    def _install_moe_oft_wrappers(self):
        """Replace each expert-OFT-target FusedMoE with a FusedMoEWithOFT wrapper
        (own peft_enabled runner). Buffers are injected later onto the wrapper's
        base_layer, unchanged. Invalidates the _find_fused_moe_modules cache so
        every later caller re-scans and sees through the wrapper to base_layer."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.oft.layers import FusedMoEWithOFT
        from sglang.srt.utils import replace_submodule

        # Only wrap FusedMoE when an expert projection is actually OFT-targeted
        # -- same gate as OFTMemoryPool._declare_expert_groups and
        # _init_identity_expert_oft_for_cuda_graph. Otherwise no expert OFT
        # buffers are declared, so wrapping applies no rotation (pure overhead)
        # and FusedMoEWithOFT.__init__ would still eagerly probe
        # quant_method.get_triton_quant_info(...), which a Marlin-quantized MoE
        # (CompressedTensorsWNA16MoE) does not implement -> boot crash. Dense-
        # only targets (e.g. the fused MLA down-proj) must leave MoE untouched.
        init_w13 = bool({"gate_up_proj", "gate_proj", "up_proj"} & self.target_modules)
        init_w2 = "down_proj" in self.target_modules
        if not (init_w13 or init_w2):
            return 0

        # Snapshot names first: replace_submodule mutates the module tree.
        # `type(...) is FusedMoE` matches only RAW modules, never the wrapper.
        moe_names = [
            name
            for name, module in self.base_model.named_modules()
            if type(module) is FusedMoE
        ]
        for name in moe_names:
            base = self.base_model.get_submodule(name)
            wrapper = FusedMoEWithOFT(base, self.oft_backend)
            replace_submodule(self.base_model, name, wrapper)
        # Drop any cache built before wrapping (e.g. if init_oft_modules touched it).
        if hasattr(self, "_moe_modules"):
            del self._moe_modules
        logger.info(f"Installed {len(moe_names)} FusedMoEWithOFT expert wrappers")
        return len(moe_names)

    def init_oft_modules(self):
        # Look-up table that maps (layer_index, module_name) to the corresponding OFT module.
        num_hidden_layers = _get_hf_config_attr(
            self.base_hf_config, "num_hidden_layers"
        )
        self.adapter_modules: List[Dict[str, BaseLayerWithOFT]] = [
            {} for _ in range(num_hidden_layers)
        ]

        self.embed_tokens_module: Optional[BaseLayerWithOFT] = None
        self.lm_head_module: Optional[BaseLayerWithOFT] = None

        # When tie_word_embeddings=True, lm_head is the same Python object as
        # embed_tokens. PyTorch's named_modules() deduplicates by object identity,
        # so lm_head will not appear as a separate entry in the scan below,
        # preventing OFT from wrapping it. To fix this, we create a new
        # ParallelLMHead that shares the same base weight tensor (no extra GPU
        # memory) so that named_modules() yields it as an independent module.
        if "lm_head" in self.target_modules:
            lm_head = getattr(self.base_model, "lm_head", None)
            embed_tokens = None
            for name, mod in self.base_model.named_modules():
                if name.endswith("embed_tokens"):
                    embed_tokens = mod
                    break
            if (
                lm_head is not None
                and embed_tokens is not None
                and lm_head is embed_tokens
            ):
                logger.info(
                    "lm_head is tied with embed_tokens. Creating a separate "
                    "ParallelLMHead that shares the base weight for OFT support."
                )
                untied_lm_head = ParallelLMHead(
                    num_embeddings=embed_tokens.org_vocab_size,
                    embedding_dim=embed_tokens.embedding_dim,
                    params_dtype=embed_tokens.weight.dtype,
                    org_num_embeddings=embed_tokens.org_vocab_size,
                )
                # Share the base weight tensor — no additional GPU memory.
                untied_lm_head.weight = embed_tokens.weight
                # Replace the model attribute so named_modules() sees it
                # independently.
                self.base_model.lm_head = untied_lm_head

        wrapped_modules = []
        skipped_by_policy = []
        skipped_without_layer_id = []
        for module_name, module in self.base_model.named_modules():
            module_suffix = module_name.split(".")[-1]
            if getattr(
                self.base_model, "should_apply_oft", None
            ) and not self.base_model.should_apply_oft(module_name):
                if module_suffix in self.target_modules:
                    skipped_by_policy.append(module_name)
                continue

            # Handle embed_tokens
            if "embed_tokens" in module_name and "embed_tokens" in self.target_modules:
                if isinstance(module, VocabParallelEmbedding) and not isinstance(
                    module, BaseLayerWithOFT
                ):
                    oft_module = self.set_oft_module(module_name, module)
                    self.embed_tokens_module = oft_module
                    wrapped_modules.append(module_name)
                    continue

            # Handle lm_head
            if "lm_head" in module_name and "lm_head" in self.target_modules:
                if isinstance(module, ParallelLMHead) and not isinstance(
                    module, BaseLayerWithOFT
                ):
                    oft_module = self.set_oft_module(module_name, module)
                    self.lm_head_module = oft_module
                    wrapped_modules.append(module_name)
                    continue

            # The module should be converted if it is included in target_names
            if module_suffix in self.target_modules:
                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    skipped_without_layer_id.append(module_name)
                    continue
                self.adapter_modules[layer_id][module_name] = self.set_oft_module(
                    module_name, module
                )
                wrapped_modules.append(module_name)

        if wrapped_modules:
            logger.info(
                "Wrapped %d OFT modules: %s",
                len(wrapped_modules),
                wrapped_modules,
            )
        if skipped_by_policy:
            sample_size = 8
            sample = skipped_by_policy[:sample_size]
            logger.info(
                "Skipped %d target-matched OFT modules due to model policy "
                "(showing first %d): %s",
                len(skipped_by_policy),
                min(sample_size, len(skipped_by_policy)),
                sample,
            )
        if skipped_without_layer_id:
            logger.info(
                "Skipped %d target-matched OFT modules without a transformer "
                "layer id: %s",
                len(skipped_without_layer_id),
                skipped_without_layer_id,
            )

    # ------------------------------------------------------------------ #
    #  Expert OFT helpers for FusedMoE layers
    # ------------------------------------------------------------------ #

    def _init_identity_expert_oft_for_cuda_graph(self):
        """Install identity expert OFT buffers before CUDA graph capture.

        Streamed training syncs expert OFT tensors after the server has
        already initialized. CUDA graph replay is only correct if the graph
        captured the expert-OFT kernels and the same R-buffer tensor objects
        are updated in place later.
        """
        target_modules = getattr(self, "target_modules", set())
        init_w13 = bool({"gate_up_proj", "gate_proj", "up_proj"} & target_modules)
        init_w2 = "down_proj" in target_modules
        if not (init_w13 or init_w2):
            return

        # oft_type (OFTArgs.oft_type, threaded through server_args into
        # self.oft_type) is the single global split-vs-fused signal -- see
        # plan Task 6. This layout HINT only applies to a module with nothing
        # loaded (CUDA-graph pre-alloc). The reliable signal for a module that
        # already has weights is the per-module buffer the loader wrote
        # (_apply_expert_oft_to_module): a loaded legacy w13_oft_r must survive
        # untouched regardless of oft_type -- short-circuited below.
        w13_is_split = self.oft_type == "canonical_oft"

        block_size = self.max_oft_block_size
        if block_size <= 0:
            return

        initialized = False
        for layer_id, moe in self._find_fused_moe_modules().items():
            if init_w13:
                if moe.hidden_size % block_size != 0:
                    raise ValueError(
                        f"MoE w13 OFT input dim {moe.hidden_size} is not "
                        f"divisible by block_size {block_size}"
                    )
                if getattr(moe, "w13_oft_r", None) is not None:
                    # Loaded legacy fused w13 — leave entirely untouched (do NOT
                    # split it, do NOT null it). This short-circuit is the fix:
                    # a loaded legacy adapter must survive regardless of
                    # oft_type.
                    pass
                elif w13_is_split:
                    # Pool-backed (mem_pool.py ``_declare_expert_groups``):
                    # each buffer is the memory pool's "w1_oft_r"/"w3_oft_r"
                    # group slot 0 (ACTIVE), not a private module-owned
                    # tensor, so a later streamed sync (which reads back and
                    # mutates ``moe.w1_oft_r``/``moe.w3_oft_r`` in place)
                    # writes the SAME pool slot.
                    for attr in ("w1_oft_r", "w3_oft_r"):
                        if getattr(moe, attr, None) is None:
                            _fill_expert_oft_identity(
                                self.memory_pool.slot(
                                    attr, layer_id, self.memory_pool.active_idx
                                )
                            )
                            setattr(
                                moe,
                                attr,
                                self.memory_pool.active_view(attr, layer_id),
                            )
                            initialized = True
                    # Split buffers supersede the legacy fused buffer.
                    moe.w13_oft_r = None
                else:
                    # Legacy, nothing loaded on this module: pre-allocate an
                    # identity fused buffer so CUDA graph replay captures an
                    # in-place-updatable tensor. Pool-backed (mem_pool.py
                    # ``_declare_expert_groups``): the buffer is the memory
                    # pool's "w13_oft_r" group slot 0 (ACTIVE), not a private
                    # module-owned tensor, so a later streamed sync (which
                    # reads back and mutates ``moe.w13_oft_r`` in place) writes
                    # the SAME pool slot.
                    _fill_expert_oft_identity(
                        self.memory_pool.slot(
                            "w13_oft_r", layer_id, self.memory_pool.active_idx
                        )
                    )
                    moe.w13_oft_r = self.memory_pool.active_view(
                        "w13_oft_r", layer_id
                    )
                    initialized = True

            if init_w2 and getattr(moe, "w2_oft_r", None) is None:
                w2_input_dim = moe.intermediate_size_per_partition
                if w2_input_dim % block_size != 0:
                    raise ValueError(
                        f"MoE w2 OFT input dim {w2_input_dim} is not "
                        f"divisible by block_size {block_size}"
                    )
                # Pool-backed, same rationale as the w13 branch above.
                _fill_expert_oft_identity(
                    self.memory_pool.slot("w2_oft_r", layer_id, self.memory_pool.active_idx)
                )
                moe.w2_oft_r = self.memory_pool.active_view("w2_oft_r", layer_id)
                initialized = True

        if initialized:
            logger.info(
                "Initialized identity expert OFT buffers for CUDA graph capture."
            )

    def _apply_expert_oft_to_module(self, moe, ew_dict, block_size, layer_id=None):
        """Compute and assign w13_oft_r / w2_oft_r on a single FusedMoE module.

        ew_dict: {global_expert_id: {"gate_proj.oft_R": tensor,
                                     "down_proj.oft_R": tensor}}.
        Block-diagonal R is kept as external PEFT — only writes
        moe.w13_oft_r / moe.w2_oft_r, never base w13_weight / w2_weight.

        w13 R rotates hidden_size (input to gate/up); hidden_size is NOT
        TP-sharded, so no TP slicing. w2 R rotates intermediate_size
        (input to down_proj); intermediate_size IS TP-sharded, so each
        TP rank takes its slice of the block-diagonal R.
        """
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        if not ew_dict:
            return

        gate_sample = _first_expert_oft_tensor(ew_dict, "gate_proj.oft_R")
        up_sample = _first_expert_oft_tensor(ew_dict, "up_proj.oft_R")
        down_sample = _first_expert_oft_tensor(ew_dict, "down_proj.oft_R")
        if gate_sample is None and up_sample is None and down_sample is None:
            return
        is_split = gate_sample is not None and up_sample is not None
        is_legacy = gate_sample is not None and up_sample is None

        device = _get_fused_moe_weight_device(moe)
        num_local = moe.num_local_experts
        tp_rank = moe.moe_tp_rank
        tp_size = moe.moe_tp_size

        # Use the compact weights' own dtype (training precision, e.g. BF16),
        # not the MoE weight dtype which may be FP8. Cayley runs in this
        # dtype to match Bridge's `_cayley_batch` for bit-identical R.
        oft_sample = gate_sample if gate_sample is not None else down_sample
        dtype = oft_sample.dtype

        num_blocks_w13 = gate_sample.shape[0] if gate_sample is not None else 0
        if down_sample is not None:
            num_blocks_w2_full = down_sample.shape[0]
            assert num_blocks_w2_full % tp_size == 0, (
                f"w2 OFT num_blocks ({num_blocks_w2_full}) must be "
                f"divisible by tp_size ({tp_size})"
            )
            blocks_per_tp = num_blocks_w2_full // tp_size
            w2_block_start = tp_rank * blocks_per_tp
            w2_block_end = w2_block_start + blocks_per_tp
        else:
            blocks_per_tp = 0

        w1_oft_r = torch.zeros(
            num_local, num_blocks_w13, block_size, block_size,
            device=device, dtype=dtype,
        ) if gate_sample is not None else None
        w3_oft_r = torch.zeros(
            num_local, num_blocks_w13, block_size, block_size,
            device=device, dtype=dtype,
        ) if up_sample is not None else None
        w2_oft_r = torch.zeros(
            num_local, blocks_per_tp, block_size, block_size,
            device=device, dtype=dtype,
        ) if blocks_per_tp > 0 else None

        # Collect per-expert compacts for batched Cayley. Without batching,
        # one Cayley call per (expert, proj) means 2*num_local launches per
        # layer (~256 for Qwen3-30B-A3B with 128 experts) and the kernel-
        # launch overhead dominates the actual GFLOPs. Batching collapses to
        # 2 launches per layer (one for w13, one for w2). This mirrors the
        # dense path's `_flush_oft_group_chunk`. w2 is sliced on the block
        # dim BEFORE Cayley so we don't waste compute on the other TP rank's
        # half (a separate ~2x win on the w2 side).
        local_ids: list[int] = []
        gate_compacts: list[torch.Tensor | None] = []
        up_compacts: list[torch.Tensor | None] = []
        down_compacts: list[torch.Tensor | None] = []
        for global_id, ew in ew_dict.items():
            local_id = moe._map_global_expert_id_to_local_expert_id(global_id)
            if local_id < 0 or local_id >= num_local:
                continue
            local_ids.append(local_id)
            gate_compacts.append(ew.get("gate_proj.oft_R"))
            up_compacts.append(ew.get("up_proj.oft_R"))
            d = ew.get("down_proj.oft_R")
            if d is not None and blocks_per_tp > 0:
                d = d[w2_block_start:w2_block_end]
            down_compacts.append(d)

        def _batched_cayley_assign(out, compacts, num_blocks):
            valid_idx = [i for i, c in enumerate(compacts) if c is not None]
            if not valid_idx:
                return
            stacked = torch.cat(
                [
                    compacts[i].to(device=device, dtype=dtype)
                    for i in valid_idx
                ],
                dim=0,
            )
            R_stacked = precompute_oft_r(stacked, block_size)
            R_per_expert = R_stacked.view(
                len(valid_idx), num_blocks, block_size, block_size
            )
            for j, i in enumerate(valid_idx):
                out[local_ids[i]] = R_per_expert[j]

        if w1_oft_r is not None:
            _batched_cayley_assign(w1_oft_r, gate_compacts, num_blocks_w13)
        if w3_oft_r is not None:
            _batched_cayley_assign(w3_oft_r, up_compacts, num_blocks_w13)
        if w2_oft_r is not None:
            _batched_cayley_assign(w2_oft_r, down_compacts, blocks_per_tp)

        # Cayley ran in the compact weights' own dtype (bit-identical to
        # Bridge's _cayley_batch), but the STORED buffers must match the
        # rotation kernel's activation dtype: apply_oft_rotation_triton feeds
        # A (model dtype) and R into one tl.dot, which rejects mixed dtypes —
        # an fp32 disk adapter otherwise crashes every expert forward. The
        # dense path already gets this cast for free by copying into the
        # oft_r_dtype memory pool; mirror it here.
        if self.oft_r_dtype is not None and dtype != self.oft_r_dtype:
            w1_oft_r = w1_oft_r.to(self.oft_r_dtype) if w1_oft_r is not None else None
            w3_oft_r = w3_oft_r.to(self.oft_r_dtype) if w3_oft_r is not None else None
            w2_oft_r = w2_oft_r.to(self.oft_r_dtype) if w2_oft_r is not None else None

        if is_split:
            moe.w1_oft_r = w1_oft_r
            moe.w3_oft_r = w3_oft_r
            moe.w13_oft_r = None
        elif is_legacy:
            # Legacy shared-R: promote the gate buffer to w13, clear w1/w3 so
            # the runner does not enter the split path.
            moe.w13_oft_r = w1_oft_r
            moe.w1_oft_r = None
            moe.w3_oft_r = None
        if w2_oft_r is not None:
            moe.w2_oft_r = w2_oft_r

    def _set_expert_oft(self, oft_adapter):
        """Set expert OFT R on FusedMoE layers from a disk-loaded adapter."""
        moe_modules = self._find_fused_moe_modules()
        if not moe_modules:
            return

        block_size = oft_adapter.block_size
        for layer_id, moe in moe_modules.items():
            if layer_id >= len(oft_adapter.layers):
                continue
            ew_dict = oft_adapter.layers[layer_id].expert_weights
            self._apply_expert_oft_to_module(moe, ew_dict, block_size, layer_id)

    def apply_streamed_expert_oft(self, expert_tensors, block_size, slot_idx=None):
        """Set FusedMoE expert OFT R from streamed-sync compact tensors.

        expert_tensors: {layer_id: {global_expert_id: {"gate_proj.oft_R": t,
                                                       "down_proj.oft_R": t}}}.
        Keeps OFT external — only writes moe.w13_oft_r / moe.w2_oft_r,
        never merges into base weights.

        ``slot_idx=None`` (default) preserves today's exact in-place-on-
        ``moe.w*_oft_r`` behavior byte-for-byte -- every existing caller
        keeps working unchanged. Passing an explicit ``slot_idx`` (e.g. the
        pool's ``staging_idx`` for double-buffer ``stage()``) bypasses the
        module attribute -- which is only ever bound to the ACTIVE slot --
        and scatters straight into ``self.memory_pool.slot(group, layer,
        slot_idx)`` instead, leaving ``moe.w*_oft_r`` untouched so forward
        keeps reading ACTIVE until ``activate()`` flips it.

        NOTE: an explicit ``slot_idx`` only isolates the WRITE; it does not
        give multi-tenant correctness by itself, since nothing on the read
        side selects a per-request ``slot_idx`` -- ``FusedMoEWithOFT.forward``
        always reads the plain ``moe.w*_oft_r`` attribute this writes when
        ``slot_idx=None``. Using a per-adapter ``slot_idx`` here without a
        matching per-token routing mechanism in forward would make that
        adapter's rotation silently never apply, not fix concurrent
        residency (`StagedOFTManager`'s ``staging_idx`` usage is safe only
        because exactly one thing is ever active at a time).

        Per-layer batched Cayley (one ``precompute_oft_r`` call per
        (layer, proj) covering all experts present in this chunk; no
        cross-layer batching). The buffers are *reused* across chunks within
        an OFT sync — orbit fans a single sync into multiple
        ``update_weights_from_tensor`` calls (one per ``get_hf_weight_chunks``
        bucket), and the same layer's experts can be split across those
        chunks. Reallocating per chunk would wipe earlier chunks' experts
        and leave most slots at zero (zero R ≠ identity → silent OFT-rotation
        loss → slow rollout/training logprob drift). Lazily allocate the
        per-FusedMoE buffer must already match the streamed tensor layout.
        If it does not, raise with diagnostics instead of silently replacing
        the graph-captured tensor and disabling CUDA Graph.
        """
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        moe_modules = self._find_fused_moe_modules()
        if not moe_modules:
            return

        for layer_id, ew_dict in expert_tensors.items():
            moe = moe_modules.get(layer_id)
            if moe is None or not ew_dict:
                continue

            gate_sample = _first_expert_oft_tensor(ew_dict, "gate_proj.oft_R")
            up_sample = _first_expert_oft_tensor(ew_dict, "up_proj.oft_R")
            down_sample = _first_expert_oft_tensor(ew_dict, "down_proj.oft_R")
            if gate_sample is None and up_sample is None and down_sample is None:
                continue
            is_split = gate_sample is not None and up_sample is not None
            is_legacy = gate_sample is not None and up_sample is None

            device = _get_fused_moe_weight_device(moe)
            num_local = moe.num_local_experts
            tp_rank, tp_size = moe.moe_tp_rank, moe.moe_tp_size
            oft_sample = gate_sample if gate_sample is not None else down_sample
            dtype = oft_sample.dtype

            num_blocks_w13 = gate_sample.shape[0] if gate_sample is not None else 0
            if down_sample is not None:
                num_blocks_w2_full = down_sample.shape[0]
                assert num_blocks_w2_full % tp_size == 0, (
                    f"w2 OFT num_blocks ({num_blocks_w2_full}) must be "
                    f"divisible by tp_size ({tp_size})"
                )
                blocks_per_tp = num_blocks_w2_full // tp_size
                w2_block_start = tp_rank * blocks_per_tp
                w2_block_end = w2_block_start + blocks_per_tp
            else:
                blocks_per_tp = 0

            # Reuse existing buffer if shape/dtype/device match — preserves
            # expert slots filled by an earlier chunk in this same sync.
            w13_shape = (num_local, num_blocks_w13, block_size, block_size)

            def _validate_w13_buffer(buf, projection):
                if num_blocks_w13 > 0 and not (
                    buf is not None
                    and tuple(buf.shape) == w13_shape
                    and buf.dtype == dtype
                    and buf.device == device
                ):
                    _raise_streamed_expert_oft_buffer_mismatch(
                        layer_id=layer_id,
                        projection=projection,
                        current_buffer=buf,
                        incoming_shape=w13_shape,
                        incoming_dtype=dtype,
                        incoming_device=device,
                    )
                if num_blocks_w13 == 0:
                    return None
                return buf

            def _resolve_expert_buffer(group_name):
                # slot_idx is None -> byte-identical to today: read the
                # already-bound module attribute (a pool ACTIVE-slot view on
                # the identity-boot path, or a disk-loaded private tensor on
                # the legacy short-circuit path -- either way, the exact
                # tensor forward reads). Any other slot_idx resolves straight
                # from the pool's group registry instead.
                if slot_idx is None:
                    return getattr(moe, group_name, None)
                groups = self.memory_pool._groups
                if group_name not in groups or layer_id not in groups[group_name]:
                    return None
                return self.memory_pool.slot(group_name, layer_id, slot_idx)

            if is_split:
                w1_oft_r = _validate_w13_buffer(_resolve_expert_buffer("w1_oft_r"), "w1")
                w3_oft_r = _validate_w13_buffer(_resolve_expert_buffer("w3_oft_r"), "w3")
                w13_oft_r = None
            elif is_legacy:
                w13_oft_r = _validate_w13_buffer(_resolve_expert_buffer("w13_oft_r"), "w13")
                w1_oft_r = None
                w3_oft_r = None
            else:
                w13_oft_r = None
                w1_oft_r = None
                w3_oft_r = None

            w2_shape = (num_local, blocks_per_tp, block_size, block_size)
            w2_oft_r = _resolve_expert_buffer("w2_oft_r")
            if blocks_per_tp > 0 and not (
                w2_oft_r is not None
                and tuple(w2_oft_r.shape) == w2_shape
                and w2_oft_r.dtype == dtype
                and w2_oft_r.device == device
            ):
                _raise_streamed_expert_oft_buffer_mismatch(
                    layer_id=layer_id,
                    projection="w2",
                    current_buffer=w2_oft_r,
                    incoming_shape=w2_shape,
                    incoming_dtype=dtype,
                    incoming_device=device,
                )
            elif blocks_per_tp == 0:
                w2_oft_r = None

            # Collect per-expert compacts in this chunk for batched Cayley
            # within this layer. Slicing w2 on the block dim happens BEFORE
            # Cayley so we don't waste compute on the other TP rank's blocks.
            local_ids: list[int] = []
            gate_compacts: list[torch.Tensor] = []
            up_compacts: list[torch.Tensor] = []
            down_compacts: list[torch.Tensor] = []
            for global_id, ew in ew_dict.items():
                local_id = moe._map_global_expert_id_to_local_expert_id(global_id)
                if local_id < 0 or local_id >= num_local:
                    continue
                local_ids.append(local_id)

                gate_compact = ew.get("gate_proj.oft_R")
                if gate_compact is not None:
                    gate_compacts.append(
                        gate_compact.to(device=device, dtype=dtype)
                    )
                else:
                    gate_compacts.append(None)

                up_compact = ew.get("up_proj.oft_R")
                if up_compact is not None:
                    up_compacts.append(
                        up_compact.to(device=device, dtype=dtype)
                    )
                else:
                    up_compacts.append(None)

                down_compact = ew.get("down_proj.oft_R")
                if down_compact is not None and blocks_per_tp > 0:
                    down_compacts.append(
                        down_compact[w2_block_start:w2_block_end].to(
                            device=device, dtype=dtype
                        )
                    )
                else:
                    down_compacts.append(None)

            def _scatter(buf, compacts, num_blocks):
                if buf is None or not any(c is not None for c in compacts):
                    return
                valid = [i for i, c in enumerate(compacts) if c is not None]
                cat = torch.cat([compacts[i] for i in valid], dim=0)
                R_stacked = precompute_oft_r(cat, block_size)
                R_per = R_stacked.view(
                    len(valid), num_blocks, block_size, block_size
                )
                for j, i in enumerate(valid):
                    buf[local_ids[i]] = R_per[j]

            # One Cayley call per (layer, proj). Cat across the experts
            # present in THIS chunk; scatter the result into the lazily
            # reused buffer. Experts not in this chunk keep whatever R the
            # earlier chunk wrote — that's the chunk-overwrite fix.
            _scatter(w1_oft_r, gate_compacts, num_blocks_w13)
            _scatter(w3_oft_r, up_compacts, num_blocks_w13)
            _scatter(w13_oft_r, gate_compacts, num_blocks_w13)
            _scatter(w2_oft_r, down_compacts, blocks_per_tp)

            if slot_idx is None:
                # Byte-identical to today: re-pin the (unchanged) module
                # attrs. Skipped for an explicit slot_idx (e.g. STAGING) --
                # the module attr stays bound to ACTIVE until activate().
                if is_split:
                    moe.w1_oft_r = w1_oft_r
                    moe.w3_oft_r = w3_oft_r
                    moe.w13_oft_r = None
                elif is_legacy:
                    moe.w13_oft_r = w13_oft_r
                    moe.w1_oft_r = None
                    moe.w3_oft_r = None
                if w2_oft_r is not None:
                    moe.w2_oft_r = w2_oft_r

            if _orbit_log_weight_sync_enabled():
                written_ids = sorted(set(local_ids))
                gate_written = sum(1 for compact in gate_compacts if compact is not None)
                down_written = sum(1 for compact in down_compacts if compact is not None)
                global_ids = sorted(ew_dict.keys())
                w13_changed, w13_max_delta = _expert_oft_delta_summary(
                    w13_oft_r, block_size
                )
                w2_changed, w2_max_delta = _expert_oft_delta_summary(
                    w2_oft_r, block_size
                )
                logger.info(
                    "OFT streamed expert apply layer=%s local_written=%s/%s "
                    "chunk_global_min=%s chunk_global_max=%s gate_compacts=%s "
                    "down_compacts=%s w13_changed=%s w13_max_delta=%.6e "
                    "w2_changed=%s w2_max_delta=%.6e",
                    layer_id,
                    len(written_ids),
                    num_local,
                    global_ids[0] if global_ids else None,
                    global_ids[-1] if global_ids else None,
                    gate_written,
                    down_written,
                    w13_changed,
                    w13_max_delta,
                    w2_changed,
                    w2_max_delta,
                )

    def _stage_fill(self, named_tensors, config, name, version):
        """Per-method hook for ``AdapterManager.stage_adapter``: partition
        ``named_tensors`` (raw checkpoint-name tensors -- the SAME format
        ``load_streamed_oft_adapter`` consumes) into dense vs expert, bake
        each dense tensor's R via Cayley (mirroring
        ``load_streamed_oft_adapter``'s dense dispatch, minus its chunking --
        chunking there is a streaming-transport perf optimization for the
        ACTIVE path's multi-RPC sync, not needed for a single-shot stage()
        fill), then write dense into STAGING via ``mem_pool.stage()`` and
        expert into STAGING via ``apply_streamed_expert_oft(slot_idx=
        staging_idx)`` (see Step 3 above).

        Scope: embed_tokens/lm_head/added_tokens are not part of the base
        buffer-group registry (never migrated onto it) -- tensors for them
        are SILENTLY SKIPPED below (the ``layer_id is None`` branch is a bare
        ``continue``: no fill written, no log emitted). DSV4-style expert
        names are unsupported and are REJECTED via assert (the fork's
        DeepSeekV4 model was dropped). Both are out of scope for the
        double-buffer phase's four fill locations; the embed/lm_head/
        added_tokens skip is a known gap, not a data-corrupting mishandling.
        """
        from sglang.srt.oft.mem_pool import normalize_merged_oft_weights
        from sglang.srt.oft.streamed_weight_loader import (
            _partition_expert_oft_tensors,
        )
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        memory_pool = self.memory_pool
        block_size = (
            config.get("oft_block_size", 32) if config else self.max_oft_block_size
        )
        if block_size != memory_pool.max_oft_block_size:
            raise ValueError(
                f"OFT staged update for '{name}' has block_size={block_size}, "
                f"but the server pool is allocated for --max-oft-block-size="
                f"{memory_pool.max_oft_block_size}; smaller or mixed block "
                f"sizes are unsupported."
            )
        other_adapters = sorted(
            r.adapter_name for r in self.refs.values() if r.adapter_name != name
        )
        if other_adapters:
            raise ValueError(
                f"Streamed OFT update for '{name}' while other adapters are "
                f"resident ({other_adapters}) is unsupported: the staged-update "
                "path targets the single active slot. Hot-swap combined with "
                "multi-tenant serving lands with the adapter_sync extension."
            )

        fused_expert_chunk, dsv4_expert_chunk, dense_named_tensors = (
            _partition_expert_oft_tensors(
                named_tensors, tp_rank=memory_pool.tp_rank
            )
        )
        assert not dsv4_expert_chunk, (
            "DSV4-style expert OFT staging is not supported (fork "
            "DeepSeekV4 model support was removed)"
        )

        # CanonicalOFT: pre-fuse split per-slice q/k/v (gate/up) tensors into
        # one stacked qkv_proj/gate_up_proj tensor when ALL siblings are
        # present in this call -- mirrors load_streamed_oft_adapter exactly.
        # Skipped entirely for an expert-only payload (nothing to normalize).
        if dense_named_tensors:
            dense_dict = dict(dense_named_tensors)
            if len(dense_dict) == len(dense_named_tensors):
                dense_named_tensors = list(
                    normalize_merged_oft_weights(
                        dense_dict, available_fused_targets=set(memory_pool.R_buffer)
                    ).items()
                )

        staged_dense = {}
        oft_modules = self.adapter_modules
        for tensor_name, tensor in dense_named_tensors:
            layer_id = get_layer_id(tensor_name)
            if layer_id is None:
                continue  # embeddings/lm_head: out of scope, see docstring.
            fused_target, slice_module, is_row_parallel, slice_index, split_count = (
                memory_pool._resolve_oft_tensor_plan(
                    tensor_name, oft_modules, layer_id
                )
            )
            compact_weight = tensor
            if is_row_parallel:
                compact_weight = memory_pool._slice_oft_compact_weight(
                    compact_weight, slice_module
                )
            target_device = memory_pool.R_buffer[fused_target][layer_id].device
            if compact_weight.device != target_device:
                compact_weight = compact_weight.to(target_device)
            r = precompute_oft_r(compact_weight, block_size)
            # FAIL LOUD on a same-key collision from DIFFERENT slices. This
            # dict is keyed by (fused_target, layer_id) and holds one whole-R
            # payload per key. If two DIFFERENT split slices of the same fused
            # target+layer (e.g. a partial-target q_proj + v_proj config)
            # reach here in one call WITHOUT all siblings present (so
            # normalize_merged_oft_weights above could not pre-fuse them into
            # one stacked tensor), keeping only the last would SILENTLY DROP
            # the earlier slice's rotation -- a weight-sync correctness bug.
            # The ACTIVE path avoids this by writing each slice into its own
            # sub-range via _write_oft_r_block(slice_index=...); a full
            # per-slice merge here is out of scope (orbit sends whole-adapter-
            # per-call today), so we detect the collision and raise instead.
            existing = staged_dense.get((fused_target, layer_id))
            if existing is not None and existing[2] != slice_index:
                raise RuntimeError(
                    f"stage_adapter: multiple split OFT slices for the same "
                    f"fused target {fused_target!r} (layer {layer_id}) arrived "
                    f"in one _stage_fill call without all siblings present to "
                    f"pre-fuse (slice_index {existing[2]} then {slice_index}); "
                    f"keeping only the last would silently drop the earlier "
                    f"slice's rotation. Per-slice staging merge is unsupported "
                    f"(orbit sends whole-adapter-per-call); send the fused "
                    f"target's siblings together in one payload."
                )
            staged_dense[(fused_target, layer_id)] = (
                r,
                block_size,
                slice_index,
                split_count,
            )

        # Always call stage() -- even with an empty dense payload -- so
        # _staged_version is set for an expert-only adapter too (otherwise a
        # later activate(version) would raise inactive_slot_busy).
        memory_pool.stage(version, staged_dense)

        if fused_expert_chunk:
            self.apply_streamed_expert_oft(
                fused_expert_chunk, block_size, slot_idx=memory_pool.staging_idx
            )

    def _bump_ref_version(self, name, version):
        for adapter_id, ref in self.refs.items():
            if ref.adapter_name == name:
                self.refs[adapter_id] = replace(ref, adapter_version=version)
                return
        raise ValueError(
            f"OFT adapter {name!r} not found in refs; cannot bump version"
        )

    def _make_streamed_ref(self, name, version, adapter_id=None, config=None):
        from sglang.srt.oft.oft_registry import OFTRef

        # Single-active convention: the id IS the name when the tokenizer supplies
        # none (mirrors _ensure_streaming_oft_adapter_slot). OFTRef rejects a None
        # adapter_id.
        if adapter_id is None:
            adapter_id = name
        oft_ref = OFTRef(
            adapter_id=adapter_id,
            adapter_name=name,
            adapter_path=name,
            # Pinned: a streamed adapter has no CPU-side OFTAdapter to re-page
            # from (the trainer pushed R straight into the slot), so evicting it
            # is unrecoverable. Upstream LoRA needs no equivalent -- every one of
            # its adapters is disk-backed. Pinning excludes the slot from
            # _acquire_buffer_slot's eviction candidates.
            pinned=True,
            adapter_version=version,
        )
        # Register per-request serving routing at the FIXED double-buffer
        # active_idx (NOT a dynamically allocate_buffer_slot()-ed slot: the DB
        # pool stages into staging_idx then copies into the fixed active_idx on
        # activate). This sets uid_to_buffer_id[adapter_id]=active_idx +
        # refs[adapter_id]=oft_ref, so the forward gather (get_buffer_id(uid))
        # resolves a /generate naming this adapter. Mirrors the proven IPC path's
        # register_streamed_adapter.
        result = self.register_streamed_adapter(
            oft_ref, self.memory_pool.active_idx, config
        )
        if not result.success:
            raise RuntimeError(
                f"Failed to register streamed OFT adapter {name!r}: "
                f"{result.error_message}"
            )
        return oft_ref

    def _clear_expert_oft(self):
        """Clear expert OFT tensors from all FusedMoE layers."""
        for moe in self._find_fused_moe_modules().values():
            moe.w13_oft_r = None
            moe.w2_oft_r = None

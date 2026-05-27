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
from contextlib import nullcontext
import os
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.oft.backend.base_backend import BaseOFTBackend
from sglang.srt.oft.backend.oft_registry import get_backend_from_name
from sglang.srt.oft.layers import BaseLayerWithOFT, get_oft_layer
from sglang.srt.oft.oft import OFTAdapter
from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.mem_pool import EMPTY_SLOT, OFTMemoryPool
from sglang.srt.oft.utils import (
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import replace_submodule
from sglang.srt.utils.hf_transformers_utils import AutoConfig

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import OFTUpdateOutput

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


class OFTManager:
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
        oft_paths: Optional[List[OFTRef]] = None,
        memory_saver_adapter=None,
        memory_saver_cpu_backup: bool = False,
    ):
        self.base_model: torch.nn.Module = base_model
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_ofts_per_batch: int = max_ofts_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.oft_r_dtype: torch.dtype = self._resolve_oft_r_dtype(
            dtype, server_args.oft_dtype
        )
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.oft_added_tokens_size: Optional[int] = None
        self.enable_oft_overlap_loading: Optional[bool] = (
            server_args.enable_oft_overlap_loading
        )
        self.memory_saver_adapter = memory_saver_adapter
        self.memory_saver_cpu_backup = memory_saver_cpu_backup

        # Store eviction policy from server args
        self.eviction_policy = server_args.oft_eviction_policy

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
            oft_paths=oft_paths,
        )

    def _weights_memory_saver_region(self):
        adapter = getattr(self, "memory_saver_adapter", None)
        if (
            adapter is None
            or not getattr(adapter, "enabled", False)
            or not getattr(self, "memory_saver_cpu_backup", False)
        ):
            return nullcontext()
        return adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=True,
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

    def create_oft_update_result(
        self, success: bool, error_message: str = ""
    ) -> "OFTUpdateOutput":
        from sglang.srt.managers.io_struct import OFTUpdateOutput

        return OFTUpdateOutput(
            success=success,
            error_message=error_message,
            loaded_adapters={
                oft_ref.oft_name: oft_ref.oft_path
                for oft_ref in self.oft_refs.values()
            },
        )

    def load_oft_adapter(self, oft_ref: OFTRef) -> "OFTUpdateOutput":
        """
        Load a single OFT adapter from the specified path.
        """
        assert (
            oft_ref.oft_name is not None and oft_ref.oft_path is not None
        ), "OFTRef must have both oft_name and oft_path set for loading."
        assert (
            oft_ref.oft_id not in self.ofts
        ), f"OFT adapter with ID {oft_ref.oft_id} is already loaded. This should have been verified before request is sent to the backend."

        try:
            # load configs
            new_adapter = OFTConfig(oft_ref.oft_path)
            self.validate_new_adapter(new_adapter, oft_ref)
            self.configs[oft_ref.oft_id] = new_adapter

            # load weights
            self.load_oft_weights(oft_ref)

            # keep metadata for displayed messages
            self.oft_refs[oft_ref.oft_id] = oft_ref
            self.num_pinned_ofts += int(oft_ref.pinned)
        except Exception as e:
            return self.create_oft_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_oft_update_result(success=True)

    def validate_new_adapter(self, oft_config: OFTConfig, oft_ref: OFTRef):
        """
        Validate if an adapter can be loaded into the current OFT memory pool and generate error if it is incompatible.
        """
        if oft_config.oft_added_tokens_size > 0:
            raise ValueError(
                f"OFT serving currently doesn't support adapters that add tokens to the vocabulary"
            )

        # Check if this OFT adapter is already loaded
        for existing_oft_ref in self.oft_refs.values():
            if oft_ref.oft_name == existing_oft_ref.oft_name:
                raise ValueError(
                    f"Failed to load OFT adapter {oft_ref.oft_name} because it is already loaded"
                )

            if oft_ref.oft_path == existing_oft_ref.oft_path:
                logger.warning(
                    f"{oft_ref.oft_path} is already loaded with name: {existing_oft_ref.oft_name}, "
                    f"but another copy is being loaded with name: {oft_ref.oft_name}"
                )

        if isinstance(oft_config.target_modules, list):
            validate_model_oft_target_modules(
                self.base_model,
                get_normalized_target_modules(oft_config.target_modules),
                source=f"adapter '{oft_ref.oft_name}' PEFT config",
            )

        # Check if the OFT adapter shape is compatible with the current OFT memory pool configuration.
        memory_pool = getattr(self, "memory_pool", None)
        incompatible = memory_pool and not memory_pool.can_support(oft_config)
        if incompatible:
            raise ValueError(
                f"OFT adapter {oft_ref.oft_name} with block_size {oft_config.block_size} is incompatible with the current "
                "OFT memory pool configuration. Please ensure that the OFT adapter's block_size is within the configured "
                "`--max-oft-block-size` and that the target modules are included in `--oft-target-modules`."
            )

        # Ensure pinned OFT adapters does not exceed maximal limit or cause starvation.
        if oft_ref.pinned and self.num_pinned_ofts >= self.max_ofts_per_batch - 1:
            raise ValueError(
                f"Failed to load OFT adapter {oft_ref.oft_name} as a pinned adapter. It is not allowed to pin all slots "
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
            config = OFTConfig.from_dict(config_dict)
            self.configs[oft_ref.oft_id] = config
            self.oft_refs[oft_ref.oft_id] = oft_ref
            # Register buffer slot mapping so inference can find this adapter
            self.memory_pool.uid_to_buffer_id[oft_ref.oft_id] = buffer_id
            self.memory_pool.buffer_id_to_uid[buffer_id] = oft_ref.oft_id
        except Exception as e:
            return self.create_oft_update_result(
                success=False,
                error_message=str(e),
            )
        return self.create_oft_update_result(success=True)

    def unload_streamed_adapter(self, oft_ref: OFTRef) -> "OFTUpdateOutput":
        """Unload an adapter that was registered via register_streamed_adapter.

        Unlike unload_oft_adapter, this does not try to access self.ofts
        since streamed adapters have no OFTAdapter object.
        """
        try:
            if oft_ref.oft_id in self.configs:
                del self.configs[oft_ref.oft_id]
            if oft_ref.oft_id in self.oft_refs:
                del self.oft_refs[oft_ref.oft_id]
            # Clean up buffer slot mapping
            if oft_ref.oft_id in self.memory_pool.uid_to_buffer_id:
                buffer_id = self.memory_pool.uid_to_buffer_id[oft_ref.oft_id]
                del self.memory_pool.uid_to_buffer_id[oft_ref.oft_id]
                self.memory_pool.buffer_id_to_uid[buffer_id] = EMPTY_SLOT
        except Exception as e:
            return self.create_oft_update_result(
                success=False,
                error_message=str(e),
            )
        return self.create_oft_update_result(success=True)

    def unload_oft_adapter(self, oft_ref: OFTRef) -> "OFTUpdateOutput":
        """
        Unload OFT adapters by their names.
        """

        adapter = self.configs.get(oft_ref.oft_id)
        oft_ref = self.oft_refs.get(oft_ref.oft_id)
        assert (
            adapter is not None and oft_ref is not None
        ), f"OFT adapter with ID {oft_ref.oft_id} is not loaded. This should have been verified before request is sent to the backend."

        if oft_ref.oft_id not in self.ofts:
            return self.unload_streamed_adapter(oft_ref)

        try:
            # Clear expert OFT weights from FusedMoE layers
            oft_adapter = self.ofts.get(oft_ref.oft_id)
            if oft_adapter is not None and any(
                hasattr(layer, "expert_weights") and layer.expert_weights
                for layer in oft_adapter.layers
            ):
                self._clear_expert_oft()

            del self.configs[oft_ref.oft_id]
            del self.ofts[oft_ref.oft_id]
            del self.oft_refs[oft_ref.oft_id]
            self.num_pinned_ofts -= int(oft_ref.pinned)
        except Exception as e:
            return self.create_oft_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_oft_update_result(success=True)

    def validate_oft_batch(self, oft_ids: set[Optional[str]]) -> bool:
        """
        Validate if the OFT IDs in the batch can be loaded into the current OFT memory pool.
        """
        if len(oft_ids) > self.max_ofts_per_batch:
            return False

        # skip pinned OFT check if no pinned OFT adapters are loaded.
        if self.num_pinned_ofts == 0:
            return True

        # counting the number of pinned OFT adapters in the batch.
        pinned_ofts_in_batch = 0
        for oft_id in oft_ids:
            if oft_id is not None:
                oft_ref = self.oft_refs.get(oft_id)
                assert (
                    oft_ref is not None
                ), f"OFT ID {oft_id} not found in oft_refs."
                pinned_ofts_in_batch += int(oft_ref.pinned)

        assert pinned_ofts_in_batch <= self.num_pinned_ofts, (
            f"Number of pinned OFT adapters in the batch ({pinned_ofts_in_batch}) exceeds the total number of pinned adapters "
            f"({self.num_pinned_ofts}). This indicates a bug in the OFT loading logic."
        )

        required_slots = len(oft_ids) - pinned_ofts_in_batch
        mem_pool_vacancy = self.memory_pool.max_ofts_per_batch - self.num_pinned_ofts

        return required_slots <= mem_pool_vacancy

    def fetch_new_ofts(
        self, new_ofts: set[Optional[str]], running_ofts: set[Optional[str]] = set()
    ):
        # Load active ofts into oft memory pool
        cur_uids = new_ofts | running_ofts

        assert len(cur_uids) <= self.max_ofts_per_batch
        self.memory_pool.prepare_oft_batch(
            cur_uids=cur_uids,
            oft_adapters=self.ofts,
            oft_modules=self.oft_modules,
            oft_refs=self.oft_refs.copy(),
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

        weight_indices = [0] * len(forward_batch.oft_ids)
        oft_block_sizes = [0] * self.max_ofts_per_batch
        for i, uid in enumerate(forward_batch.oft_ids):
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                if uid in self.ofts:
                    oft_block_sizes[weight_indices[i]] = self.ofts[uid].block_size
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
        """
        Update all OFT modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in enumerate(self.oft_modules):
            for module_name, module in layer_modules.items():
                target_module = get_target_module_name(
                    module_name, self.memory_pool.target_modules
                )
                module.set_oft_info(
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                    ),
                )

        # Update embedding layer if present
        if self.embed_tokens_module is not None:
            self.embed_tokens_module.set_oft_info(
                self.memory_pool.get_embedding_tensor("added_tokens"),
                self.memory_pool.get_embedding_tensor("embed_tokens"),
            )

        # Update lm_head layer if present
        if self.lm_head_module is not None:
            self.lm_head_module.set_oft_info(
                self.memory_pool.get_embedding_tensor("lm_head"),
            )

    def init_state(
        self,
        max_oft_block_size: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        oft_paths: Optional[List[OFTRef]] = None,
    ):
        """
        Initialize the internal (mutable) state of the OFTManager.

        When `oft_paths` is provided and not empty, it might be used for inferring OFT shape info such as
        the target modules and max_oft_block_size.
        """

        assert oft_paths or (
            max_oft_block_size is not None and target_modules is not None
        ), "When no initial --oft-paths is provided, you need to specify both --max-oft-block-size and --oft-target-modules for OFT initialization."

        self.init_oft_adapters(oft_paths)
        self.init_oft_shapes(
            max_oft_block_size=max_oft_block_size,
            target_modules=target_modules,
        )
        self.init_oft_modules()
        self.init_memory_pool()
        self.update_oft_info()
        self._init_identity_expert_oft_for_cuda_graph()

        wrapped_module_count = sum(
            len(layer_modules) for layer_modules in self.oft_modules
        )
        wrapped_layer_count = sum(
            1 for layer_modules in self.oft_modules if layer_modules
        )
        loaded_adapter_names = sorted(
            str(oft_ref.oft_name) for oft_ref in self.oft_refs.values()
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

    def init_oft_adapters(self, oft_paths: Optional[List[OFTRef]] = None):
        # Configs of all active OFT adapters, indexed by OFT ID.
        self.configs: Dict[str, OFTConfig] = {}

        # OFT adapter weights cached in CPU memory, indexed by OFT ID.
        self.ofts: Dict[str, OFTAdapter] = {}

        # Mapping from OFT ID to OFTRef object.
        self.oft_refs: Dict[str, OFTRef] = {}

        # Count of pinned OFT adapters.
        self.num_pinned_ofts: int = 0

        if oft_paths:
            for oft_ref in oft_paths:
                result = self.load_oft_adapter(oft_ref)
                if not result.success:
                    raise RuntimeError(
                        f"Failed to load OFT adapter {oft_ref.oft_name}: {result.error_message}"
                    )

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

        for oft_id, config in self.configs.items():
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
                        oft_name = self.oft_refs[oft_id].oft_name
                        raise ValueError(
                            f"OFT adapter '{oft_name}' uses "
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
            oft_name = self.oft_refs[oft_id].oft_name
            validate_model_oft_target_modules(
                self.base_model,
                adapter_target_modules,
                source=f"adapter '{oft_name}' PEFT config",
            )

            if target_modules is not None:
                # When `--oft-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    raise ValueError(
                        f"OFT adapter '{oft_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --oft-target-modules {sorted(self.target_modules)}. "
                        f"Please update --oft-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
            else:
                # Otherwise, infer target_modules from adapter configs.
                self.target_modules.update(adapter_target_modules)

        if max_oft_block_size is not None:
            self.max_oft_block_size = max_oft_block_size
        else:
            self.max_oft_block_size = max(
                [x.block_size for x in self.configs.values()],
                default=0,
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
            oft_ref.oft_id,
            self.configs[oft_ref.oft_id],
            self.base_hf_config,
            self.load_config,
            self.oft_backend,
        )
        oft_adapter.initialize_weights()

        # If we want to overlap loading OFT adapters with compute, they must be pinned in CPU memory
        if self.enable_oft_overlap_loading:
            oft_adapter.pin_weights_in_cpu()

        self.ofts[oft_ref.oft_id] = oft_adapter

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
            oft_ref.oft_id,
            self.configs[oft_ref.oft_id],
            self.base_hf_config,
            self.load_config,
            self.oft_backend,
        )
        oft_adapter.initialize_weights_from_tensors(tensors)
        self.ofts[oft_ref.oft_id] = oft_adapter

        # Set expert OFT weights on FusedMoE layers if present
        if any(
            hasattr(layer, "expert_weights") and layer.expert_weights
            for layer in oft_adapter.layers
        ):
            self._set_expert_oft(oft_adapter)

    def load_oft_adapter_from_tensors(
        self,
        oft_ref: OFTRef,
        tensors: Dict[str, torch.Tensor],
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
    ) -> "OFTUpdateOutput":
        """Not supported. Pure-inference users should register adapters via
        ``/load_oft_adapter`` (disk path); training-time adapter sync goes
        through ``update_weights_from_tensor(load_format='oft_adapter')``."""
        raise NotImplementedError(
            "OFT load-from-tensors over HTTP is not supported. Use "
            "/load_oft_adapter with a disk path for pure inference, or "
            "update_weights_from_tensor(load_format='oft_adapter') for "
            "orbit/verl-style streamed sync."
        )

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
            oft_modules=self.oft_modules,
            external_target_modules=external_target_modules,
            eviction_policy=self.eviction_policy,
            oft_added_tokens_size=self.oft_added_tokens_size,
            memory_saver_adapter=self.memory_saver_adapter,
            memory_saver_cpu_backup=self.memory_saver_cpu_backup,
        )
        logger.info(
            "Using %s for OFT R buffers (model dtype %s).",
            self.oft_r_dtype,
            self.dtype,
        )

        # Initializing memory pool with base model
        self.fetch_new_ofts({None})

    def set_oft_module(self, module_name, module):
        oft_module = get_oft_layer(module, self.oft_backend)
        replace_submodule(self.base_model, module_name, oft_module)
        return oft_module

    def init_oft_modules(self):
        # Look-up table that maps (layer_index, module_name) to the corresponding OFT module.
        num_hidden_layers = _get_hf_config_attr(
            self.base_hf_config, "num_hidden_layers"
        )
        self.oft_modules: List[Dict[str, BaseLayerWithOFT]] = [
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
                self.oft_modules[layer_id][module_name] = self.set_oft_module(
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

    def _find_fused_moe_modules(self):
        """Lazily find and cache all FusedMoE modules indexed by layer_id."""
        if hasattr(self, "_moe_modules"):
            return self._moe_modules
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        self._moe_modules = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, FusedMoE):
                layer_id = get_layer_id(name)
                if layer_id is not None:
                    self._moe_modules[layer_id] = module
        return self._moe_modules

    def _find_dsv4_moe_modules(self):
        """Lazily find DeepSeek V4 MoE modules indexed by layer_id."""
        if hasattr(self, "_dsv4_moe_modules"):
            return self._dsv4_moe_modules
        try:
            from sglang.srt.models.deepseek_v4 import DeepSeekV4MoE
        except Exception:
            self._dsv4_moe_modules = {}
            return self._dsv4_moe_modules

        self._dsv4_moe_modules = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, DeepSeekV4MoE):
                layer_id = get_layer_id(name)
                if layer_id is not None:
                    self._dsv4_moe_modules[layer_id] = module
        return self._dsv4_moe_modules

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
        dsv4_targets = target_modules & {"w1", "w2", "w3"}
        if not (init_w13 or init_w2 or dsv4_targets):
            return

        block_size = self.max_oft_block_size
        if block_size <= 0:
            return

        initialized = False
        for moe in self._find_fused_moe_modules().values():
            device = _get_fused_moe_weight_device(moe)
            dtype = self.oft_r_dtype

            if init_w13:
                if moe.hidden_size % block_size != 0:
                    raise ValueError(
                        f"MoE w13 OFT input dim {moe.hidden_size} is not "
                        f"divisible by block_size {block_size}"
                    )
                w13_shape = (
                    moe.num_local_experts,
                    moe.hidden_size // block_size,
                    block_size,
                    block_size,
                )
                with self._weights_memory_saver_region():
                    for attr in ("w1_oft_r", "w3_oft_r"):
                        if getattr(moe, attr, None) is None:
                            setattr(
                                moe,
                                attr,
                                torch.empty(*w13_shape, device=device, dtype=dtype),
                            )
                            _fill_expert_oft_identity(getattr(moe, attr))
                            initialized = True
                # Split buffers supersede the legacy fused buffer.
                moe.w13_oft_r = None

            if init_w2 and getattr(moe, "w2_oft_r", None) is None:
                w2_input_dim = moe.intermediate_size_per_partition
                if w2_input_dim % block_size != 0:
                    raise ValueError(
                        f"MoE w2 OFT input dim {w2_input_dim} is not "
                        f"divisible by block_size {block_size}"
                    )
                w2_shape = (
                    moe.num_local_experts,
                    w2_input_dim // block_size,
                    block_size,
                    block_size,
                )
                with self._weights_memory_saver_region():
                    moe.w2_oft_r = torch.empty(
                        *w2_shape, device=device, dtype=dtype
                    )
                _fill_expert_oft_identity(moe.w2_oft_r)
                initialized = True

        if initialized:
            logger.info(
                "Initialized identity expert OFT buffers for CUDA graph capture."
            )

        if not dsv4_targets:
            return

        initialized = False
        with self._weights_memory_saver_region():
            for moe in self._find_dsv4_moe_modules().values():
                for proj in sorted(dsv4_targets):
                    if moe.ensure_dsv4_expert_oft_r(
                        proj,
                        block_size=block_size,
                        dtype=self.oft_r_dtype,
                    ):
                        initialized = True

        if initialized:
            logger.info("Initialized identity DSV4 expert OFT buffers.")

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

    def apply_streamed_expert_oft(self, expert_tensors, block_size):
        """Set FusedMoE expert OFT R from streamed-sync compact tensors.

        expert_tensors: {layer_id: {global_expert_id: {"gate_proj.oft_R": t,
                                                       "down_proj.oft_R": t}}}.
        Keeps OFT external — only writes moe.w13_oft_r / moe.w2_oft_r,
        never merges into base weights.

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

            if is_split:
                w1_oft_r = _validate_w13_buffer(getattr(moe, "w1_oft_r", None), "w1")
                w3_oft_r = _validate_w13_buffer(getattr(moe, "w3_oft_r", None), "w3")
                w13_oft_r = None
            elif is_legacy:
                w13_oft_r = _validate_w13_buffer(getattr(moe, "w13_oft_r", None), "w13")
                w1_oft_r = None
                w3_oft_r = None
            else:
                w13_oft_r = None
                w1_oft_r = None
                w3_oft_r = None

            w2_shape = (num_local, blocks_per_tp, block_size, block_size)
            w2_oft_r = getattr(moe, "w2_oft_r", None)
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

    def apply_streamed_dsv4_expert_oft(self, expert_tensors, block_size):
        """Set DeepSeek V4 expert OFT R from streamed compact tensors.

        This mirrors the FusedMoE expert path, but DSV4 keeps separate
        w1/w2/w3 native-quant expert linears instead of fused w13/w2 weights.
        The active R buffers live on ``DeepSeekV4MoE`` and are applied after
        routing, so request-level OFT segment metadata is not used for
        expert-local token subsets.
        """
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        moe_modules = self._find_dsv4_moe_modules()
        if not moe_modules:
            return

        for layer_id, ew_dict in expert_tensors.items():
            moe = moe_modules.get(layer_id)
            if moe is None or not ew_dict:
                continue

            def local_compact_for_expert_tp(
                proj: str, compact: torch.Tensor
            ) -> torch.Tensor:
                if proj != "w2":
                    return compact
                moe_tp_size = int(getattr(moe, "moe_tp_size", 1))
                if moe_tp_size <= 1:
                    return compact
                moe_tp_rank = int(getattr(moe, "moe_tp_rank", 0))
                if compact.shape[0] % moe_tp_size != 0:
                    raise ValueError(
                        "DSV4 w2 OFT block mismatch for expert TP: "
                        f"layer_id={layer_id}, blocks={compact.shape[0]}, "
                        f"moe_tp_size={moe_tp_size}"
                    )
                blocks_per_tp = compact.shape[0] // moe_tp_size
                start = moe_tp_rank * blocks_per_tp
                end = start + blocks_per_tp
                return compact[start:end]

            samples = {
                proj: _first_expert_oft_tensor(ew_dict, f"{proj}.oft_R")
                for proj in ("w1", "w2", "w3")
            }
            samples = {
                proj: sample
                for proj, sample in samples.items()
                if sample is not None
            }
            if not samples:
                continue

            for proj, sample in samples.items():
                sample = local_compact_for_expert_tp(proj, sample)
                moe.ensure_dsv4_expert_oft_r(
                    proj,
                    block_size=block_size,
                    dtype=sample.dtype,
                    sample=sample,
                )

            local_ids: list[int] = []
            compacts_by_proj: dict[str, list[torch.Tensor | None]] = {
                "w1": [],
                "w2": [],
                "w3": [],
            }
            for global_id, ew in ew_dict.items():
                local_id = moe._map_global_expert_id_to_local_expert_id(global_id)
                if local_id < 0:
                    continue
                local_ids.append(local_id)
                for proj in ("w1", "w2", "w3"):
                    compact = ew.get(f"{proj}.oft_R")
                    if compact is None:
                        compacts_by_proj[proj].append(None)
                    else:
                        compact = local_compact_for_expert_tp(proj, compact)
                        compacts_by_proj[proj].append(
                            compact.to(
                                device=moe.dsv4_expert_oft_device(),
                                dtype=getattr(moe, f"{proj}_oft_r").dtype,
                            )
                        )

            for proj, compacts in compacts_by_proj.items():
                valid = [i for i, c in enumerate(compacts) if c is not None]
                if not valid:
                    continue
                out = getattr(moe, f"{proj}_oft_r")
                cat = torch.cat([compacts[i] for i in valid], dim=0)
                r_stacked = precompute_oft_r(cat, block_size)
                r_per = r_stacked.view(
                    len(valid),
                    out.shape[1],
                    block_size,
                    block_size,
                )
                for j, i in enumerate(valid):
                    out[local_ids[i]] = r_per[j]

    def _clear_expert_oft(self):
        """Clear expert OFT tensors from all FusedMoE layers."""
        for moe in self._find_fused_moe_modules().values():
            moe.w13_oft_r = None
            moe.w2_oft_r = None
        for moe in self._find_dsv4_moe_modules().values():
            moe.w1_oft_r = None
            moe.w2_oft_r = None
            moe.w3_oft_r = None

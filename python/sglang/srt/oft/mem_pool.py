import logging
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.distributed import divide
from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.utils import get_stacked_multiply as _lora_get_stacked_multiply
from sglang.srt.oft.layers import BaseLayerWithOFT
from sglang.srt.oft.oft import OFTAdapter
from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.utils import (
    EMBEDDING_NAMES,
    ROW_PARALLELISM_LINEAR_OFT_NAMES,
    get_hidden_dim,
    get_hf_config_attr,
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


class EmptySlot:
    """Singleton class to represent an empty slot in the memory pool."""

    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()


# Fused HF projection leaf -> tuple of split leaves. Inverse of the LoRA
# params_mapping in sglang.srt.lora.utils, kept local here because the OFT
# load path needs the forward direction (split-leaf -> fused).
MERGED_OFT_PROJ_GROUPS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def get_stacked_multiply(module_name: str) -> int:
    """Like :func:`sglang.srt.lora.utils.get_stacked_multiply` but accepts a
    dotted FQN (``model.layers.0.self_attn.qkv_proj``) as well as the bare
    leaf."""
    return _lora_get_stacked_multiply(module_name.rsplit(".", 1)[-1])


def _replace_leaf_module_name(key: str, old_leaf: str, new_leaf: str) -> str:
    token = "." + old_leaf + "."
    if token not in key:
        return key
    return key.replace(token, "." + new_leaf + ".", 1)


def normalize_merged_oft_weights(
    weights: Dict[str, torch.Tensor],
    *,
    available_fused_targets: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Stack split CanonicalOFT weights only when the runtime has a fused target.

    ``available_fused_targets`` should be ``set(memory_pool.R_buffer)`` on runtime
    load paths. ``None`` preserves the legacy behavior for callers that do not
    have topology context.
    """
    result: Dict[str, torch.Tensor] = {}
    consumed: Set[str] = set()
    for key, tensor in weights.items():
        if key in consumed:
            continue
        for fused_leaf, split_leaves in MERGED_OFT_PROJ_GROUPS.items():
            if available_fused_targets is not None and fused_leaf not in available_fused_targets:
                continue
            matched_split_leaf = next(
                (
                    split_leaf
                    for split_leaf in split_leaves
                    if "." + split_leaf + "." in key
                ),
                None,
            )
            if matched_split_leaf is None:
                continue
            first_leaf = split_leaves[0]
            first_key = _replace_leaf_module_name(key, matched_split_leaf, first_leaf)
            sibling_keys = [
                _replace_leaf_module_name(first_key, first_leaf, leaf)
                for leaf in split_leaves
            ]
            if not all(sibling in weights for sibling in sibling_keys):
                continue
            fused_key = _replace_leaf_module_name(first_key, first_leaf, fused_leaf)
            result[fused_key] = torch.cat(
                [weights[sibling] for sibling in sibling_keys], dim=0
            )
            consumed.update(sibling_keys)
            break
        else:
            result[key] = tensor
    return result


def resolve_fused_oft_slice(key: str) -> Tuple[str, int, int]:
    """If ``key`` references a split CanonicalOFT projection
    (``q_proj.oft_r`` etc.) return ``(fused_key, slice_index, split_count)``.
    For non-split keys, return ``(key, 0, 1)``."""
    for fused_leaf, split_leaves in MERGED_OFT_PROJ_GROUPS.items():
        for index, split_leaf in enumerate(split_leaves):
            token = "." + split_leaf + "."
            if token in key:
                fused = key.replace(token, "." + fused_leaf + ".", 1)
                return fused, index, len(split_leaves)
    return key, 0, 1


def _contains_leaf(key: str, leaf: str) -> bool:
    return f".{leaf}." in key


def _broadcast_legacy_single_R(compact: torch.Tensor, stacked_multiply: int) -> torch.Tensor:
    """Broadcast a single shared-R OFT tensor (legacy fused ``qkv_proj.oft_r``)
    into ``stacked_multiply`` identical slices along the blocks dimension.
    Used to load pre-fix checkpoints into the CanonicalOFT split buffers."""
    if stacked_multiply == 1:
        return compact
    return compact.repeat(stacked_multiply, *([1] * (compact.ndim - 1)))


def _fill_identity(buffer_view: torch.Tensor, block_size: int) -> None:
    """Fill an OFT R buffer view with block-diagonal identity (passthrough).

    OFT applies R to the input via per-block matmul. The only R that leaves
    the input unchanged is identity — a zero R would map every input to zero.
    Modules without explicit adapter weights must therefore default to
    identity, not zero.

    `buffer_view` has shape (num_blocks, max_block_size, max_block_size). Only
    the top-left (block_size, block_size) sub-block of each block is read by
    the kernel (the rest is padding for slots whose adapter uses a smaller
    block size), so we set the active sub-block to eye and leave padding zero.
    """
    buffer_view.zero_()
    if block_size <= 0:
        return
    eye = torch.eye(
        block_size, dtype=buffer_view.dtype, device=buffer_view.device
    )
    buffer_view[:, :block_size, :block_size] = eye


class OFTMemoryPool:
    """Memory pool for OFT adapter weights.

    Unlike LoRA which has separate A and B buffers, OFT uses a single R buffer
    per module, storing compact upper-triangular parameters of skew-symmetric blocks.
    """

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_ofts_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_oft_block_size: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
        eviction_policy: str,
        oft_added_tokens_size: int,
        oft_modules: Optional[List[Dict[str, BaseLayerWithOFT]]] = None,
        external_target_modules: Optional[Set[str]] = None,
        memory_saver_adapter=None,
        memory_saver_cpu_backup: bool = False,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = get_hf_config_attr(
            base_hf_config, "num_hidden_layers"
        )
        self.max_ofts_per_batch: int = max_ofts_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.max_oft_block_size: int = max_oft_block_size
        self.target_modules: Set[str] = target_modules
        self.external_target_modules: Set[str] = external_target_modules or set()
        self.oft_modules: Optional[List[Dict[str, BaseLayerWithOFT]]] = oft_modules
        self.oft_added_tokens_size: int = oft_added_tokens_size
        self.embedding_dim: int = get_hf_config_attr(base_hf_config, "hidden_size")
        self.memory_saver_adapter = memory_saver_adapter
        self.memory_saver_cpu_backup = memory_saver_cpu_backup

        # Eviction policy (reuse LoRA's adapter-agnostic implementation)
        self.eviction_policy = get_eviction_policy(eviction_policy)

        # Single R buffer per module (unlike LoRA's A + B).
        # R_buffer maps module_name -> list of per-layer tensors with shape
        #   (max_ofts_per_batch, stacked_multiply * num_blocks, block_size, block_size)
        # where num_blocks = r_dim // block_size and stacked_multiply is 3 for
        # qkv_proj, 2 for gate_up_proj, 1 elsewhere.
        # Stores precomputed orthogonal rotation matrices (not compact weights).
        #
        # CanonicalOFT (2026-05-15) attaches independent per-slice rotations:
        # qkv_proj stores [R_q ; R_k ; R_v] stacked along the block dim and the
        # forward kernel selects slice s via ``s * num_blocks + block_idx``.
        # Legacy single-R checkpoints (where Bridge HF export emitted a shared
        # R duplicated into q/k/v) load via ``_broadcast_legacy_single_R``,
        # which writes the same R into all three slices.
        self.R_buffer: Dict[str, List[torch.Tensor]] = {}

        # Embedding and lm_head R buffers (not per-layer, analogous to LoRA's embedding_A/B)
        self.embedding_R_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_R_buffer: Dict[str, torch.Tensor] = {}

        # Extra token embeddings buffer
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        # UID <-> buffer ID mapping
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}
        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
            EMPTY_SLOT
        ] * self.max_ofts_per_batch

        self.init_buffers(base_model)

    def _weights_memory_saver_region(self):
        adapter = self.memory_saver_adapter
        if (
            adapter is None
            or not getattr(adapter, "enabled", False)
            or not self.memory_saver_cpu_backup
        ):
            return nullcontext()
        return adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=True,
        )

    def can_support(self, config: Union[OFTConfig, list[OFTConfig]]) -> bool:
        """Check if the memory pool can support the given OFT adapter(s)."""

        def _can_support(config: OFTConfig) -> bool:
            if config.block_size > self.max_oft_block_size:
                return False
            if config.oft_added_tokens_size > self.oft_added_tokens_size:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            if "all" in target_module_names:
                return True
            return target_module_names.issubset(
                self.target_modules | self.external_target_modules
            )

        if isinstance(config, OFTConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def get_oft_R_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_oft_block_size: int,
        layer_idx: int,
        module: Optional[BaseLayerWithOFT] = None,
    ) -> Tuple[int]:
        """Get the R buffer shape for regular (non-embedding) modules.

        Stores precomputed orthogonal rotation matrices R with shape
        (max_ofts_per_batch, num_blocks, block_size, block_size).

        For OFT input rotation, R operates on the input dimension.
        For RowParallel layers, input is split across TP ranks.

        Fused ColumnParallel targets (qkv_proj, gate_up_proj) store a single
        shared R — see class docstring for the rationale.
        """
        if module is not None:
            input_dim = module.get_oft_input_dim()
        else:
            input_dim, _ = get_hidden_dim(
                module_name, self.base_hf_config, base_model, layer_idx
            )
            if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_OFT_NAMES:
                input_dim = divide(input_dim, self.tp_size)
        num_blocks = input_dim // max_oft_block_size
        # CanonicalOFT: fused targets (qkv_proj / gate_up_proj) hold one R per
        # slice (q/k/v or gate/up), stacked along the blocks dim.
        stacked_multiply = get_stacked_multiply(module_name)
        shape = (
            self.max_ofts_per_batch,
            stacked_multiply * num_blocks,
            max_oft_block_size,
            max_oft_block_size,
        )
        if stacked_multiply > 1 and layer_idx == 0:
            # One line per merged target on init; layer_idx==0 keeps it once
            # per module (rather than once per layer).
            logger.info(
                "OFT R buffer for %s: stacked_multiply=%d shape=%s",
                module_name,
                stacked_multiply,
                shape,
            )
        return shape

    def get_embedding_oft_R_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_oft_block_size: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """Get the R buffer shape for embedding modules (embed_tokens, lm_head).

        Stores precomputed orthogonal rotation matrices R with shape
        (max_ofts_per_batch, num_blocks, block_size, block_size).

        embed_tokens: R operates on output_dim (OFT rotates embedding output).
        lm_head: R operates on input_dim (OFT rotates lm_head input).
        """
        input_dim, output_dim = get_hidden_dim(
            module_name,
            self.base_hf_config,
            base_model,
            0,
            self.oft_added_tokens_size,
        )
        if module_name == "embed_tokens":
            r_dim = output_dim  # output rotation for embedding
        else:
            r_dim = input_dim  # input rotation for lm_head
        # TP not supported for embeddings yet.
        num_blocks = r_dim // max_oft_block_size
        return (self.max_ofts_per_batch, num_blocks, max_oft_block_size, max_oft_block_size)

    def init_buffers(self, base_model: torch.nn.Module):
        device = next(base_model.parameters()).device
        module_lookup: Dict[Tuple[str, int], BaseLayerWithOFT] = {}
        if self.oft_modules is not None:
            for layer_idx, layer_modules in enumerate(self.oft_modules):
                for full_module_name, module in layer_modules.items():
                    try:
                        target_module = get_target_module_name(
                            full_module_name, self.target_modules
                        )
                    except ValueError:
                        continue
                    module_lookup.setdefault((target_module, layer_idx), module)

        def _make_identity_r_buffer(shape, dtype, device):
            """Create R buffer initialized to identity (safe default for OFT passthrough).

            Shape: (max_ofts, c*num_blocks, block_size, block_size).
            Each (block_size, block_size) sub-matrix is set to I so that x@R = x.
            """
            buf = torch.zeros(shape, dtype=dtype, device=device)
            block_size = shape[-1]
            if block_size > 0:
                eye = torch.eye(block_size, dtype=dtype, device=device)
                # Broadcast identity into every (block_size, block_size) sub-matrix
                buf[:, :, :, :] = eye
            return buf

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            target_modules: Set[str],
            get_shape_fn: Callable[[str, torch.nn.Module, int, int], Tuple[int]],
        ):
            target_modules = (
                target_modules - set(EMBEDDING_NAMES) - self.external_target_modules
            )
            for module_name in target_modules:
                buffer[module_name] = [
                    _make_identity_r_buffer(
                        get_shape_fn(
                            module_name,
                            base_model,
                            self.max_oft_block_size,
                            idx,
                            module_lookup.get((module_name, idx)),
                        ),
                        dtype=self.dtype,
                        device=device,
                    )
                    for idx in range(self.num_layer)
                ]

        def init_embedding_buffer(
            buffer: Dict[str, torch.Tensor],
            target_modules: Set[str],
            get_shape_fn: Callable[[str, torch.nn.Module, int, int], Tuple[int]],
        ):
            target_modules = target_modules & set(EMBEDDING_NAMES)
            for module_name in target_modules:
                buffer[module_name] = _make_identity_r_buffer(
                    get_shape_fn(
                        module_name,
                        base_model,
                        self.max_oft_block_size,
                        0,
                    ),
                    dtype=self.dtype,
                    device=device,
                )

        with self._weights_memory_saver_region():
            if self.oft_added_tokens_size > 0:
                self.new_embeddings_buffer["input_embeddings"] = torch.empty(
                    (
                        self.max_ofts_per_batch,
                        self.oft_added_tokens_size,
                        self.embedding_dim,
                    ),
                    dtype=self.dtype,
                    device=device,
                )

            if "embed_tokens" in self.target_modules:
                init_embedding_buffer(
                    self.embedding_R_buffer,
                    self.target_modules,
                    self.get_embedding_oft_R_shape,
                )

            if "lm_head" in self.target_modules:
                init_embedding_buffer(
                    self.lm_head_R_buffer,
                    self.target_modules,
                    self.get_embedding_oft_R_shape,
                )

            init_buffer(
                self.R_buffer,
                self.target_modules,
                self.get_oft_R_shape,
            )

    def prepare_oft_batch(
        self,
        cur_uids: Set[Optional[str]],
        oft_adapters: Dict[str, OFTAdapter],
        oft_modules: List[Dict[str, BaseLayerWithOFT]],
        oft_refs: Dict[str, OFTRef],
        oft_embed_tokens_module: Optional[BaseLayerWithOFT],
        oft_lm_head_module: Optional[BaseLayerWithOFT],
    ):
        def get_available_buffer_slot():
            # 1. Prioritize empty slots
            for buffer_id in range(self.max_ofts_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id

            # 2. Memory pool is full, need to evict
            candidates = set()
            for buffer_id in range(self.max_ofts_per_batch):
                uid = self.buffer_id_to_uid[buffer_id]
                if uid in cur_uids:
                    continue
                if uid is not None:
                    oft_ref = oft_refs.get(uid)
                    if oft_ref and oft_ref.pinned:
                        continue
                candidates.add(uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots found. Please ensure the number of "
                    "active (pinned) OFT adapters is less than max_ofts_per_batch."
                )

            # Prefer evicting OFT adapters over base model (None)
            non_none_candidates = candidates - {None}
            candidates_to_use = (
                non_none_candidates if non_none_candidates else candidates
            )

            victim_uid = self.eviction_policy.select_victim(candidates_to_use)
            victim_buffer_id = self.uid_to_buffer_id[victim_uid]
            self.uid_to_buffer_id.pop(victim_uid)
            self.eviction_policy.remove(victim_uid)
            self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
            logger.debug(
                f"Evicting OFT {victim_uid} from buffer slot {victim_buffer_id}."
            )
            return victim_buffer_id

        # Mark all adapters in current batch as used (for LRU tracking)
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                oft_adapter = oft_adapters.get(uid, None)
                self.load_oft_weight_to_buffer(
                    uid,
                    buffer_id,
                    oft_adapter,
                    oft_modules,
                    oft_embed_tokens_module,
                    oft_lm_head_module,
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_oft_weight_to_buffer(
        self,
        uid: Optional[str],
        buffer_id: int,
        oft_adapter: Optional[OFTAdapter],
        oft_modules: List[Dict[str, BaseLayerWithOFT]],
        oft_embed_tokens_module: Optional[BaseLayerWithOFT],
        oft_lm_head_module: Optional[BaseLayerWithOFT],
    ):
        def precompute_and_store_R(
            buffer_view: torch.Tensor,
            compact_weight: Optional[torch.Tensor],
            block_size: int,
        ):
            """Precompute R from compact weights and store in buffer.

            Replaces load_weight_tensor: instead of storing compact 2D weights,
            precomputes the full orthogonal rotation matrices via Cayley transform
            and stores the 3D result (num_blocks, block_size, block_size).

            When compact_weight is None (the adapter has no weights for this
            module), the buffer is filled with the block-diagonal *identity*
            rotation. Identity is the only correct passthrough for OFT — a
            zero R would map every input to zero and silently kill the layer.

            Args:
                buffer_view: (total_blocks_buffer, block_size, block_size) GPU tensor
                compact_weight: (num_blocks_adapter, n_elements) CPU tensor, or None
                block_size: adapter's block size
            """
            from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

            if compact_weight is None:
                _fill_identity(buffer_view, block_size)
                return

            R = precompute_oft_r(
                compact_weight.to(buffer_view.device), block_size
            )
            num_blocks_adapter = R.shape[0]
            total_blocks_buffer = buffer_view.shape[0]

            buffer_view.zero_()
            if num_blocks_adapter == 1 and total_blocks_buffer > 1:
                # Block-share: replicate single R block to all positions
                buffer_view[:, :block_size, :block_size] = R[0]
            else:
                buffer_view[:num_blocks_adapter, :block_size, :block_size] = R

        if uid is None:
            # Base model: zero all R buffers (kernel does passthrough when block_size=0)
            for i in range(self.num_layer):
                for k in self.R_buffer:
                    self.R_buffer[k][i][buffer_id] = 0
            for k in self.embedding_R_buffer:
                self.embedding_R_buffer[k][buffer_id] = 0
            for k in self.lm_head_R_buffer:
                self.lm_head_R_buffer[k][buffer_id] = 0
            return

        assert oft_adapter is not None
        block_size = oft_adapter.block_size

        # Precompute R from compact weights and load into buffer
        available_fused_targets = set(self.R_buffer)
        for layer_id in range(self.num_layer):
            layer_weights = normalize_merged_oft_weights(
                oft_adapter.layers[layer_id].weights,
                available_fused_targets=available_fused_targets,
            )
            temp_R_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.R_buffer
            }

            for name, weights in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)
                temp_R_buffer[target_module] = weights

            # TP slicing (on compact weights before precompute)
            if self.tp_size > 1:
                cur_layer_modules = oft_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    target_module = get_target_module_name(
                        module_name, self.target_modules
                    )
                    if temp_R_buffer[target_module] is not None:
                        tp_rank = module.get_local_tp_rank()
                        temp_R_buffer[target_module] = module.slice_oft_r_weights(
                            temp_R_buffer[target_module], tp_rank
                        )

            for name, weights in temp_R_buffer.items():
                target_buffer = self.R_buffer[name][layer_id]
                precompute_and_store_R(
                    target_buffer[buffer_id], weights, block_size
                )

        # Load embedding layer weights (precompute R)
        if oft_adapter.embedding_layers:
            for name, weights in oft_adapter.embedding_layers.items():
                target_module = get_target_module_name(name, self.target_modules)
                if target_module == "embed_tokens" and "embed_tokens" in name:
                    precompute_and_store_R(
                        self.embedding_R_buffer[target_module][buffer_id],
                        weights,
                        block_size,
                    )
                elif target_module == "lm_head" and "lm_head" in name:
                    precompute_and_store_R(
                        self.lm_head_R_buffer[target_module][buffer_id],
                        weights,
                        block_size,
                    )

        # Load extra token embeddings (raw embeddings, no precompute needed)
        if oft_adapter.added_tokens_embeddings:
            added_tokens_size = oft_adapter.config.oft_added_tokens_size
            for name, weights in oft_adapter.added_tokens_embeddings.items():
                if "input_embeddings" in name:
                    buffer_view = self.new_embeddings_buffer["input_embeddings"][
                        buffer_id, :added_tokens_size
                    ]
                    buffer_view.copy_(weights, non_blocking=True)

    def _runtime_buffer_target_for_name(
        self, name: str
    ) -> Tuple[str, Optional[int], int]:
        """Resolve an OFT tensor name against runtime R buffers.

        Split CanonicalOFT tensors may arrive before their siblings in streamed
        sync. In that case they still target the fused runtime buffer, but only
        one stacked slice within it.
        """

        # Prefer exact fused runtime leaves first, so ``gate_up_proj`` does not
        # get interpreted as the split ``up_proj`` leaf.
        for target in self.R_buffer:
            if _contains_leaf(name, target):
                return target, None, 1

        for fused_target, split_leaves in MERGED_OFT_PROJ_GROUPS.items():
            if fused_target not in self.R_buffer:
                continue
            for index, split_leaf in enumerate(split_leaves):
                if _contains_leaf(name, split_leaf):
                    return fused_target, index, len(split_leaves)

        for target in self.target_modules:
            if _contains_leaf(name, target):
                return target, None, 1

        return get_target_module_name(name, self.target_modules), None, 1

    def _resolve_oft_tensor_plan(
        self,
        name: str,
        oft_modules: List[Dict[str, BaseLayerWithOFT]],
        layer_id: int,
    ) -> Tuple[str, Optional[BaseLayerWithOFT], bool, Optional[int], int]:
        """Resolve (fused_target, slice_module, is_row_parallel, slice_index, split_count).

        ``slice_index`` is set when a split CanonicalOFT tensor such as
        ``q_proj.oft_R`` targets one stacked slice of a fused runtime buffer
        such as ``qkv_proj``.
        """
        name_cache = getattr(self, "_oft_name_cache", None)
        if name_cache is None:
            name_cache = {}
            self._oft_name_cache = name_cache

        layer_target_cache = getattr(self, "_oft_layer_target_cache", None)
        if layer_target_cache is None:
            layer_target_cache = {}
            self._oft_layer_target_cache = layer_target_cache

        cached = name_cache.get(name)
        if cached is not None:
            return cached

        fused_target, slice_index, split_count = self._runtime_buffer_target_for_name(
            name
        )

        slice_module = None
        if self.tp_size > 1 and layer_id < len(oft_modules):
            per_layer = layer_target_cache.get(layer_id)
            if per_layer is None:
                per_layer = {}
                cur_layer_modules = oft_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    try:
                        target = get_target_module_name(module_name, self.target_modules)
                    except ValueError:
                        continue
                    per_layer.setdefault(target, module)
                layer_target_cache[layer_id] = per_layer
            slice_module = per_layer.get(fused_target)

        cached = (
            fused_target,
            slice_module,
            fused_target in ROW_PARALLELISM_LINEAR_OFT_NAMES,
            slice_index,
            split_count,
        )
        name_cache[name] = cached
        return cached

    def _slice_oft_compact_weight(
        self,
        compact_weight: torch.Tensor,
        slice_module: Optional[BaseLayerWithOFT],
    ) -> torch.Tensor:
        if slice_module is None:
            return compact_weight
        tp_rank = slice_module.get_local_tp_rank()
        return slice_module.slice_oft_r_weights(compact_weight, tp_rank)

    def _write_precomputed_oft_r(
        self,
        buffer_id: int,
        fused_target: str,
        layer_id: int,
        r: torch.Tensor,
        block_size: int,
        slice_index: Optional[int] = None,
        split_count: int = 1,
    ) -> None:
        buffer_slot = self.R_buffer[fused_target][layer_id][buffer_id]
        num_blocks_adapter = r.shape[0]
        total_blocks_buffer = buffer_slot.shape[0]

        if slice_index is not None:
            if split_count <= 1:
                raise ValueError(
                    f"slice_index={slice_index} requires split_count > 1"
                )
            if not 0 <= slice_index < split_count:
                raise ValueError(
                    f"slice_index={slice_index} out of range for split_count={split_count}"
                )
            if total_blocks_buffer % split_count != 0:
                raise ValueError(
                    f"Cannot split OFT buffer for {fused_target}: "
                    f"total_blocks={total_blocks_buffer}, split_count={split_count}"
                )
            blocks_per_slice = total_blocks_buffer // split_count
            start = slice_index * blocks_per_slice
            end = start + blocks_per_slice
            target_view = buffer_slot[start:end]

            if num_blocks_adapter == 1 and blocks_per_slice > 1:
                target_view[:, :block_size, :block_size] = r[0]
                return

            if num_blocks_adapter > blocks_per_slice:
                raise ValueError(
                    f"OFT tensor for {fused_target} slice {slice_index} has "
                    f"{num_blocks_adapter} blocks, but runtime slice only has "
                    f"{blocks_per_slice}"
                )

            if num_blocks_adapter < blocks_per_slice:
                _fill_identity(target_view, self.max_oft_block_size)

            target_view[:num_blocks_adapter, :block_size, :block_size] = r
            return

        # block_share adapters emit a single block; broadcast it to every slot.
        if num_blocks_adapter == 1 and total_blocks_buffer > 1:
            buffer_slot[:, :block_size, :block_size] = r[0]
            return

        # Reset slots the adapter doesn't touch to identity (passthrough).
        if num_blocks_adapter < total_blocks_buffer:
            _fill_identity(buffer_slot, self.max_oft_block_size)

        buffer_slot[:num_blocks_adapter, :block_size, :block_size] = r

    def load_oft_weight_direct(
        self,
        buffer_id: int,
        name: str,
        compact_weight: torch.Tensor,
        block_size: int,
        oft_modules: List[Dict[str, BaseLayerWithOFT]],
        layer_id: int,
    ):
        """Write a single OFT tensor directly into the GPU R_buffer.

        FT-style per-tensor load path. Each tensor is precomputed to an
        orthogonal R on GPU and written into the R_buffer slot.

        Fused Megatron layers (linear_qkv, linear_fc1) use a shared R. Bridge's
        HF export duplicates that R into q/k/v (or gate/up), so only the
        primary sub-module writes; the duplicates are skipped.
        """
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        # Handle embedding layers separately
        if "embed_tokens" in name or "lm_head" in name:
            self._load_embedding_weight_direct(
                buffer_id, name, compact_weight, block_size
            )
            return

        fused_target, slice_module, _, slice_index, split_count = self._resolve_oft_tensor_plan(
            name,
            oft_modules,
            layer_id,
        )

        compact_weight = self._slice_oft_compact_weight(compact_weight, slice_module)

        # Precompute R on GPU
        target_buffer = self.R_buffer[fused_target][layer_id]
        device = target_buffer.device
        if compact_weight.device != device:
            compact_weight = compact_weight.to(device)
        r = precompute_oft_r(compact_weight, block_size)
        self._write_precomputed_oft_r(
            buffer_id,
            fused_target,
            layer_id,
            r,
            block_size,
            slice_index=slice_index,
            split_count=split_count,
        )

    def _load_embedding_weight_direct(
        self,
        buffer_id: int,
        name: str,
        compact_weight: torch.Tensor,
        block_size: int,
    ):
        """Write an embedding OFT tensor directly into the GPU embedding R_buffer."""
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        if "embed_tokens" in name and "embed_tokens" in self.embedding_R_buffer:
            buffer = self.embedding_R_buffer["embed_tokens"]
            R = precompute_oft_r(compact_weight.to(buffer.device), block_size)
            num_blocks_adapter = R.shape[0]
            _fill_identity(buffer[buffer_id], self.max_oft_block_size)
            if num_blocks_adapter == 1 and buffer[buffer_id].shape[0] > 1:
                buffer[buffer_id, :, :block_size, :block_size] = R[0]
            else:
                buffer[buffer_id, :num_blocks_adapter, :block_size, :block_size] = R
        elif "lm_head" in name and "lm_head" in self.lm_head_R_buffer:
            buffer = self.lm_head_R_buffer["lm_head"]
            R = precompute_oft_r(compact_weight.to(buffer.device), block_size)
            num_blocks_adapter = R.shape[0]
            _fill_identity(buffer[buffer_id], self.max_oft_block_size)
            if num_blocks_adapter == 1 and buffer[buffer_id].shape[0] > 1:
                buffer[buffer_id, :, :block_size, :block_size] = R[0]
            else:
                buffer[buffer_id, :num_blocks_adapter, :block_size, :block_size] = R

    def allocate_buffer_slot(self) -> int:
        """Allocate a buffer slot for direct-to-GPU loading.

        Returns the buffer_id of an available slot (preferring empty slots).
        Raises ValueError if no slot is available.
        """
        for buffer_id in range(self.max_ofts_per_batch):
            if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                return buffer_id

        # Diagnostic: dump the actual contents of each slot on failure so we
        # can tell what's holding things up. With max_ofts_per_batch=1 this is
        # what surfaces "the streamed identity adapter can't allocate even at
        # init"-type bugs. Logged at error level since we're about to raise.
        slot_summary = ", ".join(
            f"slot[{i}]={self.buffer_id_to_uid[i]!r}"
            for i in range(self.max_ofts_per_batch)
        )
        uid_summary = ", ".join(
            f"{uid!r}->buffer[{bid}]"
            for uid, bid in self.uid_to_buffer_id.items()
        )
        logger.error(
            "allocate_buffer_slot: pool full at max_ofts_per_batch=%d. "
            "Slot contents: {%s}. UID->buffer map: {%s}.",
            self.max_ofts_per_batch,
            slot_summary,
            uid_summary,
        )
        raise ValueError(
            "No available buffer slots for direct OFT loading. "
            f"All slots are occupied. (max_ofts_per_batch={self.max_ofts_per_batch}, "
            f"slot contents: {{{slot_summary}}}, "
            f"uid_to_buffer_id={dict(self.uid_to_buffer_id)})"
        )

    def reset_buffer_slot_to_identity(self, buffer_id: int):
        """Reset all R buffers for a given slot to block-diagonal identity.

        Identity is the OFT passthrough — applying it to any input leaves the
        input unchanged. This is the correct baseline before writing per-module
        adapter weights, because modules that the adapter does not touch must
        still let their inputs flow through unchanged. A zero R would map
        every input to zero and silently kill the layer.
        """
        bs = self.max_oft_block_size
        for layer_id in range(self.num_layer):
            for k in self.R_buffer:
                _fill_identity(self.R_buffer[k][layer_id][buffer_id], bs)
        for k in self.embedding_R_buffer:
            _fill_identity(self.embedding_R_buffer[k][buffer_id], bs)
        for k in self.lm_head_R_buffer:
            _fill_identity(self.lm_head_R_buffer[k][buffer_id], bs)

    def get_tensor(self, target_module: str, layer_id: int) -> torch.Tensor:
        """Get the R buffer tensor for a given module and layer."""
        return self.R_buffer[target_module][layer_id]

    def get_embedding_tensor(
        self, target_module: str
    ) -> Optional[torch.Tensor]:
        """Get OFT tensor for non-layer modules (embed_tokens, lm_head, added_tokens)."""
        if target_module == "added_tokens":
            if self.oft_added_tokens_size > 0:
                return self.new_embeddings_buffer["input_embeddings"]
            return None
        elif target_module == "embed_tokens":
            return self.embedding_R_buffer.get(target_module)
        elif target_module == "lm_head":
            return self.lm_head_R_buffer.get(target_module)
        raise ValueError(
            f"Invalid target_module '{target_module}'. "
            f"Expected 'embed_tokens', 'lm_head', or 'added_tokens'."
        )

    def get_buffer_id(self, oft_uid: Optional[str]) -> int:
        return self.uid_to_buffer_id[oft_uid]

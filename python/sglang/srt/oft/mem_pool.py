import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.distributed import divide
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.utils import get_stacked_multiply as _lora_get_stacked_multiply
from sglang.srt.oft.base.mem_pool import EMPTY_SLOT, AdapterMemPool, EmptySlot
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


# Fused HF projection leaf -> tuple of split leaves. Inverse of the LoRA
# params_mapping in sglang.srt.lora.utils, kept local here because the OFT
# load path needs the forward direction (split-leaf -> fused).
MERGED_OFT_PROJ_GROUPS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "fused_qkv_a_proj_with_mqa": ("q_a_proj", "kv_a_proj_with_mqa"),
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


def _fill_expert_oft_identity(buffer: torch.Tensor) -> None:
    """Fill an expert-OFT R buffer view with identity (OFT passthrough).

    Expert-OFT groups (``w13_oft_r``/``w1_oft_r``/``w3_oft_r``/``w2_oft_r``)
    have no max-block-size padding concept unlike the dense ``_fill_identity``
    above -- ``buffer.shape[-1]`` (the block dim) IS the block size, so the
    whole trailing (block_size, block_size) sub-matrix is set to eye, not just
    a top-left sub-block. Moved here (from ``oft_manager.py``) so this module
    -- which already owns ``_fill_identity``, the dense equivalent -- can call
    it from ``reset_buffer_slot_to_identity`` without a circular import
    (``oft_manager.py`` imports from this module, not the reverse);
    ``oft_manager.py`` re-imports it under the same name for its own
    boot-time ``active_idx`` fill.
    """
    buffer.zero_()
    if buffer.numel() == 0:
        return
    block_size = buffer.shape[-1]
    eye = torch.eye(block_size, dtype=buffer.dtype, device=buffer.device)
    buffer[...] = eye


def _write_oft_r_block(
    buffer_slot: torch.Tensor,
    r: torch.Tensor,
    block_size: int,
    max_oft_block_size: int,
    slice_index: Optional[int] = None,
    split_count: int = 1,
) -> None:
    """Write a precomputed OFT rotation ``r`` into ``buffer_slot``.

    ``buffer_slot`` is a single destination view of shape
    (total_blocks_buffer, bs, bs) — one buffer-pool slot's worth of blocks for
    one (target_module, layer) pair. Blocks the adapter doesn't cover are
    reset to block-diagonal identity (OFT passthrough); a zero R would map
    every input to zero and silently kill the layer.

    Shared by the legacy per-adapter streamed-load path
    (``OFTMemoryPool._write_precomputed_oft_r``, used by both
    ``load_oft_weight_direct`` and ``streamed_weight_loader._flush_oft_group_chunk``)
    and ``OFTMemoryPool._fill_slot`` — same math, only the destination differs.
    """
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
                f"Cannot split OFT buffer: total_blocks={total_blocks_buffer}, "
                f"split_count={split_count}"
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
                f"OFT tensor for slice {slice_index} has {num_blocks_adapter} "
                f"blocks, but runtime slice only has {blocks_per_slice}"
            )

        if num_blocks_adapter < blocks_per_slice:
            _fill_identity(target_view, max_oft_block_size)

        target_view[:num_blocks_adapter, :block_size, :block_size] = r
        return

    # block_share adapters emit a single block; broadcast it to every slot.
    if num_blocks_adapter == 1 and total_blocks_buffer > 1:
        buffer_slot[:, :block_size, :block_size] = r[0]
        return

    # Reset slots the adapter doesn't touch to identity (passthrough).
    if num_blocks_adapter < total_blocks_buffer:
        _fill_identity(buffer_slot, max_oft_block_size)

    buffer_slot[:num_blocks_adapter, :block_size, :block_size] = r


class OFTMemoryPool(AdapterMemPool):
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
        oft_type: str,
        oft_modules: Optional[List[Dict[str, BaseLayerWithOFT]]] = None,
        external_target_modules: Optional[Set[str]] = None,
        memory_saver_adapter=None,
        memory_saver_cpu_backup: bool = False,
        double_buffer: bool = False,
    ):
        super().__init__(
            max_adapters_per_batch=max_ofts_per_batch,
            dtype=dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            eviction_policy=eviction_policy,
            memory_saver_adapter=memory_saver_adapter,
            memory_saver_cpu_backup=memory_saver_cpu_backup,
        )
        if double_buffer:
            # base=0 (unchanged), active=1, staging=max_ofts_per_batch-1
            # (=2 when orbit passes max_ofts_per_batch=3).
            self.active_idx = 1
            self.staging_idx = max_ofts_per_batch - 1
        # else: inherits base defaults (0/1); stage/activate never invoked.
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = get_hf_config_attr(
            base_hf_config, "num_hidden_layers"
        )
        self.max_ofts_per_batch: int = max_ofts_per_batch
        self.max_oft_block_size: int = max_oft_block_size
        self.target_modules: Set[str] = target_modules
        self.external_target_modules: Set[str] = external_target_modules or set()
        self.oft_modules: Optional[List[Dict[str, BaseLayerWithOFT]]] = oft_modules
        self.oft_added_tokens_size: int = oft_added_tokens_size
        # Single global split(canonical)-vs-fused signal (sglang.srt.oft.config.
        # OFTArgs.oft_type); drives which MoE expert gate/up group layout
        # _declare_expert_groups registers (w1_oft_r+w3_oft_r split vs w13_oft_r
        # fused).
        self.oft_type: str = oft_type
        self.embedding_dim: int = get_hf_config_attr(base_hf_config, "hidden_size")

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
        #
        # The tensors themselves now live in the base ``AdapterMemPool`` group
        # registry (``self._groups["R:{target}"]``, populated by
        # ``_declare_groups``); ``self.R_buffer`` below is a read-only accessor
        # kept for existing internal readers.

        # Embedding and lm_head R buffers (not per-layer, analogous to LoRA's embedding_A/B)
        self.embedding_R_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_R_buffer: Dict[str, torch.Tensor] = {}

        # Extra token embeddings buffer
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        self.init_buffers(base_model)

    @property
    def R_buffer(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """Thin accessor over the base ``AdapterMemPool`` group registry.

        ``R_buffer[target][layer]`` returns the group's full
        ``[max_ofts_per_batch, ...]`` tensor for that target/layer — the same
        object the legacy dict-of-lists used to hold (a rename of the
        allocation site, not a data change; see ``_declare_groups``). This is
        a read accessor only: buffers are (re)allocated exclusively through
        ``register_buffer_group`` in ``_declare_groups``.
        """
        prefix = "R:"
        return {
            name[len(prefix):]: tensors
            for name, tensors in self._groups.items()
            if name.startswith(prefix)
        }

    def can_support(self, config: Union[OFTConfig, list[OFTConfig]]) -> bool:
        """Check if the memory pool can support the given OFT adapter(s)."""

        def _can_support(config: OFTConfig) -> bool:
            # Equality, not <=: pool tiles have one (bs, bs) geometry, and a
            # smaller-block adapter's R has the wrong shape for the tile
            # (block size is geometry, unlike LoRA ranks, which pad).
            if config.block_size != self.max_oft_block_size:
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
        self.base_model = base_model
        self.device = device

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

            self._declare_groups()

    def _declare_groups(self):
        """Register one buffer group per dense fused-target module (``R:{target}``)
        on the base ``AdapterMemPool`` registry.

        Same shape and initial values (block-diagonal identity) the legacy
        ``self.R_buffer`` dict-of-lists construction produced — this only
        changes where the tensors live (the base group registry instead of a
        private dict), not their shape or values. ``self.R_buffer[target][layer]``
        (see the property above) resolves to exactly the tensor registered
        here.
        """
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

        target_modules = (
            self.target_modules - set(EMBEDDING_NAMES) - self.external_target_modules
        )
        for module_name in target_modules:
            per_key_shape: Dict[int, Tuple[int, ...]] = {}
            for layer_idx in range(self.num_layer):
                full_shape = self.get_oft_R_shape(
                    module_name,
                    self.base_model,
                    self.max_oft_block_size,
                    layer_idx,
                    module_lookup.get((module_name, layer_idx)),
                )
                # Drop the leading max_ofts_per_batch dim: register_buffer_group
                # re-adds it (as the slot dim) via max_adapters_per_batch.
                per_key_shape[layer_idx] = full_shape[1:]
            self.register_buffer_group(
                f"R:{module_name}",
                per_key_shape,
                dtype=self.dtype,
                device=self.device,
            )
            # Default every slot to block-diagonal identity (OFT passthrough),
            # matching the legacy `_make_identity_r_buffer` init exactly.
            for tensor in self._groups[f"R:{module_name}"].values():
                tensor.zero_()
                if self.max_oft_block_size > 0:
                    eye = torch.eye(
                        self.max_oft_block_size,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    tensor[:, :, :, :] = eye

        self._declare_expert_groups()

    def _find_fused_moe_layers(self) -> Dict[int, torch.nn.Module]:
        """Layer-id-indexed FusedMoE modules found in the base model.

        Mirrors ``AdapterManager._find_fused_moe_modules`` (base/manager.py);
        duplicated here (not imported) because the pool only has
        ``self.base_model`` to scan, not an ``AdapterManager`` instance.
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        moe_layers: Dict[int, torch.nn.Module] = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, FusedMoE):
                layer_idx = get_layer_id(name)
                if layer_idx is not None:
                    moe_layers[layer_idx] = module
        return moe_layers

    def _declare_expert_groups(self):
        """Register the per-layer expert OFT groups for FusedMoE layers.

        ``w2_oft_r`` is registered ALWAYS (both oft_type layouts use it
        identically). For gate/up, ``self.oft_type`` (the single global
        split-vs-fused signal, ``sglang.srt.oft.config.OFTArgs.oft_type``)
        selects EXACTLY ONE layout: ``oft_type=="canonical_oft"`` registers the
        per-sub-projection SPLIT groups ``w1_oft_r`` (gate) / ``w3_oft_r`` (up)
        -- orbit's only trained variant; ``oft_type=="oft"`` registers the
        legacy shared-R FUSED group ``w13_oft_r``. Only the selected layout is
        registered (zero wasted allocation); both layouts remain supported.

        Shapes match the module-attribute ``torch.empty(...)`` calls
        ``OFTManager._init_identity_expert_oft_for_cuda_graph`` used to
        allocate directly on ``moe`` (oft_manager.py, legacy w13 + w2
        branches, and w1/w3 as of Task 6) — this only moves where the tensors
        live (a pool group instead of a private module attribute), not their
        shape or dtype. Gated on ``self.target_modules`` exactly like that
        function's own ``init_w13``/``init_w2`` so OFT deployments that don't
        target MoE experts don't allocate unused buffers. Group names
        deliberately do NOT start with ``"R:"`` so they stay out of the dense
        ``R_buffer`` accessor.
        """
        if self.max_oft_block_size <= 0:
            return
        init_w13 = bool(
            {"gate_up_proj", "gate_proj", "up_proj"} & self.target_modules
        )
        init_w2 = "down_proj" in self.target_modules
        if not (init_w13 or init_w2):
            return

        moe_layers = self._find_fused_moe_layers()
        if not moe_layers:
            return

        block_size = self.max_oft_block_size
        if init_w13:
            w13_shape_by_layer = {
                layer_idx: (
                    moe.num_local_experts,
                    moe.hidden_size // block_size,
                    block_size,
                    block_size,
                )
                for layer_idx, moe in moe_layers.items()
            }
            if self.oft_type == "canonical_oft":
                self.register_buffer_group(
                    "w1_oft_r",
                    w13_shape_by_layer,
                    dtype=self.dtype,
                    device=self.device,
                )
                self.register_buffer_group(
                    "w3_oft_r",
                    w13_shape_by_layer,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                self.register_buffer_group(
                    "w13_oft_r",
                    w13_shape_by_layer,
                    dtype=self.dtype,
                    device=self.device,
                )
        if init_w2:
            self.register_buffer_group(
                "w2_oft_r",
                {
                    layer_idx: (
                        moe.num_local_experts,
                        moe.intermediate_size_per_partition // block_size,
                        block_size,
                        block_size,
                    )
                    for layer_idx, moe in moe_layers.items()
                },
                dtype=self.dtype,
                device=self.device,
            )

    def _fill_slot(
        self,
        slot_idx: int,
        named_tensors: Dict[
            Tuple[str, int], Tuple[torch.Tensor, int, Optional[int], int]
        ],
    ) -> None:
        """Write precomputed dense OFT R blocks into ``slot_idx``.

        ``named_tensors`` maps ``(target_module, layer_id) -> (r, block_size,
        slice_index, split_count)`` — the same precomputed-R payload the
        legacy streamed path (``load_oft_weight_direct`` /
        ``_write_precomputed_oft_r``) writes, targeting an explicit slot
        instead of a resolved ``buffer_id``. Reuses ``_write_oft_r_block``
        (no math change) so both paths write identically.
        """
        for (target_module, layer_id), (
            r,
            block_size,
            slice_index,
            split_count,
        ) in named_tensors.items():
            buffer_slot = self.slot(f"R:{target_module}", layer_id, slot_idx)
            _write_oft_r_block(
                buffer_slot,
                r,
                block_size,
                self.max_oft_block_size,
                slice_index=slice_index,
                split_count=split_count,
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
        # Mark all adapters in current batch as used (for LRU tracking)
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        def get_available_buffer_slot() -> int:
            """Lazy-admission fallback for a real uid with already-
            materialized weights (``oft_adapters``) but no serving slot yet
            -- e.g. a brand-new adapter that the staged path (StagedOFTManager
            .activate_adapter) just registered CPU-side only, with no on-disk
            preload path left to have admitted it eagerly. Mirrors
            LoRAMemoryPool.prepare_lora_batch's inline ``get_available_buffer_
            slot`` closure, adapted to OFT's own eviction conventions
            (established by allocate_buffer_slot_with_eviction, the native-RPC
            admission path): the base/identity placeholder (uid=None) is
            never an eviction candidate, and neither is a pinned ref nor a
            non-reloadable one (an adapter loaded over the wire has no CPU-
            side artifact to re-page from).
            """
            # 1. Prioritize empty slots
            for buffer_id in range(self.max_ofts_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id

            # 2. Memory pool is full, need to evict using policy
            candidates: Set[Optional[str]] = set()
            for buffer_id in range(self.max_ofts_per_batch):
                victim_uid = self.buffer_id_to_uid[buffer_id]
                if victim_uid is None:
                    continue
                if victim_uid in cur_uids:
                    continue
                ref = oft_refs.get(victim_uid)
                if ref is not None and (ref.pinned or not ref.reloadable):
                    continue
                candidates.add(victim_uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots for lazy OFT admission. "
                    "Please ensure the number of active (pinned) adapters "
                    "and adapters loaded over the wire (no on-disk artifact "
                    "to reload from, never evicted) is less than "
                    f"max_ofts_per_batch={self.max_ofts_per_batch}."
                )

            victim_uid = self.eviction_policy.select_victim(candidates)
            victim_buffer_id = self.uid_to_buffer_id.pop(victim_uid)
            self.eviction_policy.remove(victim_uid)
            self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
            logger.debug(
                f"Evicting OFT adapter {victim_uid} from buffer slot "
                f"{victim_buffer_id} for lazy admission."
            )
            return victim_buffer_id

        # Deterministic order: cur_uids is a set, and per-process hash
        # randomization would otherwise let each TP rank map the same adapters
        # to different slots -- silently wrong weights, no shape error.
        # Mirrors upstream LoRAMemoryPool.prepare_lora_batch.
        for uid in sorted(cur_uids, key=lambda uid: (uid is not None, uid or "")):
            if uid not in self.uid_to_buffer_id:
                if uid is None:
                    # The base/identity placeholder's own one-time boot
                    # registration (OFTManager.init_memory_pool's
                    # fetch_new_ofts({None}) call, against a freshly
                    # constructed, entirely empty pool) -- not admission for
                    # a real adapter, so it needs no eviction fallback and is
                    # unrelated to the (now-retired) on-disk (--peft-paths)
                    # lazy admission this loop used to also perform. Always
                    # lands in the first empty slot (slot 0 on a fresh pool);
                    # never evicted afterward (see allocate_buffer_slot_with_
                    # eviction's uid=None exclusion), so this branch never
                    # runs again post-boot.
                    buffer_id = next(
                        (
                            i
                            for i in range(self.max_ofts_per_batch)
                            if self.buffer_id_to_uid[i] == EMPTY_SLOT
                        ),
                        None,
                    )
                    if buffer_id is None:
                        raise ValueError(
                            "No empty buffer slot available for the base/"
                            "identity placeholder's boot registration -- this "
                            "should be unreachable (it is the very first uid "
                            "ever admitted, against a freshly constructed, "
                            "entirely empty pool)."
                        )
                    self.reset_buffer_slot_to_identity(buffer_id)
                    self.uid_to_buffer_id[uid] = buffer_id
                    self.buffer_id_to_uid[buffer_id] = uid
                    continue
                # A real uid with no serving slot yet: admit it lazily from
                # its already-materialized CPU-side adapter (``oft_adapters``
                # -- populated by StagedOFTManager.activate_adapter's
                # deferred-registration path for a brand-new staged adapter,
                # or already present for any other resident real uid). The
                # native-RPC path (OFTManager.load_adapter_from_tensors)
                # still admits eagerly at load time via
                # allocate_buffer_slot_with_eviction, so this fallback is
                # normally only reached by the staged path.
                buffer_id = get_available_buffer_slot()
                # Identity-fill BEFORE writing real weights: load_oft_weight_
                # to_buffer only overwrites the dense R_buffer/embedding
                # groups, never the expert-OFT groups (w1/w3/w13/w2_oft_r), so
                # a freshly-acquired (or evicted-and-reused) slot must be
                # identity-safe first -- mirrors the native-RPC path's own
                # allocate_buffer_slot_with_eviction -> reset_buffer_slot_to_
                # identity call pattern (Task 4b).
                self.reset_buffer_slot_to_identity(buffer_id)
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
        uid: str,
        buffer_id: int,
        oft_adapter: Optional[OFTAdapter],
        oft_modules: List[Dict[str, BaseLayerWithOFT]],
        oft_embed_tokens_module: Optional[BaseLayerWithOFT],
        oft_lm_head_module: Optional[BaseLayerWithOFT],
    ):
        """Write an already-materialized real adapter's compact checkpoint
        weights into buffer slot ``buffer_id``, precomputing each module's R
        matrix on the way in.

        General-purpose admission helper for any not-yet-resident real uid
        with a materialized ``OFTAdapter`` -- used by ``prepare_oft_batch``'s
        lazy-admission fallback (analogous to ``LoRAMemoryPool.
        load_lora_weight_to_buffer``). ``uid`` is never ``None`` here: the
        base/identity placeholder is registered by ``prepare_oft_batch``'s own
        boot-registration branch instead, which never calls this method.
        """

        def precompute_and_store_R(
            buffer_view: torch.Tensor,
            compact_weight: Optional[torch.Tensor],
            block_size: int,
        ):
            """Precompute R from compact weights and store in buffer.

            Instead of storing compact 2D weights, precomputes the full
            orthogonal rotation matrices via Cayley transform and stores the
            3D result (num_blocks, block_size, block_size).

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

            R = precompute_oft_r(compact_weight.to(buffer_view.device), block_size)
            num_blocks_adapter = R.shape[0]
            total_blocks_buffer = buffer_view.shape[0]

            buffer_view.zero_()
            if num_blocks_adapter == 1 and total_blocks_buffer > 1:
                # Block-share: replicate single R block to all positions
                buffer_view[:, :block_size, :block_size] = R[0]
            else:
                buffer_view[:num_blocks_adapter, :block_size, :block_size] = R

        if oft_adapter is None:
            # A registry/GPU-pool divergence (e.g. a batch names a uid whose
            # CPU-side OFTAdapter is missing from oft_adapters even though the
            # pool's admission bookkeeping expects one) slipping through some
            # path not yet foreseen. Raise a clear, catchable error instead of
            # asserting, so this can never crash the engine outright.
            raise ValueError(
                f"No OFTAdapter available to load for uid={uid!r} into buffer "
                f"slot {buffer_id}. This uid is resident in the memory pool's "
                "admission bookkeeping but has no corresponding CPU-side "
                "adapter to load weights from."
            )
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
                        temp_R_buffer[target_module] = module.slice_oft_r_weights(
                            temp_R_buffer[target_module]
                        )

            for name, weights in temp_R_buffer.items():
                target_buffer = self.R_buffer[name][layer_id]
                precompute_and_store_R(target_buffer[buffer_id], weights, block_size)

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
        return slice_module.slice_oft_r_weights(compact_weight)

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
        _write_oft_r_block(
            buffer_slot,
            r,
            block_size,
            self.max_oft_block_size,
            slice_index=slice_index,
            split_count=split_count,
        )

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

    def allocate_buffer_slot_with_eviction(
        self, refs: Dict[str, OFTRef]
    ) -> Tuple[int, Optional[str]]:
        """Admission for the native multi-tenant RPC path: if no slot is
        empty, LRU-evict an unpinned, reloadable resident real adapter to
        make room. The base/identity slot (uid=None) is never a candidate
        here (Task 4b review fix), and neither is a pinned ref (the caller's
        explicit "keep this one resident" request) nor a non-reloadable one
        (an adapter loaded over the wire has no CPU-side artifact to re-page
        from, so evicting it would be unrecoverable -- mirrors the retired
        streamed path's _make_streamed_ref, which always pinned such
        adapters for exactly this reason).

        Returns (buffer_id, evicted_uid); evicted_uid is None when an empty
        slot was found and no eviction was needed.
        """
        for buffer_id in range(self.max_ofts_per_batch):
            if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                return buffer_id, None

        candidates: Set[Optional[str]] = set()
        for uid in self.uid_to_buffer_id:
            if uid is None:
                continue
            ref = refs.get(uid)
            if ref is not None and (ref.pinned or not ref.reloadable):
                continue
            candidates.add(uid)

        if not candidates:
            raise ValueError(
                "No available buffer slots for direct OFT loading, and no "
                "evictable resident adapter to make room for a new one. "
                "Pinned adapters and adapters loaded over the wire (no "
                "on-disk artifact to reload from) are never evicted. "
                f"(max_ofts_per_batch={self.max_ofts_per_batch})"
            )

        try:
            victim_uid = self.eviction_policy.select_victim(candidates)
        except AssertionError:
            # select_victim asserts if NONE of the candidates were ever
            # eviction_policy.mark_used(...)'d and None isn't a candidate
            # (it never is here). OFTManager.load_adapter_from_tensors marks
            # every adapter used at admission time specifically to avoid
            # this, but this method shouldn't crash even for a future
            # caller that doesn't -- fall back to a deterministic choice
            # among the untracked candidates instead of propagating the
            # policy's internal "nothing to select" assertion.
            victim_uid = sorted(candidates, key=lambda uid: (uid is None, uid or ""))[0]
        victim_buffer_id = self.uid_to_buffer_id.pop(victim_uid)
        self.eviction_policy.remove(victim_uid)
        self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
        logger.info(
            "Evicting OFT adapter %s from buffer slot %d to admit a new "
            "adapter via the native RPC path (pool full at "
            "max_ofts_per_batch=%d).",
            victim_uid,
            victim_buffer_id,
            self.max_ofts_per_batch,
        )
        return victim_buffer_id, victim_uid

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

        # Expert-OFT groups (w13_oft_r / w1_oft_r / w3_oft_r / w2_oft_r) are
        # NOT covered by the R_buffer loop above (_declare_expert_groups
        # registers them separately in self._groups) and were never reset
        # per-slot before -- only slot active_idx ever got identity-filled,
        # at boot (_init_identity_expert_oft_for_cuda_graph). Without this, a
        # token whose adapter has no MoE weights for a layer (or a freshly
        # (re)assigned slot before any real weights are written) reads
        # uninitialized torch.empty memory here instead of identity.
        for group_name in ("w13_oft_r", "w1_oft_r", "w3_oft_r", "w2_oft_r"):
            groups = self._groups.get(group_name)
            if groups is None:
                continue
            for tensor in groups.values():
                _fill_expert_oft_identity(tensor[buffer_id])

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

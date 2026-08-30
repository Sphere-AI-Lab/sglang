"""Staged weight updates for UPSTREAM LoRA, by subclassing rather than editing.

Upstream sglang's LoRA adapters are immutable: loaded from disk, never changed.
Reinforcement learning needs the opposite -- one adapter republished every step
while the sampler keeps serving. This adds that to upstream's pool WITHOUT
touching ``srt/lora``: subclass ``LoRAMemoryPool``, supply the two primitives
``VersionedStaging`` needs, and inherit the whole stage/activate state machine.

Two facts about upstream's layout drive the implementation:

* weights are spread over seven dicts (dense A/B per module per layer, plus
  embedding, lm_head and added-token buffers), not one registered group set, so
  ``_copy_slot`` walks all of them;
* the pool has no spare slot, so one is allocated ON TOP of the advertised
  capacity and hidden from every loop upstream runs -- serving capacity is
  unchanged, at the cost of one slot's worth of memory.
"""

import logging
from typing import Dict, List, Optional

import torch

from sglang.srt.adapter_sync.versioning import VersionedStaging
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool

logger = logging.getLogger(__name__)


class StagedLoRAMemoryPool(VersionedStaging, LoRAMemoryPool):
    """Upstream's pool plus a reserved staging slot and versioned activation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_idx = 0          # single-active convention (uid=None)
        self.staging_idx = None      # assigned in init_buffers
        self._init_versioning()

    def init_buffers(self, base_model) -> None:
        """Allocate one MORE slot than the pool advertises, and hide the extra.

        The staging slot cannot simply be the last of N, because upstream picks
        serving slots in a nested closure inside ``prepare_lora_batch`` that
        scans ``range(max_loras_per_batch)`` -- not overridable by subclassing,
        and it would hand the staging slot to a serving adapter for the next
        stage to overwrite. Instead the buffers are allocated with N+1 slots
        while the pool continues to advertise N, so index N exists in every
        buffer but is outside every loop upstream runs and outside
        ``buffer_id_to_uid`` (sized N in ``__init__``). It is reachable only
        through ``staging_idx``.
        """
        advertised = self.max_loras_per_batch
        self.max_loras_per_batch = advertised + 1
        try:
            super().init_buffers(base_model)
        finally:
            self.max_loras_per_batch = advertised
        self.staging_idx = advertised

    def _slot_buffers(self):
        """Every container whose leading dimension is the adapter slot."""
        for d in (self.A_buffer, self.B_buffer):
            for per_layer in d.values():
                for t in per_layer:
                    yield t
        for d in (
            self.embedding_A_buffer,
            self.embedding_B_buffer,
            self.lm_head_A_buffer,
            self.lm_head_B_buffer,
            self.new_embeddings_buffer,
        ):
            for t in d.values():
                yield t

    # ---- VersionedStaging primitives --------------------------------------
    def _copy_slot(self, src_idx: int, dst_idx: int) -> None:
        for t in self._slot_buffers():
            t[dst_idx].copy_(t[src_idx])

    def _fill_slot(
        self, slot_idx: int, named_tensors: List[tuple]
    ) -> None:
        """Write incoming ``(buffer_name, layer_id, kind, tensor)`` rows into a slot.

        ``kind`` is "A" or "B". Ranks below ``max_lora_rank`` occupy the leading
        sub-slice, matching how upstream pages a smaller adapter into a
        max-rank buffer; the tail is zeroed so a previous occupant cannot bleed
        through (upstream's own load path relies on the same invariant).
        """
        for buffer_name, layer_id, kind, tensor in named_tensors:
            buf = (self.A_buffer if kind == "A" else self.B_buffer).get(buffer_name)
            if buf is None:
                logger.warning(
                    "staged LoRA update names %r (%s), which this pool has no buffer "
                    "for; skipping", buffer_name, kind
                )
                continue
            view = buf[layer_id][slot_idx]
            view.zero_()
            src = tensor.to(view.device, dtype=view.dtype, non_blocking=True)
            view[tuple(slice(0, n) for n in src.shape)].copy_(src)

    def available_serving_slots(self) -> int:
        """Slots usable for serving. The staging slot is extra, not carved out of
        the advertised capacity, so this is simply what the caller asked for."""
        return self.max_loras_per_batch


class StagedLoRAManager(LoRAManager):
    """Upstream's LoRA manager plus staged, versioned weight updates.

    Deliberately subclasses UPSTREAM's manager rather than
    ``adapter_sync.AdapterManager``. Upstream already implements everything a
    LoRA manager needs -- load, unload, init_state, batch preparation -- so
    reusing the shared AdapterManager would mean reimplementing all of it under
    different method names. What LoRA is actually missing is only the staged
    update, so that is all this adds. (``adapter_sync/manager.py`` remains the
    lifecycle scaffolding that ``srt/oft`` migrates onto in WS2-4; the piece
    genuinely shared by both methods is ``VersionedStaging``.)

    Two overrides, both at non-nested seams:
      init_memory_pool  -> build the staging-capable pool
      stage/activate    -> the new capability, delegated to that pool
    """

    def init_memory_pool(self) -> None:
        """Same construction as upstream, with the staging-capable pool class."""
        self.memory_pool = StagedLoRAMemoryPool(
            base_hf_config=self.base_hf_config,
            max_loras_per_batch=self.max_loras_per_batch,
            dtype=self.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            attn_tp_size=self.attn_tp_size,
            max_lora_rank=self.max_lora_rank,
            target_modules=self.target_modules,
            base_model=self.base_model,
            eviction_policy=self.eviction_policy,
            lora_added_tokens_size=self.lora_added_tokens_size,
            experts_shared_outer_loras=self.experts_shared_outer_loras,
            strict_loading=self.lora_strict_loading,
            enable_lora_overlap_loading=self.enable_lora_overlap_loading,
        )

    def stage_adapter(self, named_tensors, config, name, version, adapter_id=None):
        """Fill the staging slot from raw trainer tensors. Lock-free.

        ``named_tensors`` arrives as checkpoint-style ``(name, tensor)`` rows;
        they are resolved to (buffer, layer, A|B) here because that mapping is
        LoRA's business, not the shared core's.
        """
        uid = adapter_id if adapter_id is not None else name
        self.memory_pool.stage(
            version, self._resolve_named_tensors(named_tensors), uid=uid
        )

    def activate_adapter(self, name, version, adapter_id=None):
        """Promote the staged weights into this adapter's slot."""
        uid = adapter_id if adapter_id is not None else name
        self.memory_pool.activate(version, uid=uid)

    def _resolve_named_tensors(self, named_tensors):
        """Checkpoint names -> (buffer_name, layer_id, "A"|"B", tensor).

        Mirrors how upstream's own loader interprets adapter weight names, so a
        staged update and a disk load agree on where a tensor belongs.
        """
        from sglang.srt.layers.utils import get_layer_id

        from sglang.srt.adapter_sync.utils import get_target_module_name

        resolved = []
        for weight_name, tensor in named_tensors:
            layer_id = get_layer_id(weight_name)
            if layer_id is None:
                logger.warning(
                    "staged LoRA update names %r, which has no layer id "
                    "(embedding/lm_head staging is not supported yet); skipping",
                    weight_name,
                )
                continue
            if "lora_A" in weight_name:
                kind = "A"
            elif "lora_B" in weight_name:
                kind = "B"
            else:
                logger.warning(
                    "staged LoRA update names %r, which is neither lora_A nor "
                    "lora_B; skipping", weight_name
                )
                continue
            try:
                buffer_name = get_target_module_name(
                    weight_name, set(self.memory_pool.A_buffer)
                )
            except ValueError:
                logger.warning(
                    "staged LoRA update names %r, which no buffer matches; skipping",
                    weight_name,
                )
                continue
            resolved.append((buffer_name, layer_id, kind, tensor))
        return resolved

    # ---- compatibility with the fork's call sites --------------------------
    #
    # model_runner and the fork's streamed loader were written against
    # srt/peft/lora's manager API, which upstream's manager does not share.
    # Bridging here keeps those (upstream-tracked) files unedited.

    def init_cuda_graph_moe_buffers(self, *args, **kwargs):
        """Serve both calling conventions for MoE CUDA-graph buffer pre-alloc.

        The fork calls ``(max_bs, disable_cuda_graph=...)`` and expects the
        manager to walk the MoE layers itself. Upstream's method is per-layer,
        ``(max_bs, max_loras, compute_dtype, moe_layer)``, normally driven by
        upstream's own walker. Dispatch on which one arrived, so neither caller
        has to change.
        """
        upstream_call = len(args) >= 4 or "moe_layer" in kwargs
        if upstream_call:
            return super().init_cuda_graph_moe_buffers(*args, **kwargs)

        max_bs = kwargs.get("max_bs", args[0] if args else None)
        if kwargs.get("disable_cuda_graph", False):
            return
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_loras = kwargs.get("max_loras", self.max_loras_per_batch)
        for module in self.base_model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                super().init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    "Pre-allocated shared MoE LoRA CUDA graph buffers "
                    "(max_bs=%s, max_loras=%s)", max_bs, max_loras
                )
                break   # all MoE LoRA layers share one buffer set

    def _unsupported_ipc(self, what):
        raise NotImplementedError(
            f"{what} belongs to the IPC/in-place streaming transport, which is not "
            "ported to the staged upstream-LoRA backend. This backend implements the "
            "NCCL double-buffer transport (stage_adapter/activate_adapter). Use "
            "--lora-impl peft for the IPC path."
        )

    def _active_adapter(self):
        self._unsupported_ipc("_active_adapter")

    def apply_streamed_update(self, *args, **kwargs):
        self._unsupported_ipc("apply_streamed_update")

    def apply_streamed_expert_lora(self, *args, **kwargs):
        self._unsupported_ipc("apply_streamed_expert_lora")

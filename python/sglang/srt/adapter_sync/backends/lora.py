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
from sglang.srt.lora.mem_pool import EMPTY_SLOT, LoRAMemoryPool

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
        """Every container whose leading dimension is the adapter slot.

        Yields ``(label, tensor)`` so a failure can name the buffer rather than
        surfacing as an anonymous tensor error.
        """
        for fam, d in (("A", self.A_buffer), ("B", self.B_buffer)):
            for name, per_layer in d.items():
                for layer, t in enumerate(per_layer):
                    yield f"{fam}_buffer[{name}][layer {layer}]", t
        for fam, d in (
            ("embedding_A", self.embedding_A_buffer),
            ("embedding_B", self.embedding_B_buffer),
            ("lm_head_A", self.lm_head_A_buffer),
            ("lm_head_B", self.lm_head_B_buffer),
            ("new_embeddings", self.new_embeddings_buffer),
        ):
            for name, t in d.items():
                yield f"{fam}[{name}]", t

    # ---- VersionedStaging primitives --------------------------------------
    def _copy_slot(self, src_idx: int, dst_idx: int) -> None:
        for label, t in self._slot_buffers():
            if t.shape[0] <= max(src_idx, dst_idx):
                raise RuntimeError(
                    f"slot_dim_too_small: {label} has shape {tuple(t.shape)}, so slot "
                    f"{max(src_idx, dst_idx)} does not exist. Its leading dimension is "
                    "not the adapter slot, or it was allocated before the staging "
                    "slot was reserved."
                )
            if t[dst_idx].data_ptr() == t[src_idx].data_ptr():
                raise RuntimeError(
                    f"aliased_slots: {label} shape {tuple(t.shape)} strides "
                    f"{tuple(t.stride())} -- slots {src_idx} and {dst_idx} share "
                    "memory, so this buffer is broadcast/expanded across slots "
                    "rather than being per-slot storage."
                )
            t[dst_idx].copy_(t[src_idx])

    def _fill_slot(self, slot_idx, staged) -> None:
        """Place a staged adapter into ``slot_idx`` using UPSTREAM's own routine.

        Earlier this method parsed checkpoint names into buffers itself, which
        was wrong: upstream FUSES projections -- q/k/v share one ``qkv_proj``
        buffer (stacked x3), gate/up share ``gate_up_proj`` (x2), each occupying
        a different row range at a max-rank stride -- so name-driven placement
        matched nothing and silently skipped every q/k/v tensor.

        Both existing implementations in this repo work the other way round:
        drive from the adapter object and let the placement routine decide where
        each tensor belongs. So does this now -- it calls
        ``LoRAMemoryPool.load_lora_weight_to_buffer`` with ``buffer_id=slot_idx``,
        which is the exact code path a disk load uses, and therefore handles
        fusion, rank padding, TP slicing and the MoE variants identically. A
        staged update and a disk load cannot drift apart, because they are the
        same code.
        """
        adapter, modules, embed_module, lm_head_module = staged
        self.load_lora_weight_to_buffer(
            uid=None,                 # placement is by slot; uid is only logged
            buffer_id=slot_idx,
            lora_adapter=adapter,
            lora_modules=modules,
            lora_embed_tokens_module=embed_module,
            lora_lm_head_module=lm_head_module,
        )

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

    def _uid_for(self, name, adapter_id):
        """Resolve one adapter identity for both stage and activate.

        The trainer may supply ``adapter_id`` on one call and not the other --
        orbit's stage carries the tokenizer-minted id while its activate sends
        only the name. Resolving each call independently made them disagree, and
        the pool (correctly) refused to promote weights into a slot staged under
        a different uid. So remember what a name was staged under and reuse it.
        """
        if adapter_id is not None:
            return adapter_id
        staged = getattr(self, "_staged_uid_by_name", {}).get(name)
        if staged is not None:
            return staged
        for uid, ref in getattr(self, "lora_refs", {}).items():
            if getattr(ref, "lora_name", None) == name:
                return uid
        return name

    def stage_adapter(self, named_tensors, config, name, version, adapter_id=None):
        """Fill the staging slot from raw trainer tensors. Lock-free.

        Builds the same ``LoRAAdapter`` object a from-tensors disk load builds,
        then lets the pool place it with upstream's own routine. Nothing here
        interprets weight names or buffer layouts -- that was the mistake in the
        first version, which assumed one buffer per projection while upstream
        fuses q/k/v and gate/up.
        """
        from sglang.srt.lora.lora import LoRAAdapter
        from sglang.srt.lora.lora_config import LoRAConfig

        uid = self._uid_for(name, adapter_id)
        if not hasattr(self, "_staged_uid_by_name"):
            self._staged_uid_by_name = {}
        self._staged_uid_by_name[name] = uid
        lora_config = self.configs.get(uid)
        if lora_config is None:
            if not config:
                raise ValueError(
                    "stage_adapter needs an adapter_config the first time an "
                    f"adapter is staged (adapter {name!r} is not yet known)."
                )
            lora_config = LoRAConfig.from_dict(config)
            self.configs[uid] = lora_config

        adapter = LoRAAdapter(
            uid,
            lora_config,
            self.base_hf_config,
            self.load_config,
            self.lora_backend,
            base_model=self.base_model,
        )
        adapter.initialize_weights_from_tensors(dict(named_tensors))

        # A trainer-pushed adapter was never loaded from disk, so nothing has
        # given it a serving slot or a CPU-side object. Register both, mirroring
        # OFT's register_streamed_adapter:
        #   * uid -> slot, or activate has nowhere to promote into (and
        #     get_buffer_id raises a bare KeyError);
        #   * self.loras[uid], because upstream's prepare_lora_batch reads
        #     lora.config.r and lora.scaling from it to fill lora_ranks/scalings
        #     -- without it a served request would find rank 0 and the kernels
        #     would no-op, i.e. the adapter would silently not apply.
        pool = self.memory_pool
        if uid not in pool.uid_to_buffer_id:
            # Take a SERVING slot, never the staging slot. Upstream registers the
            # base model (uid None) at slot 0 during init, so claiming
            # active_idx unconditionally would both evict base routing and, when
            # the pool advertises a single slot, collide with staging.
            slot = next(
                (i for i in range(pool.max_loras_per_batch)
                 if pool.buffer_id_to_uid[i] is EMPTY_SLOT),
                None,
            )
            if slot is None:
                raise RuntimeError(
                    "no free serving slot for a staged adapter "
                    f"(pool advertises {pool.max_loras_per_batch}; the staging slot "
                    "is separate). Raise --max-loras-per-batch."
                )
            pool.uid_to_buffer_id[uid] = slot
            pool.buffer_id_to_uid[slot] = uid
            logger.info("staged adapter %s registered at serving slot %d", uid, slot)
        self.loras[uid] = adapter

        self.memory_pool.stage(
            version,
            (adapter, self.lora_modules, self.embed_tokens_module, self.lm_head_module),
            uid=uid,
        )

    def activate_adapter(self, name, version, adapter_id=None):
        """Promote the staged weights into this adapter's slot."""
        self.memory_pool.activate(version, uid=self._uid_for(name, adapter_id))

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

    def prepare_lora_batch(self, forward_batch):
        """Skip when the batch carries no per-request adapter ids.

        Upstream's implementation indexes ``forward_batch.lora_ids``; the fork's
        facade calls this unconditionally because the FROZEN peft/lora manager
        ignores that field entirely (single-active: it applies the adapter to
        every request and sizes off input_ids). During CUDA-graph capture there
        are no requests and the field is None, which upstream's version cannot
        take. Upstream guards identically in its own capture path.

        KNOWN GAP: upstream also seeds dummy ``lora_ids`` during capture so its
        batch metadata is recorded in the graph. The fork's capture path does
        not, so with decode CUDA graphs enabled a replayed batch would find no
        prepared metadata and skip LoRA silently. Until that is wired, run the
        staged stack with --disable-cuda-graph.
        """
        if getattr(forward_batch, "lora_ids", None) is None:
            return
        return super().prepare_lora_batch(forward_batch)

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

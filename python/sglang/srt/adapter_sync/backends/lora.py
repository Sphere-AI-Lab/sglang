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

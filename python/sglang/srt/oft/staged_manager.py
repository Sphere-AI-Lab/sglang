"""Symmetric counterpart to lora/staged_manager.py: OFT staging through one
hidden memory-pool slot, alongside B1's existing multi-tenant admission and
eviction (unaffected by this file)."""

import logging
from typing import Optional, Tuple

from sglang.srt.oft.mem_pool import OFTMemoryPool

logger = logging.getLogger(__name__)


class StagedOFTMemoryPool(OFTMemoryPool):
    """OFT pool with one physical slot hidden from serving, and per-uid
    stage/activate (unlike the inherited AdapterMemPool.stage/activate,
    which are pool-wide single-slot and used only by the non-multi-tenant
    double-buffer path)."""

    def __init__(self, *args, **kwargs):
        self.staging_idx = None
        self._staged_uid = None
        self._staged_version = None
        self._active_versions = {}
        super().__init__(*args, **kwargs)

    def init_buffers(self, base_model) -> None:
        """Allocate one extra physical slot in every dense/expert OFT buffer
        group while continuing to advertise ``max_ofts_per_batch`` serving
        slots.

        Unlike LoRAMemoryPool (whose ``init_buffers`` sizes every buffer
        directly off a single ``max_loras_per_batch`` field),
        ``OFTMemoryPool``'s buffer families are sized off two independent
        fields:

        - ``AdapterMemPool.register_buffer_group`` (called from
          ``_declare_groups``/``_declare_expert_groups``) allocates the dense
          ``R:{target}`` groups and the expert ``w1/w3/w13/w2`` groups using
          ``self.max_adapters_per_batch`` as the leading (slot) dimension.
        - ``embedding_R_buffer``, ``lm_head_R_buffer``, and
          ``new_embeddings_buffer`` are allocated directly off
          ``self.max_ofts_per_batch``.

        ``stage()``/``activate()`` (below) only ever read/write
        ``self._groups`` (mirroring ``_fill_slot``'s existing scope, which
        never touches the embedding/lm_head/added-token buffers), so only
        the ``register_buffer_group`` family needs the extra hidden row.
        Widening ``max_ofts_per_batch`` too would grow the embedding buffers
        for no reason, since the staging index is never used against them.
        """
        advertised = self.max_adapters_per_batch
        self.max_adapters_per_batch = advertised + 1
        try:
            super().init_buffers(base_model)
        finally:
            self.max_adapters_per_batch = advertised
        self.staging_idx = advertised

    def available_serving_slots(self) -> int:
        return self.max_ofts_per_batch

    def staged_identity(self) -> Optional[Tuple[str, int]]:
        if self._staged_uid is None:
            return None
        return self._staged_uid, self._staged_version

    def _require_staged_identity(self, uid: str, version: int) -> None:
        current = self.staged_identity()
        if current != (uid, version):
            detail = (
                "the staging slot is empty"
                if current is None
                else f"it holds uid={current[0]} version={current[1]}"
            )
            raise ValueError(
                f"No staged OFT adapter matches uid={uid} version={version}; {detail}."
            )

    def stage(self, uid: str, version: int, named_tensors) -> None:
        current = self.staged_identity()
        if current == (uid, version):
            return
        if current is not None:
            raise ValueError(
                f"Staging slot already holds uid={current[0]} version={current[1]}."
            )
        self._fill_slot(self.staging_idx, named_tensors)
        self._staged_uid = uid
        self._staged_version = version

    def activate(self, uid: str, version: int, destination: int) -> None:
        self._require_staged_identity(uid, version)
        if (
            destination < 0
            or destination >= self.max_ofts_per_batch
            or destination == self.staging_idx
        ):
            raise ValueError(
                f"OFT activation destination {destination} is not a serving slot."
            )
        for name, keyed in self._groups.items():
            for key in keyed:
                self.slot(name, key, destination).copy_(
                    self.slot(name, key, self.staging_idx)
                )
        self._active_versions[uid] = version
        self._staged_uid = None
        self._staged_version = None

    def discard_stage(self, uid: str, version: int) -> None:
        self._require_staged_identity(uid, version)
        self._staged_uid = None
        self._staged_version = None

    def active_version_for(self, uid: str) -> Optional[int]:
        return self._active_versions.get(uid)

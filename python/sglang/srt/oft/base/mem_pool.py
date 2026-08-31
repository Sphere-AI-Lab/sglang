import logging
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.oft.base.eviction_policy import get_eviction_policy

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


class AdapterMemPool:
    """Generic slot/eviction bookkeeping shared by adapter memory pools.

    Tracks which adapter UID occupies which buffer slot and how to evict a
    slot when the pool is full. Subclasses (e.g. OFTMemoryPool) own the actual
    weight buffers and the logic to load an adapter's weights into a slot.
    """

    # Default slot layout: single-active adapter at 0, staging twin at 1.
    # OFT overrides active_idx/staging_idx (base identity occupies slot 0).
    def __init__(
        self,
        max_adapters_per_batch,
        dtype,
        tp_size,
        tp_rank,
        eviction_policy,
        memory_saver_adapter=None,
        memory_saver_cpu_backup=False,
    ):
        self.max_adapters_per_batch = max_adapters_per_batch
        self.dtype = dtype
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.memory_saver_adapter = memory_saver_adapter
        self.memory_saver_cpu_backup = memory_saver_cpu_backup
        self.eviction_policy = get_eviction_policy(eviction_policy)
        self.uid_to_buffer_id = {}
        self.buffer_id_to_uid = [EMPTY_SLOT] * max_adapters_per_batch
        self._groups = {}
        self._active_version = None
        self._staged_version = None
        self.active_idx = 0
        self.staging_idx = 1

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

    def get_buffer_id(self, uid):
        return self.uid_to_buffer_id[uid]

    def register_buffer_group(self, name, per_key_shape, dtype=None, device=None):
        dt = dtype or self.dtype
        self._groups[name] = {
            key: torch.empty(
                self.max_adapters_per_batch, *shape, dtype=dt, device=device or "cuda"
            )
            for key, shape in per_key_shape.items()
        }

    def slot(self, name, key, slot_idx):
        return self._groups[name][key][slot_idx]

    def active_view(self, name, key):
        return self.slot(name, key, self.active_idx)

    def staging_view(self, name, key):
        return self.slot(name, key, self.staging_idx)

    def stage(self, version, named_tensors):
        self._fill_slot(self.staging_idx, named_tensors)
        self._staged_version = version

    def activate(self, version):
        if self._staged_version != version:
            raise RuntimeError(
                f"inactive_slot_busy: staged={self._staged_version} requested={version}"
            )
        for name, keyed in self._groups.items():
            for key in keyed:
                self.slot(name, key, self.active_idx).copy_(
                    self.slot(name, key, self.staging_idx)
                )
        self._active_version = version
        self._staged_version = None

    def _init_staging_from_active(self):
        """Boot-time hardening: seed each group's STAGING slot with its
        neutral-initialized ACTIVE slot, so a PARTIAL-coverage stage (a sync
        that fills only a subset of the registered groups) does not leave the
        staging slot as ``torch.empty`` garbage that ``activate()`` would then
        blanket-copy into the active slot.

        Call ONLY when double-buffer is on, AFTER the active slot has been
        neutral-initialized (end of the manager's init_state). It is a pure
        no-op for orbit's full-coverage syncs (every stage overwrites staging
        before activate); it only affects the never-orbit-hit partial-coverage
        case. The ``staging_idx == active_idx`` guard makes it safe to call when
        double-buffer collapses the two indices."""
        if self.staging_idx == self.active_idx:
            return
        for name, keyed in self._groups.items():
            for key in keyed:
                self.slot(name, key, self.staging_idx).copy_(
                    self.slot(name, key, self.active_idx)
                )

    def _declare_groups(self):
        raise NotImplementedError

    def _fill_slot(self, slot_idx, named_tensors):
        raise NotImplementedError

    def _acquire_buffer_slot(self, cur_uids, refs):
        # 1. Prioritize empty slots
        for buffer_id in range(self.max_adapters_per_batch):
            if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                return buffer_id

        # 2. Memory pool is full, need to evict
        candidates = set()
        for buffer_id in range(self.max_adapters_per_batch):
            uid = self.buffer_id_to_uid[buffer_id]
            if uid in cur_uids:
                continue
            if uid is not None:
                ref = refs.get(uid)
                if ref and ref.pinned:
                    continue
            candidates.add(uid)

        if not candidates:
            raise ValueError(
                "No available buffer slots found. Please ensure the number of "
                "active (pinned) adapters is less than max_adapters_per_batch."
            )

        # Prefer evicting adapters over base model (None)
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
            f"Evicting adapter {victim_uid} from buffer slot {victim_buffer_id}."
        )
        return victim_buffer_id

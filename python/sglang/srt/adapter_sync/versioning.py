"""Staged, per-adapter versioned activation -- independent of buffer layout.

This is the part of the weight-sync core that genuinely generalises across PEFT
methods. It owns *when* weights become live and *which* adapter they belong to;
it knows nothing about how those weights are stored.

That separation is what lets upstream LoRA use it. ``srt/adapter_sync``'s own
pool keeps weights in registered buffer groups, while upstream's
``LoRAMemoryPool`` keeps them in ``A_buffer``/``B_buffer`` dicts -- and the
never-edit-srt/lora rule means upstream's layout cannot be changed to match. A
host class supplies two primitives and gets the whole state machine:

    _fill_slot(slot_idx, named_tensors)   write incoming tensors into a slot
    _copy_slot(src_idx, dst_idx)          promote one slot's contents into another

Concurrency model: ONE shared staging slot, not one per adapter. The trainer
republishes a single adapter at a time, so serialising stages costs nothing real
and avoids a staging buffer per slot (which on MoE models, where expert buffers
dominate, would be expensive). ``_staged_uid`` records who currently owns that
slot so an activate cannot promote one adapter's weights into another's.
"""


class VersionedStaging:
    """Mixin: stage into a shared slot, then activate into one adapter's slot."""

    def _init_versioning(self):
        self._active_versions = {}   # uid -> version currently live in its slot
        self._staged_version = None
        self._staged_uid = None

    # ---- host class must provide ------------------------------------------
    def _fill_slot(self, slot_idx, named_tensors):
        raise NotImplementedError

    def _copy_slot(self, src_idx, dst_idx):
        raise NotImplementedError

    def _slot_for_uid(self, uid):
        """Slot holding ``uid``. ``None`` means the single-active convention."""
        return self.active_idx if uid is None else self.get_buffer_id(uid)

    # ---- the state machine -------------------------------------------------
    def stage(self, version, named_tensors, uid=None):
        """Fill the staging slot for ``uid``. Lock-free: generation keeps running."""
        self._fill_slot(self.staging_idx, named_tensors)
        self._staged_version = version
        self._staged_uid = uid

    def activate(self, version, uid=None):
        """Promote the staged weights into ``uid``'s slot, and only that slot."""
        if self._staged_version != version:
            raise RuntimeError(
                f"inactive_slot_busy: staged={self._staged_version} requested={version}"
            )
        if self._staged_uid != uid:
            raise RuntimeError(
                f"staged_adapter_mismatch: staged={self._staged_uid} requested={uid}. "
                "Refusing to promote one adapter's weights into another's slot."
            )
        self._copy_slot(self.staging_idx, self._slot_for_uid(uid))
        self._active_versions[uid] = version
        self._staged_version = None
        self._staged_uid = None

    def active_version(self, uid=None):
        """Version currently live in ``uid``'s slot, or None if never activated."""
        return self._active_versions.get(uid)

    @property
    def _active_version(self):
        """Single-active compatibility shim: the version of the ``None`` adapter."""
        return self._active_versions.get(None)

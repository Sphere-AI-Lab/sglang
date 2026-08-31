"""Symmetric counterpart to lora/staged_manager.py: OFT staging through one
hidden memory-pool slot, alongside B1's existing multi-tenant admission and
eviction (unaffected by this file)."""

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft import OFTAdapter
from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.oft_manager import OFTManager

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import OFTUpdateOutput

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


class PendingOFTStage:
    """CPU metadata retained until a staged OFT adapter is activated."""

    __slots__ = ("uid", "version", "named_tensors", "config", "name")

    def __init__(self, uid, version, named_tensors, config, name):
        self.uid = uid
        self.version = version
        self.named_tensors = named_tensors
        self.config = config
        self.name = name


class StagedOFTManager(OFTManager):
    """OFT manager with an explicit stage/activate transaction, alongside
    B1's existing multi-tenant admission and eviction (unaffected).

    ``named_tensors`` here is the memory pool's own per-uid ``stage()``
    payload -- ``Dict[(target_module, layer_id), (r, block_size,
    slice_index, split_count)]`` (see ``StagedOFTMemoryPool.stage`` and
    ``OFTMemoryPool._fill_slot``) -- NOT raw checkpoint-name tensors.
    Translating a raw streamed payload into that shape is the existing
    ``OFTManager._stage_fill`` machinery (Cayley precompute + dense/expert
    partitioning) built for the single-slot double-buffer path; wiring that
    translation onto this per-uid transaction is out of scope here (this
    class only wires the already-staged tensors through the new pool).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_oft_stage = None

    def init_memory_pool(self) -> None:
        """Override of OFTManager.init_memory_pool: same construction, but
        builds a StagedOFTMemoryPool so the extra hidden staging slot exists."""
        external_target_modules = set()
        getter = getattr(self.base_model, "get_oft_external_target_modules", None)
        if getter is not None:
            external_target_modules = set(getter())
        self.memory_pool = StagedOFTMemoryPool(
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
            double_buffer=self.peft_double_buffer,
        )
        logger.info(
            "Using %s for OFT R buffers (model dtype %s).",
            self.oft_r_dtype,
            self.dtype,
        )

        # Initializing memory pool with base model
        self.fetch_new_ofts({None})

    def stage_adapter(
        self, named_tensors, config, name, version, adapter_id=None
    ) -> "OFTUpdateOutput":
        uid = adapter_id if adapter_id is not None else name
        try:
            version = int(version)
            pending = self._pending_oft_stage
            if pending is not None:
                if (pending.uid, pending.version) == (uid, version):
                    return self.create_oft_update_result(success=True)
                raise ValueError(
                    f"An OFT stage is already pending for uid={pending.uid} "
                    f"version={pending.version}."
                )
            self.memory_pool.stage(uid, version, named_tensors)
            self._pending_oft_stage = PendingOFTStage(
                uid=uid,
                version=version,
                named_tensors=named_tensors,
                config=config,
                name=name,
            )
        except Exception as error:
            return self.create_oft_update_result(
                success=False, error_message=str(error)
            )
        return self.create_oft_update_result(success=True)

    def activate_adapter(
        self, name, version, adapter_id=None
    ) -> "OFTUpdateOutput":
        uid = adapter_id if adapter_id is not None else name
        try:
            version = int(version)
        except Exception as error:
            return self.create_oft_update_result(
                success=False, error_message=str(error)
            )

        pending = self._pending_oft_stage
        if pending is None or (pending.uid, pending.version) != (uid, version):
            detail = (
                "no OFT stage is pending"
                if pending is None
                else f"pending uid={pending.uid} version={pending.version}"
            )
            return self.create_oft_update_result(
                success=False,
                error_message=(
                    f"Cannot activate uid={uid} version={version}; {detail}."
                ),
            )

        destination = self.memory_pool.uid_to_buffer_id.get(uid)
        if destination is None:
            return self.create_oft_update_result(
                success=False,
                error_message=f"No serving slot is reserved for adapter uid={uid}.",
            )
        try:
            self.memory_pool.activate(uid, version, destination)
        except Exception as activation_error:
            return self.create_oft_update_result(
                success=False, error_message=str(activation_error)
            )

        # REQUIRED, not optional: OFTManager.prepare_oft_batch reads
        # self.adapters[uid].block_size / self.configs[uid].block_size for
        # every resident uid on every batch. Activating a new uid without
        # populating these leaves it physically live in the GPU slot but
        # invisible to the manager's own bookkeeping. Mirrors
        # load_oft_weights_from_tensors's construction (OFTConfig + OFTAdapter
        # from the adapter's own config), except no raw checkpoint tensors are
        # available here (named_tensors is the pool's already-Cayley-baked
        # per-uid payload, not the raw HF weights `initialize_weights_from_
        # tensors` expects) -- the R data itself is already correctly resident
        # in the memory pool via stage()/activate() above, so the OFTAdapter
        # built here is a metadata shell (block_size, target_modules) rather
        # than a fully weight-populated one.
        try:
            oft_config = OFTConfig.from_dict(pending.config)
            oft_adapter = OFTAdapter(
                uid, oft_config, self.base_hf_config, self.load_config, self.oft_backend
            )
        except Exception as bookkeeping_error:
            return self.create_oft_update_result(
                success=False, error_message=str(bookkeeping_error)
            )
        self.configs[uid] = oft_config
        self.adapters[uid] = oft_adapter

        # No discard_stage() call here, unlike StagedLoRAManager: unlike
        # StagedLoRAMemoryPool.activate (which leaves _staged_uid/_staged_
        # version set so the manager must clear them separately),
        # StagedOFTMemoryPool.activate (Task 2) already clears them itself
        # as its last step. Calling discard_stage() again here would re-run
        # _require_staged_identity against an already-empty staging slot and
        # raise unconditionally.
        self._pending_oft_stage = None
        return self.create_oft_update_result(success=True)

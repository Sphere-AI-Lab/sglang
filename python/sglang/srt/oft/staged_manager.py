"""Symmetric counterpart to lora/staged_manager.py: OFT staging through one
hidden memory-pool slot, alongside B1's existing multi-tenant admission and
eviction (unaffected by this file)."""

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft import OFTAdapter
from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.oft_registry import OFTRef

if TYPE_CHECKING:
    from sglang.srt.oft.io_types import OFTUpdateOutput

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
    """CPU metadata retained until a staged OFT adapter is activated.

    ``config``/``adapter`` are already fully constructed and validated (see
    ``StagedOFTManager.stage_adapter``) -- activation is then a trivial dict
    commit, mirroring ``PendingLoRAStage``/``StagedLoRAManager.activate_adapter``
    exactly, with no adapter-construction work (and no way for it to fail)
    left to do after the pool-level copy has already succeeded.
    """

    __slots__ = (
        "uid",
        "version",
        "config",
        "adapter",
        "name",
        "ref",
    )

    def __init__(self, uid, version, config, adapter, name, ref):
        self.uid = uid
        self.version = version
        self.config = config
        self.adapter = adapter
        self.name = name
        self.ref = ref


class StagedOFTManager(OFTManager):
    """OFT manager with an explicit stage/activate transaction, alongside
    B1's existing multi-tenant admission and eviction (unaffected).

    ``stage_adapter``'s ``named_tensors`` is raw checkpoint-name tensors --
    the SAME format ``OFTManager._stage_fill`` consumes, and the SAME format
    ``weight_updater.py`` -> ``oft/
    integration.py`` -> ``oft_manager.stage_adapter(...)`` actually supplies
    in production. This class reuses every transformation primitive
    ``_stage_fill`` (oft_manager.py:1412-1538, unedited) itself uses --
    ``_partition_expert_oft_tensors``, ``normalize_merged_oft_weights``,
    ``memory_pool._resolve_oft_tensor_plan``/``_slice_oft_compact_weight``,
    ``precompute_oft_r``, and the inherited ``apply_streamed_expert_oft`` --
    only the orchestration loop is duplicated here (see
    ``_partition_and_precompute``), because ``_stage_fill`` itself ends by
    calling the OLD pool-wide, single-slot ``AdapterMemPool.stage(version,
    named_tensors)`` (2 args), which is incompatible with
    ``StagedOFTMemoryPool.stage(uid, version, named_tensors)`` (3 args,
    per-uid). ``_stage_fill`` is untouched and still serves the original
    single-slot double-buffer path.
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
            double_buffer=False,
        )
        logger.info(
            "Using %s for OFT R buffers (model dtype %s).",
            self.oft_r_dtype,
            self.dtype,
        )

        # Initializing memory pool with base model
        self.fetch_new_ofts({None})

    def _partition_and_precompute(self, named_tensors, config):
        """Raw checkpoint-name tensors -> (staged_dense, fused_expert_chunk,
        block_size), the per-uid analogue of ``OFTManager._stage_fill``
        (oft_manager.py:1412-1538, unedited -- still the hook for the
        original single-slot ``AdapterManager.stage_adapter``). Reuses every
        transformation primitive that method uses; only the final pool call
        differs (per-uid ``memory_pool.stage(uid, version, staged_dense)``
        here vs. the pool-wide ``memory_pool.stage(version, staged_dense)``
        there), which is why this loop is duplicated rather than shared.

        Unlike ``_stage_fill``, this does NOT raise when other adapters are
        already resident: that guard existed because the OLD pool-wide
        single-slot path cannot represent more than one resident adapter --
        exactly the limitation ``StagedOFTMemoryPool`` (per-uid destination
        slots, Task 2) removes. Concurrent/conflicting use of the ONE hidden
        staging slot itself is still guarded, by
        ``StagedOFTMemoryPool.stage()``'s own ``_require_staged_identity``
        and by this class's own ``_pending_oft_stage`` check in
        ``stage_adapter``.
        """
        from sglang.srt.oft.mem_pool import normalize_merged_oft_weights
        from sglang.srt.oft.streamed_weight_loader import (
            _partition_expert_oft_tensors,
        )
        from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r

        memory_pool = self.memory_pool
        block_size = (
            config.get("oft_block_size", 32) if config else self.max_oft_block_size
        )
        if block_size != memory_pool.max_oft_block_size:
            raise ValueError(
                f"OFT staged update has block_size={block_size}, but the "
                f"server pool is allocated for --max-oft-block-size="
                f"{memory_pool.max_oft_block_size}; smaller or mixed block "
                f"sizes are unsupported."
            )

        fused_expert_chunk, dsv4_expert_chunk, dense_named_tensors = (
            _partition_expert_oft_tensors(named_tensors, tp_rank=memory_pool.tp_rank)
        )
        assert not dsv4_expert_chunk, (
            "DSV4-style expert OFT staging is not supported (fork "
            "DeepSeekV4 model support was removed)"
        )

        if dense_named_tensors:
            dense_dict = dict(dense_named_tensors)
            if len(dense_dict) == len(dense_named_tensors):
                dense_named_tensors = list(
                    normalize_merged_oft_weights(
                        dense_dict, available_fused_targets=set(memory_pool.R_buffer)
                    ).items()
                )

        staged_dense = {}
        oft_modules = self.adapter_modules
        for tensor_name, tensor in dense_named_tensors:
            layer_id = get_layer_id(tensor_name)
            if layer_id is None:
                continue  # embeddings/lm_head: out of scope, see _stage_fill.
            fused_target, slice_module, is_row_parallel, slice_index, split_count = (
                memory_pool._resolve_oft_tensor_plan(tensor_name, oft_modules, layer_id)
            )
            compact_weight = tensor
            if is_row_parallel:
                compact_weight = memory_pool._slice_oft_compact_weight(
                    compact_weight, slice_module
                )
            target_device = memory_pool.R_buffer[fused_target][layer_id].device
            if compact_weight.device != target_device:
                compact_weight = compact_weight.to(target_device)
            r = precompute_oft_r(compact_weight, block_size)
            existing = staged_dense.get((fused_target, layer_id))
            if existing is not None and existing[2] != slice_index:
                raise RuntimeError(
                    f"stage_adapter: multiple split OFT slices for the same "
                    f"fused target {fused_target!r} (layer {layer_id}) arrived "
                    f"in one call without all siblings present to pre-fuse "
                    f"(slice_index {existing[2]} then {slice_index}); send "
                    f"the fused target's siblings together in one payload."
                )
            staged_dense[(fused_target, layer_id)] = (
                r,
                block_size,
                slice_index,
                split_count,
            )

        return staged_dense, fused_expert_chunk, block_size

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

            # Construct and validate the adapter's identity FIRST, before the
            # pool is touched at all: a failure here (e.g. a missing
            # peft_type/target_modules/oft_block_size key) must leave the
            # hidden staging slot untouched, or it jams permanently -- there
            # is no rollback for memory_pool.stage() once it has run, and
            # _pending_oft_stage would stay None (this call never reaches the
            # assignment below), so no later stage_adapter call for ANY uid
            # could ever re-occupy the slot. Matches
            # StagedLoRAManager.stage_adapter's order exactly: LoRAConfig.
            # from_dict -> validate -> _create_lora_adapter_from_tensors, all
            # strictly before memory_pool.stage(...).
            oft_config = OFTConfig.from_dict(config)
            old_ref = self.refs.get(uid)
            if old_ref is not None and version <= old_ref.adapter_version:
                raise ValueError(
                    f"OFT adapter version {version} must be newer than active "
                    f"version {old_ref.adapter_version}."
                )
            oft_adapter = OFTAdapter(
                uid, oft_config, self.base_hf_config, self.load_config, self.oft_backend
            )
            oft_adapter.initialize_weights_from_tensors(dict(named_tensors))

            staged_dense, fused_expert_chunk, block_size = (
                self._partition_and_precompute(named_tensors, config)
            )
            pending_ref = OFTRef(
                adapter_id=uid,
                adapter_name=name,
                adapter_path="__distributed__",
                pinned=old_ref.pinned if old_ref is not None else False,
                adapter_version=version,
                reloadable=False,
            )
            # memory_pool.stage()/apply_streamed_expert_oft() are the two
            # calls that actually mutate the hidden staging slot; either can
            # still raise (e.g. apply_streamed_expert_oft's shape/dtype/
            # device mismatch or tp_size divisibility checks on a bad expert
            # chunk) AFTER the dense stage() call has already run. Neither
            # has its own rollback, and self._pending_oft_stage is not
            # assigned until both succeed -- so a bare exception here would
            # leave the pool's _staged_uid/_staged_version set for this
            # (uid, version) with no pending transaction pointing at it,
            # jamming the one hidden slot for every future stage_adapter
            # call (any uid) until someone retries this exact identity.
            # discard_stage() clears that regardless of which of the two
            # calls failed, so the slot is clean again before the failure
            # result is returned.
            try:
                self.memory_pool.stage(uid, version, staged_dense)
                if fused_expert_chunk:
                    self.apply_streamed_expert_oft(
                        fused_expert_chunk,
                        block_size,
                        slot_idx=self.memory_pool.staging_idx,
                    )
            except Exception as mutation_error:
                try:
                    self.memory_pool.discard_stage(uid, version)
                except Exception:
                    # stage() itself may have raised before ever setting
                    # _staged_uid/_staged_version (e.g. a bad tensor shape
                    # inside _fill_slot), in which case there is nothing to
                    # discard and this would raise "the staging slot is
                    # empty" -- the original mutation_error is what matters.
                    pass
                raise mutation_error

            self._pending_oft_stage = PendingOFTStage(
                uid=uid,
                version=version,
                config=oft_config,
                adapter=oft_adapter,
                name=name,
                ref=pending_ref,
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

        # Existing adapters already have a serving slot, so commit the hidden
        # staging slot into it immediately. A newly introduced adapter has no
        # serving slot yet; publish its CPU-side state now and let the first
        # request admit it through OFTMemoryPool.prepare_oft_batch.
        destination = self.memory_pool.uid_to_buffer_id.get(uid)
        if destination is not None:
            try:
                self.memory_pool.activate(uid, version, destination)
            except Exception as activation_error:
                return self.create_oft_update_result(
                    success=False, error_message=str(activation_error)
                )
        else:
            self.memory_pool.discard_stage(uid, version)

        # Lazy admission needs the materialized adapter and its configuration;
        # eviction filtering also needs the corresponding reference.
        self.configs[uid] = pending.config
        self.adapters[uid] = pending.adapter
        self.refs[uid] = pending.ref

        # Both activate() and the new-adapter discard branch clear the hidden
        # staging identity before the transaction is published.
        self._pending_oft_stage = None
        return self.create_oft_update_result(success=True)


from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend


class OFTStagingBackend(AdapterStagingBackend):
    """Tokenizer-layer two-phase staging for canonical OFT."""

    def __init__(self, tm):
        self._tm = tm

    async def reserve_stage(self, obj) -> None:
        async with self._tm.peft_update_lock:
            await self._reserve_locked(obj)

    async def _reserve_locked(self, obj) -> None:
        if obj.load_format != "oft_adapter" or not obj.adapter_name:
            raise ValueError(
                "canonical OFT staging requires load_format=oft_adapter "
                "and adapter_name"
            )
        try:
            version = int(obj.adapter_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical OFT staging requires an integer adapter_version"
            ) from exc
        if obj.adapter_name in self._tm.failed_oft_activations:
            raise ValueError(
                f"OFT adapter '{obj.adapter_name}' is quarantined; restart required"
            )

        pending = self._tm.pending_oft_stage
        if pending is not None:
            if (
                pending.adapter_name == obj.adapter_name
                and pending.adapter_version == version
            ):
                obj.adapter_id = pending.adapter_id
                return
            raise ValueError(
                "staging slot already reserved for "
                f"name={pending.adapter_name} id={pending.adapter_id} "
                f"version={pending.adapter_version}"
            )

        active = self._tm.peft_registry.get_all_adapters().get(obj.adapter_name)
        if active is not None and version <= active.adapter_version:
            raise ValueError(
                f"OFT adapter version {version} must be newer than active "
                f"version {active.adapter_version}."
            )
        if active is None:
            candidate = OFTRef(
                adapter_name=obj.adapter_name,
                adapter_path="__distributed__",
                pinned=False,
                adapter_version=version,
                reloadable=False,
            )
        else:
            candidate = replace(
                active,
                adapter_path="__distributed__",
                adapter_version=version,
                reloadable=False,
            )

        self._tm.pending_oft_stage = candidate
        obj.adapter_id = candidate.adapter_id

    def prepare_activation(self, obj) -> None:
        try:
            version = int(obj.adapter_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical OFT activation requires an integer adapter_version"
            ) from exc
        pending = self._tm.pending_oft_stage
        if pending is None or (
            pending.adapter_name,
            pending.adapter_version,
        ) != (obj.adapter_name, version):
            detail = (
                "no OFT stage is pending"
                if pending is None
                else f"pending name={pending.adapter_name} id={pending.adapter_id} "
                f"version={pending.adapter_version}"
            )
            raise ValueError(
                f"Cannot activate name={obj.adapter_name} version={version}; {detail}"
            )
        obj.adapter_id = pending.adapter_id

    async def _publish(self) -> None:
        pending = self._tm.pending_oft_stage
        if pending is None:
            raise RuntimeError("No OFT stage is pending for publication")
        registered = self._tm.peft_registry.get_all_adapters().get(
            pending.adapter_name
        )
        if registered is None:
            await self._tm.peft_registry.register(pending)
        else:
            await self._tm.peft_registry.refresh(pending)
        self._tm.peft_ref_cache[pending.adapter_name] = pending
        self._tm.failed_oft_activations.pop(pending.adapter_name, None)
        self._tm.pending_oft_stage = None

    async def finish_activation(self, obj, results):
        from sglang.srt.managers.communicator import FanOutCommunicator

        pending = self._tm.pending_oft_stage
        if pending is None:
            raise RuntimeError("No OFT stage is pending during activation")
        success, message = FanOutCommunicator.merge_results(results)
        expected_version = int(obj.adapter_version)

        def version_matches(result) -> bool:
            try:
                return int(result.active_adapter_version) == expected_version
            except (TypeError, ValueError):
                return False

        versions_match = bool(results) and all(version_matches(r) for r in results)
        if success and versions_match:
            await self._publish()
            return True, message

        active_versions = [getattr(r, "active_adapter_version", None) for r in results]
        failure = (
            "OFT activation consistency failure for "
            f"adapter '{pending.adapter_name}' version={pending.adapter_version}: "
            f"{message}; worker active versions={active_versions}; restart required"
        )
        self._tm.failed_oft_activations[pending.adapter_name] = failure
        return False, failure

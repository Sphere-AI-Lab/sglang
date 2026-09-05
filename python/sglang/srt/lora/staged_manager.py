"""Native LoRA staging through one hidden memory-pool slot."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.managers.io_struct import LoRAUpdateOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingLoRAStage:
    """CPU metadata retained until a staged adapter is activated."""

    ref: LoRARef
    config: LoRAConfig
    adapter: LoRAAdapter
    old_ref: Optional[LoRARef]
    old_config: Optional[LoRAConfig]
    old_adapter: Optional[LoRAAdapter]


class StagedLoRAMemoryPool(LoRAMemoryPool):
    """Native LoRA pool with one physical slot hidden from serving."""

    def __init__(self, *args, **kwargs):
        self.staging_idx = None
        self._staged_uid = None
        self._staged_version = None
        super().__init__(*args, **kwargs)

    def init_buffers(self, base_model) -> None:
        """Allocate N+1 physical slots while continuing to advertise N."""
        advertised = self.max_loras_per_batch
        self.max_loras_per_batch = advertised + 1
        try:
            super().init_buffers(base_model)
        finally:
            self.max_loras_per_batch = advertised
        self.staging_idx = advertised

    def _slot_buffers(self):
        """Yield every tensor whose leading dimension is the adapter slot."""
        for buffers in (self.A_buffer, self.B_buffer):
            for per_layer in buffers.values():
                yield from per_layer
        for buffers in (
            self.embedding_A_buffer,
            self.embedding_B_buffer,
            self.lm_head_A_buffer,
            self.lm_head_B_buffer,
            self.new_embeddings_buffer,
        ):
            yield from buffers.values()

    def _copy_slot(self, source: int, destination: int) -> None:
        for tensor in self._slot_buffers():
            tensor[destination].copy_(tensor[source])

    def available_serving_slots(self) -> int:
        return self.max_loras_per_batch

    def get_tensor(self, target_module, layer_id, lora_type):
        """Expose only serving slots to forward-time LoRA kernels."""
        tensor = super().get_tensor(target_module, layer_id, lora_type)
        return tensor[: self.max_loras_per_batch]

    def get_embedding_tensor(self, target_module, lora_type):
        """Hide the staging slot from embedding and lm-head kernels too."""
        tensor = super().get_embedding_tensor(target_module, lora_type)
        if tensor is None:
            return None
        return tensor[: self.max_loras_per_batch]

    def staged_identity(self) -> Optional[tuple[str, int]]:
        if self._staged_uid is None:
            return None
        return self._staged_uid, self._staged_version

    def _require_staged_identity(self, uid: str, version: int) -> None:
        current = self.staged_identity()
        if current != (uid, version):
            if current is None:
                detail = "the staging slot is empty"
            else:
                detail = f"it holds uid={current[0]} version={current[1]}"
            raise ValueError(
                f"No staged LoRA matches uid={uid} version={version}; {detail}."
            )

    def stage(
        self,
        uid: str,
        version: int,
        adapter: LoRAAdapter,
        lora_modules,
        embed_module,
        lm_head_module,
    ) -> None:
        current = self.staged_identity()
        if current == (uid, version):
            return
        if current is not None:
            raise ValueError(
                f"Staging slot already holds uid={current[0]} version={current[1]}."
            )
        if self.staging_idx is None:
            raise RuntimeError("LoRA staging buffers have not been initialized.")
        if self.staging_idx < self.max_loras_per_batch:
            raise RuntimeError("LoRA staging slot overlaps advertised serving slots.")

        self.load_lora_weight_to_buffer(
            uid,
            self.staging_idx,
            adapter,
            lora_modules,
            embed_module,
            lm_head_module,
        )
        self._staged_uid = uid
        self._staged_version = version

    def activate(self, uid: str, version: int, destination: int) -> None:
        self._require_staged_identity(uid, version)
        if (
            destination < 0
            or destination >= self.max_loras_per_batch
            or destination == self.staging_idx
        ):
            raise ValueError(
                f"LoRA activation destination {destination} is not a serving slot."
            )
        self._copy_slot(self.staging_idx, destination)

    def discard_stage(self, uid: str, version: int) -> None:
        self._require_staged_identity(uid, version)
        self._staged_uid = None
        self._staged_version = None


class StagedLoRAManager(LoRAManager):
    """Native LoRA manager with an explicit stage/activate transaction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_lora_stage = None

    def init_memory_pool(self) -> None:
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
        self.fetch_new_loras({None})

    def stage_adapter(
        self, named_tensors, config, name, version, adapter_id=None
    ) -> LoRAUpdateOutput:
        uid = adapter_id if adapter_id is not None else name
        try:
            version = int(version)
            pending = getattr(self, "_pending_lora_stage", None)
            if pending is not None:
                current = (pending.ref.lora_id, pending.ref.version)
                if current == (uid, version):
                    return self.create_lora_update_result(success=True)
                raise ValueError(
                    f"A LoRA stage is already pending for uid={current[0]} "
                    f"version={current[1]}."
                )

            new_config = LoRAConfig.from_dict(
                config,
                base_vocab_size=self.base_hf_config.vocab_size,
            )
            old_ref = self.lora_refs.get(uid)
            old_config = self.configs.get(uid)
            old_adapter = self.loras.get(uid)
            new_ref = LoRARef(
                lora_id=uid,
                lora_name=name,
                lora_path="__distributed__",
                pinned=old_ref.pinned if old_ref is not None else False,
                reloadable=False,
                version=version,
            )
            self.validate_new_adapter(
                new_config,
                new_ref,
                is_update=uid in self.loras,
                old_ref=old_ref,
            )
            new_adapter = self._create_lora_adapter_from_tensors(
                new_ref, new_config, dict(named_tensors)
            )
            self.memory_pool.stage(
                uid,
                version,
                new_adapter,
                self.lora_modules,
                self.embed_tokens_module,
                self.lm_head_module,
            )
            self._pending_lora_stage = PendingLoRAStage(
                ref=new_ref,
                config=new_config,
                adapter=new_adapter,
                old_ref=old_ref,
                old_config=old_config,
                old_adapter=old_adapter,
            )
        except Exception as error:
            return self.create_lora_update_result(
                success=False, error_message=str(error)
            )

        return self.create_lora_update_result(success=True)

    def activate_adapter(
        self, name, version, adapter_id=None
    ) -> LoRAUpdateOutput:
        uid = adapter_id if adapter_id is not None else name
        try:
            version = int(version)
        except Exception as error:
            return self.create_lora_update_result(
                success=False, error_message=str(error)
            )

        pending = getattr(self, "_pending_lora_stage", None)
        if pending is None or (
            pending.ref.lora_id,
            pending.ref.version,
        ) != (uid, version):
            if pending is None:
                detail = "no LoRA stage is pending"
            else:
                detail = (
                    f"pending uid={pending.ref.lora_id} "
                    f"version={pending.ref.version}"
                )
            return self.create_lora_update_result(
                success=False,
                error_message=(
                    f"Cannot activate uid={uid} version={version}; {detail}."
                ),
            )

        destination = self.memory_pool.uid_to_buffer_id.get(uid)
        if destination is not None:
            try:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.memory_pool.activate(uid, version, destination)
            except Exception as activation_error:
                try:
                    if pending.old_adapter is None:
                        raise RuntimeError("the previous adapter is unavailable")
                    self.memory_pool.load_lora_weight_to_buffer(
                        uid,
                        destination,
                        pending.old_adapter,
                        self.lora_modules,
                        self.embed_tokens_module,
                        self.lm_head_module,
                    )
                except Exception as restore_error:
                    logger.exception(
                        "Failed to restore LoRA adapter %s in serving slot %s; "
                        "the worker must restart.",
                        uid,
                        destination,
                    )
                    return self.create_lora_update_result(
                        success=False,
                        error_message=(
                            f"{activation_error}; restoring the previous adapter "
                            f"failed: {restore_error}; worker restart required"
                        ),
                    )
                return self.create_lora_update_result(
                    success=False,
                    error_message=str(activation_error),
                )

        self.configs[uid] = pending.config
        self.loras[uid] = pending.adapter
        self.lora_refs[uid] = pending.ref
        old_pinned = int(bool(pending.old_ref.pinned)) if pending.old_ref else 0
        self.num_pinned_loras += int(bool(pending.ref.pinned)) - old_pinned
        self.memory_pool.discard_stage(uid, version)
        self._pending_lora_stage = None
        return self.create_lora_update_result(success=True)


class LoRAStagingBackend:
    """Tokenizer-layer staging for native LoRA. Wraps TokenizerManager's
    lora_registry/lora_ref_cache/failed_lora_activations/pending_lora_stage
    state — same objects tokenizer_control_mixin.py used directly before
    this extraction; the fields did not move, only which class reads them."""

    def __init__(self, tm):
        self._tm = tm

    def _assert_available(self, lora_path) -> None:
        # Shared with the always-on check in _resolve_lora_path
        # (tokenizer_manager.py) -- one quarantine check, read off the same
        # self._tm.failed_lora_activations dict either way.
        self._tm._assert_native_lora_available(lora_path)

    def _quarantine(self, name: str, message: str) -> None:
        self._tm.failed_lora_activations[name] = message

    async def reserve_stage(self, obj) -> None:
        # Reservation must be atomic across concurrent stage requests. The
        # communicator serializes dispatch, but reservation happens before it.
        async with self._tm.lora_update_lock:
            await self._reserve_locked(obj)

    async def _reserve_locked(self, obj) -> None:
        if obj.load_format != "lora_adapter" or not obj.adapter_name:
            raise ValueError(
                "native LoRA staging requires load_format=lora_adapter "
                "and adapter_name"
            )
        try:
            version = int(obj.adapter_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native LoRA staging requires an integer adapter_version"
            ) from exc
        if self._tm.server_args.tokenizer_worker_num > 1:
            raise ValueError("native LoRA staging requires tokenizer_worker_num == 1")
        if obj.adapter_name in self._tm.failed_lora_activations:
            raise ValueError(
                f"LoRA adapter '{obj.adapter_name}' is quarantined; restart required"
            )

        pending = self._tm.pending_lora_stage
        if pending is not None:
            if pending.lora_name == obj.adapter_name and pending.version == version:
                obj.adapter_id = pending.lora_id
                return
            raise ValueError(
                "staging slot already reserved for "
                f"name={pending.lora_name} id={pending.lora_id} "
                f"version={pending.version}"
            )

        candidate, _ = await self._tm.lora_registry.register_or_reuse(
            LoRARef(
                lora_name=obj.adapter_name,
                lora_path="__distributed__",
                pinned=False,
                reloadable=False,
                version=version,
            ),
            upsert=True,
            preserve_pinned=True,
        )
        self._tm.pending_lora_stage = candidate
        obj.adapter_id = candidate.lora_id

    def prepare_activation(self, obj) -> None:
        try:
            version = int(obj.adapter_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native LoRA activation requires an integer adapter_version"
            ) from exc

        pending = self._tm.pending_lora_stage
        if pending is None or (pending.lora_name, pending.version) != (
            obj.adapter_name,
            version,
        ):
            if pending is None:
                detail = "no native LoRA stage is pending"
            else:
                detail = (
                    f"pending name={pending.lora_name} id={pending.lora_id} "
                    f"version={pending.version}"
                )
            raise ValueError(
                f"Cannot activate name={obj.adapter_name} version={version}; {detail}"
            )
        self._assert_available(obj.adapter_name)
        obj.adapter_id = pending.lora_id

    async def _publish(self) -> None:
        pending = self._tm.pending_lora_stage
        if pending is None:
            raise RuntimeError("No native LoRA stage is pending for publication")
        registered = self._tm.lora_registry.get_all_adapters().get(pending.lora_name)
        if registered is None:
            await self._tm.lora_registry.register(pending)
        else:
            await self._tm.lora_registry.refresh(pending)
        self._tm.lora_ref_cache[pending.lora_name] = pending
        self._tm.failed_lora_activations.pop(pending.lora_name, None)
        self._tm.pending_lora_stage = None

    async def finish_activation(self, obj, results):
        from sglang.srt.managers.communicator import FanOutCommunicator

        pending = self._tm.pending_lora_stage
        if pending is None:
            raise RuntimeError("No native LoRA stage is pending during activation")
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
            "Native LoRA activation consistency failure for "
            f"adapter '{pending.lora_name}' version={pending.version}: "
            f"{message}; worker active versions={active_versions}; restart required"
        )
        self._quarantine(pending.lora_name, failure)
        return False, failure

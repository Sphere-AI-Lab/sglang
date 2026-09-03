from contextlib import nullcontext

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.oft.utils import get_target_module_name
from sglang.srt.utils import replace_submodule


class AdapterManager:
    """Generic lifecycle/utility methods for adapter managers. Originally
    intended to be shared with LoRA (several method docstrings below still
    describe LoRA-specific behavior), but LoRAManager evolved independently
    and does not subclass this -- today only OFTManager does."""

    # ------------------------------------------------------------------ #
    #  Hooks — subclasses must implement these to specialize the generic
    #  load/unload lifecycle below for their adapter method.
    # ------------------------------------------------------------------ #

    def _update_output_cls(self):
        raise NotImplementedError

    def _build_config(self, path):
        raise NotImplementedError

    def _clear_expert_on_unload(self, adapter):
        raise NotImplementedError

    def _unload_streamed_adapter(self, ref):
        raise NotImplementedError

    def _set_module_info(self, module, target_module, layer_id):
        raise NotImplementedError

    def _update_embedding_info(self):
        raise NotImplementedError

    def _prepare_mem_pool_batch(self, cur_uids):
        raise NotImplementedError

    def _get_adapter_layer(self, module):
        raise NotImplementedError

    def _stage_fill(self, named_tensors, config, name, version):
        raise NotImplementedError

    def _bump_ref_version(self, name, version):
        raise NotImplementedError

    def _make_streamed_ref(self, name, version, oft_id=None, config=None):
        """Construct the method-specific AdapterRef for an adapter first
        introduced via ``stage_adapter`` (double-buffer / streamed identity
        boot). Called by ``stage_adapter`` only when ``name`` is not already
        registered in ``self.oft_refs``.

        Multi-slot methods (OFT) additionally register per-request serving
        routing here (``memory_pool.uid_to_buffer_id[oft_id]=active_idx``,
        via ``register_streamed_adapter``) so a ``/generate`` naming the adapter
        resolves; single-active methods (LoRA) never route per-request and
        ignore ``oft_id``/``config``."""
        raise NotImplementedError

    def _make_update_result(self, success, error_message=""):
        cls = self._update_output_cls()
        return cls(
            success=success,
            error_message=error_message,
            loaded_adapters={
                ref.oft_name: ref.oft_path for ref in self.oft_refs.values()
            },
        )

    def init_adapters(self):
        self.configs = {}
        self.adapters = {}
        self.oft_refs = {}
        self.num_pinned = 0
        # Overlap-loading (see oft_overlap_loader.py): maps an adapter id
        # currently being materialized on the load stream to the CUDA event
        # marking that copy's completion. Mirrors LoRAManager's
        # pending_lora_load_events exactly.
        self.pending_oft_load_events = {}

    def unload_adapter(self, ref):
        adapter = self.configs.get(ref.oft_id)
        stored_ref = self.oft_refs.get(ref.oft_id)
        if adapter is None or stored_ref is None:
            # Should have been verified before the request was sent to the
            # backend (e.g. a registry/GPU-pool divergence, such as the GPU
            # side having already evicted this adapter) -- return a graceful
            # failure instead of asserting, so this can never crash the
            # engine outright.
            return self._make_update_result(
                success=False,
                error_message=(
                    f"Adapter with ID {ref.oft_id} is not loaded. This "
                    "should have been verified before request is sent to "
                    "the backend."
                ),
            )
        if ref.oft_id not in self.adapters:
            return self._unload_streamed_adapter(stored_ref)
        try:
            # An in-flight overlap load for this exact id must finish before
            # the adapter's state is torn down, or the load-stream copy could
            # write into memory that's already been freed/reassigned.
            pending_events = getattr(self, "pending_oft_load_events", {})
            pending_event = pending_events.pop(ref.oft_id, None)
            if pending_event is not None:
                pending_event.synchronize()

            self._clear_expert_on_unload(self.adapters.get(ref.oft_id))
            del self.configs[ref.oft_id]
            del self.adapters[ref.oft_id]
            del self.oft_refs[ref.oft_id]
            self.num_pinned -= int(stored_ref.pinned)
        except Exception as e:
            return self._make_update_result(success=False, error_message=str(e))
        return self._make_update_result(success=True)

    def stage_adapter(self, named_tensors, config, name, version, oft_id=None):
        """Fill the staging slot (lock-free). Reuses _load_weights/expert fills
        with slot_idx=staging_idx via mem_pool.stage()."""
        self._stage_fill(named_tensors, config, name, version)
        # Identity-boot deployments (e.g. orbit) start with empty refs; the
        # adapter name first arrives with the stage request. Register it so
        # activate_adapter's _bump_ref_version can find and bump it, and (for
        # multi-slot OFT) so per-request /generate routing resolves. No-op once
        # registered (subsequent syncs of the same adapter reuse the ref).
        if not any(ref.oft_name == name for ref in self.oft_refs.values()):
            ref = self._make_streamed_ref(name, version, oft_id, config)
            # setdefault, not assignment: OFT's _make_streamed_ref already stored
            # the ref via register_streamed_adapter (which also sets the routing);
            # LoRA's did not, so this registers the fresh-uuid ref for it. Avoids
            # double-registering the OFT ref.
            self.oft_refs.setdefault(ref.oft_id, ref)

    def activate_adapter(self, name, version):
        self.memory_pool.activate(version)
        self._bump_ref_version(name, version)

    def _weights_memory_saver_region(self):
        adapter = getattr(self, "memory_saver_adapter", None)
        if (
            adapter is None
            or not getattr(adapter, "enabled", False)
            or not getattr(self, "memory_saver_cpu_backup", False)
        ):
            return nullcontext()
        return adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=True,
        )

    def _find_fused_moe_modules(self):
        """Lazily find and cache all FusedMoE modules indexed by layer_id.

        Sees through adapter wrappers (FusedMoEWithOFT / FusedMoEWithLoRA): a
        wrapper exposes the real module as ``.base_layer``, so buffer injection,
        streamed sync, double-buffer and cuda-graph init keep operating on the
        underlying FusedMoE regardless of whether it has been wrapped.
        """
        if hasattr(self, "_moe_modules"):
            return self._moe_modules
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        self._moe_modules = {}
        for name, module in self.base_model.named_modules():
            candidate = getattr(module, "base_layer", module)
            if isinstance(candidate, FusedMoE):
                layer_id = get_layer_id(name)
                if layer_id is not None:
                    self._moe_modules[layer_id] = candidate
        return self._moe_modules

    def validate_batch(self, oft_ids):
        """
        Validate if the OFT IDs in the batch can be loaded into the current OFT memory pool.
        """
        # Buffer slot 0 is always reserved for the base/identity placeholder
        # (uid=None) -- allocate_buffer_slot_with_eviction never evicts it
        # (Task 4b review fix) -- so real per-batch adapter capacity is
        # max_adapters_per_batch - 1, not max_adapters_per_batch. A None in
        # oft_ids never competes for a slot, so it must not count toward
        # this bound either; admitting it here without correcting for that
        # let a batch referencing more distinct real adapters than the pool
        # could ever hold resident simultaneously reach prepare_oft_batch,
        # which raises ValueError for any adapter not already resident (no
        # on-disk preload path exists anymore to lazily seat it) with no
        # handler for that error -- SIGQUITs the whole engine.
        real_adapter_ids = {a for a in oft_ids if a is not None}
        if len(real_adapter_ids) > self.max_adapters_per_batch - 1:
            return False

        # skip pinned OFT check if no pinned OFT adapters are loaded.
        if self.num_pinned == 0:
            return True

        # counting the number of pinned OFT adapters in the batch.
        pinned_ofts_in_batch = 0
        for oft_id in real_adapter_ids:
            oft_ref = self.oft_refs.get(oft_id)
            assert (
                oft_ref is not None
            ), f"adapter ID {oft_id} not found in refs."
            pinned_ofts_in_batch += int(oft_ref.pinned)

        assert pinned_ofts_in_batch <= self.num_pinned, (
            f"Number of pinned adapters in the batch ({pinned_ofts_in_batch}) exceeds the total number of pinned adapters "
            f"({self.num_pinned}). This indicates a bug in the adapter loading logic."
        )

        required_slots = len(real_adapter_ids) - pinned_ofts_in_batch
        mem_pool_vacancy = (
            self.memory_pool.max_adapters_per_batch - 1
        ) - self.num_pinned

        return required_slots <= mem_pool_vacancy

    def set_adapter_module(self, module_name, module):
        adapter_module = self._get_adapter_layer(module)
        replace_submodule(self.base_model, module_name, adapter_module)
        return adapter_module

    def update_info(self):
        """Associate all adapter modules with the latest memory buffer."""
        for layer_id, layer_modules in enumerate(self.adapter_modules):
            for module_name, module in layer_modules.items():
                target_module = get_target_module_name(
                    module_name, self.memory_pool.target_modules
                )
                self._set_module_info(module, target_module, layer_id)
        self._update_embedding_info()

    def fetch_new_adapters(self, new_adapters, running_adapters=set()):
        cur_uids = new_adapters | running_adapters
        # Real (non-None) adapter capacity is max_adapters_per_batch - 1 --
        # buffer slot 0 is always reserved for the base/identity placeholder
        # and never evicted (see validate_batch above for the full
        # rationale). A None in cur_uids never competes for a slot.
        real_uids = {uid for uid in cur_uids if uid is not None}
        assert len(real_uids) <= self.max_adapters_per_batch - 1
        self._prepare_mem_pool_batch(cur_uids)

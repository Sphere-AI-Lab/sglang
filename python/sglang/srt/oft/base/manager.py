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

    def _load_weights(self, ref):
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

    def _make_streamed_ref(self, name, version, adapter_id=None, config=None):
        """Construct the method-specific AdapterRef for an adapter first
        introduced via ``stage_adapter`` (double-buffer / streamed identity
        boot). Called by ``stage_adapter`` only when ``name`` is not already
        registered in ``self.refs``.

        Multi-slot methods (OFT) additionally register per-request serving
        routing here (``memory_pool.uid_to_buffer_id[adapter_id]=active_idx``,
        via ``register_streamed_adapter``) so a ``/generate`` naming the adapter
        resolves; single-active methods (LoRA) never route per-request and
        ignore ``adapter_id``/``config``."""
        raise NotImplementedError

    def _make_update_result(self, success, error_message=""):
        cls = self._update_output_cls()
        return cls(
            success=success,
            error_message=error_message,
            loaded_adapters={
                ref.adapter_name: ref.adapter_path for ref in self.refs.values()
            },
        )

    def init_adapters(self, refs=None):
        self.configs = {}
        self.adapters = {}
        self.refs = {}
        self.num_pinned = 0
        if refs:
            for ref in refs:
                result = self.load_adapter(ref)
                if not result.success:
                    raise RuntimeError(
                        f"Failed to load adapter {ref.adapter_name}: {result.error_message}"
                    )

    def load_adapter(self, ref):
        assert ref.adapter_name is not None and ref.adapter_path is not None, (
            "Adapter ref must have both name and path set for loading."
        )
        assert ref.adapter_id not in self.adapters, (
            f"Adapter with ID {ref.adapter_id} is already loaded. This should have been verified before request is sent to the backend."
        )
        try:
            new_adapter = self._build_config(ref.adapter_path)
            self.validate_new_adapter(new_adapter, ref)
            self.configs[ref.adapter_id] = new_adapter
            self._load_weights(ref)
            self.refs[ref.adapter_id] = ref
            self.num_pinned += int(ref.pinned)
        except Exception as e:
            return self._make_update_result(success=False, error_message=str(e))
        return self._make_update_result(success=True)

    def unload_adapter(self, ref):
        adapter = self.configs.get(ref.adapter_id)
        stored_ref = self.refs.get(ref.adapter_id)
        assert adapter is not None and stored_ref is not None, (
            f"Adapter with ID {ref.adapter_id} is not loaded. This should have been verified before request is sent to the backend."
        )
        if ref.adapter_id not in self.adapters:
            return self._unload_streamed_adapter(stored_ref)
        try:
            self._clear_expert_on_unload(self.adapters.get(ref.adapter_id))
            del self.configs[ref.adapter_id]
            del self.adapters[ref.adapter_id]
            del self.refs[ref.adapter_id]
            self.num_pinned -= int(stored_ref.pinned)
        except Exception as e:
            return self._make_update_result(success=False, error_message=str(e))
        return self._make_update_result(success=True)

    def stage_adapter(self, named_tensors, config, name, version, adapter_id=None):
        """Fill the staging slot (lock-free). Reuses _load_weights/expert fills
        with slot_idx=staging_idx via mem_pool.stage()."""
        self._stage_fill(named_tensors, config, name, version)
        # Identity-boot deployments (e.g. orbit) start with empty refs; the
        # adapter name first arrives with the stage request. Register it so
        # activate_adapter's _bump_ref_version can find and bump it, and (for
        # multi-slot OFT) so per-request /generate routing resolves. No-op once
        # registered (subsequent syncs of the same adapter reuse the ref).
        if not any(ref.adapter_name == name for ref in self.refs.values()):
            ref = self._make_streamed_ref(name, version, adapter_id, config)
            # setdefault, not assignment: OFT's _make_streamed_ref already stored
            # the ref via register_streamed_adapter (which also sets the routing);
            # LoRA's did not, so this registers the fresh-uuid ref for it. Avoids
            # double-registering the OFT ref.
            self.refs.setdefault(ref.adapter_id, ref)

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

    def validate_batch(self, adapter_ids):
        """
        Validate if the OFT IDs in the batch can be loaded into the current OFT memory pool.
        """
        if len(adapter_ids) > self.max_adapters_per_batch:
            return False

        # skip pinned OFT check if no pinned OFT adapters are loaded.
        if self.num_pinned == 0:
            return True

        # counting the number of pinned OFT adapters in the batch.
        pinned_ofts_in_batch = 0
        for adapter_id in adapter_ids:
            if adapter_id is not None:
                oft_ref = self.refs.get(adapter_id)
                assert (
                    oft_ref is not None
                ), f"adapter ID {adapter_id} not found in refs."
                pinned_ofts_in_batch += int(oft_ref.pinned)

        assert pinned_ofts_in_batch <= self.num_pinned, (
            f"Number of pinned adapters in the batch ({pinned_ofts_in_batch}) exceeds the total number of pinned adapters "
            f"({self.num_pinned}). This indicates a bug in the adapter loading logic."
        )

        required_slots = len(adapter_ids) - pinned_ofts_in_batch
        mem_pool_vacancy = self.memory_pool.max_adapters_per_batch - self.num_pinned

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
        assert len(cur_uids) <= self.max_adapters_per_batch
        self._prepare_mem_pool_batch(cur_uids)

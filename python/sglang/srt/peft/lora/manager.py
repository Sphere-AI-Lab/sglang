# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Single-active LoRAManager (own implementation).

Makes LoRA a working method on top of ``peft.base.manager.AdapterManager``:
fills the generic load/unload hooks and wires the single loaded adapter's
A/B tensors directly into the L2 single-active layers (``peft/lora/layers.py``).

SINGLE-ACTIVE MVP:
  * One loaded adapter, applied to the WHOLE batch. No per-request routing,
    no ``weight_indices``, no double-buffer.
  * ``_set_module_info`` reads the active adapter's weights DIRECTLY from
    ``self.adapters[uid].layers[layer_id].weights`` and pushes them into the
    layer via ``set_lora_info``, which ``copy_``s into the layer's dense
    A/B attributes -- VIEWS into ``LoRAMemoryPool``'s single slot, bound once
    by ``_bind_dense_lora_views``. Afterwards, when orbit streams a fresh
    adapter, ``apply_streamed_update`` copies the new A/B *in place* into that
    same slot via ``LoRAMemoryPool._fill_slot`` (``copy_``, CUDA-graph-safe)
    -- still single-active, still one buffer slot.
  * Linear attn+MLP only -- no embed/lm_head, no expert-LoRA.
  Multi-adapter serving (buffer-backed mem pool, per-request routing) is
  deferred to the later integration phase.

The base ``AdapterManager.init_state`` template was never extracted (it
stayed OFT-flavored in ``OFTManager.init_state``), so this file defines its
own ``init_state`` calling the LoRA-flavored steps in the same order OFT's
does: ``init_adapters`` -> ``init_lora_shapes`` -> ``init_lora_modules`` ->
``init_memory_pool`` -> ``update_info``.
"""

import logging
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.peft.base.manager import AdapterManager
from sglang.srt.peft.lora.layers import (
    BaseLayerWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    get_lora_layer,
)
from sglang.srt.peft.lora.lora import LoRAAdapter
from sglang.srt.peft.lora.lora_config import LoRAConfig
from sglang.srt.peft.lora.lora_registry import LoRARef
from sglang.srt.peft.lora.mem_pool import (
    DENSE_SUB_NAMES,
    LoRAMemoryPool,
)
from sglang.srt.peft.lora.moe_backend import (
    SingleActiveMoEBackend,
    build_single_active_batch_info,
)
from sglang.srt.peft.oft.utils import (
    get_hf_config_attr,
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


def fill_moe_lora_buffers(
    gate_up_A: torch.Tensor,
    gate_up_B: torch.Tensor,
    down_A: torch.Tensor,
    down_B: torch.Tensor,
    expert_weights: Dict[int, Dict[str, torch.Tensor]],
    rank: int,
    scaling: float,
    moe,
) -> None:
    """Fill the 4 upstream-v0.5.14-layout MoE-LoRA buffers (buffer_id=0,
    `mem_pool._declare_expert_groups`) from a per-expert adapter dict, per
    `phase3-integration-spec.md` §2's "Concrete index map for OUR fork"
    (single-active: `max_lora_rank == rank`, so there is no padding tail).

    A missing expert (a key absent from `expert_weights`) or a missing
    per-expert projection leaves that slice zero (the buffers are
    preallocated via `torch.zeros`). Scaling is baked into B only --
    upstream's `fused_moe_lora` kernel has no scaling arg -- via `.mul_`
    after the copy (avoids an allocating `src * scaling`); A is never
    scaled.

    This function itself never zeroes anything -- it is a pure per-expert
    scatter that only touches the slices explicitly present in
    `expert_weights`. The "leaves that slice zero" behavior above holds when
    the caller zeroed the buffers first (the disk-load full-replace path,
    `_set_expert_lora`); `apply_streamed_expert_lora` reuses this same
    function WITHOUT zeroing first, so an absent expert/projection there
    keeps its prior value instead (scatter-only sync).

    `expert_weights`: {expert_id: {"gate_proj.lora_A": [rank,H],
    "gate_proj.lora_B": [I,rank], "up_proj.lora_A": [rank,H],
    "up_proj.lora_B": [I,rank], "down_proj.lora_A": [rank,I],
    "down_proj.lora_B": [H,rank]}}.
    """
    R = rank
    # inter_size is the buffer's MoE-TP-LOCAL intermediate (buffers are alloced
    # with moe_intermediate_size = intermediate_size_per_partition). orbit sends
    # the FULL adapter, so the intermediate-sharded projections (gate/up lora_B
    # out=I, down lora_A in=I) are sliced to this rank's [i0:i1] window; the
    # hidden-sized ones (gate/up lora_A in=H, down lora_B out=H) copy full.
    # No-op when moe_tp_size == 1 (i0=0, i1=inter_size=full).
    inter_size = gate_up_B.shape[2] // 2
    num_local = moe.num_local_experts
    i0 = moe.moe_tp_rank * inter_size
    i1 = i0 + inter_size

    for e, projs in expert_weights.items():
        # EP: map the incoming GLOBAL expert id to this rank's LOCAL slot; skip
        # experts owned by another EP rank (mirrors OFT oft_manager). At ep=1,
        # local_id == e and nothing is skipped.
        local_id = moe._map_global_expert_id_to_local_expert_id(e)
        if local_id < 0 or local_id >= num_local:
            continue

        gate_A = projs.get("gate_proj.lora_A")
        if gate_A is not None:
            gate_up_A[0, local_id, 0:R, :].copy_(gate_A)
        up_A = projs.get("up_proj.lora_A")
        if up_A is not None:
            gate_up_A[0, local_id, R : 2 * R, :].copy_(up_A)

        gate_B = projs.get("gate_proj.lora_B")
        if gate_B is not None:
            view = gate_up_B[0, local_id, 0:inter_size, :R]
            view.copy_(gate_B[i0:i1, :])
            view.mul_(scaling)
        up_B = projs.get("up_proj.lora_B")
        if up_B is not None:
            view = gate_up_B[0, local_id, inter_size : 2 * inter_size, :R]
            view.copy_(up_B[i0:i1, :])
            view.mul_(scaling)

        down_A_src = projs.get("down_proj.lora_A")
        if down_A_src is not None:
            down_A[0, local_id, 0:R, :].copy_(down_A_src[:, i0:i1])
        down_B_src = projs.get("down_proj.lora_B")
        if down_B_src is not None:
            view = down_B[0, local_id, :, :R]
            view.copy_(down_B_src)
            view.mul_(scaling)


class LoRAManager(AdapterManager):
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        tp_size: int = 1,
        tp_rank: int = 0,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
        eviction_policy: str = "lru",
        memory_saver_adapter=None,
        memory_saver_cpu_backup: bool = False,
        peft_double_buffer: bool = False,
    ):
        self.base_model: torch.nn.Module = base_model
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        # SINGLE-ACTIVE MVP: exactly one adapter is ever active, regardless
        # of `max_loras_per_batch` (kept for structural/logging parity with
        # OFTManager; multi-adapter batching is a later integration phase).
        self.max_adapters_per_batch = 1
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_added_tokens_size: Optional[int] = None
        self.eviction_policy = eviction_policy
        self.memory_saver_adapter = memory_saver_adapter
        self.memory_saver_cpu_backup = memory_saver_cpu_backup
        # Double-buffer sizing signal (PEFTArgs.peft_double_buffer), threaded
        # into LoRAMemoryPool below so it reserves a staging slot iff set.
        # Named peft_double_buffer to match OFTManager (attribute symmetry).
        self.peft_double_buffer: bool = peft_double_buffer

        # Initialize mutable internal state of the LoRAManager.
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )

    # ------------------------------------------------------------------ #
    #  AdapterManager hooks
    # ------------------------------------------------------------------ #

    def _update_output_cls(self):
        from sglang.srt.managers.io_struct import LoRAUpdateOutput

        return LoRAUpdateOutput

    def _build_config(self, path):
        return LoRAConfig(path)

    def _load_weights(self, ref):
        lora_adapter = LoRAAdapter(
            ref.adapter_id,
            self.configs[ref.adapter_id],
            self.base_hf_config,
            self.load_config,
        )
        lora_adapter.initialize_weights()
        self.adapters[ref.adapter_id] = lora_adapter
        self._set_expert_lora(lora_adapter)

    def validate_new_adapter(self, lora_config: LoRAConfig, lora_ref: LoRARef):
        """Validate if an adapter can be loaded into the current LoRA memory
        pool and raise if it is incompatible. Mirrors OFTManager's dup-name/
        path + can_support + pinned-starvation checks (generic member names).
        """
        for existing_ref in self.refs.values():
            if lora_ref.adapter_name == existing_ref.adapter_name:
                raise ValueError(
                    f"Failed to load LoRA adapter {lora_ref.adapter_name} because it is already loaded"
                )

            if lora_ref.adapter_path == existing_ref.adapter_path:
                logger.warning(
                    f"{lora_ref.adapter_path} is already loaded with name: {existing_ref.adapter_name}, "
                    f"but another copy is being loaded with name: {lora_ref.adapter_name}"
                )

        memory_pool = getattr(self, "memory_pool", None)
        incompatible = memory_pool and not memory_pool.can_support(lora_config)
        if incompatible:
            raise ValueError(
                f"LoRA adapter {lora_ref.adapter_name} with rank {lora_config.r} is incompatible with the "
                "current LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within "
                "the configured `--max-lora-rank` and that the target modules are included in "
                "`--lora-target-modules`."
            )

        # Ensure pinned adapters does not exceed maximal limit or cause
        # starvation (single-active MVP: the pool has exactly one slot, so
        # pinning it would starve the base model / any other adapter).
        if lora_ref.pinned and self.num_pinned >= self.max_adapters_per_batch - 1:
            raise ValueError(
                f"Failed to load LoRA adapter {lora_ref.adapter_name} as a pinned adapter. It is not allowed "
                "to pin all slots in the LoRA memory pool to avoid starvation for unpinned adapters and base "
                "models. Please increase your `--max-loras-per-batch` or load it as unpinned LoRA adapters."
            )

    def _clear_expert_on_unload(self, adapter):
        pass  # MVP: no expert-LoRA support.

    def _unload_streamed_adapter(self, ref):
        raise NotImplementedError(
            "Streamed (direct-to-GPU) LoRA adapter unload is not supported in the single-active MVP."
        )

    def _get_adapter_layer(self, module):
        if isinstance(module, FusedMoE):
            return get_lora_layer(module, SingleActiveMoEBackend(self.device))
        return get_lora_layer(module)

    def _update_embedding_info(self):
        pass  # MVP: no embed_tokens/lm_head LoRA.

    def _prepare_mem_pool_batch(self, cur_uids):
        # MVP: single-active layers already read the adapter directly (see
        # _set_module_info); there is no buffer to load per-batch.
        pass

    # ------------------------------------------------------------------ #
    #  The crux: per-sub-projection A/B mapping (single-active, no buffer)
    # ------------------------------------------------------------------ #

    def _active_adapter(self) -> Optional[LoRAAdapter]:
        """MVP single-active: whichever adapter is currently loaded, if any."""
        return next(iter(self.adapters.values()), None)

    @staticmethod
    def _find_sub_proj_weight(
        weights: Dict[str, torch.Tensor], candidate_name: str, lora_type: str
    ) -> Optional[torch.Tensor]:
        """Find the checkpoint tensor whose key matches `candidate_name`'s
        lora_{A,B} weight.

        `candidate_name` is derived from the wrapped module's own full name
        in the base model (e.g. "...self_attn.q_proj"); matched by substring
        containment against each checkpoint key, since checkpoint keys carry
        an outer PEFT/model-wrapper prefix the base model's own module names
        don't have (same substring-containment pattern as
        ``oft.utils.get_target_module_name``). MVP: expert-LoRA tensors
        (keys containing "experts.") are skipped -- L3's loader does not
        route them into a separate expert_weights dict, so they could
        otherwise appear here under a name that happens to contain the
        candidate substring.
        """
        suffix = f"lora_{lora_type}"
        for name, tensor in weights.items():
            if "experts." in name:
                continue
            if candidate_name in name and suffix in name:
                return tensor
        return None

    def _to_dev(self, t):
        # The base model is on GPU when update_info() runs, but the loaded
        # adapter tensors (L4) are still CPU; move them to the base model's
        # device/dtype so forward does x(cuda) @ A(cuda).T rather than crashing.
        return t.to(device=self.device, dtype=self.dtype)

    def _set_module_info(self, module, target_module, layer_id):
        active = self._active_adapter()
        if active is None:
            # No adapter loaded yet (e.g. server started with only
            # --lora-target-modules / --max-lora-rank); leave the module in
            # base-only mode.
            return
        args = self._resolve_module_args(
            module,
            target_module,
            layer_id,
            active.layers[layer_id].weights,
            active.scaling,
        )
        if args is not None:
            module.set_lora_info(*args)

    def _bind_dense_lora_views(self):
        """Hand out the pool's per-sub "A:<sub>"/"B:<sub>" buffer-group views
        onto every dense wrapper's own A/B attribute name(s) (`DENSE_SUB_NAMES`
        gives the sub order per fused target), ONCE, right after
        `init_memory_pool()` creates the pool and BEFORE `update_info()` (disk
        adapter) or `_init_identity_dense_lora_for_cuda_graph()` (identity
        boot) ever call `set_lora_info` -- see the call in `init_state()`.
        `set_lora_info`/`update_lora_info` then only ever `copy_` into these --
        never reassign -- so `apply_lora` keeps reading the same storage under
        CUDA graph.

        Unconditional (not gated on whether a disk adapter is active), unlike
        a separate standalone method call would allow -- so both call sites
        above always find self.A/self.B (etc.) already bound, regardless of
        which one runs first. This avoids the dead-pool-memory pattern the
        OFT-expert migration hit (identity-init's short-circuit skipped
        rebinding an already-loaded module, leaving the disk-loaded raw
        tensor as the module attribute and the pool's group unused -- see the
        WATCH note in the task brief); dense LoRA's only populate call site is
        set_lora_info, so binding unconditionally here has no such gap.
        """
        pool = self.memory_pool
        for layer_id, layer_modules in enumerate(self.adapter_modules):
            for module_name, module in layer_modules.items():
                target_module = get_target_module_name(
                    module_name, pool.target_modules
                )
                if isinstance(module, QKVParallelLinearWithLoRA):
                    module.A_q = pool.active_view("A:q", layer_id)
                    module.B_q = pool.active_view("B:q", layer_id)
                    module.A_k = pool.active_view("A:k", layer_id)
                    module.B_k = pool.active_view("B:k", layer_id)
                    module.A_v = pool.active_view("A:v", layer_id)
                    module.B_v = pool.active_view("B:v", layer_id)
                elif isinstance(module, MergedColumnParallelLinearWithLoRA):
                    module.A_gate = pool.active_view("A:gate", layer_id)
                    module.B_gate = pool.active_view("B:gate", layer_id)
                    module.A_up = pool.active_view("A:up", layer_id)
                    module.B_up = pool.active_view("B:up", layer_id)
                elif isinstance(module, ReplicatedLinearWithLoRA):
                    module.A_q_a = pool.active_view("A:q_a", layer_id)
                    module.B_q_a = pool.active_view("B:q_a", layer_id)
                    module.A_kv_a = pool.active_view("A:kv_a", layer_id)
                    module.B_kv_a = pool.active_view("B:kv_a", layer_id)
                else:
                    (sub,) = DENSE_SUB_NAMES[target_module]
                    module.A = pool.active_view(f"A:{sub}", layer_id)
                    module.B = pool.active_view(f"B:{sub}", layer_id)

    def _resolve_module_args(self, module, target_module, layer_id, weights, scaling):
        """Compute the positional set_lora_info/update_lora_info args for one
        wrapped module from `weights` (a per-layer checkpoint-name -> tensor
        dict), or None to leave the module base-only. Extracted verbatim from
        the former _set_module_info body so numerics are unchanged.
        """
        full_name = module.full_module_name
        assert full_name.endswith(target_module), (
            f"LoRA module full name {full_name!r} does not end with its "
            f"resolved target module {target_module!r}"
        )
        name_prefix = full_name[: -len(target_module)]

        if target_module == "qkv_proj":
            q_name = name_prefix + "q_proj"
            k_name = name_prefix + "k_proj"
            v_name = name_prefix + "v_proj"

            A_q = self._find_sub_proj_weight(weights, q_name, "A")
            B_q = self._find_sub_proj_weight(weights, q_name, "B")
            assert A_q is not None and B_q is not None, (
                f"LoRA adapter is missing q_proj lora_A/lora_B for {full_name!r} "
                f"(layer {layer_id}); every LoRA adapter targeting qkv_proj must include q_proj."
            )
            A_k = self._find_sub_proj_weight(weights, k_name, "A")
            B_k = self._find_sub_proj_weight(weights, k_name, "B")
            A_v = self._find_sub_proj_weight(weights, v_name, "A")
            B_v = self._find_sub_proj_weight(weights, v_name, "B")

            # lora_A's shape only depends on rank/in_features, which is
            # shared by q/k/v, so a missing A can always be zero-filled from
            # q's shape. lora_B's shape depends on the sub-projection's own
            # output dim (q's differs from k/v's under GQA), so a missing B
            # must borrow its k/v SIBLING's shape, not q's. Mirrors
            # srt/lora's normalize_qkv_proj zero-fill (lora.py:160+),
            # generalized to treat k and v symmetrically as "may be missing"
            # (srt/lora only special-cases missing k).
            if A_k is None:
                A_k = torch.zeros_like(A_q)
            if A_v is None:
                A_v = torch.zeros_like(A_q)
            if B_k is None:
                assert B_v is not None, (
                    f"LoRA adapter has neither k_proj nor v_proj lora_B for {full_name!r} "
                    f"(layer {layer_id}); cannot infer the kv output shape to zero-fill."
                )
                B_k = torch.zeros_like(B_v)
            if B_v is None:
                assert B_k is not None, (
                    f"LoRA adapter has neither k_proj nor v_proj lora_B for {full_name!r} "
                    f"(layer {layer_id}); cannot infer the kv output shape to zero-fill."
                )
                B_v = torch.zeros_like(B_k)

            # TP>1: orbit sends the FULL adapter (weight-sync all-gathers TP), so
            # slice each per-sub-proj lora_B to this rank's output shard here (A is
            # replicated). At tp=1 this branch is skipped. Shared disk+streamed path.
            if module.base_layer.tp_size > 1:
                tp_rank = module.get_local_tp_rank()
                B_q, B_k, B_v = module.slice_streamed_b_weights(B_q, B_k, B_v, tp_rank)

            return (
                self._to_dev(A_q),
                self._to_dev(B_q),
                self._to_dev(A_k),
                self._to_dev(B_k),
                self._to_dev(A_v),
                self._to_dev(B_v),
                scaling,
            )

        if target_module == "gate_up_proj":
            gate_name = name_prefix + "gate_proj"
            up_name = name_prefix + "up_proj"

            A_gate = self._find_sub_proj_weight(weights, gate_name, "A")
            B_gate = self._find_sub_proj_weight(weights, gate_name, "B")
            assert A_gate is not None and B_gate is not None, (
                f"LoRA adapter is missing gate_proj lora_A/lora_B for {full_name!r} "
                f"(layer {layer_id}); every LoRA adapter targeting gate_up_proj must include gate_proj."
            )
            A_up = self._find_sub_proj_weight(weights, up_name, "A")
            B_up = self._find_sub_proj_weight(weights, up_name, "B")
            # gate/up always share the same output dim (no GQA-like
            # asymmetry), so zero-filling a missing up from gate's own shape
            # is always correct.
            if A_up is None:
                A_up = torch.zeros_like(A_gate)
            if B_up is None:
                B_up = torch.zeros_like(B_gate)

            # TP>1: slice each per-sub-proj lora_B to this rank's output shard
            # (A replicated). At tp=1 skipped. See qkv_proj branch above.
            if module.base_layer.tp_size > 1:
                tp_rank = module.get_local_tp_rank()
                B_gate, B_up = module.slice_streamed_b_weights(B_gate, B_up, tp_rank)

            return (
                self._to_dev(A_gate),
                self._to_dev(B_gate),
                self._to_dev(A_up),
                self._to_dev(B_up),
                scaling,
            )

        if target_module == "fused_qkv_a_proj_with_mqa":
            # MLA fused down-proj (ReplicatedLinearWithLoRA): q_a and kv_a
            # have UNEQUAL output sizes (q_lora_rank vs kv_lora_rank+rope),
            # unlike gate_up_proj's equal halves.
            q_a_name = name_prefix + "q_a_proj"
            kv_a_name = name_prefix + "kv_a_proj_with_mqa"

            A_q_a = self._find_sub_proj_weight(weights, q_a_name, "A")
            B_q_a = self._find_sub_proj_weight(weights, q_a_name, "B")
            assert A_q_a is not None and B_q_a is not None, (
                f"LoRA adapter is missing q_a_proj lora_A/lora_B for {full_name!r} "
                f"(layer {layer_id}); every LoRA adapter targeting "
                f"fused_qkv_a_proj_with_mqa must include q_a_proj."
            )
            A_kv_a = self._find_sub_proj_weight(weights, kv_a_name, "A")
            B_kv_a = self._find_sub_proj_weight(weights, kv_a_name, "B")
            # A_kv_a shares q_a's own shape (rank/in_features, same for both
            # sub-projections), so it can be zero-filled from A_q_a like
            # gate_up's A_up. B_kv_a's output dim is kv_a's OWN (UNEQUAL) size
            # though -- there's no kv_a sibling tensor in the adapter to
            # borrow it from (unlike qkv_proj's k/v), so pull it from the
            # base layer's own geometry instead (the same
            # `base_layer.output_sizes` source ReplicatedLinearWithLoRA's own
            # `dense_lora_shapes`/`bind_zero_lora` use, layers.py).
            if A_kv_a is None:
                A_kv_a = torch.zeros_like(A_q_a)
            if B_kv_a is None:
                kv_a_out = module.base_layer.output_sizes[1]
                B_kv_a = torch.zeros(kv_a_out, A_q_a.shape[0], dtype=B_q_a.dtype)

            # NO TP slice: ReplicatedLinear is replicated on every rank (no
            # tensor-parallel output sharding), unlike gate_up_proj's
            # ColumnParallel sharding above -- skip slice_streamed_b_weights
            # entirely, regardless of tp_size.

            return (
                self._to_dev(A_q_a),
                self._to_dev(B_q_a),
                self._to_dev(A_kv_a),
                self._to_dev(B_kv_a),
                scaling,
            )

        # o_proj / down_proj: single Column/Row linear, own lora_A/lora_B.
        A = self._find_sub_proj_weight(weights, full_name, "A")
        B = self._find_sub_proj_weight(weights, full_name, "B")
        if A is None or B is None:
            # This adapter doesn't target this module; leave it base-only.
            return None
        # TP>1: single Column/Row linear. The per-tensor slice helpers do the
        # right thing by wrapper type -- Row: slice_lora_a shards A on the input
        # partition, slice_lora_b is identity; plain Column: the reverse. At tp=1
        # both are identity. Shared disk+streamed path.
        if module.base_layer.tp_size > 1:
            tp_rank = module.get_local_tp_rank()
            A = module.slice_lora_a_weights(A, tp_rank)
            B = module.slice_lora_b_weights(B, tp_rank)
        return (self._to_dev(A), self._to_dev(B), scaling)

    def _set_expert_lora(self, adapter):
        """Populate FusedMoE expert LoRA buffers from a disk-loaded adapter.
        No-op on dense models, or when the adapter's target modules don't
        cover MoE experts (no FusedMoE wrapped by `init_lora_modules`).

        For each `FusedMoEWithLoRA` wrapper discovered in
        `self.moe_lora_modules`: fetch the pool's "A_expert"/"B_expert"/
        "A_expert_down"/"B_expert_down" active-slot views ONCE per layer
        (cached in `self._moe_lora_buffers` -- stable pointers, never
        reallocated; Task 5 moved the allocation itself into
        `LoRAMemoryPool._declare_expert_groups`), zero + refill them from the
        adapter's `expert_weights` via `fill_moe_lora_buffers` (full replace:
        an expert/projection absent from this adapter becomes zero, not stale
        from a previously-loaded adapter -- contrast with the streamed
        scatter-only sync in `apply_streamed_expert_lora`), then push them
        into the wrapper via `set_lora_info` (per
        `.superpowers/sdd/phase3-integration-spec.md` §2/§6 step 3).
        """
        moe_lora_modules = getattr(self, "moe_lora_modules", None)
        if not moe_lora_modules:
            return

        rank = adapter.config.r
        scaling = adapter.scaling
        for layer_id, layer_modules in enumerate(moe_lora_modules):
            for wrapper in layer_modules.values():
                buffers = self._moe_lora_buffers.get(layer_id)
                if buffers is None:
                    pool = self.memory_pool
                    buffers = (
                        pool.active_view("A_expert", layer_id),
                        pool.active_view("B_expert", layer_id),
                        pool.active_view("A_expert_down", layer_id),
                        pool.active_view("B_expert_down", layer_id),
                    )
                    self._moe_lora_buffers[layer_id] = buffers
                gate_up_A, gate_up_B, down_A, down_B = buffers
                for buf in buffers:
                    buf.zero_()
                fill_moe_lora_buffers(
                    gate_up_A,
                    gate_up_B,
                    down_A,
                    down_B,
                    adapter.layers[layer_id].expert_weights,
                    rank,
                    scaling,
                    wrapper.base_layer,
                )
                wrapper.set_lora_info(gate_up_A, gate_up_B, down_A, down_B)

    def _init_identity_moe_lora_for_cuda_graph(self):
        """Bind ZERO MoE-LoRA buffers on every FusedMoEWithLoRA wrapper that has
        no adapter loaded, so (a) the base forward runs a zero delta instead of
        reading an unset ``down_lora_a_weights`` (identity-boot crash in
        ``FusedMoEWithLoRA._get_lora_info``), and (b) CUDA-graph capture pins the
        same buffer objects a later streamed sync fills IN PLACE.

        Mirrors the OFT manager's ``_init_identity_expert_oft_for_cuda_graph``:
        called last in ``init_state``, and SKIPS any wrapper an initial disk
        adapter already established (``set_lora=True``) so it never clobbers
        loaded weights. This is what lets orbit boot LoRA identity (no
        ``peft_paths`` in normal RL) and stream the first adapter, symmetric
        to how orbit boots OFT.
        """
        moe_lora_modules = getattr(self, "moe_lora_modules", None)
        if not moe_lora_modules:
            return
        for layer_id, layer_modules in enumerate(moe_lora_modules):
            for wrapper in layer_modules.values():
                if getattr(wrapper, "set_lora", False):
                    continue  # an initial adapter already bound this wrapper
                buffers = self._moe_lora_buffers.get(layer_id)
                if buffers is None:
                    pool = self.memory_pool
                    buffers = (
                        pool.active_view("A_expert", layer_id),
                        pool.active_view("B_expert", layer_id),
                        pool.active_view("A_expert_down", layer_id),
                        pool.active_view("B_expert_down", layer_id),
                    )
                    self._moe_lora_buffers[layer_id] = buffers
                for buf in buffers:
                    buf.zero_()
                wrapper.set_lora_info(*buffers)

    def _init_identity_dense_lora_for_cuda_graph(self):
        """Bind ZERO dense-LoRA buffers on every dense wrapper that has no adapter
        loaded, so (a) CUDA-graph capture records the LoRA path (``set_lora=True``,
        pinned zero A/B) instead of the base-only branch its ``forward`` would
        otherwise take, and (b) a later streamed sync fills those SAME pinned
        buffers in place (``update_lora_info`` ``copy_``). Without this, an
        identity boot captures the base path and the first streamed adapter is
        silently ignored on replay.

        The dense analogue of ``_init_identity_moe_lora_for_cuda_graph``: called
        last in ``init_state`` and SKIPS any wrapper an initial disk adapter
        already established (``set_lora=True``) so it never clobbers loaded
        weights. This is what lets orbit boot LoRA identity (no
        ``peft_paths``) and stream the first adapter under CUDA graph.
        """
        adapter_modules = getattr(self, "adapter_modules", None)
        if not adapter_modules:
            return
        for layer_modules in adapter_modules:
            for wrapper in layer_modules.values():
                if getattr(wrapper, "set_lora", False):
                    continue  # an initial adapter already bound this wrapper
                wrapper.bind_zero_lora(self.max_lora_rank, self.dtype, self.device)

    def apply_streamed_expert_lora(self, expert_tensors, scaling, slot_idx=None):
        """Streamed CUDA IPC expert sync: scatter each present expert's new
        A/B into the already-installed upstream-v0.5.14-layout MoE-LoRA
        buffers (`self._moe_lora_buffers`, §2 index map) IN PLACE, via the
        SAME per-expert slice/scaling logic as the disk-load path
        (`fill_moe_lora_buffers`) -- re-baking `scaling` into B on each copy
        since the buffers carry no separate scaling field
        (`.superpowers/sdd/phase3-integration-spec.md` §7). SCATTER
        semantics — only experts/projections present in `expert_tensors` are
        overwritten; absent ones are left untouched (matching the dense
        path's constant-target-set invariant and OFT's
        apply_streamed_expert_oft), so a sync chunked across multiple calls
        does not wipe experts delivered in other chunks.

        INVARIANT: because of this scatter semantics, a streamed sync must carry
        the FULL expert set that the initial (disk-loaded) adapter established --
        a partial/shrinking payload silently leaves the ABSENT experts at their
        prior values rather than clearing them. Contrast with the disk-load path
        (`_set_expert_lora`), which zeros + full-replaces the buffers each time
        (an expert absent from the checkpoint becomes zero, not stale).
        expert_tensors: {layer_id: {expert_id: {"{proj}.lora_{A|B}": tensor}}}.

        ``slot_idx=None`` (default) preserves today's exact behavior byte-
        for-byte: resolve buffers from the ``self._moe_lora_buffers`` cache
        (ACTIVE-slot views, never re-touching ``self.memory_pool``). Passing
        an explicit ``slot_idx`` (e.g. the pool's ``staging_idx`` for
        double-buffer ``stage()``) bypasses that cache -- which only ever
        holds ACTIVE views -- and resolves fresh views straight from the
        pool's group registry at that slot instead.
        """
        moe_lora_modules = getattr(self, "moe_lora_modules", None)
        if not moe_lora_modules:
            return

        # The buffers hold no notion of scaling themselves (it's baked into
        # B on write), but the manager still tracks the initial adapter's
        # scaling on `self.adapters` -- mirror the old code's scaling-change
        # guard against that, when available.
        active = self._active_adapter()
        if active is not None:
            assert active.scaling == scaling, (
                f"streamed expert LoRA changed scaling {active.scaling} -> "
                f"{scaling}; rank/alpha must be constant across syncs under CUDA graph"
            )

        for layer_id, ew in expert_tensors.items():
            if not ew or layer_id >= len(moe_lora_modules):
                continue
            layer_modules = moe_lora_modules[layer_id]
            if not layer_modules:
                continue
            if slot_idx is None:
                buffers = self._moe_lora_buffers.get(layer_id)
                assert buffers is not None, (
                    f"streamed expert LoRA sync for layer {layer_id} before an "
                    "initial adapter established the FusedMoE buffers"
                )
            else:
                pool = self.memory_pool
                buffers = (
                    pool.slot("A_expert", layer_id, slot_idx),
                    pool.slot("B_expert", layer_id, slot_idx),
                    pool.slot("A_expert_down", layer_id, slot_idx),
                    pool.slot("B_expert_down", layer_id, slot_idx),
                )
            gate_up_A, gate_up_B, down_A, down_B = buffers
            for wrapper in layer_modules.values():
                assert wrapper.set_lora, (
                    f"streamed expert LoRA sync for layer {layer_id} before an "
                    "initial adapter established the FusedMoE buffers"
                )
                # Scatter this layer's streamed experts into the shared buffers,
                # sliced per this FusedMoE's EP/TP (wrapper.base_layer) -- same as
                # the disk-load path. Scatter-only, so a hypothetical >1-wrapper
                # layer (one FusedMoE per layer in practice) is idempotent.
                fill_moe_lora_buffers(
                    gate_up_A,
                    gate_up_B,
                    down_A,
                    down_B,
                    ew,
                    self.max_lora_rank,
                    scaling,
                    wrapper.base_layer,
                )

    def _build_dense_lora_payload(
        self, named_tensors: List[Tuple[str, torch.Tensor]], scaling: float
    ) -> Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor, float]]:
        """Build the (sub, layer_id) -> (A, B, scaling) payload `_fill_slot`
        expects from raw (checkpoint_name, gpu_tensor) pairs, via the SAME
        per-module `_resolve_module_args` that feeds `update_lora_info`
        directly on the disk-load path -- `_resolve_module_args` always
        returns (A0, B0, A1, B1, ..., scaling) in DENSE_SUB_NAMES order.
        Extracted from `apply_streamed_update` (verbatim) so `_stage_fill`
        can reuse the exact same payload-building logic for STAGING.
        """
        # Route (checkpoint_name, gpu_tensor) into per-layer dicts, mirroring
        # LoRAAdapter._process_weight's get_layer_id routing but staying on GPU.
        layer_weights = [
            dict()
            for _ in range(get_hf_config_attr(self.base_hf_config, "num_hidden_layers"))
        ]
        for name, tensor in named_tensors:
            layer_id = get_layer_id(name)
            if layer_id is not None:
                layer_weights[layer_id][name] = tensor

        dense_payload: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor, float]] = {}
        for layer_id, layer_modules in enumerate(self.adapter_modules):
            for module_name, module in layer_modules.items():
                target_module = get_target_module_name(
                    module_name, self.memory_pool.target_modules
                )
                args = self._resolve_module_args(
                    module, target_module, layer_id, layer_weights[layer_id], scaling
                )
                if args is None:
                    continue
                *ab_flat, _scaling = args
                for i, sub in enumerate(DENSE_SUB_NAMES[target_module]):
                    dense_payload[(sub, layer_id)] = (
                        ab_flat[2 * i],
                        ab_flat[2 * i + 1],
                        scaling,
                    )
        return dense_payload

    def apply_streamed_update(
        self, named_tensors: List[Tuple[str, torch.Tensor]], scaling: float
    ) -> None:
        """In-place streamed weight-sync: copy_ new per-layer A/B into the
        pool's ACTIVE slot (routed through `LoRAMemoryPool._fill_slot`, the
        SAME storage the already-bound single-active layers' self.A/self.B
        (etc.) view). Keeps tensors on GPU (no .cpu() bounce) and never
        reassigns layer tensors (CUDA-graph safe).

        Constant-target-set invariant: a module bound at init but ABSENT from
        this payload is left untouched (its previous weights are NOT zeroed),
        so the set of streamed target modules must stay constant across syncs.
        """
        dense_payload = self._build_dense_lora_payload(named_tensors, scaling)
        if dense_payload:
            self.memory_pool._fill_slot(self.memory_pool.active_idx, dense_payload)

    def _stage_fill(self, named_tensors, config, name, version):
        """Per-method hook for ``AdapterManager.stage_adapter``: derive
        ``scaling`` from ``config`` (mirrors ``streamed_weight_loader.
        _scaling_from_config``, inlined here since the manager has no
        ``model_runner`` reference to fall back through), partition
        ``named_tensors`` into dense vs expert
        (``_partition_expert_lora_tensors``), then write dense into STAGING
        via ``mem_pool.stage()`` (reusing ``_build_dense_lora_payload``) and
        expert into STAGING via ``apply_streamed_expert_lora(slot_idx=
        staging_idx)`` (Step 3 above).
        """
        from sglang.srt.peft.lora.streamed_weight_loader import (
            _partition_expert_lora_tensors,
        )

        if config and "lora_alpha" in config and "r" in config:
            scaling = float(config["lora_alpha"]) / float(config["r"])
        else:
            active = self._active_adapter()
            assert active is not None, (
                "stage_adapter needs an initial adapter (or an adapter_config "
                "with lora_alpha/r) to determine scaling"
            )
            scaling = active.scaling

        expert_layer_dict, dense_named_tensors = _partition_expert_lora_tensors(
            named_tensors
        )

        # Always call stage() -- even with an empty dense payload -- so
        # _staged_version is set for an expert-only adapter too (otherwise a
        # later activate(version) would raise inactive_slot_busy).
        dense_payload = self._build_dense_lora_payload(dense_named_tensors, scaling)
        self.memory_pool.stage(version, dense_payload)

        if expert_layer_dict:
            self.apply_streamed_expert_lora(
                expert_layer_dict, scaling, slot_idx=self.memory_pool.staging_idx
            )

    def _bump_ref_version(self, name, version):
        for lora_id, ref in self.refs.items():
            if ref.adapter_name == name:
                self.refs[lora_id] = replace(ref, adapter_version=version)
                return
        raise ValueError(
            f"LoRA adapter {name!r} not found in refs; cannot bump version"
        )

    def _make_streamed_ref(self, name, version, adapter_id=None, config=None):
        # LoRA is always-active and never routes per-request, so adapter_id/config
        # are ignored (no uid_to_buffer_id registration): the fresh-uuid ref is
        # sufficient for the version bump. Unchanged from the pre-routing-fix
        # behavior; only the signature widened to match the base hook.
        from sglang.srt.peft.lora.lora_registry import LoRARef

        return LoRARef(
            adapter_name=name, adapter_path=None, pinned=False, adapter_version=version
        )

    # ------------------------------------------------------------------ #
    #  init_state (own; the base template stayed OFT-flavored, see the
    #  module docstring) + the LoRA-flavored steps it calls
    # ------------------------------------------------------------------ #

    def init_state(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        """
        Initialize the internal (mutable) state of the LoRAManager.

        When `lora_paths` is provided and not empty, it might be used for inferring LoRA shape info such as
        the target modules and max_lora_rank.
        """

        assert lora_paths or (
            max_lora_rank is not None and target_modules is not None
        ), (
            "When no initial --lora-paths is provided, you need to specify both "
            "--max-lora-rank and --lora-target-modules for LoRA initialization."
        )

        self.init_adapters(lora_paths)
        self.init_lora_shapes(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
        )
        self.init_lora_modules()
        self.init_memory_pool()
        # Backfill: `init_state`'s call order is init_adapters() ->
        # init_lora_shapes() -> init_lora_modules() -> init_memory_pool()
        # (this line), so any adapter(s) loaded from startup `--lora-paths`
        # already ran `_load_weights` -> `_set_expert_lora` TWICE too early --
        # once before `self.moe_lora_modules` existed (guarded no-op), and
        # again from within `init_lora_modules()` before `init_memory_pool()`
        # had created the pool's expert buffer-group registry
        # (`_declare_expert_groups`, Task 5) for `_set_expert_lora`'s
        # cache-miss branch to read. Re-run now that BOTH
        # `self.moe_lora_modules` AND `self.memory_pool` exist, so startup-
        # loaded adapters' expert weights are not silently dropped. No-op for
        # the common case of no adapters yet (`--max-lora-rank`/
        # `--lora-target-modules` with no `--lora-paths`); a NEWLY
        # (dynamically) loaded adapter after server startup already gets this
        # for free straight from `_load_weights` (pool already exists by then).
        # Single-active: apply ONLY the active adapter (the first-loaded, same
        # one the dense wrappers bind to via `_active_adapter()`). Looping over
        # all `self.adapters` would leave the shared MoE buffers holding the
        # LAST adapter while dense holds the first -> a silent cross-adapter mix
        # under multi-`--lora-paths` startup.
        active = self._active_adapter()
        if active is not None:
            self._set_expert_lora(active)
        # Hand out the pool's dense A/B views onto every wrapper BEFORE
        # update_info() (disk adapter) or _init_identity_dense_lora_for_cuda_
        # graph() (identity boot) below ever call set_lora_info -- see
        # _bind_dense_lora_views' docstring for why this must be unconditional.
        self._bind_dense_lora_views()
        self.update_info()
        # Identity boot (no lora_paths): bind zero MoE-LoRA + dense-LoRA buffers
        # so the base forward is a no-op AND CUDA-graph capture records the LoRA
        # path (set_lora=True) whose pinned buffers a later streamed sync fills in
        # place (orbit's normal-RL pattern; mirrors OFT). No-op with a disk
        # adapter (both guard on the wrapper's set_lora).
        self._init_identity_moe_lora_for_cuda_graph()
        self._init_identity_dense_lora_for_cuda_graph()
        # Double-buffer hardening: seed the STAGING slot from the now-neutral
        # (zero) ACTIVE slot so a partial-coverage stage can't leave garbage for
        # activate() to promote. No-op for full-coverage orbit syncs.
        if self.peft_double_buffer:
            self.memory_pool._init_staging_from_active()

    def init_lora_shapes(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
    ):
        """Infer LoRA target modules and max_lora_rank from loaded adapters if not provided."""

        self.target_modules = (
            get_normalized_target_modules(target_modules) if target_modules else set()
        )

        for lora_id, config in self.configs.items():
            # Handle PEFT shorthand strings like "all-linear" or "all".
            # These cannot be resolved to concrete module names without
            # inspecting the base model, so we require the user to specify
            # --lora-target-modules explicitly when such shorthands are used.
            if isinstance(config.target_modules, str):
                if config.target_modules in ("all-linear", "all"):
                    if target_modules is not None:
                        # CLI --lora-target-modules already provided; skip
                        # per-adapter inference for this adapter.
                        continue
                    else:
                        lora_name = self.refs[lora_id].adapter_name
                        raise ValueError(
                            f"LoRA adapter '{lora_name}' uses "
                            f"target_modules='{config.target_modules}' which cannot "
                            "be resolved automatically. Please explicitly specify "
                            "--lora-target-modules during server startup. You can "
                            "specify 'all' to enable all supported module types."
                        )
                else:
                    raise ValueError(
                        f"SGLang does not recognize target_modules="
                        f"'{config.target_modules}'. Please use a list of module "
                        "name suffixes in the adapter's PEFT config, or explicitly "
                        "specify --lora-target-modules during server startup."
                    )

            if not isinstance(config.target_modules, list):
                raise ValueError(
                    f"SGLang currently only supports inferring LoRA target modules when a list of "
                    "suffixes is provided in `target_modules` field of PEFT config. Please explicitly "
                    "specify `--lora-target-modules` during server startup. You can specify `all` to "
                    "enable all support modules types. "
                )

            adapter_target_modules = get_normalized_target_modules(
                config.target_modules
            )

            if target_modules is not None:
                # When `--lora-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    lora_name = self.refs[lora_id].adapter_name
                    raise ValueError(
                        f"LoRA adapter '{lora_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --lora-target-modules {sorted(self.target_modules)}. "
                        f"Please update --lora-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
            else:
                # Otherwise, infer target_modules from adapter configs.
                self.target_modules.update(adapter_target_modules)

        if max_lora_rank is not None:
            self.max_lora_rank = max_lora_rank
        else:
            self.max_lora_rank = max(
                [x.r for x in self.configs.values()],
                default=0,
            )

    def init_lora_modules(self):
        # Look-up table that maps (layer_index, module_name) to the
        # corresponding LoRA module.
        num_hidden_layers = get_hf_config_attr(
            self.base_hf_config, "num_hidden_layers"
        )
        self.adapter_modules: List[Dict[str, BaseLayerWithLoRA]] = [
            {} for _ in range(num_hidden_layers)
        ]

        wrapped_modules = []
        for module_name, module in self.base_model.named_modules():
            module_suffix = module_name.split(".")[-1]
            if module_suffix not in self.target_modules:
                continue

            # MVP: linear attn+MLP only. Modules without a numeric
            # transformer layer index (embed_tokens, lm_head, ...) and any
            # expert sub-module are out of scope and skipped.
            layer_id = get_layer_id(module_name)
            if layer_id is None:
                continue

            adapter_module = self.set_adapter_module(module_name, module)
            # Record the module's own full name so `_set_module_info` can
            # derive the checkpoint sub-projection lookup keys from it.
            adapter_module.full_module_name = module_name
            self.adapter_modules[layer_id][module_name] = adapter_module
            wrapped_modules.append(module_name)

        if wrapped_modules:
            logger.info(
                "Wrapped %d LoRA modules: %s",
                len(wrapped_modules),
                wrapped_modules,
            )

        # MoE-LoRA: wrap each FusedMoE with upstream's own `FusedMoEWithLoRA`
        # (Option A, `.superpowers/sdd/phase3-integration-spec.md` §5/§6),
        # via the SAME install mechanism as the dense wrappers above
        # (`set_adapter_module` -> `_get_adapter_layer` -> `replace_submodule`).
        # Recorded in `self.moe_lora_modules` rather than `self.adapter_modules`:
        # a FusedMoE's own module name ends in "experts", which doesn't match
        # any of the dense qkv_proj/gate_up_proj/o_proj/down_proj suffixes, so
        # routing it through `update_info`'s generic `get_target_module_name`
        # -> `_resolve_module_args` pipeline (built for per-sub-projection
        # dense lookup) would raise. `_set_expert_lora`/`prepare_lora_batch`
        # read `self.moe_lora_modules` directly instead.
        self.moe_lora_modules: List[Dict[str, FusedMoEWithLoRA]] = [
            {} for _ in range(num_hidden_layers)
        ]
        self._moe_lora_buffers: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

        if {"gate_up_proj", "down_proj"}.issubset(self.target_modules):
            moe_modules = self._find_fused_moe_modules()
            if moe_modules:
                name_by_id = {id(m): n for n, m in self.base_model.named_modules()}
                wrapped_moe = []
                for layer_id, moe in moe_modules.items():
                    module_name = name_by_id[id(moe)]
                    wrapper = self.set_adapter_module(module_name, moe)
                    wrapper.experts_shared_outer_loras = False
                    wrapper.lora_use_virtual_experts = False
                    self.moe_lora_modules[layer_id][module_name] = wrapper
                    wrapped_moe.append(module_name)

                if wrapped_moe:
                    logger.info(
                        "Wrapped %d MoE-LoRA modules: %s",
                        len(wrapped_moe),
                        wrapped_moe,
                    )

    def init_memory_pool(self):
        """(Re)initialize the LoRA memory pool based on the current configurations."""
        self.memory_pool = LoRAMemoryPool(
            dtype=self.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            target_modules=self.target_modules,
            max_lora_rank=self.max_lora_rank,
            eviction_policy=self.eviction_policy,
            # `self.adapter_modules`/`self.moe_lora_modules` were already
            # built by init_lora_modules() (init_state calls it before
            # init_memory_pool); the pool uses them ONLY to size the dense
            # "A:<sub>"/"B:<sub>" groups and the expert "A_expert"/"B_expert"/
            # "A_expert_down"/"B_expert_down" groups from each wrapper's own
            # base_layer geometry (mem_pool._declare_groups/_declare_expert_groups).
            adapter_modules=self.adapter_modules,
            moe_lora_modules=self.moe_lora_modules,
            device=self.device,
            memory_saver_adapter=self.memory_saver_adapter,
            memory_saver_cpu_backup=self.memory_saver_cpu_backup,
            double_buffer=self.peft_double_buffer,
        )

    def init_cuda_graph_moe_buffers(
        self,
        max_bs: int,
        disable_cuda_graph: bool,
        max_loras: int = 1,
    ) -> None:
        """Pre-capture CUDA-graph buffer alloc for the MoE-LoRA wrappers.

        Mirrors upstream `srt/lora/lora_manager.py:138-151`'s
        `LoRAManager.init_cuda_graph_moe_buffers`, which forwards
        `(max_bs, max_loras, compute_dtype, a_representative_moe_layer)` to
        `self.lora_backend.init_cuda_graph_moe_buffers(...)` ONCE (all of
        upstream's `FusedMoEWithLoRA` layers share a single backend). Our
        `_get_adapter_layer` gives each wrapped MoE its OWN
        `SingleActiveMoEBackend` instance (`moe_backend.py`), so we call it
        once PER wrapper instead -- using the wrapper itself as its own
        "representative layer" (each wrapper's own `base_layer`/`_quant_info`
        determine its own buffer sizes).

        NO-OP when CUDA graph is disabled (`disable_cuda_graph=True`, as
        under the CPU/GPU parity gates which boot with
        `disable_cuda_graph=True`) or when the model has no MoE-LoRA
        wrappers (dense-only LoRA) -- the same `any(self.moe_lora_modules)`
        guard `prepare_lora_batch` already uses.
        """
        moe_lora_modules = getattr(self, "moe_lora_modules", None)
        if disable_cuda_graph or not moe_lora_modules or not any(moe_lora_modules):
            return

        for layer_modules in moe_lora_modules:
            for wrapper in layer_modules.values():
                wrapper.lora_backend.init_cuda_graph_moe_buffers(
                    max_bs=max_bs,
                    max_loras=max_loras,
                    compute_dtype=self.dtype,
                    moe_layer=wrapper,
                )

    def prepare_lora_batch(self, forward_batch):
        # MVP: the dense single-active layers already have the active
        # adapter pushed in by update_info() at init time and apply it
        # unconditionally to the whole batch (per-request routing /
        # weight_indices are deferred to the later multi-adapter integration
        # phase) -- nothing to do for them here. The MoE-LoRA wrappers are
        # different: upstream's `FusedMoEWithLoRA` reads its routing off
        # `self.lora_backend.batch_info` fresh every forward, so push a
        # minimal single-active `batch_info` (one lora, index 0, covering
        # every token) onto each wrapped MoE's own backend.
        moe_lora_modules = getattr(self, "moe_lora_modules", None)
        # `moe_lora_modules` is a per-layer list; for a dense-only model it is a
        # list of empty dicts (truthy) -> use `any()` so we skip the per-step
        # `batch_info` allocs entirely. (`or not moe_lora_modules` guards None.)
        if not moe_lora_modules or not any(moe_lora_modules):
            return

        num_tokens = (
            forward_batch.input_ids.shape[0]
            if forward_batch.input_ids is not None
            else 0
        )
        active = self._active_adapter()
        rank = active.config.r if active is not None else self.max_lora_rank
        batch_info = build_single_active_batch_info(num_tokens, rank, self.device)
        for layer_modules in moe_lora_modules:
            for wrapper in layer_modules.values():
                wrapper.lora_backend.batch_info = batch_info

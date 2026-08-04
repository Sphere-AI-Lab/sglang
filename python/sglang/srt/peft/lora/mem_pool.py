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
"""Minimal single-active LoRA memory pool (own implementation).

Single-active MVP: the one loaded adapter's weights are read directly off
``LoRAAdapter.layers[...].weights`` by ``LoRAManager._set_module_info`` and
pushed into the L2 layers via ``set_lora_info``. This class hosts the DENSE
``A:<sub>``/``B:<sub>`` buffer groups (Task 4) on the base ``AdapterMemPool``
registry -- one group per dense sub-projection (q,k,v,o,gate,up,down); see
``DENSE_SUB_NAMES`` below. ``LoRAManager._bind_dense_lora_views`` hands out a
VIEW (``active_view``) into each group onto the matching L2 wrapper's own
``self.A``/``self.B`` (etc.) attribute, so ``apply_lora`` keeps reading the
same storage while ``set_lora_info``/``update_lora_info`` write it via
``copy_`` (CUDA-graph safe). Expert (FusedMoE) A/B buffers (Task 5) are
hosted the same way -- ``"A_expert"``/``"B_expert"``/``"A_expert_down"``/
``"B_expert_down"`` groups, one entry per FusedMoEWithLoRA-wrapped layer, in
upstream's exact 4D layout (see ``_declare_expert_groups`` below);
``LoRAManager._set_expert_lora``/``_init_identity_moe_lora_for_cuda_graph``
hand the pool's ``active_view`` tensors to ``FusedMoEWithLoRA.set_lora_info``
instead of allocating freestanding tensors.

Double-buffer (async-RL NCCL weight-sync) is a ctor-level sizing signal
(``double_buffer=True``): the pool then allocates a staging twin
(``max_adapters_per_batch=2``, active=0/staging=1) alongside the single
active slot. Off by default (``max_adapters_per_batch=1``), byte-identical
to today; the stage/activate endpoints themselves land in a later task.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.peft.base.mem_pool import AdapterMemPool
from sglang.srt.peft.lora.layers import BaseLayerWithLoRA
from sglang.srt.peft.lora.lora_config import LoRAConfig
from sglang.srt.peft.oft.utils import (
    get_normalized_target_modules,
    get_target_module_name,
)

# Fused target module -> its dense sub-projection names, in the fixed order
# each wrapper's own `dense_lora_shapes` (layers.py) returns their shapes.
# Single source of truth for both `_declare_groups` (below) and
# `LoRAManager._bind_dense_lora_views` (manager.py).
DENSE_SUB_NAMES: Dict[str, Tuple[str, ...]] = {
    "qkv_proj": ("q", "k", "v"),
    "gate_up_proj": ("gate", "up"),
    "fused_qkv_a_proj_with_mqa": ("q_a", "kv_a"),
    "o_proj": ("o",),
    "down_proj": ("down",),
}


class LoRAMemoryPool(AdapterMemPool):
    def __init__(
        self,
        dtype,
        tp_size: int,
        tp_rank: int,
        target_modules: Set[str],
        max_lora_rank: int,
        eviction_policy: str,
        adapter_modules: Optional[List[Dict[str, BaseLayerWithLoRA]]] = None,
        moe_lora_modules: Optional[List[Dict[str, torch.nn.Module]]] = None,
        device=None,
        memory_saver_adapter=None,
        memory_saver_cpu_backup: bool = False,
        double_buffer: bool = False,
    ):
        super().__init__(
            # SINGLE-ACTIVE MVP: one adapter slot, plus a staging twin under
            # double-buffer (async-RL NCCL weight-sync).
            max_adapters_per_batch=2 if double_buffer else 1,
            dtype=dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            eviction_policy=eviction_policy,
            memory_saver_adapter=memory_saver_adapter,
            memory_saver_cpu_backup=memory_saver_cpu_backup,
        )
        # active=0 (viewed by dense wrappers), staging=1 under double-buffer.
        self.active_idx = 0
        self.staging_idx = 1 if double_buffer else 0
        self.target_modules: Set[str] = target_modules
        self.max_lora_rank: int = max_lora_rank
        # Per-layer dense wrapper lookup (LoRAManager.adapter_modules), used
        # ONLY by _declare_groups below to size the "A:<sub>"/"B:<sub>"
        # groups from each wrapper's own base_layer geometry -- the SAME
        # rank/in_features/out formulas bind_zero_lora used to allocate
        # directly (see layers.py:dense_lora_shapes).
        self.adapter_modules = adapter_modules
        # Per-layer FusedMoEWithLoRA wrapper lookup (LoRAManager.moe_lora_modules),
        # used ONLY by _declare_expert_groups below to size the "A_expert"/
        # "B_expert"/"A_expert_down"/"B_expert_down" groups from each wrapper's
        # own base_layer geometry -- the SAME num_local_experts/hidden_size/
        # intermediate_size_per_partition formulas the expert buffers were
        # allocated from before Task 5 moved them into this pool.
        self.moe_lora_modules = moe_lora_modules
        self.device = device

        with self._weights_memory_saver_region():
            self._declare_groups()

    def _declare_groups(self):
        """Register the dense "A:<sub>"/"B:<sub>" groups (q,k,v,o,gate,up,down)
        on the base ``AdapterMemPool`` registry. Only moves where the tensors
        live (the base group registry instead of a wrapper-owned self.A/
        self.B attribute) -- same shapes, dtype, and (via bind_zero_lora /
        set_lora_info) values as before.
        """
        module_lookup: Dict[Tuple[str, int], BaseLayerWithLoRA] = {}
        if self.adapter_modules is not None:
            for layer_idx, layer_modules in enumerate(self.adapter_modules):
                for full_module_name, module in layer_modules.items():
                    try:
                        target_module = get_target_module_name(
                            full_module_name, self.target_modules
                        )
                    except ValueError:
                        continue
                    module_lookup.setdefault((target_module, layer_idx), module)

        num_layers = len(self.adapter_modules) if self.adapter_modules else 0
        for target_module, subs in DENSE_SUB_NAMES.items():
            if target_module not in self.target_modules:
                continue
            per_sub_a: Dict[str, Dict[int, Tuple[int, ...]]] = {
                sub: {} for sub in subs
            }
            per_sub_b: Dict[str, Dict[int, Tuple[int, ...]]] = {
                sub: {} for sub in subs
            }
            for layer_idx in range(num_layers):
                module = module_lookup.get((target_module, layer_idx))
                if module is None:
                    continue
                for sub, (shape_a, shape_b) in zip(
                    subs, module.dense_lora_shapes(self.max_lora_rank)
                ):
                    per_sub_a[sub][layer_idx] = shape_a
                    per_sub_b[sub][layer_idx] = shape_b
            for sub in subs:
                if not per_sub_a[sub]:
                    continue
                self.register_buffer_group(
                    f"A:{sub}", per_sub_a[sub], dtype=self.dtype, device=self.device
                )
                self.register_buffer_group(
                    f"B:{sub}", per_sub_b[sub], dtype=self.dtype, device=self.device
                )

        self._declare_expert_groups()

    def _declare_expert_groups(self):
        """Register the per-layer expert MoE-LoRA groups (``"A_expert"``/
        ``"B_expert"``/``"A_expert_down"``/``"B_expert_down"``) for
        FusedMoEWithLoRA-wrapped layers, in upstream v0.5.14's EXACT 4D
        layout: gate_up_A ``[1,E,2R,H]``, gate_up_B ``[1,E,2I,R]``, down_A
        ``[1,E,R,I]``, down_B ``[1,E,H,R]`` (E=num_local_experts,
        R=max_lora_rank, H=hidden_size, I=moe_intermediate_size).

        The FULL 4D shape (including the kernel's own leading num_loras=1
        axis) is the per-key shape passed to ``register_buffer_group``,
        which prepends ONLY the slot dim (``max_adapters_per_batch=1``
        here) -- so ``active_view("A_expert", layer_id)`` collapses back to
        the exact upstream 4D tensor ``FusedMoEWithLoRA`` reads. No-op if no
        FusedMoEWithLoRA wrapper was installed (dense-only LoRA, or
        ``--lora-target-modules`` doesn't cover MoE experts).
        """
        if not self.moe_lora_modules:
            return

        R = self.max_lora_rank
        gate_up_a_shapes: Dict[int, Tuple[int, ...]] = {}
        gate_up_b_shapes: Dict[int, Tuple[int, ...]] = {}
        down_a_shapes: Dict[int, Tuple[int, ...]] = {}
        down_b_shapes: Dict[int, Tuple[int, ...]] = {}
        for layer_id, layer_modules in enumerate(self.moe_lora_modules):
            if not layer_modules:
                continue
            wrapper = next(iter(layer_modules.values()))
            base = wrapper.base_layer
            E = base.num_local_experts
            H = base.hidden_size
            I = base.intermediate_size_per_partition
            gate_up_a_shapes[layer_id] = (1, E, 2 * R, H)
            gate_up_b_shapes[layer_id] = (1, E, 2 * I, R)
            down_a_shapes[layer_id] = (1, E, R, I)
            down_b_shapes[layer_id] = (1, E, H, R)

        if not gate_up_a_shapes:
            return

        self.register_buffer_group(
            "A_expert", gate_up_a_shapes, dtype=self.dtype, device=self.device
        )
        self.register_buffer_group(
            "B_expert", gate_up_b_shapes, dtype=self.dtype, device=self.device
        )
        self.register_buffer_group(
            "A_expert_down", down_a_shapes, dtype=self.dtype, device=self.device
        )
        self.register_buffer_group(
            "B_expert_down", down_b_shapes, dtype=self.dtype, device=self.device
        )

    def _fill_slot(
        self,
        slot_idx: int,
        named_tensors: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor, float]],
    ) -> None:
        """Write dense LoRA A/B into ``slot_idx``. ``named_tensors`` maps
        ``(sub, layer_id) -> (A, B, scaling)``. Scaling is folded into B only
        (never A), via an in-place ``mul_`` after the copy (avoids an
        allocating ``B * scaling``) -- mirrors
        ``lora.manager.fill_moe_lora_buffers``'s convention.
        """
        for (sub, layer_id), (A, B, scaling) in named_tensors.items():
            self.slot(f"A:{sub}", layer_id, slot_idx).copy_(A)
            b_slot = self.slot(f"B:{sub}", layer_id, slot_idx)
            b_slot.copy_(B)
            b_slot.mul_(scaling)

    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:
        """Check if this memory pool can support the given LoRA adapter(s)."""

        def _can_support(config: LoRAConfig) -> bool:
            if config.r > self.max_lora_rank:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            if "all" in target_module_names:
                return True
            return target_module_names.issubset(self.target_modules)

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        return all(_can_support(c) for c in config)

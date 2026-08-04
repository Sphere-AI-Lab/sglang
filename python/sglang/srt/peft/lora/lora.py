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
"""Single-active LoRA adapter loader (own implementation).

Loads a LoRA checkpoint into per-layer A/B tensors for the single-active
LoRA layers in ``peft/lora/layers.py``. Structurally mirrors ``srt/lora``'s
``LoRALayer``/``LoRAAdapter`` (multi-adapter serving loader), but:

  * Does NOT fuse q/k/v or gate/up LoRA weights. Our single-active layers
    apply LoRA PER SUB-PROJECTION to the matching slice of the fused base
    linear's output, so ``layer.weights`` keeps q_proj/k_proj/v_proj and
    gate_proj/up_proj tensors separate, keyed by the checkpoint tensor name
    (srt/lora's ``normalize_qkv_proj``/``normalize_gate_up_proj`` are not
    used here — the sub-proj -> fused-module association is the mem pool's
    job, not the loader's).
  * MVP = linear attn+MLP + grouped-MoE experts. Expert-LoRA routing IS
    implemented: ``_process_weight`` routes ``experts.<id>.{gate,up,down}_proj
    .lora_{A,B}`` keys into ``LoRALayer.expert_weights`` (consumed by
    ``LoRAManager``'s expert buffers). embed/lm_head LoRA is still not
    implemented; those fields are kept for structural parity but stay
    empty/unused.
"""

import dataclasses
import re
from typing import Dict, List

import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.peft.lora.lora_config import LoRAConfig
from sglang.srt.peft.oft.utils import get_hf_config_attr
from sglang.srt.utils.hf_transformers_utils import AutoConfig

_EXPERT_LORA_RE = re.compile(
    r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)"
)


class LoRALayer:
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # LoRA A/B weights in cpu, keyed by the checkpoint tensor name (e.g.
        # "...layers.3.self_attn.q_proj.lora_A.weight"). Per sub-projection —
        # no q/k/v or gate/up fusion.
        self.weights: Dict[str, torch.Tensor] = {}
        # expert-LoRA weights, keyed by expert_id (unused in MVP; kept for
        # structural parity with peft/oft's OFTLayer).
        self.expert_weights: Dict[int, Dict[str, torch.Tensor]] = {}


class LoRAAdapter:
    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
    ):
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.scaling: float = self.config.lora_alpha / self.config.r

        self.layers: List[LoRALayer] = [
            LoRALayer(config, base_hf_config)
            for _ in range(get_hf_config_attr(base_hf_config, "num_hidden_layers"))
        ]

        # Unused in MVP (embed/lm_head LoRA + added-token embeddings); kept
        # for structural parity with srt/lora and peft/oft.
        self.embedding_layers: Dict[str, torch.Tensor] = {}
        self.added_tokens_embeddings: Dict[str, torch.Tensor] = {}

    def initialize_weights(self):
        model_path = self.config.path
        # An adapter's own weights always exist on disk, so they must never be
        # loaded as "dummy". When the base model was booted with
        # load_format="dummy" (e.g. perf/parity fixtures), that format leaks in
        # via the shared load_config and DefaultModelLoader._prepare_weights
        # hard-raises on DUMMY -- override it to AUTO for the adapter's real
        # safetensors. No-op for real bases (already AUTO/safetensors).
        load_config = self.load_config
        if load_config.load_format == LoadFormat.DUMMY:
            load_config = dataclasses.replace(load_config, load_format=LoadFormat.AUTO)
        loader = DefaultModelLoader(load_config)
        revision = getattr(self.config.hf_config, "revision", None)

        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            self._process_weight(name, loaded_weight)

    def initialize_weights_from_tensors(self, tensors: Dict[str, torch.Tensor]):
        for name, tensor in tensors.items():
            self._process_weight(name, tensor)

    def _process_weight(self, name: str, loaded_weight: torch.Tensor):
        # Remap PEFT "unembed_tokens" key to "lm_head" (mirrors srt/lora).
        if "unembed_tokens" in name:
            name = name.replace("unembed_tokens", "lm_head")

        layer_id = get_layer_id(name)
        if layer_id is not None:
            # MVP: linear attn+MLP only. Every per-layer LoRA A/B tensor is
            # stored under its own checkpoint name — no fusion. Expert-LoRA
            # keys are routed into LoRALayer.expert_weights instead (see
            # _EXPERT_LORA_RE above).
            m = _EXPERT_LORA_RE.search(name)
            if m is not None:
                expert_id = int(m.group(1))
                proj_name = m.group(2)
                ab = m.group(3)
                ew = self.layers[layer_id].expert_weights
                ew.setdefault(expert_id, {})[f"{proj_name}.lora_{ab}"] = (
                    loaded_weight.cpu()
                )
            else:
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
        # else: embed_tokens/lm_head/added-token tensors are not consumed by
        # the single-active linear layers in MVP; skip silently.

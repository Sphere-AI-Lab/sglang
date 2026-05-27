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

# OFT (Orthogonal Finetuning) adapter layers for SGLang serving.
# OFT applies learned orthogonal transformations to pretrained weight matrices,
# as opposed to LoRA's low-rank additive decomposition.

# OFT layers class structure adapted from LoRA implementation in:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import logging
import re
from typing import Dict, List

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.oft.backend.base_backend import BaseOFTBackend
from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.utils import get_hf_config_attr
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)

_EXPERT_OFT_RE = re.compile(
    r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.oft_R"
)
_DSV4_EXPERT_OFT_RE = re.compile(
    r"(?:mlp|ffn)\.experts\.(\d+)\.(w1|w2|w3)\.oft_R"
)

class OFTLayer(nn.Module):
    def __init__(self, config: OFTConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: OFTConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # OFT weights in cpu. The weights are loaded from checkpoint.
        # For OFT, the key weight is "oft_r" — the orthogonal rotation parameters
        # (block-diagonal skew-symmetric or Cayley-parameterized matrices).
        self.weights: Dict[str, torch.Tensor] = {}
        # expert OFT weights, keyed by expert_id
        self.expert_weights: Dict[int, Dict[str, torch.Tensor]] = {}


class OFTAdapter(nn.Module):

    def __init__(
        self,
        uid: str,
        config: OFTConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        oft_backend: BaseOFTBackend,
    ):
        super().__init__()
        self.uid: str = uid
        self.config: OFTConfig = config
        assert self.config.hf_config["peft_type"].lower() == "oft"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.oft_backend: BaseOFTBackend = oft_backend
        # OFT does not use a scaling factor like LoRA (lora_alpha / r).
        # Instead, it uses eps (constraint coefficient) and coft (constrained OFT) flag.
        self.eps: float = self.config.eps
        self.coft: bool = self.config.coft
        self.block_size: int = self.config.block_size

        num_hidden_layers = get_hf_config_attr(
            base_hf_config, "num_hidden_layers"
        )
        self.layers: List[OFTLayer] = nn.ModuleList(
            [OFTLayer(config, base_hf_config) for _ in range(num_hidden_layers)]
        )

        self.embedding_layers: Dict[str, torch.Tensor] = {}
        self.added_tokens_embeddings: Dict[str, torch.Tensor] = {}

    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)

        # Get normalized target modules for filtering
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            self._process_weight(name, loaded_weight)

        self._normalize_weights()

    def initialize_weights_from_tensors(self, tensors: Dict[str, torch.Tensor]):
        for name, tensor in tensors.items():
            self._process_weight(name, tensor)

        self._normalize_weights()

    def _process_weight(self, name: str, loaded_weight: torch.Tensor):
        from sglang.srt.oft.utils import get_normalized_target_modules

        normalized_target_modules = get_normalized_target_modules(
            self.config.target_modules
        )

        # Remap PEFT "unembed_tokens" key to "lm_head" so the weight is
        # recognized and loaded into the correct buffer.
        if "unembed_tokens" in name:
            name = name.replace("unembed_tokens", "lm_head")

        layer_id = get_layer_id(name)
        if layer_id is not None:
            m = _EXPERT_OFT_RE.search(name)
            if m is None:
                m = _DSV4_EXPERT_OFT_RE.search(name)
            if m:
                expert_id = int(m.group(1))
                proj_name = m.group(2)
                ew = self.layers[layer_id].expert_weights
                if expert_id not in ew:
                    ew[expert_id] = {}
                ew[expert_id][f"{proj_name}.oft_R"] = loaded_weight.cpu()
            else:
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
        elif "embed_tokens" in name or "lm_head" in name:
            # Check if this module is declared in target_modules before loading.
            # When normalized_target_modules is {"all"} (e.g. target_modules was
            # "all-linear"), we allow loading since the server-level
            # --oft-target-modules will govern which modules are active.
            module_name = "embed_tokens" if "embed_tokens" in name else "lm_head"
            if (
                "all" in normalized_target_modules
                or module_name in normalized_target_modules
            ):
                self.embedding_layers[name] = loaded_weight.cpu()
            else:
                logger.debug(
                    f"Skipping {name} as '{module_name}' is not in adapter's target_modules: {self.config.target_modules}"
                )
        elif "input_embeddings" in name or "output_embeddings" in name:
            # added/extra token emb
            self.added_tokens_embeddings[name] = loaded_weight.cpu()
            assert loaded_weight.shape[0] == self.config.oft_added_tokens_size, (
                f"OFT adapter {self.uid} has extra_vocab_size {self.config.extra_vocab_size} specified in the config, "
                f"but the loaded weight has {loaded_weight.shape[0]} extra vocab size"
            )

    def _normalize_weights(self):
        # Topology-aware normalization is done at load time inside OFTMemoryPool
        # / streamed_weight_loader, where the runtime R_buffer keys are known.
        return

    def pin_weights_in_cpu(self):
        for layer in self.layers:
            for name, weight in layer.weights.items():
                layer.weights[name] = weight.pin_memory()

            for expert_id, expert_dict in layer.expert_weights.items():
                for name, weight in expert_dict.items():
                    expert_dict[name] = weight.pin_memory()

        for name, weight in self.embedding_layers.items():
            self.embedding_layers[name] = weight.pin_memory()

        for name, weight in self.added_tokens_embeddings.items():
            self.added_tokens_embeddings[name] = weight.pin_memory()

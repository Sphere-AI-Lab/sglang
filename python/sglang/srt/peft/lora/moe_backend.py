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
"""Single-active MoE-LoRA backend + per-batch routing info (own implementation).

Drives upstream v0.5.14's ``FusedMoEWithLoRA`` (`srt/lora/layers.py`) from the
single-active manager, per `.superpowers/sdd/phase3-integration-spec.md` §3/§5
(Option A: reuse upstream's forward/hooks/align/CUDA-graph path verbatim).

``SingleActiveMoEBackend`` subclasses upstream's ``BaseLoRABackend`` (rather
than a bare object) so it gets ``init_cuda_graph_moe_buffers`` and
``_compute_moe_lora_info`` for free -- both are wired up in a later task
(3.3/3.5), not here.

``build_single_active_batch_info`` packs the minimal single-active routing
values (one lora, index 0, covering every token in the batch) into a new
``SingleActiveMoEBatchInfo``. Upstream's ``MoELoRABatchInfo`` (grandfathered
``@dataclass``, `srt/lora/utils.py`) only carries
``seg_indptr/req_to_lora/adapter_enabled/token_lora_mapping`` -- it has no
``lora_ranks``/``has_active_lora`` fields, so it cannot be reused standalone
as the object `FusedMoEWithLoRA._get_lora_info` reads
(`batch_info.lora_ranks`, `.moe_lora_info`, `.has_active_lora`). Per
`.claude/rules/no-dataclasses.md`, this new holder is a ``msgspec.Struct``.
"""

import msgspec
import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import MoELoRABatchInfo


class SingleActiveMoEBackend(BaseLoRABackend):
    """Minimal MoE-LoRA backend for the single-active manager.

    Always exactly one active lora (index 0), so ``max_loras_per_batch=1``.
    """

    def __init__(self, device: torch.device):
        super().__init__(max_loras_per_batch=1, device=device)


class SingleActiveMoEBatchInfo(msgspec.Struct, frozen=True):
    """Per-batch routing info consumed by ``FusedMoEWithLoRA._get_lora_info``."""

    lora_ranks: torch.Tensor
    moe_lora_info: MoELoRABatchInfo
    has_active_lora: bool


def build_single_active_batch_info(
    num_tokens: int, rank: int, device: torch.device
) -> SingleActiveMoEBatchInfo:
    """Build the minimal single-active per-batch routing info (spec §3):
    one lora (index 0) covering all ``num_tokens`` tokens.
    """
    moe_lora_info = MoELoRABatchInfo(
        seg_indptr=torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        req_to_lora=torch.tensor([0], dtype=torch.int32, device=device),
        adapter_enabled=torch.tensor([1], dtype=torch.int32, device=device),
        token_lora_mapping=torch.zeros(num_tokens, dtype=torch.int32, device=device),
    )
    return SingleActiveMoEBatchInfo(
        lora_ranks=torch.tensor([rank], dtype=torch.int32, device=device),
        moe_lora_info=moe_lora_info,
        has_active_lora=True,
    )

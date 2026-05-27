# Copyright 2024-2026 SGLang Team
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

"""Inference-only DeepSeek V4 model implementation for SGLang.

The model loads DeepSeek V4 native-quant checkpoints and keeps the parameter
layout aligned with the training/reference implementation. The attention,
compressor, indexer, and MoE paths use DeepSeek V4-specific kernels where the
public checkpoint format requires FP4/FP8 native-quant math.

Reference implementation:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py

Runtime controls used for deterministic parity and kernel selection:
  * ``SGLANG_DSV4_CANONICAL_RMSNORM=1`` enables the persistent
    batch-invariant RMSNorm path.
  * ``SGLANG_DSV4_CANONICAL_COMPRESSOR_LINEAR=1`` computes compressor
    ``wkv``/``wgate`` with the configured canonical linear helper.
  * ``SGLANG_DSV4_CANONICAL_*_IMPL`` selects a canonical linear helper:
    ``persistent`` (batch-invariant generic Triton), ``fixed_order``
    (batch-invariant fp32 Triton), or ``torch``.
  * ``SGLANG_DSV4_USE_TILEKERNELS=1`` uses DeepSeek TileKernels for the mHC
    and router paths. Missing TileKernels or incompatible dtype/shape is a
    hard error.
  * ``SGLANG_DSV4_FAST_MHC_PRE=1`` uses the PyTorch/cuBLAS mHC pre path in
    decoder layers while keeping the TileKernels mHC post path.
  * ``SGLANG_DSV4_GATE_LINEAR_IMPL=persistent`` feeds the TileKernels router
    with a batch-invariant projection.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_moe_tp_group,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_partial,
    dp_scatter,
    get_global_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.token_dispatcher import (
    DeepEPDispatcher,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
    DispatchOutputChecker,
)
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.models import deepseek_v4_kernels
from sglang.srt.models.deepseek_v4_q_rmsnorm import deepseek_v4_q_rmsnorm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek V4 FP4/FP8 block sizes. These match the native checkpoint layout
# and the official inference kernels.
# ---------------------------------------------------------------------------
_FP8_BLOCK_SIZE = 128
_FP4_BLOCK_SIZE = 32
_DEFAULT_SCALE_FMT = "ue8m0"
_DEFAULT_SCALE_DTYPE = torch.float8_e8m0fnu
_DSV4_CANONICAL_LM_HEAD_ATTR = "_dsv4_canonical_lm_head_wrapped"


# Top-level native checkpoint rewrites that do not depend on a layer index.
_TOP_LEVEL_REWRITES = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_fn": "model.hc_head_params.hc_head_fn",
    "hc_head_base": "model.hc_head_params.hc_head_base",
    "hc_head_scale": "model.hc_head_params.hc_head_scale",
}


def _is_ignore(key: str) -> bool:
    """Keys explicitly excluded from the forward inference graph."""
    if key.startswith("mtp."):
        return True
    if key.endswith(".kv_cache"):
        return True
    return False


def _rewrite_native_key(key: str) -> Optional[str]:
    """Translate a native-V4 checkpoint key to the SGLang parameter name."""
    if key in _TOP_LEVEL_REWRITES:
        return _TOP_LEVEL_REWRITES[key]

    if not key.startswith("layers."):
        return None

    parts = key.split(".")
    if len(parts) < 3:
        return None
    layer_idx = parts[1]
    sub_top = parts[2]
    rest = ".".join(parts[3:])

    if sub_top == "attn_norm":
        return f"model.layers.{layer_idx}.input_layernorm.{rest}"
    if sub_top == "ffn_norm":
        return f"model.layers.{layer_idx}.post_attention_layernorm.{rest}"
    if sub_top == "attn":
        sub = rest
        # Indexer linears are wrapped explicitly in the SGLang module tree.
        sub = sub.replace("indexer.wq_b", "indexer.linear_wq_b")
        sub = sub.replace("indexer.weights_proj", "indexer.linear_weights_proj")
        return f"model.layers.{layer_idx}.self_attn.{sub}"
    if sub_top == "ffn":
        return f"model.layers.{layer_idx}.mlp.{rest}"
    if sub_top.startswith("hc_"):
        return f"model.layers.{layer_idx}.{'.'.join(parts[2:])}"

    return None


def _try_load_packed_routed_expert_param(
    model: torch.nn.Module,
    target: str,
    tensor: torch.Tensor,
) -> bool:
    parts = target.split(".")
    if (
        len(parts) != 8
        or parts[0] != "model"
        or parts[1] != "layers"
        or parts[3] != "mlp"
        or parts[4] != "experts"
        or parts[6] not in {"w1", "w2", "w3"}
        or parts[7] not in {"weight", "scale"}
    ):
        return False

    try:
        layer_id = int(parts[2])
        expert_id = int(parts[5])
    except ValueError:
        return False

    try:
        mlp = model.model.layers[layer_id].mlp
    except (AttributeError, IndexError, TypeError):
        return False

    loader = getattr(mlp, "load_routed_expert_param", None)
    if loader is None:
        return False
    return bool(loader(expert_id, parts[6], parts[7], tensor))


def _deepseek_v4_update_cache_slots_(
    cache: torch.Tensor,
    req_indices: torch.Tensor,
    write_idx: torch.Tensor,
    values: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Update one cache slot per active request without cloning full rows."""
    if values.dim() == 3 and values.shape[1] == 1:
        values = values.squeeze(1)
    if active.dtype != torch.bool:
        active = active.to(torch.bool)
    if cache.is_cuda and req_indices.is_cuda and write_idx.is_cuda and values.is_cuda:
        from sglang.srt.models.deepseek_v4_cache_kernels import (
            deepseek_v4_update_cache_slots_,
        )

        deepseek_v4_update_cache_slots_(cache, req_indices, write_idx, values, active)
        return

    old_values = cache[req_indices, write_idx]
    new_values = torch.where(active.unsqueeze(-1), values.to(cache.dtype), old_values)
    cache[req_indices, write_idx] = new_values


def _deepseek_v4_require_metadata_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if tensor.device != device or tensor.dtype != dtype:
        raise RuntimeError(
            f"DeepSeek V4 decode metadata {name} must be {dtype} on {device}; "
            f"got dtype={tensor.dtype} device={tensor.device}."
        )
    return tensor


def _deepseek_v4_require_forward_batch(forward_batch: Any, name: str) -> Any:
    assert forward_batch is not None, f"{name} requires non-None forward_batch"
    return forward_batch


def _quantized_linear(
    x: torch.Tensor,
    weight: nn.Parameter,
    scale: Optional[nn.Parameter],
) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_empty(*x.shape[:-1], weight.shape[0])
    if weight.dtype == torch.float4_e2m1fn_x2:
        # Activation block stays at 128 (FP8 quant) even when weight is FP4 block-32.
        # Matches official ``inference/model.py:linear`` exactly.
        x_q, s_x = deepseek_v4_kernels.act_quant(
            x, _FP8_BLOCK_SIZE, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE
        )
        return deepseek_v4_kernels.fp4_gemm(x_q, s_x, weight, scale, _DEFAULT_SCALE_DTYPE)
    if weight.dtype == torch.float8_e4m3fn:
        if os.environ.get("SGLANG_DSV4_DET_FP8_GEMM", "0") == "1":
            from sglang.srt.models.det_fp8_gemm import (
                det_act_quant,
                det_fp8_gemm,
            )

            x_q, s_x = det_act_quant(
                x, _FP8_BLOCK_SIZE, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE
            )
            return det_fp8_gemm(x_q, s_x, weight, scale, _DEFAULT_SCALE_DTYPE)
        x_q, s_x = deepseek_v4_kernels.act_quant(
            x, _FP8_BLOCK_SIZE, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE
        )
        return deepseek_v4_kernels.fp8_gemm(x_q, s_x, weight, scale, _DEFAULT_SCALE_DTYPE)
    if weight.dtype != x.dtype and weight.is_floating_point() and x.is_floating_point():
        return F.linear(x, weight.to(x.dtype))
    return F.linear(x, weight)


def _deepseek_v4_env_impl(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default).lower()


def _deepseek_v4_canonical_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    env_name: str,
    *,
    default: str,
) -> torch.Tensor:
    impl = _deepseek_v4_env_impl(env_name, default)
    if impl == "torch":
        return F.linear(x, weight)
    if impl == "fixed_order":
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            matmul_fixed_order,
        )

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if flat.shape[0] == 0:
            return x.new_empty(*x.shape[:-1], weight.shape[0])
        weight_t = weight.t().contiguous()
        if flat.dtype != torch.float32:
            flat = flat.float()
        if weight_t.dtype != torch.float32:
            weight_t = weight_t.float()
        out = matmul_fixed_order(flat, weight_t)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    if impl == "persistent":
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if flat.shape[0] == 0:
            return x.new_empty(*x.shape[:-1], weight.shape[0])
        weight_t = weight.t().contiguous()
        if weight_t.dtype != flat.dtype:
            weight_t = weight_t.to(flat.dtype)
        out = matmul_persistent(flat, weight_t)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    raise ValueError(
        f"Unsupported {env_name}={impl!r}; expected persistent, fixed_order, or torch"
    )


def _deepseek_v4_best_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    env_name: str,
) -> torch.Tensor:
    impl = _deepseek_v4_env_impl(env_name, "persistent")
    if impl == "torch":
        return F.linear(x, weight)
    if impl == "persistent":
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if flat.shape[0] == 0:
            return x.new_empty(*x.shape[:-1], weight.shape[0])
        weight_t = weight.t().contiguous()
        if weight_t.dtype != flat.dtype:
            weight_t = weight_t.to(flat.dtype)
        out = matmul_persistent(flat, weight_t)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    if impl == "fixed_order":
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            matmul_fixed_order,
        )

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if flat.shape[0] == 0:
            return x.new_empty(*x.shape[:-1], weight.shape[0])
        weight_t = weight.t().contiguous()
        if flat.dtype != torch.float32:
            flat = flat.float()
        if weight_t.dtype != torch.float32:
            weight_t = weight_t.float()
        out = matmul_fixed_order(flat, weight_t)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    raise ValueError(
        f"Unsupported {env_name}={impl!r}; expected persistent, fixed_order, or torch"
    )


def _maybe_wrap_deepseek_v4_lm_head_canonical(lm_head: nn.Module) -> bool:
    if os.environ.get("SGLANG_DSV4_CANONICAL_LM_HEAD", "0") != "1":
        return False
    if getattr(lm_head, _DSV4_CANONICAL_LM_HEAD_ATTR, False):
        return False

    def _canonical_lm_head_forward(input_: torch.Tensor) -> torch.Tensor:
        out = _deepseek_v4_canonical_linear(
            input_,
            lm_head.weight,
            "SGLANG_DSV4_CANONICAL_LM_HEAD_IMPL",
            default="persistent",
        )
        bias = getattr(lm_head, "bias", None)
        if bias is not None:
            out = out + bias
        return out

    lm_head.forward = _canonical_lm_head_forward
    setattr(lm_head, _DSV4_CANONICAL_LM_HEAD_ATTR, True)
    return True


def _deepseek_v4_use_tilekernels() -> bool:
    return os.environ.get("SGLANG_DSV4_USE_TILEKERNELS", "1") != "0"


def _deepseek_v4_fast_mhc_pre_enabled() -> bool:
    return os.environ.get("SGLANG_DSV4_FAST_MHC_PRE", "0") == "1"


@lru_cache(maxsize=None)
def _deepseek_v4_tile_mhc_ops():
    try:
        from tile_kernels.modeling.mhc.ops.head_compute_mix import (
            mhc_head_compute_mix,
        )
        from tile_kernels.modeling.mhc.ops.norm_fn import mhc_pre_norm_fn
        from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix
        from tile_kernels.modeling.mhc.ops.pre_big_fuse import mhc_pre_big_fuse
        from tile_kernels.modeling.mhc.ops.pre_split_mixes import (
            mhc_pre_split_mixes,
        )
        from tile_kernels.modeling.mhc.ops.sinkhorn import sinkhorn_normalize
    except Exception as exc:
        raise RuntimeError(
            "SGLANG_DSV4_USE_TILEKERNELS=1 requires DeepSeek TileKernels "
            "installed in the active Python environment."
        ) from exc
    return (
        mhc_pre_big_fuse,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
        mhc_pre_apply_mix,
        mhc_head_compute_mix,
    )


@lru_cache(maxsize=None)
def _deepseek_v4_tile_router_op():
    try:
        from tile_kernels.moe.top2_sum_gate_kernel import top2_sum_gate
    except Exception as exc:
        raise RuntimeError(
            "SGLANG_DSV4_USE_TILEKERNELS=1 requires "
            "tile_kernels.moe.top2_sum_gate_kernel."
        ) from exc
    return top2_sum_gate


# ---------------------------------------------------------------------------
# DeepSeek V4 native-quant linear primitives.
# ---------------------------------------------------------------------------


class DeepSeekV4Linear(nn.Module):
    """DeepSeek V4 native-quant linear layer.

    Holds a ``weight`` Parameter plus an optional sibling ``scale`` Parameter
    when the weight dtype is one of FP4 / FP8. The native-quant checkpoint
    stores per-block scale tensors; both the weight and the scale must be
    loadable by name.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16

        if dtype == torch.float4_e2m1fn_x2:
            # FP4 weight is packed two-per-byte: storage shape uses ``in // 2``.
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2)
            )
            scale = nn.Parameter(
                torch.empty(
                    out_features,
                    in_features // _FP4_BLOCK_SIZE,
                    dtype=_DEFAULT_SCALE_DTYPE,
                )
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale = nn.Parameter(
                torch.empty(
                    (out_features + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
                    (in_features + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
                    dtype=_DEFAULT_SCALE_DTYPE,
                )
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale = None

        if scale is not None:
            # Keep both the registered Parameter and the convenience alias on
            # ``weight.scale`` so the loader can dispatch to either name.
            self.weight.scale = scale
            self.scale = scale
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _quantized_linear(x, self.weight, self.scale)
        if self.bias is not None:
            y = y + self.bias
        return y

    def apply_input_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Identity hook for the V4 attention einsum-bypass on ``wo_a``.

        wo_a is consumed via ``self.wo_a.weight.view(...)`` + a manual
        ``einsum`` (see ``DeepseekV4Attention.forward``), bypassing
        ``self.wo_a(x)``. Without an explicit hook the OFT wrapper's
        ``forward`` is never called for wo_a and any non-identity
        adapter is silently dropped during serving. The base layer
        returns ``x`` unchanged; ``DeepSeekV4LinearWithOFT`` overrides this
        with a per-group OFT rotation matching the training-side adapter
        convention.
        """
        return x


def _deepseek_v4_linear_storage_shapes(
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
) -> tuple[tuple[int, ...], Optional[tuple[int, ...]]]:
    if dtype == torch.float4_e2m1fn_x2:
        return (
            (out_features, in_features // 2),
            (out_features, in_features // _FP4_BLOCK_SIZE),
        )
    if dtype == torch.float8_e4m3fn:
        return (
            (out_features, in_features),
            (
                (out_features + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
                (in_features + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
            ),
        )
    return (out_features, in_features), None


def _tp_group_world(tp_group: Any) -> int:
    if tp_group is None:
        return 1
    if hasattr(tp_group, "world_size"):
        return int(tp_group.world_size)
    if hasattr(tp_group, "size"):
        return int(tp_group.size())
    raise TypeError(f"Unsupported TP group object: {type(tp_group)!r}")


def _tp_group_rank(tp_group: Any) -> int:
    if tp_group is None:
        return 0
    if hasattr(tp_group, "rank_in_group"):
        return int(tp_group.rank_in_group)
    if hasattr(tp_group, "rank"):
        rank = tp_group.rank
        return int(rank() if callable(rank) else rank)
    try:
        return int(dist.get_rank(group=tp_group))
    except Exception:
        return 0


def _mark_deepseek_v4_tp_shard(param: Optional[nn.Parameter], dim: int, tp_group: Any) -> None:
    if param is None:
        return
    param._dsv4_tp_partition_dim = dim
    param._dsv4_tp_rank = _tp_group_rank(tp_group)
    param._dsv4_tp_world = _tp_group_world(tp_group)


def _copy_deepseek_v4_param_data(
    param: nn.Parameter,
    tensor: torch.Tensor,
    *,
    tp_rank: Optional[int] = None,
    tp_size: Optional[int] = None,
) -> None:
    """Copy a full-checkpoint tensor into a possibly TP-sharded DeepSeek V4 parameter."""
    loaded = tensor.to(param.device).to(param.dtype)
    if tuple(param.shape) == tuple(loaded.shape):
        param.data.copy_(loaded)
        return

    if tp_rank is None:
        tp_rank = int(getattr(param, "_dsv4_tp_rank", 0))
    if tp_size is None:
        tp_size = int(getattr(param, "_dsv4_tp_world", 1))
    if tp_size <= 1:
        raise RuntimeError(
            "DeepSeek V4 parameter shape mismatch without TP sharding: "
            f"param={tuple(param.shape)} loaded={tuple(loaded.shape)}"
        )

    dim = getattr(param, "_dsv4_tp_partition_dim", None)
    if dim is None:
        candidates = [
            idx
            for idx, (local, full) in enumerate(zip(param.shape, loaded.shape, strict=True))
            if local != full and local * tp_size == full
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Cannot infer DeepSeek V4 TP shard dimension: "
                f"param={tuple(param.shape)} loaded={tuple(loaded.shape)} "
                f"{tp_rank=} {tp_size=}"
            )
        dim = candidates[0]

    shard_size = param.shape[dim]
    start = tp_rank * shard_size
    if start + shard_size > loaded.shape[dim]:
        raise RuntimeError(
            "DeepSeek V4 TP shard is out of range: "
            f"param={tuple(param.shape)} loaded={tuple(loaded.shape)} "
            f"{dim=} {tp_rank=} {tp_size=}"
        )
    loaded = loaded.narrow(dim, start, shard_size)
    if tuple(param.shape) != tuple(loaded.shape):
        raise RuntimeError(
            "DeepSeek V4 TP-sharded load produced wrong shape: "
            f"param={tuple(param.shape)} shard={tuple(loaded.shape)}"
        )
    param.data.copy_(loaded)


class DeepSeekV4PackedExpertLinear(nn.Module):
    """DeepSeekV4Linear-compatible view over a packed routed-expert tensor.

    The MoE module owns ``[num_local_experts, ...]`` parameters. Each routed
    expert keeps lightweight ``w1/w2/w3`` modules so eager code can continue to
    call ``expert.w1(x)`` while all storage, state_dict, and memory-saver
    ownership lives on the packed tensors.
    """

    def __init__(
        self,
        owner: nn.Module,
        proj: str,
        local_expert_id: int,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.proj = proj
        self.local_expert_id = local_expert_id
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        return getattr(self._owner, f"routed_experts_{self.proj}_weight")[
            self.local_expert_id
        ]

    @property
    def scale(self) -> Optional[torch.Tensor]:
        scale = getattr(self._owner, f"routed_experts_{self.proj}_scale", None)
        if scale is None:
            return None
        return scale[self.local_expert_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _quantized_linear(x, self.weight, self.scale)


class DeepSeekV4ColumnParallelLinear(DeepSeekV4Linear):
    """Shards ``out_features`` across TP. No reduction on output.

    With TP=1 this is equivalent to ``DeepSeekV4Linear``. The shape divisibility
    check keeps TP-sharded checkpoint loading explicit.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        tp_group: Any = None,
    ):
        world = _tp_group_world(tp_group)
        assert out_features % world == 0, (
            f"DeepSeekV4ColumnParallelLinear: out_features={out_features} "
            f"not divisible by tp_world={world}"
        )
        super().__init__(in_features, out_features // world, bias, dtype)
        self.tp_group = tp_group
        _mark_deepseek_v4_tp_shard(self.weight, 0, tp_group)
        _mark_deepseek_v4_tp_shard(self.scale, 0, tp_group)
        _mark_deepseek_v4_tp_shard(self.bias, 0, tp_group)


class DeepSeekV4RowParallelLinear(DeepSeekV4Linear):
    """Shards ``in_features`` across TP and all-reduces output.

    ``expert_skip_comm=True`` mirrors stock Megatron's ``explicit_expert_comm``:
    the linear returns the partial (un-reduced) output and the caller (the
    surrounding MoE wrapper) takes responsibility for the cross-TP reduce.
    Used by ``DeepSeekV4MoE`` under TP>1 to fuse "per-expert all_reduce + per-
    expert bf16 cast" into "one outer all_reduce + one bf16 cast", matching
    Megatron's MoE precision pattern. Bias is forbidden with skip_comm because
    adding bias before the outer reduce would scale it by ``tp_world``.

    The bias allocation lives on ``self.bias`` (NOT a separate ``row_bias``)
    and is added post-reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        tp_group: Any = None,
        expert_skip_comm: bool = False,
    ):
        world = _tp_group_world(tp_group)
        assert in_features % world == 0, (
            f"DeepSeekV4RowParallelLinear: in_features={in_features} "
            f"not divisible by tp_world={world}"
        )
        # Disable bias on base init; bias is added after any TP reduction.
        super().__init__(in_features // world, out_features, bias=False, dtype=dtype)
        if expert_skip_comm and bias:
            raise ValueError(
                "DeepSeekV4RowParallelLinear: expert_skip_comm=True forbids bias — "
                "adding bias before the outer reduce would scale it by tp_world."
            )
        self.tp_group = tp_group
        self.expert_skip_comm = expert_skip_comm
        _mark_deepseek_v4_tp_shard(self.weight, 1, tp_group)
        _mark_deepseek_v4_tp_shard(self.scale, 1, tp_group)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _quantized_linear(x, self.weight, self.scale)
        if (
            self.tp_group is not None
            and _tp_group_world(self.tp_group) > 1
            and not self.expert_skip_comm
        ):
            y = y.float()
            if hasattr(self.tp_group, "all_reduce"):
                y = self.tp_group.all_reduce(y)
            else:
                dist.all_reduce(y, group=self.tp_group)
            y = y.type_as(x)
        if self.bias is not None:
            y = y + self.bias
        return y


# Backward-compatible aliases for code that imported the earlier short names.
DSV4Linear = DeepSeekV4Linear
DSV4PackedExpertLinear = DeepSeekV4PackedExpertLinear
DSV4ColumnParallelLinear = DeepSeekV4ColumnParallelLinear
DSV4RowParallelLinear = DeepSeekV4RowParallelLinear


# ---------------------------------------------------------------------------
# Hyperconnection utilities.
# ---------------------------------------------------------------------------


class HCHeadParams(nn.Module):
    """Decoder-level mHC head parameters stored in fp32."""

    def __init__(self, hc_mult: int, hidden_size: int):
        super().__init__()
        hc_dim = hc_mult * hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))


class DeepSeekV4HyperConnectionUtil:
    """Stateless helper for DeepSeek V4 manifold-HyperConnection ops.

    Carries no parameters of its own; the per-layer mixers (``hc_attn_*``,
    ``hc_ffn_*``) live on ``DeepSeekV4DecoderLayer`` and the decoder-level
    head (``hc_head_*``) lives on ``HCHeadParams``.

    The actual TileLang Sinkhorn split kernel is pulled from
    ``sglang.srt.models.deepseek_v4_kernels`` at forward time.
    """

    def __init__(
        self,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float,
    ):
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps

    # ---- Raw helpers operating on `[b, s, hc, d]`. ----
    def hc_pre_raw(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        use_fast_mhc_pre: bool = False,
    ):
        shape, dtype = x.size(), x.dtype
        if x.numel() == 0:
            post = x.new_empty(*shape[:-2], self.hc_mult, dtype=torch.float32)
            comb = x.new_empty(
                *shape[:-2], self.hc_mult, self.hc_mult, dtype=torch.float32
            )
            y = x.new_empty(*shape[:-2], shape[-1])
            return y, post, comb
        if use_fast_mhc_pre:
            return self.hc_pre_raw_official(x, hc_fn, hc_scale, hc_base)
        if _deepseek_v4_use_tilekernels():
            if dtype != torch.bfloat16:
                raise RuntimeError(
                    "TileKernels MHC pre expects bf16 residual input; "
                    f"got {dtype}."
                )
            (
                mhc_pre_big_fuse,
                mhc_pre_norm_fn,
                mhc_pre_split_mixes,
                sinkhorn_normalize,
                mhc_pre_apply_mix,
                _,
            ) = _deepseek_v4_tile_mhc_ops()
            hc_fn = hc_fn.contiguous()
            hc_scale = hc_scale.contiguous()
            hc_base = hc_base.contiguous()
            if not torch.is_grad_enabled() or not x.requires_grad:
                with torch.no_grad():
                    post, comb, y = mhc_pre_big_fuse(
                        x,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        rms_eps=self.norm_eps,
                        mhc_pre_eps=self.hc_eps,
                        mhc_sinkhorn_eps=self.hc_eps,
                        mhc_post_mult_value=2.0,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                        n_splits=16,
                    )
                return y.to(dtype), post.squeeze(-1), comb

            with torch.no_grad():
                mixes = mhc_pre_norm_fn(
                    x,
                    hc_fn,
                    None,
                    self.norm_eps,
                    fuse_grad_acc=False,
                    n_splits=16,
                )
                pre, post, comb = mhc_pre_split_mixes(
                    mixes,
                    hc_scale,
                    hc_base,
                    self.hc_mult,
                    2.0,
                    self.hc_eps,
                )
                comb = sinkhorn_normalize(
                    comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps
                )
            y = mhc_pre_apply_mix(x, pre)
            return y.to(dtype), post.squeeze(-1), comb

        return self.hc_pre_raw_official(x, hc_fn, hc_scale, hc_base)

    def hc_pre_raw_official(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        with torch.no_grad():
            rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(x_flat, hc_fn) * rsqrt
            pre, post, comb = deepseek_v4_kernels.hc_split_sinkhorn(
                mixes,
                hc_scale,
                hc_base,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
            )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post_raw(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if _deepseek_v4_use_tilekernels() and (
            x.is_cuda and residual.is_cuda and post.is_cuda and comb.is_cuda
        ):
            hidden = x.shape[-1]
            hc_mult = residual.shape[-2]
            n_tokens = x.numel() // hidden
            from tile_kernels.modeling.mhc.ops import post as post_ops

            out = post_ops.mhc_post(
                x.contiguous().reshape(1, n_tokens, hidden),
                residual.contiguous().reshape(1, n_tokens, hc_mult, hidden),
                post.contiguous().reshape(1, n_tokens, hc_mult, 1),
                comb.contiguous().reshape(1, n_tokens, hc_mult, hc_mult),
            )
            return out.reshape(*x.shape[:-1], hc_mult, hidden)

        y = post.unsqueeze(-1) * x.unsqueeze(-2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def hc_head_raw(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        if x.numel() == 0:
            return x.new_empty(*shape[:-2], shape[-1])
        if _deepseek_v4_use_tilekernels():
            if dtype != torch.bfloat16:
                raise RuntimeError(
                    "TileKernels MHC head expects bf16 residual input; "
                    f"got {dtype}."
                )
            (
                _,
                mhc_pre_norm_fn,
                _,
                _,
                mhc_pre_apply_mix,
                mhc_head_compute_mix,
            ) = _deepseek_v4_tile_mhc_ops()
            mhc_mult3 = self.hc_mult * (2 + self.hc_mult)
            hc_fn = hc_fn.contiguous()
            if hc_fn.shape[0] < mhc_mult3:
                hc_fn = F.pad(hc_fn, (0, 0, 0, mhc_mult3 - hc_fn.shape[0]))
            with torch.no_grad():
                mixes = mhc_pre_norm_fn(
                    x,
                    hc_fn,
                    None,
                    self.norm_eps,
                    fuse_grad_acc=False,
                    n_splits=16,
                )[..., : self.hc_mult].contiguous()
                pre = mhc_head_compute_mix(
                    mixes,
                    hc_scale.reshape(1).contiguous(),
                    hc_base.contiguous(),
                    self.hc_eps,
                )
            y = mhc_pre_apply_mix(x, pre.unsqueeze(-1))
            return y.to(dtype)

        x_flat = x.flatten(2).float()
        with torch.no_grad():
            rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(x_flat, hc_fn) * rsqrt
            pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)

    # ---- Block-level helpers (preserved Megatron-LM `[s, b, ...]` convention). ----
    def block_expand(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

    def layer_pre(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        use_fast_mhc_pre: Optional[bool] = None,
    ):
        if use_fast_mhc_pre is None:
            use_fast_mhc_pre = _deepseek_v4_fast_mhc_pre_enabled()
        return self.layer_pre_impl(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            use_fast_mhc_pre=use_fast_mhc_pre,
        )

    def layer_pre_impl(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        use_fast_mhc_pre: bool = False,
    ):
        x = hidden_states.permute(1, 0, 2, 3).contiguous()
        x, post, comb = self.hc_pre_raw(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            use_fast_mhc_pre=use_fast_mhc_pre,
        )
        return x.permute(1, 0, 2).contiguous(), post, comb

    def layer_post(
        self,
        out,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(out, tuple):
            out, bias = out
            assert bias is None, "DeepSeek V4 native layers do not support bias outputs."
        out = out.permute(1, 0, 2).contiguous()
        residual = residual.permute(1, 0, 2, 3).contiguous()
        hidden_states = self.hc_post_raw(out, residual, post, comb)
        return hidden_states.permute(1, 0, 2, 3).contiguous()

    def block_head(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        return self.block_head_impl(hidden_states, hc_fn, hc_scale, hc_base)

    def block_head_impl(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states.permute(1, 0, 2, 3).contiguous()
        x = self.hc_head_raw(x, hc_fn, hc_scale, hc_base)
        return x.permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------
# RMSNorm.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """DeepSeek-style RMSNorm.

    The native checkpoint may store these tensors as bf16, but the official
    inference code keeps RMSNorm weights in fp32 in-flight and loads into that
    dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match reference RMSNorm numerics: fp32 variance accumulation,
        # original dtype output.
        dtype = x.dtype
        if os.environ.get("SGLANG_DSV4_CANONICAL_RMSNORM", "1") == "1":
            canonical_impl = _deepseek_v4_env_impl(
                "SGLANG_DSV4_CANONICAL_RMSNORM_IMPL", "persistent"
            )
            if canonical_impl != "persistent":
                raise ValueError(
                    "Unsupported SGLANG_DSV4_CANONICAL_RMSNORM_IMPL="
                    f"{canonical_impl!r}; expected persistent"
                )
            from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm

            return rms_norm(x.float(), self.weight, self.eps).to(dtype)
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


# ---------------------------------------------------------------------------
# RoPE / window / compress-index helpers.
# ---------------------------------------------------------------------------


@lru_cache(2)
def _precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, rope_dim, rope_base, max_seq_len):
        return (
            rope_dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(rope_base))
        )

    def find_correction_range(low_rot, high_rot, rope_dim, rope_base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, rope_dim, rope_base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, rope_dim, rope_base, max_seq_len))
        return max(low, 0), min(high, rope_dim - 1)

    def linear_ramp_factor(min_value, max_value, rope_dim):
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(rope_dim, dtype=torch.float32) - min_value) / (
            max_value - min_value
        )
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _wrapped_precompute_freqs_cis(
    *,
    rope_head_dim: int,
    base: float,
    rotary_scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: int,
    beta_slow: int,
    yarn_disabled: bool = False,
) -> torch.Tensor:
    max_seq_len = 65536
    original_seq_len = 0 if yarn_disabled else original_max_position_embeddings
    return _precompute_freqs_cis(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=rotary_scaling_factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """Apply RoPE in-place to the last dimension of ``x``.

    Uses the fused Triton kernel by default (fp32 math in registers, single
    pass, in-place). Set ``SGLANG_DSV4_ROPE_TRITON=0`` to fall back to eager.
    """
    if os.environ.get("SGLANG_DSV4_ROPE_TRITON", "1") == "1" and x.is_cuda:
        from sglang.srt.models.deepseek_v4_rope import apply_rotary_emb_triton
        if freqs_cis.device != x.device:
            raise RuntimeError(
                "DeepSeek V4 RoPE freqs_cis must already be on the activation device; "
                f"got freqs_cis.device={freqs_cis.device}, x.device={x.device}."
            )
        return apply_rotary_emb_triton(x, freqs_cis, inverse=inverse)
    y = x
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if freqs_cis.device != x.device:
        raise RuntimeError(
            "DeepSeek V4 RoPE freqs_cis must already be on the activation device; "
            f"got freqs_cis.device={freqs_cis.device}, x.device={x.device}."
        )
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2).to(dtype=x.dtype)
    y.copy_(x_out)
    return y


def _apply_rotary_emb_decode(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
):
    """Apply RoPE for graph decode with per-batch dynamic positions.

    `x` is `[b, 1, ..., rope_dim]`; `freqs_cis` is `[b, rope_dim / 2]`.
    The regular helper is sequence-position based and broadcasts the same
    position across the batch, which would freeze CUDA graph replay to the
    capture-time sequence length. Uses the fused Triton kernel by default;
    set ``SGLANG_DSV4_ROPE_TRITON=0`` to fall back to eager.
    """
    if os.environ.get("SGLANG_DSV4_ROPE_TRITON", "1") == "1" and x.is_cuda:
        from sglang.srt.models.deepseek_v4_rope import apply_rotary_emb_decode_triton
        if freqs_cis.device != x.device:
            raise RuntimeError(
                "DeepSeek V4 decode RoPE freqs_cis must already be on the activation device; "
                f"got freqs_cis.device={freqs_cis.device}, x.device={x.device}."
            )
        return apply_rotary_emb_decode_triton(x, freqs_cis, inverse=inverse)
    y = x
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if freqs_cis.device != x.device:
        raise RuntimeError(
            "DeepSeek V4 decode RoPE freqs_cis must already be on the activation device; "
            f"got freqs_cis.device={freqs_cis.device}, x.device={x.device}."
        )
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(x_complex.size(0), 1, x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(
            x_complex.size(0), 1, 1, x_complex.size(-1)
        )
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2).to(dtype=x.dtype)
    y.copy_(x_out)
    return y


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used by the DeepSeek V4 indexer FP4 path."""
    from fast_hadamard_transform import hadamard_transform

    assert x.dtype == torch.bfloat16, "DeepSeek V4 rotation currently only supports bf16"
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


def _maybe_fp8_simulate_qat(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """FP8 fused quant+dequant helper.

    Gated on ``DSV4_USE_KV_QAT``; default ``"1"`` matches the official inference
    and reference implementation. ``"0"`` skips smoke tests with
    non-conforming dims.
    Callers must consume the return value: some kernels return an out-of-place
    tensor instead of mutating the input view.
    """
    if os.environ.get("DSV4_USE_KV_QAT", "1") == "0":
        return x
    return deepseek_v4_kernels.act_quant(
        x, block_size, _DEFAULT_SCALE_FMT, _DEFAULT_SCALE_DTYPE, True
    )


def _merge_fp8_qat_non_rope(
    kv: torch.Tensor,
    kv_non_rope: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if kv_non_rope.data_ptr() == kv.data_ptr():
        return kv
    return torch.cat([kv_non_rope, kv[..., -rope_dim:]], dim=-1)


def _maybe_fp4_simulate_qat(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """FP4 fused quant+dequant helper for indexer Q and compressor tensors."""
    if os.environ.get("DSV4_USE_KV_QAT", "1") == "0":
        return x
    return deepseek_v4_kernels.fp4_act_quant(x, block_size, True)


def _overlap_transform(
    tensor: torch.Tensor,
    *,
    compress_ratio: int,
    head_dim: int,
    value=0,
) -> torch.Tensor:
    batch, groups, _, _ = tensor.size()
    new_tensor = tensor.new_full((batch, groups, 2 * compress_ratio, head_dim), value)
    new_tensor[:, :, compress_ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :compress_ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


def _get_q_positions(seqlen: int, *, device) -> torch.Tensor:
    return torch.arange(0, seqlen, device=device)


def _maybe_canonicalize_compress_topk_order(topk_idxs: torch.Tensor) -> torch.Tensor:
    if os.environ.get("SGLANG_DSV4_CANONICAL_COMPRESS_TOPK_ORDER", "0") != "1":
        return topk_idxs
    valid = topk_idxs >= 0
    sentinel = torch.full_like(topk_idxs, torch.iinfo(topk_idxs.dtype).max)
    sort_keys = torch.where(valid, topk_idxs, sentinel)
    sorted_keys = torch.sort(sort_keys, dim=-1).values
    return torch.where(sorted_keys == sentinel, -1, sorted_keys)


# --- Stateful top-k index helpers. ---
# The stateless helpers above operate on a tensor of q-positions (prefill only).
# The stateful variants take (window_size/ratio, bsz, seqlen, start_pos) and
# branch on `start_pos == 0` (prefill) vs `> 0` (decode of seqlen=1).
def _get_window_topk_idxs_stateful(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    *,
    device,
    pad_short_to_window: bool = False,
) -> torch.Tensor:
    if start_pos >= window_size - 1:
        wrap = start_pos % window_size
        matrix = torch.cat(
            [
                torch.arange(wrap + 1, window_size, device=device),
                torch.arange(0, wrap + 1, device=device),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = torch.nn.functional.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        window_topk = (
            window_size if pad_short_to_window and seqlen < window_size
            else min(seqlen, window_size)
        )
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            window_topk, device=device
        )
        matrix = torch.where(matrix > base, -1, matrix)
    # 1D matrix at decode (-> [bsz, 1, window_size] via leading-dim expand);
    # 2D matrix at prefill (-> [bsz, seqlen, k_count]).
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _get_compress_topk_idxs_stateful(
    ratio: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    *,
    device,
) -> torch.Tensor:
    if start_pos > 0:
        matrix = (
            torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
        )
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= (
            torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        )
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# ---------------------------------------------------------------------------
# Compressor + Indexer.
# ---------------------------------------------------------------------------


class DeepSeekV4Compressor(nn.Module):
    """Compressed-KV builder for C4 (overlap=True) and C128 layers.

    Mirrors the official inference ``Compressor``. The native checkpoint may
    store ``wkv.weight``, ``wgate.weight``, and ``norm.weight`` as bf16, but
    inference keeps them in fp32 in-flight and loads into that dtype.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        norm_eps: float,
        rotate: bool = False,
        rope_theta: float = 160000.0,
        rotary_scaling_factor: float = 16.0,
        original_max_position_embeddings: int = 65536,
        beta_fast: int = 32,
        beta_slow: int = 1,
        max_batch_size: int = 1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        # Overlap doubles the projection width.
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        coeff = 1 + int(self.overlap)

        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, coeff * head_dim, dtype=torch.float32)
        )
        self.wkv = DeepSeekV4Linear(dim, coeff * head_dim, bias=False, dtype=torch.float32)
        self.wgate = DeepSeekV4Linear(dim, coeff * head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, eps=norm_eps, dtype=torch.float32)

        # Stateful-decode buffers:
        # `kv_state` is the running fp32 accumulator over the current compress
        # window; `score_state` is the parallel score accumulator initialised
        # to -inf so unfilled slots contribute 0 weight under softmax.
        # Shapes match `inference/model.py:303-304`. Stored as `nn.Parameter`
        # (with `requires_grad=False`) so TMS recognises them as parameters
        # for offload — they aren't trainable, just stateful per-request KV.
        self.kv_state = nn.Parameter(
            torch.zeros(
                max_batch_size,
                coeff * compress_ratio,
                coeff * head_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.score_state = nn.Parameter(
            torch.full(
                (max_batch_size, coeff * compress_ratio, coeff * head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "_batch_positions",
            torch.arange(max_batch_size, dtype=torch.int64),
            persistent=False,
        )
        # Compressed-tier write target — assigned lazily from the parent
        # Attention's `kv_cache[:, win:]` view (or from an Indexer's own
        # `kv_cache` when the Compressor is owned by an Indexer). Mirrors
        # `inference/model.py:300`.
        self.kv_cache: Optional[torch.Tensor] = None

        # RoPE freqs for the compressed-KV path (uses compress_rope_theta).
        freqs_cis = _wrapped_precompute_freqs_cis(
            rope_head_dim=rope_head_dim,
            base=rope_theta,
            rotary_scaling_factor=rotary_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _canonical_project(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if os.environ.get("SGLANG_DSV4_CANONICAL_COMPRESSOR_LINEAR", "0") != "1":
            return linear(x)
        weight = linear.weight
        out = _deepseek_v4_canonical_linear(
            x,
            weight,
            "SGLANG_DSV4_CANONICAL_COMPRESSOR_LINEAR_IMPL",
            default="fixed_order",
        )
        if linear.bias is not None:
            out = out + linear.bias
        return out

    # ---- Inner forward: input is `[b, s, d]`. ----
    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen_local, _ = x.size()
        ratio = self.compress_ratio
        dtype = x.dtype
        # Mirror official `Compressor.forward` short-circuit: when seqlen < ratio
        # there's no full compression group; return an empty `[batch, 0, head_dim]`
        # so the attention call site's `torch.cat([kv, kv_compress], dim=1)` is well-defined.
        if seqlen_local < ratio:
            return x.new_zeros((batch, 0, self.head_dim), dtype=dtype)
        # Stateless full-prefill has no compression state to carry a partial
        # tail into the next decode step, so match Megatron / official full
        # prefill and only emit complete compression groups.
        seqlen_compress = seqlen_local - (seqlen_local % ratio)
        if seqlen_compress == 0:
            return x.new_zeros((batch, 0, self.head_dim), dtype=dtype)

        x_fp32 = x.float()
        kv = self._canonical_project(self.wkv, x_fp32)
        score = self._canonical_project(self.wgate, x_fp32)
        if seqlen_compress != seqlen_local:
            kv = kv[:, :seqlen_compress]
            score = score[:, :seqlen_compress]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if self.overlap:
            kv = _overlap_transform(
                kv, compress_ratio=ratio, head_dim=self.head_dim, value=0
            )
            score = _overlap_transform(
                score, compress_ratio=ratio, head_dim=self.head_dim, value=float("-inf")
            )

        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))

        freqs_cis = self.freqs_cis[:seqlen_compress:ratio]
        _apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = _rotate_activation(kv)
            kv = _maybe_fp4_simulate_qat(kv, 32)
        else:
            kv = kv.clone()
            kv_non_rope = _maybe_fp8_simulate_qat(
                kv[..., : -self.rope_head_dim], 64
            )
            kv = torch.cat([kv_non_rope, kv[..., -self.rope_head_dim :]], dim=-1)

        return kv

    # ---- Outer forward: input is `[s, b, d]` (Megatron-LM convention). ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bsd = x.permute(1, 0, 2).contiguous()
        k_bsd = self.forward_raw(x_bsd)
        return k_bsd.permute(1, 0, 2).contiguous()

    # ---- Stateful forward. Input is `[bsz, seqlen, dim]`. Returns the
    # compressed-token tensor when a compression group finished this call,
    # else None. The returned tensor's shape is
    # `[bsz, seqlen // ratio, head_dim]` at prefill or `[bsz, 1, head_dim]`
    # at a decode boundary.
    #
    # Side effects (all on this Compressor's own buffers):
    #   * updates `kv_state` / `score_state` (running compression accumulator);
    #   * writes the finished compressed token into `self.kv_cache` (which is
    #     a view of the parent Attention's `kv_cache[:, win:]`, or, if this
    #     Compressor is owned by an Indexer, the Indexer's `kv_cache`).
    #
    # Caller contract: `self.kv_cache` MUST be assigned before the first call
    # (the parent does this lazily at the start of its own `forward_stateful`).
    def forward_stateful(
        self,
        x: torch.Tensor,
        start_pos: int,
        req_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        assert self.kv_cache is not None, (
            "DeepSeekV4Compressor.forward_stateful: parent must assign "
            "`compressor.kv_cache` before invoking forward_stateful "
            "(mirrors `inference/model.py:317`)."
        )
        bsz, seqlen, _ = x.size()
        # Per-request slot range. The scheduler dispatch hands us a specific
        # `req_idx` so each in-flight request has its own Compressor state.
        r0, r1 = req_idx, req_idx + bsz
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype

        # bf16-weight wkv/wgate matmul on fp32 input. Some checkpoint paths
        # store these weights as bf16 on disk while the reference inference
        # path keeps fp32 in memory; both produce fp32 projection outputs.
        x = x.float()
        kv = self._canonical_project(self.wkv, x)
        score = self._canonical_project(self.wgate, x)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[r0:r1, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[r0:r1, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, self.kv_state[r0:r1, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[r0:r1, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = _overlap_transform(
                    kv, compress_ratio=ratio, head_dim=d, value=0
                )
                score = _overlap_transform(
                    score, compress_ratio=ratio, head_dim=d, value=float("-inf")
                )
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[r0:r1, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[r0:r1, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[r0:r1, :ratio, :d],
                            self.kv_state[r0:r1, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[r0:r1, :ratio, :d],
                            self.score_state[r0:r1, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    self.kv_state[r0:r1, :ratio] = self.kv_state[r0:r1, ratio:]
                    self.score_state[r0:r1, :ratio] = self.score_state[r0:r1, ratio:]
            else:
                self.kv_state[r0:r1, start_pos % ratio] = kv.squeeze(1)
                self.score_state[r0:r1, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[r0:r1] * self.score_state[r0:r1].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)

        if self.rotate:
            kv = _rotate_activation(kv)
            kv = _maybe_fp4_simulate_qat(kv, 32)
        else:
            kv_non_rope = _maybe_fp8_simulate_qat(kv[..., :-rd], 64)
            kv = torch.cat([kv_non_rope, kv[..., -rd:]], dim=-1)

        if start_pos == 0:
            self.kv_cache[r0:r1, : seqlen // ratio] = kv
        else:
            self.kv_cache[r0:r1, start_pos // ratio] = kv.squeeze(1)
        return kv

    def forward_stateful_cuda_graph_decode(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        req_indices: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode-only compressor update with tensor metadata.

        CUDA graph replay updates `start_pos` and `req_indices` in-place before
        replay; this path keeps those values as tensors so the graph does not
        bake in capture-time Python scalars.
        """
        assert self.kv_cache is not None, (
            "DeepSeekV4Compressor.forward_stateful_cuda_graph_decode: parent "
            "must assign `compressor.kv_cache` before invoking the graph path."
        )
        bsz, seqlen, _ = x.size()
        if seqlen != 1:
            raise RuntimeError("DeepSeek V4 compressor CUDA graph path is decode-only.")

        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype
        device = x.device
        rows = self._batch_positions[:bsz]
        if rows.device != device:
            raise RuntimeError(
                "DeepSeek V4 compressor batch positions must already be on the decode device; "
                f"got rows.device={rows.device}, decode device={device}."
            )
        req_indices = _deepseek_v4_require_metadata_tensor(
            req_indices, device=device, dtype=torch.int64, name="req_indices"
        )
        start_pos = _deepseek_v4_require_metadata_tensor(
            start_pos, device=device, dtype=torch.int64, name="start_pos"
        )
        active = _deepseek_v4_require_metadata_tensor(
            active_mask, device=device, dtype=torch.bool, name="active_mask"
        )
        phase = start_pos.remainder(ratio)

        x = x.float()
        kv = self._canonical_project(self.wkv, x)
        score = self._canonical_project(self.wgate, x)
        score = score + self.ape.index_select(0, phase).unsqueeze(1)
        kv_token = kv.squeeze(1)
        score_token = score.squeeze(1)
        should_compress = phase.eq(ratio - 1) & active

        kv_state = self.kv_state.index_select(0, req_indices).clone()
        score_state = self.score_state.index_select(0, req_indices).clone()
        if overlap:
            write_idx = ratio + phase
            old_kv = kv_state[rows, write_idx]
            old_score = score_state[rows, write_idx]
            kv_state[rows, write_idx] = torch.where(
                active.unsqueeze(1), kv_token, old_kv
            )
            score_state[rows, write_idx] = torch.where(
                active.unsqueeze(1), score_token, old_score
            )
            kv_for_reduce = torch.cat(
                [kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], dim=1
            )
            score_for_reduce = torch.cat(
                [score_state[:, :ratio, :d], score_state[:, ratio:, d:]], dim=1
            )
            kv_out = (kv_for_reduce * score_for_reduce.softmax(dim=1)).sum(
                dim=1, keepdim=True
            )
            shifted_kv = kv_state.clone()
            shifted_score = score_state.clone()
            shifted_kv[:, :ratio] = kv_state[:, ratio:]
            shifted_score[:, :ratio] = score_state[:, ratio:]
            kv_state = torch.where(
                should_compress.view(bsz, 1, 1), shifted_kv, kv_state
            )
            score_state = torch.where(
                should_compress.view(bsz, 1, 1), shifted_score, score_state
            )
        else:
            write_idx = phase
            old_kv = kv_state[rows, write_idx]
            old_score = score_state[rows, write_idx]
            kv_state[rows, write_idx] = torch.where(
                active.unsqueeze(1), kv_token, old_kv
            )
            score_state[rows, write_idx] = torch.where(
                active.unsqueeze(1), score_token, old_score
            )
            kv_out = (kv_state * score_state.softmax(dim=1)).sum(
                dim=1, keepdim=True
            )

        self.kv_state.index_copy_(0, req_indices, kv_state)
        self.score_state.index_copy_(0, req_indices, score_state)

        kv_out = self.norm(kv_out.to(dtype))
        rope_idx = torch.clamp(start_pos + 1 - ratio, min=0)
        freqs_cis = self.freqs_cis.index_select(0, rope_idx)
        _apply_rotary_emb_decode(kv_out[..., -rd:], freqs_cis)

        if self.rotate:
            kv_out = _rotate_activation(kv_out)
            kv_out = _maybe_fp4_simulate_qat(kv_out, 32)
        else:
            kv_non_rope = _maybe_fp8_simulate_qat(kv_out[..., :-rd], 64)
            kv_out = _merge_fp8_qat_non_rope(
                kv_out,
                kv_non_rope,
                rd,
            )

        compressed_idx = start_pos // ratio
        _deepseek_v4_update_cache_slots_(
            self.kv_cache,
            req_indices,
            compressed_idx,
            kv_out.squeeze(1),
            should_compress,
        )
        return kv_out


class _DeepSeekV4Indexer(nn.Module):
    """Compressed-KV top-k selector (only fires at C4 layers).

    The native checkpoint exposes ``wq_b`` and ``weights_proj`` linears. In
    SGLang they are named ``linear_wq_b`` and ``linear_weights_proj`` to keep
    module names unambiguous.
    """

    def __init__(
        self,
        dim: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_head_dim: int,
        norm_eps: float,
        compress_rope_theta: float = 160000.0,
        rotary_scaling_factor: float = 16.0,
        original_max_position_embeddings: int = 65536,
        beta_fast: int = 32,
        beta_slow: int = 1,
        max_batch_size: int = 1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = 4
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.linear_wq_b = DeepSeekV4Linear(
            q_lora_rank,
            index_n_heads * index_head_dim,
            bias=False,
            dtype=torch.float8_e4m3fn,
        )
        self.linear_weights_proj = DeepSeekV4Linear(
            dim,
            index_n_heads,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.compressor = DeepSeekV4Compressor(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=self.compress_ratio,
            norm_eps=norm_eps,
            rotate=True,
            rope_theta=compress_rope_theta,
            rotary_scaling_factor=rotary_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Stateful-decode buffer. The Indexer owns its own compressed-KV ring
        # separate from the parent Attention's ring, and the inner
        # `compressor.kv_cache` is assigned lazily at forward time. Stored as
        # `nn.Parameter` with `requires_grad=False` so memory management treats
        # it as an offloadable parameter.
        self.kv_cache = nn.Parameter(
            torch.zeros(
                max_batch_size,
                max_seq_len // self.compress_ratio,
                index_head_dim,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        freqs_cis = _wrapped_precompute_freqs_cis(
            rope_head_dim=rope_head_dim,
            base=compress_rope_theta,
            rotary_scaling_factor=rotary_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        """Inputs are `[s, b, d]` and `[s, b, q_lora_rank]` (Megatron-LM convention).

        Returns `[b, s, topk]` int64 indices into the compressor KV stream.
        """
        seqlen, batch, _ = x.size()
        q = self.linear_wq_b(qr).reshape(
            seqlen, batch, self.index_n_heads, self.index_head_dim
        )

        freqs_cis = self.freqs_cis[:seqlen]
        q = q.clone().permute(1, 0, 2, 3).contiguous()
        _apply_rotary_emb(q[..., -self.rope_head_dim :], freqs_cis)
        q = q.permute(1, 0, 2, 3).contiguous()

        q = _rotate_activation(q)
        q = _maybe_fp4_simulate_qat(q, block_size=32)

        k = self.compressor(x)
        weights = self.linear_weights_proj(x) * (self.index_n_heads ** -0.5) * (self.index_head_dim ** -0.5)

        scores = torch.einsum("sbhd,tbd->bsth", q, k)
        scores = (F.relu(scores) * weights.permute(1, 0, 2).unsqueeze(2)).sum(dim=-1)

        q_positions = _get_q_positions(seqlen, device=x.device)
        kv_positions = torch.arange(k.shape[0], device=x.device)
        valid_end = (q_positions + 1).unsqueeze(1) // self.compress_ratio
        scores = scores.masked_fill(
            kv_positions.view(1, 1, -1) >= valid_end.view(1, seqlen, 1), float("-inf")
        )

        topk_k = min(self.index_topk, scores.size(-1))
        return scores.topk(topk_k, dim=-1).indices

    # ---- Stateful Indexer forward. Input is `[bsz, seqlen, dim]` and
    # `[bsz, seqlen, q_lora_rank]`. Returns
    # `topk_idxs` of shape `[bsz, seqlen, topk]` (prefill) or `[bsz, 1, topk]`
    # (decode), already offset by `offset` (for downstream concat with the
    # vanilla-window topk indices).
    #
    # Side effects: drives the inner Compressor's stateful forward, which
    # updates `self.kv_cache` (the Indexer's own compressed ring) at compress
    # boundaries.
    def forward_stateful(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
        req_idx: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        # Per-request slot range for state buffers.
        r0, r1 = req_idx, req_idx + bsz
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        # Lazy-link the inner Compressor's kv_cache to the Indexer's own
        # compressed ring. Mirrors `inference/model.py:408-410`. Skipped if
        # already wired (idempotent).
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache

        # Q path. `wq_b(qr)` produces `[bsz, seqlen, n_heads * head_dim]`;
        # reshape to `[bsz, seqlen, n_heads, head_dim]`. RoPE is applied to
        # the rope-dim slice in place.
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        q = self.linear_wq_b(qr)
        q = q.unflatten(-1, (self.index_n_heads, self.index_head_dim))
        # Clone before in-place rotary so we don't poison upstream qr.
        q = q.clone()
        _apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = _rotate_activation(q)
        q = _maybe_fp4_simulate_qat(q, block_size=32)

        # Drive the compressor's state update + kv_cache write at our slot.
        self.compressor.forward_stateful(x, start_pos, req_idx=req_idx)

        # Score path. The official reads back `self.kv_cache[:bsz, :end_pos // ratio]`
        # for the einsum; we do the same. The stateless path uses the just-returned
        # `k` from `self.compressor(x)` instead — which is equivalent ONLY at
        # prefill (where every compressed token is freshly produced). For
        # decode, the kv_cache holds the entire history so we must read it.
        # Score normalisation matches the stateless `forward` and the
        # official Indexer: `index_head_dim ** -0.5 * index_n_heads ** -0.5`.
        weights = self.linear_weights_proj(x) * (self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[r0:r1, : end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        if start_pos == 0:
            # Prefill mask: a query at position s can only see compressed
            # tokens at index < (s+1) // ratio.
            mask = torch.arange(
                seqlen // ratio, device=x.device
            ).repeat(seqlen, 1) >= torch.arange(
                1, seqlen + 1, device=x.device
            ).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0.0)

        topk_idxs = index_score.topk(
            min(self.index_topk, end_pos // ratio), dim=-1
        )[1]

        if start_pos == 0:
            mask = topk_idxs >= (
                torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

    def forward_stateful_cuda_graph_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: torch.Tensor,
        offset: int,
        req_indices: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        if seqlen != 1:
            raise RuntimeError("DeepSeek V4 indexer CUDA graph path is decode-only.")

        device = x.device
        req_indices = _deepseek_v4_require_metadata_tensor(
            req_indices, device=device, dtype=torch.int64, name="req_indices"
        )
        start_pos = _deepseek_v4_require_metadata_tensor(
            start_pos, device=device, dtype=torch.int64, name="start_pos"
        )
        active = _deepseek_v4_require_metadata_tensor(
            active_mask, device=device, dtype=torch.bool, name="active_mask"
        )
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        group_count = (start_pos + 1) // ratio
        max_groups = self.kv_cache.shape[1]

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache

        freqs_cis = self.freqs_cis.index_select(0, start_pos)
        q = self.linear_wq_b(qr)
        q = q.unflatten(-1, (self.index_n_heads, self.index_head_dim))
        _apply_rotary_emb_decode(q[..., -rd:], freqs_cis)
        q = _rotate_activation(q)
        q = _maybe_fp4_simulate_qat(q, block_size=32)

        self.compressor.forward_stateful_cuda_graph_decode(
            x,
            start_pos,
            req_indices=req_indices,
            active_mask=active,
        )

        weights = self.linear_weights_proj(x) * (self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5)

        from sglang.srt.models.deepseek_v4_decode_kernels import (
            deepseek_v4_indexer_score_decode,
        )

        index_score = deepseek_v4_indexer_score_decode(
            q,
            self.kv_cache,
            weights,
            req_indices,
            group_count,
            active,
        )
        topk_k = min(self.index_topk, max_groups)
        topk_idxs = index_score.topk(topk_k, dim=-1)[1]
        topk_valid = topk_idxs < group_count.view(bsz, 1, 1)
        topk_valid = topk_valid & active.view(bsz, 1, 1)
        return torch.where(topk_valid, topk_idxs + int(offset), -1)


# ---------------------------------------------------------------------------
# Attention.
# ---------------------------------------------------------------------------


class DeepSeekV4Attention(nn.Module):
    """V4 native attention block (MLA + sparse window + compressor + indexer).

    The native-quant linears follow the official inference layout:

      * ``wq_a``: FP8 ``[dim -> q_lora_rank]``
      * ``wq_b``: FP8 ``[q_lora_rank -> n_heads*head_dim]`` (column-parallel)
      * ``wkv``:  FP8 ``[dim -> head_dim]``
      * ``wo_a``: bf16 ``[(n_heads*head_dim)/n_groups -> n_groups*o_lora_rank]``
                  (column-parallel)
      * ``wo_b``: FP8 ``[n_groups*o_lora_rank -> dim]`` (row-parallel)

    Per-layer ``compress_ratio`` (from ``compress_ratios[layer_id]``) gates
    the optional ``compressor`` (built when ``compress_ratio > 0``) and
    ``indexer`` (built only when ``compress_ratio == 4``). ``compress_ratio``
    of 128 has compressor but no indexer. Layer 0/1/... with ratio 0 have
    neither — the ``compressor``/``indexer`` attributes are simply not
    constructed (and not registered as None children either; we keep the
    audit stable by *not* declaring them on ratio-0 layers).
    """

    def __init__(
        self,
        config,
        layer_id: int,
        tp_group: Any = None,
    ):
        super().__init__()
        if tp_group is None:
            try:
                tp_group = get_attn_tp_group()
            except AssertionError:
                tp_group = None
        self.tp_group = tp_group
        self.tp_size = _tp_group_world(tp_group)
        self.layer_id = layer_id
        self.dim = config.dim
        self.total_n_heads = config.n_heads
        if self.total_n_heads % self.tp_size != 0:
            raise ValueError(
                "DeepSeek V4 attention heads must divide evenly across attention TP: "
                f"n_heads={self.total_n_heads} tp_size={self.tp_size}"
            )
        self.n_heads = self.total_n_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.total_n_groups = config.o_groups
        if self.total_n_groups % self.tp_size != 0:
            raise ValueError(
                "DeepSeek V4 O-projection groups must divide evenly across attention TP: "
                f"o_groups={self.total_n_groups} tp_size={self.tp_size}"
            )
        self.n_groups = self.total_n_groups // self.tp_size
        self.window_size = config.window_size
        self.eps = config.norm_eps
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = (
            config.compress_ratios[layer_id] if config.compress_ratios else 0
        )
        # Thread inference-time capacity from the V4 native config so the
        # per-layer `kv_cache` buffer and compressor decode-state buffers are
        # sized at construction time.
        self.max_batch_size = getattr(config, "max_batch_size", 1)
        self.max_seq_len = getattr(config, "max_seq_len", 256)

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        _mark_deepseek_v4_tp_shard(self.attn_sink, 0, tp_group)

        self.wq_a = DeepSeekV4Linear(
            self.dim, self.q_lora_rank, bias=False, dtype=torch.float8_e4m3fn
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps, dtype=torch.float32)
        self.wq_b = DeepSeekV4ColumnParallelLinear(
            self.q_lora_rank,
            self.total_n_heads * self.head_dim,
            bias=False,
            dtype=torch.float8_e4m3fn,
            tp_group=tp_group,
        )
        self.wkv = DeepSeekV4Linear(
            self.dim, self.head_dim, bias=False, dtype=torch.float8_e4m3fn
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps, dtype=torch.float32)

        self.wo_a = DeepSeekV4ColumnParallelLinear(
            self.total_n_heads * self.head_dim // self.total_n_groups,
            self.total_n_groups * self.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
            tp_group=tp_group,
        )
        self.wo_b = DeepSeekV4RowParallelLinear(
            self.total_n_groups * self.o_lora_rank,
            self.dim,
            bias=False,
            dtype=torch.float8_e4m3fn,
            tp_group=tp_group,
        )

        compress_rope_theta = getattr(config, "compress_rope_theta", 160000.0)
        rope_theta = getattr(config, "rope_theta", 10000.0)
        rotary_scaling_factor = getattr(config, "rope_factor", 16.0)
        original_max_position_embeddings = getattr(config, "original_seq_len", 0)
        beta_fast = getattr(config, "beta_fast", 32)
        beta_slow = getattr(config, "beta_slow", 1)

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = DeepSeekV4Compressor(
                dim=self.dim,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                norm_eps=self.eps,
                rotate=False,
                rope_theta=compress_rope_theta,
                rotary_scaling_factor=rotary_scaling_factor,
                original_max_position_embeddings=original_max_position_embeddings,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
            )
            if self.compress_ratio == 4:
                self.indexer = _DeepSeekV4Indexer(
                    dim=self.dim,
                    q_lora_rank=self.q_lora_rank,
                    index_n_heads=config.index_n_heads,
                    index_head_dim=config.index_head_dim,
                    index_topk=config.index_topk,
                    rope_head_dim=self.rope_head_dim,
                    norm_eps=self.eps,
                    compress_rope_theta=compress_rope_theta,
                    rotary_scaling_factor=rotary_scaling_factor,
                    original_max_position_embeddings=original_max_position_embeddings,
                    beta_fast=beta_fast,
                    beta_slow=beta_slow,
                    max_batch_size=self.max_batch_size,
                    max_seq_len=self.max_seq_len,
                )

        # RoPE freqs for the dense path. Compressed layers use
        # `compress_rope_theta`; pure sliding-window layers use `rope_theta`.
        # YaRN remains enabled here to match the checkpoint schema used by the
        # native-quant weights.
        rope_base = compress_rope_theta if self.compress_ratio else rope_theta
        freqs_cis = _wrapped_precompute_freqs_cis(
            rope_head_dim=self.rope_head_dim,
            base=rope_base,
            rotary_scaling_factor=rotary_scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            yarn_disabled=False,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Stateful decode follows the official inference RoPE convention,
        # which disables YaRN on ratio-0 layers. At position 0 the dense and
        # stateful variants are identical; at later positions they diverge, so
        # decode keeps its own buffer.
        if self.compress_ratio:
            # ratio-4 / ratio-128 layers use the same freqs_cis in dense and
            # stateful paths. Alias to avoid duplicate memory.
            self.freqs_cis_stateful = self.freqs_cis
        else:
            freqs_cis_stateful = _wrapped_precompute_freqs_cis(
                rope_head_dim=self.rope_head_dim,
                base=rope_base,
                rotary_scaling_factor=rotary_scaling_factor,
                original_max_position_embeddings=original_max_position_embeddings,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                yarn_disabled=True,
            )
            self.register_buffer(
                "freqs_cis_stateful", freqs_cis_stateful, persistent=False
            )

        # Per-layer KV ring. The buffer holds the vanilla sliding window in
        # `[:, :win]` and the compressed history in `[:, win:]`. For ratio-0
        # layers the compressed slab is empty. Stored as `nn.Parameter` with
        # `requires_grad=False` so memory management treats it as offloadable.
        kv_cache_size = self.window_size + (
            (self.max_seq_len // self.compress_ratio) if self.compress_ratio else 0
        )
        self.kv_cache = nn.Parameter(
            torch.zeros(
                self.max_batch_size,
                kv_cache_size,
                self.head_dim,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        """DeepSeek V4 attention forward.

        The scheduler always supplies ``DeepSeekV4AttentionBackend``. This
        wrapper only dispatches to the stateful extend/decode implementations;
        q/kv construction, model-owned state updates, official TileLang
        sparse attention, and output projection all happen inside those paths.

        ``hidden_states`` is `[s, b, d]` (Megatron-LM convention). The
        block runs in `[b, s, ...]` internally and emits `[s, b, d]` to
        match the reference convention.
        """
        forward_batch = _deepseek_v4_require_forward_batch(
            forward_batch, "DeepSeekV4Attention.forward"
        )

        from sglang.srt.layers.attention.deepseek_v4_attention_backend import (
            DeepSeekV4AttentionBackend,
        )

        attn_backend = getattr(forward_batch, "attn_backend", None)
        assert isinstance(attn_backend, DeepSeekV4AttentionBackend), (
            "DeepSeekV4Attention.forward requires DeepSeekV4AttentionBackend; "
            f"got {type(attn_backend).__name__}."
        )

        meta = attn_backend.metadata
        if getattr(meta, "start_pos", None) is None:
            raise RuntimeError(
                "DeepSeekV4Attention.forward requires DeepSeek V4 metadata with "
                "`start_pos`; call DeepSeekV4AttentionBackend.init_forward_metadata "
                "before model forward."
            )
        if meta.is_decode:
            return attn_backend.forward_decode(
                q=None,
                k=None,
                v=None,
                layer=self,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
            )
        return attn_backend.forward_extend(
            q=None,
            k=None,
            v=None,
            layer=self,
            forward_batch=forward_batch,
            hidden_states=hidden_states,
        )

    # ---- Stateful attention forward. `hidden_states` is `[s, b, d]`
    # (Megatron convention); the body works in
    # `[b, s, d]` and returns `[s, b, d]`. `start_pos` is the global decode
    # position of the first new token in this call (0 at prefill;
    # prompt_len + k - 1 at decode step k).
    #
    # The path mirrors the official Attention.forward 1:1: it owns its own
    # `kv_cache` ring (vanilla window in `[:, :win]`, compressed history in
    # `[:, win:]`), calls the stateful Compressor + Indexer, and runs the
    # official TileLang `deepseek_v4_kernels.sparse_attn`.
    def forward_stateful(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        req_idx: int = 0,
    ) -> torch.Tensor:
        x = hidden_states.permute(1, 0, 2).contiguous()
        bsz, seqlen, _ = x.size()
        if start_pos > 0 and seqlen > 1:
            raise RuntimeError(
                "DeepSeek V4 stateful attention only supports multi-token "
                "suffix replay at the full-model level. Call "
                "DeepSeekV4Model.forward_stateful so HC/MLP/layer state "
                "advance token by token."
            )

        # Per-request slot range so each in-flight request has its own
        # kv_cache / compressor / indexer slot.
        r0, r1 = req_idx, req_idx + bsz
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        # Use the stateful-only freqs_cis (YaRN-off on C0 layers, matching
        # the official inference). See __init__ comment for why this differs
        # from `self.freqs_cis`.
        freqs_cis = self.freqs_cis_stateful[start_pos : start_pos + seqlen]

        # Lazy-link the compressor's compressed-tier kv_cache to the
        # back-half of this layer's vanilla+compressed ring.
        # Mirrors `inference/model.py:490-494`.
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]

        # Q path. `qr` is the post-q_norm latent that the indexer also reads.
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = deepseek_v4_q_rmsnorm(q, self.eps)
        q = q.clone()
        _apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV (vanilla / window) path.
        kv = self.kv_norm(self.wkv(x)).clone()
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)
        kv_non_rope = _maybe_fp8_simulate_qat(kv[..., :-rd], 64)
        kv = torch.cat([kv_non_rope, kv[..., -rd:]], dim=-1)

        # topk_idxs assembly: window + (optional) compressed.
        use_fixed_short_prefill_layout = (
            ratio
            and start_pos == 0
            and seqlen < win
            and os.environ.get("SGLANG_DSV4_PREFILL_FIXED_WINDOW_LAYOUT", "0") == "1"
        )
        topk_idxs = _get_window_topk_idxs_stateful(
            win,
            bsz,
            seqlen,
            start_pos,
            device=x.device,
            pad_short_to_window=use_fixed_short_prefill_layout,
        )
        if ratio:
            offset = win if (start_pos > 0 or use_fixed_short_prefill_layout) else kv.size(1)
            if self.indexer is not None:
                compress_topk_idxs = self.indexer.forward_stateful(
                    x, qr, start_pos, offset, req_idx=req_idx
                )
            else:
                compress_topk_idxs = _get_compress_topk_idxs_stateful(
                    ratio, bsz, seqlen, start_pos, offset, device=x.device
                )
            compress_topk_idxs = _maybe_canonicalize_compress_topk_order(
                compress_topk_idxs
            )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Vanilla-window write into this layer's kv_cache + sparse_attn call.
        # Prefill writes the trailing `min(seqlen, win)` tokens; decode writes
        # the new token at the circular slot `start_pos % win`.
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[r0:r1, :seqlen] = kv
            else:
                cutoff = seqlen % win
                # Split kv[:, -win:] into two pieces and place them so that
                # the circular index `start_pos % win` (0 here, advancing as
                # we decode) is consistent post-prefill. Mirrors
                # `inference/model.py:521-523`.
                tail = kv[:, -win:]
                head, tail2 = tail.split([win - cutoff, cutoff], dim=1)
                self.kv_cache[r0:r1, cutoff:win] = head
                self.kv_cache[r0:r1, :cutoff] = tail2
            if ratio:
                kv_compress = self.compressor.forward_stateful(
                    x, start_pos, req_idx=req_idx
                )
                if use_fixed_short_prefill_layout and kv.size(1) < win:
                    kv = F.pad(kv, (0, 0, 0, win - kv.size(1)))
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            out = deepseek_v4_kernels.sparse_attn(
                q, kv, self.attn_sink, topk_idxs, self.softmax_scale
            )
        else:
            # Decode of a single new token (seqlen == 1).
            self.kv_cache[r0:r1, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor.forward_stateful(x, start_pos, req_idx=req_idx)
            out = deepseek_v4_kernels.sparse_attn(
                q,
                self.kv_cache[r0:r1],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )

        _apply_rotary_emb(out[..., -rd:], freqs_cis, inverse=True)

        # O projection.
        # OFT input-rotation hook for wo_a — see prefill forward above
        # for rationale; the einsum-bypass reads weight directly so the
        # hook is required to apply non-identity wo_a adapters at decode.
        out = out.flatten(-2)
        out = self.wo_a.apply_input_rotation(out)
        out = out.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out, wo_a)
        out = out.flatten(2)
        x = self.wo_b(out)
        return x.permute(1, 0, 2).contiguous()

    def forward_stateful_cuda_graph(
        self,
        hidden_states: torch.Tensor,
        *,
        start_pos: torch.Tensor,
        req_indices: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode-only stateful attention with tensor metadata.

        This mirrors `forward_stateful(..., start_pos:int, req_idx:int)` for
        decode, but keeps the absolute position and request slot as tensors so
        CUDA graph replay can update them without recapturing.
        """
        x = hidden_states.permute(1, 0, 2).contiguous()
        bsz, seqlen, _ = x.size()
        if seqlen != 1:
            raise RuntimeError("DeepSeek V4 CUDA graph attention is decode-only.")

        device = x.device
        start_pos = _deepseek_v4_require_metadata_tensor(
            start_pos, device=device, dtype=torch.int64, name="start_pos"
        )
        req_indices = _deepseek_v4_require_metadata_tensor(
            req_indices, device=device, dtype=torch.int64, name="req_indices"
        )
        active = _deepseek_v4_require_metadata_tensor(
            active_mask, device=device, dtype=torch.bool, name="active_mask"
        )
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        freqs_cis = self.freqs_cis_stateful.index_select(0, start_pos)

        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]

        qr = self.q_norm(self.wq_a(x))
        q_raw = self.wq_b(qr)
        q = q_raw.unflatten(-1, (self.n_heads, self.head_dim))
        q = deepseek_v4_q_rmsnorm(q, self.eps)
        _apply_rotary_emb_decode(q[..., -rd:], freqs_cis)

        kv = self.kv_norm(self.wkv(x))
        _apply_rotary_emb_decode(kv[..., -rd:], freqs_cis)
        kv_non_rope = _maybe_fp8_simulate_qat(kv[..., :-rd], 64)
        kv = _merge_fp8_qat_non_rope(kv, kv_non_rope, rd)

        from sglang.srt.models.deepseek_v4_decode_kernels import (
            deepseek_v4_window_topk_decode,
        )

        topk_idxs = deepseek_v4_window_topk_decode(win, start_pos, active)
        if ratio:
            offset = win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer.forward_stateful_cuda_graph_decode(
                    x,
                    qr,
                    start_pos,
                    offset,
                    req_indices=req_indices,
                    active_mask=active,
                )
            else:
                max_groups = self.kv_cache.shape[1] - win
                from sglang.srt.models.deepseek_v4_decode_kernels import (
                    deepseek_v4_compress_topk_decode,
                )

                compress_topk_idxs = deepseek_v4_compress_topk_decode(
                    ratio,
                    start_pos,
                    offset,
                    max_groups,
                    active,
                )
            compress_topk_idxs = _maybe_canonicalize_compress_topk_order(
                compress_topk_idxs
            )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        write_idx = start_pos.remainder(win)
        _deepseek_v4_update_cache_slots_(
            self.kv_cache,
            req_indices,
            write_idx,
            kv.squeeze(1),
            active,
        )

        if ratio:
            self.compressor.forward_stateful_cuda_graph_decode(
                x,
                start_pos,
                req_indices=req_indices,
                active_mask=active,
            )
        from sglang.srt.models.deepseek_v4_decode_kernels import (
            deepseek_v4_gather_active_cache_decode,
        )

        cache = deepseek_v4_gather_active_cache_decode(self.kv_cache, req_indices, active)

        out = deepseek_v4_kernels.sparse_attn(
            q,
            cache,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
        )

        _apply_rotary_emb_decode(out[..., -rd:], freqs_cis, inverse=True)

        out = out.flatten(-2)
        out = self.wo_a.apply_input_rotation(out)
        out = out.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out, wo_a)
        out = out.flatten(2)
        x = self.wo_b(out)
        return x.permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------
# MoE: gate, expert, MoE block.
# ---------------------------------------------------------------------------


class DeepSeekV4Gate(nn.Module):
    """V4 router: hash routing (first ``n_hash_layers``) or score routing.

    Native-checkpoint shapes (per audit):
      * ``weight``: ``[n_routed_experts, dim]`` fp32 in-flight, matching
        official inference.
      * ``bias``: ``[n_routed_experts]`` fp32 — score-based layers only.
      * ``tid2eid``: ``[vocab_size, n_activated]`` int32 — hash layers only.

    The ``tid2eid`` table is registered as a non-trainable Parameter (not
    a buffer) so it surfaces in ``named_parameters()`` — the audit lists
    it as a parameter target name.
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated: int,
        score_func: str,
        route_scale: float,
        vocab_size: int,
        is_hash_layer: bool,
        layer_id: int,
    ):
        super().__init__()
        self.topk = n_activated
        self.score_func = score_func
        self.route_scale = route_scale
        self.is_hash_layer = is_hash_layer
        self.layer_id = layer_id
        self.capture_dp_local = True

        self.weight = nn.Parameter(
            torch.empty(n_routed_experts, dim, dtype=torch.float32)
        )
        if is_hash_layer:
            # Official inference keeps tid2eid int32 in-flight. Staged
            # checkpoints may store int64; loading casts once into this dtype.
            self.tid2eid = nn.Parameter(
                torch.empty(vocab_size, n_activated, dtype=torch.int32),
                requires_grad=False,
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                torch.empty(n_routed_experts, dtype=torch.float32)
            )
            self.register_parameter("tid2eid", None)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        """DeepSeek V4 router forward.

        ``x`` is ``[N, dim]`` (flat token stream); ``input_ids`` is ``[N]``.
        Returns ``(weights, indices)`` for the top-k routed-expert dispatch,
        with ``route_scale`` already applied. Hash-gate path uses the
        ``tid2eid`` int32 table directly with NO scoring; score-gate path
        runs ``F.linear(x.float(), weight)`` then sqrtsoftplus /
        sigmoid normalisation, biased top-k, gather, renorm,
        ``route_scale``.

        Gate weight is fp32 in-flight, matching official inference.
        """
        x = x.float()
        weight = self.weight
        if _deepseek_v4_use_tilekernels():
            scores = _deepseek_v4_best_linear(
                x, weight, "SGLANG_DSV4_GATE_LINEAR_IMPL"
            )
        else:
            scores = F.linear(x, weight)
        if not self.is_hash_layer:
            if self.score_func not in ("sigmoid", "sqrtsoftplus"):
                raise RuntimeError(
                    "top2_sum_gate DeepSeek V4 router is the production path and "
                    f"requires sigmoid/sqrtsoftplus; got {self.score_func!r}."
                )
            top2_sum_gate = _deepseek_v4_tile_router_op()
            indices, weights = top2_sum_gate(
                scores.contiguous(),
                self.bias.contiguous(),
                self.topk,
                0,
                0,
                False,
                0,
                float(self.route_scale),
                0,
                1,
                0,
                1,
                self.score_func,
            )
            get_global_experts_capturer().capture(
                layer_id=self.layer_id,
                topk_ids=indices.to(torch.int32),
                dp_local=self.capture_dp_local,
            )
            return weights, indices

        if self.score_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            raise RuntimeError(
                f"Unsupported DeepSeek V4 router score function {self.score_func!r}."
            )
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.is_hash_layer:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        get_global_experts_capturer().capture(
            layer_id=self.layer_id,
            topk_ids=indices.to(torch.int32),
            dp_local=self.capture_dp_local,
        )
        weights = original_scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class DeepSeekV4Expert(nn.Module):
    """Single SwiGLU FFN expert.

    ``dtype`` controls FP4 (routed) vs FP8 (shared). The native ckpt scale
    layouts are determined by ``DeepSeekV4Linear``; both ``weight`` and ``scale``
    surface in ``named_parameters()`` under ``w{1,2,3}.weight`` /
    ``w{1,2,3}.scale``.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        dtype: Optional[torch.dtype] = None,
        swiglu_limit: float = 0.0,
        linears: Optional[dict[str, nn.Module]] = None,
    ):
        super().__init__()
        if linears is None:
            self.w1 = DeepSeekV4Linear(dim, inter_dim, bias=False, dtype=dtype)
            self.w2 = DeepSeekV4Linear(inter_dim, dim, bias=False, dtype=dtype)
            self.w3 = DeepSeekV4Linear(dim, inter_dim, bias=False, dtype=dtype)
        else:
            self.w1 = linears["w1"]
            self.w2 = linears["w2"]
            self.w3 = linears["w3"]
        self.swiglu_limit = swiglu_limit

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        w1_oft_r: Optional[torch.Tensor] = None,
        w2_oft_r: Optional[torch.Tensor] = None,
        w3_oft_r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SwiGLU FFN expert forward.

        Compute graph: ``w2( weights * silu(clamp(w1(x))) * clamp(w3(x)) )``.
        The clamp is applied iff ``swiglu_limit > 0`` (matching official:
        gate clipped above by limit, up clipped both ways). ``weights`` is
        the per-token routing weight tensor (``[N, 1]``); when ``None``
        (shared expert path) the multiply is skipped.
        """
        dtype = x.dtype
        w1_x = _deepseek_v4_apply_expert_oft_r(x, w1_oft_r) if w1_oft_r is not None else x
        w3_x = _deepseek_v4_apply_expert_oft_r(x, w3_oft_r) if w3_oft_r is not None else x
        if weights is not None:
            from sglang.srt.models.deepseek_v4_moe_kernels import (
                deepseek_v4_clamp_silu_mul_preexpanded,
            )

            y = deepseek_v4_clamp_silu_mul_preexpanded(
                self.w1(w1_x),
                self.w3(w3_x),
                weights,
                self.swiglu_limit,
                dtype,
            )
        else:
            gate = self.w1(w1_x).float()
            up = self.w3(w3_x).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            y = (F.silu(gate) * up).to(dtype)
        if w2_oft_r is not None:
            y = _deepseek_v4_apply_expert_oft_r(y, w2_oft_r)
        return self.w2(y)


def _deepseek_v4_apply_expert_oft_r(
    x: torch.Tensor,
    oft_r: Optional[torch.Tensor],
) -> torch.Tensor:
    if oft_r is None:
        return x
    if x.numel() == 0:
        return x
    block_size = oft_r.shape[-1]
    input_dim = oft_r.shape[0] * block_size
    if x.shape[-1] != input_dim:
        raise ValueError(
            f"DeepSeek V4 expert OFT input dim mismatch: x last dim {x.shape[-1]} "
            f"vs R dim {input_dim}"
        )
    orig_shape = x.shape
    required_dtype = x.dtype
    if required_dtype != oft_r.dtype:
        x = x.to(oft_r.dtype)
    flat = x.contiguous().reshape(-1, input_dim)
    oft_r = oft_r.contiguous()

    if flat.is_cuda:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
            apply_oft_rotation_triton,
        )
        from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
            moe_align_block_size,
        )

        topk_ids = torch.zeros((flat.shape[0], 1), device=flat.device, dtype=torch.int32)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, 64, 1
        )
        rotated = apply_oft_rotation_triton(
            flat,
            oft_r.unsqueeze(0),
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k=1,
            block_m=64,
        )
    else:
        x_blocks = flat.reshape(-1, oft_r.shape[0], block_size)
        rotated = torch.einsum("...bk,bkc->...bc", x_blocks, oft_r).reshape_as(flat)
    return rotated.reshape(orig_shape).to(required_dtype)


def _deepseek_v4_moe_ep_info() -> tuple[int, int]:
    try:
        ep_size = int(get_moe_expert_parallel_world_size())
        ep_rank = int(get_moe_expert_parallel_rank())
    except Exception:
        ep_size = 1
        ep_rank = 0
    return ep_size, ep_rank


def _deepseek_v4_moe_tp_info() -> tuple[int, int]:
    try:
        tp_size = int(get_moe_tensor_parallel_world_size())
        tp_rank = int(get_moe_tensor_parallel_rank())
    except Exception:
        tp_size = 1
        tp_rank = 0
    return tp_size, tp_rank


def _slice_deepseek_v4_routed_expert_param_for_tp(
    tensor: torch.Tensor,
    dest: torch.Tensor,
    *,
    proj: str,
    kind: str,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if tuple(tensor.shape) == tuple(dest.shape):
        return tensor
    if tp_size <= 1:
        raise RuntimeError(
            "DeepSeek V4 routed expert tensor shape mismatch without MoE TP: "
            f"proj={proj} kind={kind} tensor={tuple(tensor.shape)} "
            f"dest={tuple(dest.shape)}"
        )

    if proj in {"w1", "w3"}:
        shard_dim = 0
    elif proj == "w2":
        shard_dim = 1
    else:
        raise RuntimeError(f"Unsupported DeepSeek V4 routed expert projection: {proj}")

    if tensor.shape[shard_dim] != dest.shape[shard_dim] * tp_size:
        raise RuntimeError(
            "DeepSeek V4 routed expert TP shard dimension mismatch: "
            f"proj={proj} kind={kind} tensor={tuple(tensor.shape)} "
            f"dest={tuple(dest.shape)} shard_dim={shard_dim} "
            f"{tp_rank=} {tp_size=}"
        )

    shard_size = dest.shape[shard_dim]
    start = tp_rank * shard_size
    tensor = tensor.narrow(shard_dim, start, shard_size)
    if tuple(tensor.shape) != tuple(dest.shape):
        raise RuntimeError(
            "DeepSeek V4 routed expert TP shard produced wrong shape: "
            f"proj={proj} kind={kind} shard={tuple(tensor.shape)} "
            f"dest={tuple(dest.shape)}"
        )
    return tensor


class DeepSeekV4MoE(nn.Module):
    """V4 MoE block: hash-or-score gate -> top-k routed experts (FP4) +
    1 shared expert (FP8).
    """

    def __init__(self, config, layer_id: int, is_hash_layer: bool):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_activated = config.n_activated_experts
        self.moe_ep_size, self.moe_ep_rank = _deepseek_v4_moe_ep_info()
        self.moe_tp_size, self.moe_tp_rank = _deepseek_v4_moe_tp_info()
        if self.moe_ep_size < 1:
            raise ValueError(f"Invalid DeepSeek V4 MoE EP size: {self.moe_ep_size}")
        if self.moe_tp_size < 1:
            raise ValueError(f"Invalid DeepSeek V4 MoE TP size: {self.moe_tp_size}")
        if self.n_routed_experts % self.moe_ep_size != 0:
            raise ValueError(
                "DeepSeek V4 routed experts must divide evenly across EP ranks: "
                f"{self.n_routed_experts=} {self.moe_ep_size=}"
            )
        self.dispatcher_backend = "none"
        self._deepep_dispatcher = None
        self.num_local_routed_experts = self.n_routed_experts // self.moe_ep_size
        start_expert = self.moe_ep_rank * self.num_local_routed_experts
        end_expert = start_expert + self.num_local_routed_experts
        self.local_routed_expert_ids = tuple(range(start_expert, end_expert))
        a2a_backend = get_moe_a2a_backend()
        if self.moe_ep_size > 1:
            if a2a_backend.is_none():
                self.dispatcher_backend = "naive"
            elif a2a_backend.is_deepep():
                self.dispatcher_backend = "deepep"
                self._deepep_dispatcher = DeepEPDispatcher(
                    group=get_moe_ep_group().device_group,
                    router_topk=self.n_activated,
                    permute_fusion=True,
                    num_experts=self.n_routed_experts,
                    num_local_experts=self.num_local_routed_experts,
                    hidden_size=self.dim,
                    params_dtype=getattr(config, "torch_dtype", torch.bfloat16),
                    deepep_mode=get_deepep_mode(),
                    async_finish=False,
                    return_recv_hook=False,
                    force_bf16_dispatch=True,
                )
            else:
                raise NotImplementedError(
                    "DeepSeek V4 MoE EP currently supports only "
                    "--moe-a2a-backend none or deepep; got "
                    f"{a2a_backend.value!r}."
                )
        self.gate = DeepSeekV4Gate(
            dim=self.dim,
            n_routed_experts=self.n_routed_experts,
            n_activated=self.n_activated,
            score_func=config.score_func,
            route_scale=config.route_scale,
            vocab_size=config.vocab_size,
            is_hash_layer=is_hash_layer,
            layer_id=layer_id,
        )
        expert_dtype_name = getattr(config, "expert_dtype", "fp4")
        if expert_dtype_name != "fp4":
            raise ValueError(
                "DeepSeek V4 routed experts only supports expert_dtype='fp4'; "
                f"got {expert_dtype_name!r}."
            )
        expert_dtype = torch.float4_e2m1fn_x2
        moe_inter = config.moe_inter_dim
        if moe_inter % self.moe_tp_size != 0:
            raise ValueError(
                "DeepSeek V4 routed expert intermediate dim must divide across MoE TP: "
                f"{moe_inter=} {self.moe_tp_size=}"
            )
        self.moe_inter_dim = moe_inter
        self.moe_inter_dim_per_partition = moe_inter // self.moe_tp_size
        self._init_packed_routed_expert_params(moe_inter, expert_dtype)
        local_experts = [
            DeepSeekV4Expert(
                dim=self.dim,
                inter_dim=self.moe_inter_dim_per_partition,
                dtype=expert_dtype,
                swiglu_limit=getattr(config, "swiglu_limit", 0.0) or 0.0,
                linears={
                    "w1": DeepSeekV4PackedExpertLinear(
                        self,
                        "w1",
                        local_id,
                        self.dim,
                        self.moe_inter_dim_per_partition,
                    ),
                    "w2": DeepSeekV4PackedExpertLinear(
                        self,
                        "w2",
                        local_id,
                        self.moe_inter_dim_per_partition,
                        self.dim,
                    ),
                    "w3": DeepSeekV4PackedExpertLinear(
                        self,
                        "w3",
                        local_id,
                        self.dim,
                        self.moe_inter_dim_per_partition,
                    ),
                },
            )
            for local_id, _ in enumerate(self.local_routed_expert_ids)
        ]
        if self.moe_ep_size == 1:
            self.experts = nn.ModuleList(local_experts)
        else:
            # Preserve global expert ids in module keys for routing/OFT helper
            # lookups. Routed expert weights themselves are packed above.
            self.experts = nn.ModuleDict(
                {
                    str(global_expert_id): expert
                    for global_expert_id, expert in zip(
                        self.local_routed_expert_ids, local_experts, strict=True
                    )
                }
            )
        # Shared expert: FP8 storage; ``shared_inter = moe_inter * n_shared``.
        shared_inter = config.moe_inter_dim * config.n_shared_experts
        if shared_inter % self.moe_tp_size != 0:
            raise ValueError(
                "DeepSeek V4 shared expert intermediate dim must divide across MoE TP: "
                f"{shared_inter=} {self.moe_tp_size=}"
            )
        moe_tp_group = get_moe_tp_group() if self.moe_tp_size > 1 else None
        # When MoE TP > 1 AND we're not on the DeepEP dispatcher, fold the
        # per-expert all_reduce + per-expert bf16 cast into a single outer
        # all_reduce + single bf16 cast at the end of the MoE block. Matches
        # stock Megatron MoE precision pattern (one bf16 round-trip per MoE
        # call instead of one per expert). Shared expert w2 also returns
        # partial under this flag so its contribution gets summed with routed
        # partials before the outer combine. DeepEP paths keep the legacy
        # per-expert all_reduce (their combine semantics are different).
        self._use_outer_tp_reduce = (
            self.moe_tp_size > 1 and self.dispatcher_backend != "deepep"
        )
        self.shared_experts = DeepSeekV4Expert(
            dim=self.dim,
            inter_dim=shared_inter,
            dtype=torch.float8_e4m3fn,
            swiglu_limit=0.0,
            linears={
                "w1": DeepSeekV4ColumnParallelLinear(
                    self.dim,
                    shared_inter,
                    bias=False,
                    dtype=torch.float8_e4m3fn,
                    tp_group=moe_tp_group,
                ),
                "w2": DeepSeekV4RowParallelLinear(
                    shared_inter,
                    self.dim,
                    bias=False,
                    dtype=torch.float8_e4m3fn,
                    tp_group=moe_tp_group,
                    expert_skip_comm=self._use_outer_tp_reduce,
                ),
                "w3": DeepSeekV4ColumnParallelLinear(
                    self.dim,
                    shared_inter,
                    bias=False,
                    dtype=torch.float8_e4m3fn,
                    tp_group=moe_tp_group,
                ),
            },
        )
        self.w1_oft_r: Optional[torch.Tensor] = None
        self.w2_oft_r: Optional[torch.Tensor] = None
        self.w3_oft_r: Optional[torch.Tensor] = None
        self._deep_gemm_fp4_scale_cache: dict[
            str, tuple[tuple[Any, ...], torch.Tensor]
        ] = {}

    def packed_routed_expert_param_names(self) -> list[str]:
        return [
            f"routed_experts_{proj}_{kind}"
            for proj in ("w1", "w2", "w3")
            for kind in ("weight", "scale")
            if getattr(self, f"routed_experts_{proj}_{kind}", None) is not None
        ]

    def _init_packed_routed_expert_params(
        self,
        moe_inter: int,
        dtype: torch.dtype,
    ) -> None:
        specs = {
            "w1": (self.dim, self.moe_inter_dim_per_partition),
            "w2": (self.moe_inter_dim_per_partition, self.dim),
            "w3": (self.dim, self.moe_inter_dim_per_partition),
        }
        for proj, (in_features, out_features) in specs.items():
            weight_shape, scale_shape = _deepseek_v4_linear_storage_shapes(
                in_features, out_features, dtype
            )
            weight = nn.Parameter(
                torch.empty(
                    self.num_local_routed_experts,
                    *weight_shape,
                    dtype=dtype,
                )
            )
            self.register_parameter(f"routed_experts_{proj}_weight", weight)
            if scale_shape is None:
                self.register_parameter(f"routed_experts_{proj}_scale", None)
            else:
                scale = nn.Parameter(
                    torch.empty(
                        self.num_local_routed_experts,
                        *scale_shape,
                        dtype=_DEFAULT_SCALE_DTYPE,
                    )
                )
                self.register_parameter(f"routed_experts_{proj}_scale", scale)
                weight.scale = scale

    def load_routed_expert_param(
        self,
        global_expert_id: int,
        proj: str,
        kind: str,
        tensor: torch.Tensor,
    ) -> bool:
        if proj not in {"w1", "w2", "w3"} or kind not in {"weight", "scale"}:
            return False
        local_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)
        if local_id < 0:
            return False
        packed = getattr(self, f"routed_experts_{proj}_{kind}", None)
        if packed is None:
            return False
        dest = packed[local_id]
        with torch.no_grad():
            shard = _slice_deepseek_v4_routed_expert_param_for_tp(
                tensor,
                dest,
                proj=proj,
                kind=kind,
                tp_rank=self.moe_tp_rank,
                tp_size=self.moe_tp_size,
            )
            dest.copy_(shard.to(dest.device).to(dest.dtype))
        return True

    def get_moe_weights(self) -> list[torch.Tensor]:
        return [
            getattr(self, name).data
            for name in self.packed_routed_expert_param_names()
            if getattr(self, name, None) is not None
        ]

    def _iter_local_experts(self):
        if isinstance(self.experts, nn.ModuleDict):
            return self.experts.values()
        return self.experts

    def _map_global_expert_id_to_local_expert_id(self, global_expert_id: int) -> int:
        try:
            return self.local_routed_expert_ids.index(global_expert_id)
        except ValueError:
            return -1

    def _local_expert(self, global_expert_id: int) -> Optional[DeepSeekV4Expert]:
        if isinstance(self.experts, nn.ModuleDict):
            key = str(global_expert_id)
            if key in self.experts:
                return self.experts[key]
            return None
        if 0 <= global_expert_id < len(self.experts):
            return self.experts[global_expert_id]
        return None

    def dsv4_expert_oft_device(self) -> torch.device:
        first = next(iter(self._iter_local_experts()))
        return first.w1.weight.device

    def _dsv4_expert_oft_input_dim(self, proj: str) -> int:
        first = next(iter(self._iter_local_experts()))
        return int(getattr(first, proj).in_features)

    def ensure_dsv4_expert_oft_r(
        self,
        proj: str,
        *,
        block_size: int,
        dtype: torch.dtype,
        sample: Optional[torch.Tensor] = None,
    ) -> bool:
        input_dim = self._dsv4_expert_oft_input_dim(proj)
        if input_dim % block_size != 0:
            raise ValueError(
                f"DeepSeek V4 {proj} OFT input dim {input_dim} is not divisible "
                f"by block_size {block_size}"
            )
        num_blocks = input_dim // block_size
        if sample is not None and sample.shape[0] != num_blocks:
            raise ValueError(
                f"DeepSeek V4 {proj} OFT block mismatch: expected {num_blocks}, "
                f"got {sample.shape[0]}"
            )

        attr = f"{proj}_oft_r"
        current = getattr(self, attr)
        shape = (
            self.num_local_routed_experts,
            num_blocks,
            block_size,
            block_size,
        )
        device = self.dsv4_expert_oft_device()
        if current is not None:
            if (
                tuple(current.shape) == shape
                and current.dtype == dtype
                and current.device == device
            ):
                return False
            raise RuntimeError(
                "DeepSeek V4 expert OFT update would replace an existing R buffer: "
                f"proj={proj}, current_shape={tuple(current.shape)}, "
                f"current_dtype={current.dtype}, current_device={current.device}, "
                f"incoming_shape={shape}, incoming_dtype={dtype}, "
                f"incoming_device={device}."
            )
        buf = torch.empty(shape, dtype=dtype, device=device)
        block_eye = torch.eye(block_size, dtype=dtype, device=device)
        buf[...] = block_eye
        setattr(self, attr, buf)
        return True

    def _local_expert_oft_r(
        self,
        proj: str,
        global_expert_id: int,
    ) -> Optional[torch.Tensor]:
        all_r = getattr(self, f"{proj}_oft_r", None)
        if all_r is None:
            return None
        local_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)
        if local_id < 0:
            return None
        return all_r[local_id]

    def has_dsv4_expert_oft(self) -> bool:
        return (
            self.w1_oft_r is not None
            or self.w2_oft_r is not None
            or self.w3_oft_r is not None
        )

    def _local_topk_ids(self, indices: torch.Tensor) -> torch.Tensor:
        start_expert = self.local_routed_expert_ids[0]
        end_expert = start_expert + self.num_local_routed_experts
        local = indices.to(torch.int64) - start_expert
        is_local = (indices >= start_expert) & (indices < end_expert)
        return torch.where(is_local, local, torch.full_like(local, -1)).to(torch.int32)

    @staticmethod
    def _routed_expert_oft_rotation(
        x: torch.Tensor,
        oft_r: torch.Tensor,
        local_topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        *,
        top_k: int,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x.reshape(-1, x.shape[-1])
        required_dtype = x.dtype
        if required_dtype != oft_r.dtype:
            x = x.to(oft_r.dtype)
        flat = x.contiguous().reshape(-1, x.shape[-1])
        oft_r = oft_r.contiguous()

        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
            apply_oft_rotation_triton,
        )

        rotated = apply_oft_rotation_triton(
            flat,
            oft_r,
            local_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k=top_k,
            block_m=64,
        )
        return rotated.to(required_dtype)

    def _all_reduce_routed_expert_tp_output(self, y: torch.Tensor) -> torch.Tensor:
        # Per-expert TP all_reduce. Under the new "outer-reduce" pattern
        # (``self._use_outer_tp_reduce``) we *skip* this for the naive/CUDA-
        # graph paths and let the partials accumulate; the cross-TP combine
        # fires once at the end of the MoE block via
        # ``_outer_tp_reduce_if_needed``. DeepEP paths keep their own
        # per-expert all_reduce because their combine semantics are different.
        if self._use_outer_tp_reduce and self.dispatcher_backend != "deepep":
            return y
        if self.moe_tp_size > 1:
            out_dtype = y.dtype
            y = get_moe_tp_group().all_reduce(y.float()).to(out_dtype)
        return y

    def _all_reduce_routed_output(self, y: torch.Tensor) -> torch.Tensor:
        if self.moe_ep_size > 1:
            y = get_moe_ep_group().all_reduce(y)
        return y

    def _outer_tp_reduce_if_needed(self, y: torch.Tensor) -> torch.Tensor:
        """Outer TP combine at the end of the MoE block (new pattern).

        ``y`` arrives as fp32 (routed accumulator) plus the shared-expert
        partial added in-place. Reduces in fp32 and returns fp32; the caller
        does the final ``.type_as(x)`` bf16 cast — keeping the total bf16
        round-trip count at one per MoE call.
        """
        if self._use_outer_tp_reduce and self.moe_tp_size > 1:
            if hasattr(get_moe_tp_group(), "all_reduce"):
                y = get_moe_tp_group().all_reduce(y)
            else:
                dist.all_reduce(y, group=get_moe_tp_group())
        return y

    @staticmethod
    def _canonicalize_deepep_metadata(
        weights: torch.Tensor,
        indices: torch.Tensor,
        num_experts: int,
        router_topk: int,
        capacity_factor: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.zeros(
            weights.shape[0],
            num_experts,
            dtype=torch.float32,
            device=weights.device,
        )
        probs.scatter_(1, indices.to(torch.int64), weights.float())
        token_probs, token_indices = torch.topk(probs, router_topk, dim=-1)
        if capacity_factor is not None:
            token_indices = token_indices.masked_fill(token_probs == 0, -1)
        return token_probs.contiguous(), token_indices.contiguous()

    def _graph_expert_linear_params(
        self, linear_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(self, f"routed_experts_{linear_name}_weight")
        scale = getattr(self, f"routed_experts_{linear_name}_scale")
        if weight.dtype != torch.float4_e2m1fn_x2:
            raise RuntimeError(
                "DeepSeek V4 MoE CUDA graph path currently requires FP4 routed "
                f"experts, got {weight.dtype} for {linear_name}."
            )
        if scale is None:
            raise RuntimeError(f"DeepSeek V4 MoE {linear_name} packed scale is missing.")
        return weight, scale

    def _graph_expert_linear_versions(
        self, linear_name: str
    ) -> tuple[tuple[int, int], ...]:
        weight = getattr(self, f"routed_experts_{linear_name}_weight")
        scale = getattr(self, f"routed_experts_{linear_name}_scale")
        return (
            (
                getattr(weight, "_version", 0),
                getattr(scale, "_version", 0) if scale is not None else 0,
            ),
        )

    def _graph_expert_linear_deep_gemm_scale(
        self, linear_name: str, *, force_refresh: bool = False
    ) -> torch.Tensor:
        _, scale = self._graph_expert_linear_params(linear_name)
        cache_key = (
            scale.data_ptr(),
            getattr(scale, "_version", 0),
            tuple(scale.shape),
            tuple(scale.stride()),
            scale.dtype,
            scale.device,
        )
        cached = self._deep_gemm_fp4_scale_cache.get(linear_name)
        if cached is not None and not force_refresh and cached[0] == cache_key:
            return cached[1]

        from sglang.srt.models.deepseek_v4_moe_kernels import (
            pack_fp8_e8m0_scale_for_deep_gemm,
        )

        packed_scale = pack_fp8_e8m0_scale_for_deep_gemm(scale)
        if cached is not None:
            _, cached_scale = cached
            if (
                tuple(cached_scale.shape) == tuple(packed_scale.shape)
                and cached_scale.dtype == packed_scale.dtype
                and cached_scale.device == packed_scale.device
            ):
                cached_scale.copy_(packed_scale)
                self._deep_gemm_fp4_scale_cache[linear_name] = (
                    cache_key,
                    cached_scale,
                )
                return cached_scale
            raise RuntimeError(
                "DeepSeek V4 DeepGEMM packed scale refresh would replace a cached "
                "CUDA-graph tensor. Recapture is required for layout changes: "
                f"{linear_name=}, cached_shape={tuple(cached_scale.shape)}, "
                f"new_shape={tuple(packed_scale.shape)}, "
                f"cached_dtype={cached_scale.dtype}, new_dtype={packed_scale.dtype}, "
                f"cached_device={cached_scale.device}, new_device={packed_scale.device}."
            )
        self._deep_gemm_fp4_scale_cache[linear_name] = (cache_key, packed_scale)
        return packed_scale

    def refresh_cuda_graph_expert_weights(self) -> None:
        if not self._deep_gemm_fp4_scale_cache:
            return None

        from sglang.srt.models.deepseek_v4_moe_kernels import (
            has_deep_gemm_official_fp8_fp4,
        )

        if not has_deep_gemm_official_fp8_fp4():
            self._deep_gemm_fp4_scale_cache.clear()
            return None

        for linear_name in tuple(self._deep_gemm_fp4_scale_cache):
            self._graph_expert_linear_deep_gemm_scale(
                linear_name, force_refresh=True
            )
        return None

    def prepare_cuda_graph_capture(self) -> None:
        from sglang.srt.models.deepseek_v4_moe_kernels import (
            has_deep_gemm_official_fp8_fp4,
        )

        use_deep_gemm = has_deep_gemm_official_fp8_fp4()
        for linear_name in ("w1", "w2", "w3"):
            self._graph_expert_linear_params(linear_name)
            if use_deep_gemm:
                self._graph_expert_linear_deep_gemm_scale(linear_name)

    def _graph_grouped_fp4_linear(
        self,
        x: torch.Tensor,
        linear_name: str,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.models.deepseek_v4_moe_kernels import (
            deepseek_v4_deep_gemm_act_quant,
            grouped_fp4_gemm,
            has_deep_gemm_official_fp8_fp4,
        )

        weight, scale = self._graph_expert_linear_params(linear_name)
        if has_deep_gemm_official_fp8_fp4():
            x_q, x_s = deepseek_v4_deep_gemm_act_quant(x.contiguous(), _FP8_BLOCK_SIZE)
            scale = self._graph_expert_linear_deep_gemm_scale(linear_name)
        else:
            x_q, x_s = deepseek_v4_kernels.act_quant(
                x.contiguous(),
                _FP8_BLOCK_SIZE,
                _DEFAULT_SCALE_FMT,
                _DEFAULT_SCALE_DTYPE,
            )
        return grouped_fp4_gemm(
            x_q,
            x_s,
            weight,
            scale,
            pos_to_expert.contiguous(),
            _DEFAULT_SCALE_DTYPE,
        )

    @staticmethod
    def _graph_expand_rotated_route_rows(
        rotated_rows: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        pos_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        from tile_kernels.moe.expand_to_fused_kernel import expand_to_fused

        return expand_to_fused(
            rotated_rows,
            token_topk_to_pos.reshape(-1, 1).contiguous(),
            pos_to_expert,
        )

    @staticmethod
    def _graph_num_expanded_tokens(
        num_routed_items: int,
        num_experts: int,
        alignment: int,
    ) -> int:
        max_active_experts = min(num_experts, num_routed_items)
        return (
            (
                num_routed_items
                + (alignment - 1) * max_active_experts
                + alignment - 1
            )
            // alignment
            * alignment
        )

    def _forward_cuda_graph(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        from tile_kernels.moe.expand_to_fused_kernel import expand_to_fused
        from tile_kernels.moe.get_fused_mapping_kernel import get_fused_mapping
        from sglang.srt.models.deepseek_v4_moe_kernels import (
            has_deep_gemm_official_fp8_fp4,
            reduce_fused_topk_fp32,
        )

        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        route_indices = (
            self._local_topk_ids(indices) if self.moe_ep_size > 1 else indices
        ).to(torch.int64).contiguous()
        route_num_experts = (
            self.num_local_routed_experts
            if self.moe_ep_size > 1
            else self.n_routed_experts
        )

        alignment = 128 if has_deep_gemm_official_fp8_fp4() else 32
        num_expanded_tokens = self._graph_num_expanded_tokens(
            route_indices.numel(),
            route_num_experts,
            alignment,
        )
        (
            pos_to_expert,
            _pos_to_token,
            pos_to_token_topk,
            token_topk_to_pos,
            _expert_start,
            _expert_end,
            _expert_count,
            _counts_list,
        ) = get_fused_mapping(
            route_indices,
            route_num_experts,
            num_expanded_tokens,
            alignment,
            force_no_sync=True,
        )

        expanded_x = expand_to_fused(x, token_topk_to_pos, pos_to_expert)
        w1_input = expanded_x
        w3_input = expanded_x
        if self.has_dsv4_expert_oft():
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size,
            )

            local_topk_ids = route_indices.to(torch.int32)
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(
                    local_topk_ids,
                    64,
                    self.num_local_routed_experts,
                )
            )
            if self.w1_oft_r is not None:
                w1_input = self._graph_expand_rotated_route_rows(
                    self._routed_expert_oft_rotation(
                        x,
                        self.w1_oft_r,
                        local_topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        top_k=indices.shape[1],
                    ),
                    token_topk_to_pos,
                    pos_to_expert,
                )
            if self.w3_oft_r is not None:
                w3_input = self._graph_expand_rotated_route_rows(
                    self._routed_expert_oft_rotation(
                        x,
                        self.w3_oft_r,
                        local_topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        top_k=indices.shape[1],
                    ),
                    token_topk_to_pos,
                    pos_to_expert,
                )

        gate = self._graph_grouped_fp4_linear(w1_input, "w1", pos_to_expert)
        up = self._graph_grouped_fp4_linear(w3_input, "w3", pos_to_expert)
        first_expert = next(iter(self._iter_local_experts()))
        from sglang.srt.models.deepseek_v4_moe_kernels import deepseek_v4_clamp_silu_mul_topk

        expanded_y = deepseek_v4_clamp_silu_mul_topk(
            gate,
            up,
            weights,
            pos_to_token_topk,
            first_expert.swiglu_limit,
            x.dtype,
        )
        if self.w2_oft_r is not None:
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size,
            )

            w2_topk_ids = pos_to_expert.reshape(-1, 1).contiguous()
            w2_sorted_ids, w2_expert_ids, w2_num_tokens_post_padded = (
                moe_align_block_size(
                    w2_topk_ids,
                    64,
                    self.num_local_routed_experts,
                )
            )
            expanded_y = self._routed_expert_oft_rotation(
                expanded_y,
                self.w2_oft_r,
                w2_topk_ids,
                w2_sorted_ids,
                w2_expert_ids,
                w2_num_tokens_post_padded,
                top_k=1,
            )
        expanded_out = self._graph_grouped_fp4_linear(
            expanded_y,
            "w2",
            pos_to_expert,
        )
        expanded_out = self._all_reduce_routed_expert_tp_output(expanded_out)
        routed = reduce_fused_topk_fp32(expanded_out, token_topk_to_pos)
        routed = self._all_reduce_routed_output(routed)
        y = routed + self.shared_experts(x)
        y = self._outer_tp_reduce_if_needed(y)
        return y.type_as(x).view(shape)

    def _forward_deepep_normal_core_cuda_graph(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ) -> torch.Tensor:
        from tile_kernels.moe.expand_to_fused_kernel import expand_to_fused
        from tile_kernels.moe.get_fused_mapping_kernel import get_fused_mapping
        from sglang.srt.models.deepseek_v4_moe_kernels import (
            has_deep_gemm_official_fp8_fp4,
            reduce_fused_topk_fp32,
        )

        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states

        topk_ids = dispatch_output.topk_ids.to(torch.int64).contiguous()
        topk_weights = dispatch_output.topk_weights.contiguous()
        alignment = 128 if has_deep_gemm_official_fp8_fp4() else 32
        num_expanded_tokens = self._graph_num_expanded_tokens(
            topk_ids.numel(),
            self.num_local_routed_experts,
            alignment,
        )
        (
            pos_to_expert,
            _pos_to_token,
            pos_to_token_topk,
            token_topk_to_pos,
            _expert_start,
            _expert_end,
            _expert_count,
            _counts_list,
        ) = get_fused_mapping(
            topk_ids,
            self.num_local_routed_experts,
            num_expanded_tokens,
            alignment,
            force_no_sync=True,
        )

        expanded_x = expand_to_fused(hidden_states, token_topk_to_pos, pos_to_expert)
        w1_input = expanded_x
        w3_input = expanded_x
        if self.has_dsv4_expert_oft():
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size,
            )

            local_topk_ids = topk_ids.to(torch.int32).contiguous()
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(
                    local_topk_ids,
                    64,
                    self.num_local_routed_experts,
                )
            )
            if self.w1_oft_r is not None:
                w1_input = self._graph_expand_rotated_route_rows(
                    self._routed_expert_oft_rotation(
                        hidden_states,
                        self.w1_oft_r,
                        local_topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        top_k=topk_ids.shape[1],
                    ),
                    token_topk_to_pos,
                    pos_to_expert,
                )
            if self.w3_oft_r is not None:
                w3_input = self._graph_expand_rotated_route_rows(
                    self._routed_expert_oft_rotation(
                        hidden_states,
                        self.w3_oft_r,
                        local_topk_ids,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        top_k=topk_ids.shape[1],
                    ),
                    token_topk_to_pos,
                    pos_to_expert,
                )

        gate = self._graph_grouped_fp4_linear(w1_input, "w1", pos_to_expert)
        up = self._graph_grouped_fp4_linear(w3_input, "w3", pos_to_expert)
        first_expert = next(iter(self._iter_local_experts()))
        from sglang.srt.models.deepseek_v4_moe_kernels import deepseek_v4_clamp_silu_mul_topk

        expanded_y = deepseek_v4_clamp_silu_mul_topk(
            gate,
            up,
            topk_weights,
            pos_to_token_topk,
            first_expert.swiglu_limit,
            hidden_states.dtype,
        )
        if self.w2_oft_r is not None:
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size,
            )

            w2_topk_ids = pos_to_expert.reshape(-1, 1).contiguous()
            w2_sorted_ids, w2_expert_ids, w2_num_tokens_post_padded = (
                moe_align_block_size(
                    w2_topk_ids,
                    64,
                    self.num_local_routed_experts,
                )
            )
            expanded_y = self._routed_expert_oft_rotation(
                expanded_y,
                self.w2_oft_r,
                w2_topk_ids,
                w2_sorted_ids,
                w2_expert_ids,
                w2_num_tokens_post_padded,
                top_k=1,
            )
        expanded_out = self._graph_grouped_fp4_linear(
            expanded_y,
            "w2",
            pos_to_expert,
        )
        expanded_out = self._all_reduce_routed_expert_tp_output(expanded_out)
        return reduce_fused_topk_fp32(expanded_out, token_topk_to_pos).to(
            hidden_states.dtype
        )

    def _forward_deepep(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        dispatcher = self._deepep_dispatcher
        assert dispatcher is not None
        token_probs, token_indices = self._canonicalize_deepep_metadata(
            weights,
            indices,
            self.n_routed_experts,
            self.n_activated,
        )
        topk_output = StandardTopKOutput(
            topk_weights=token_probs,
            topk_ids=token_indices,
            router_logits=x.new_empty(0),
        )
        dispatch_output = dispatcher.dispatch(x, topk_output)

        if not DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            raise NotImplementedError(
                f"Unsupported DeepSeek V4 DeepEP dispatch output: {dispatch_output.format}."
            )
        restored_y = self._forward_deepep_normal_core_cuda_graph(dispatch_output)
        combine_input = DeepEPNormalCombineInput(
            hidden_states=restored_y,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

        y = dispatcher.combine(combine_input)
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        capture_dp_local: bool = True,
    ) -> torch.Tensor:
        old_capture_dp_local = self.gate.capture_dp_local
        self.gate.capture_dp_local = capture_dp_local
        try:
            if not x.is_cuda:
                raise RuntimeError(
                    "DeepSeek V4 FP4 routed-expert MoE requires CUDA input."
                )
            if self.dispatcher_backend != "deepep":
                return self._forward_cuda_graph(x, input_ids)

            shape = x.size()
            x = x.view(-1, self.dim)
            weights, indices = self.gate(x, input_ids.flatten())
            return self._forward_deepep(x, weights, indices, shape)
        finally:
            self.gate.capture_dp_local = old_capture_dp_local


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class DeepSeekV4DecoderLayer(nn.Module):
    """V4 transformer layer: per-layer mHC mixers around attention + MoE.

    Native checkpoint keys are translated to SGLang module names:
      * ``input_layernorm`` maps attention norm weights.
      * ``post_attention_layernorm`` maps FFN norm weights.
      * ``self_attn`` maps the native attention block.
      * ``mlp`` maps the native MoE block.
      * ``hc_attn_{fn,base,scale}``, ``hc_ffn_{fn,base,scale}`` — fp32.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = config.hc_mult
        hc_dim = self.hc_mult * config.dim
        mix_hc = (2 + self.hc_mult) * self.hc_mult

        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        self.input_layernorm = RMSNorm(
            config.dim, eps=config.norm_eps, dtype=torch.float32
        )
        self.post_attention_layernorm = RMSNorm(
            config.dim, eps=config.norm_eps, dtype=torch.float32
        )
        self.self_attn = DeepSeekV4Attention(config=config, layer_id=layer_id)
        is_hash = layer_id < (config.n_hash_layers or 0)
        self.mlp = DeepSeekV4MoE(
            config=config, layer_id=layer_id, is_hash_layer=is_hash
        )
        # Stateless HC helper. Configuration constants come from the same
        # config namespace the model uses; the helper carries no parameters.
        self.hc_util = DeepSeekV4HyperConnectionUtil(
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
            norm_eps=config.norm_eps,
        )

    def _should_gather_dp_moe(self, forward_batch: Any) -> bool:
        dispatcher_backend = getattr(self.mlp, "dispatcher_backend", None)
        moe_ep_size = getattr(self.mlp, "moe_ep_size", 1)
        moe_tp_size = getattr(self.mlp, "moe_tp_size", 1)
        return (
            is_dp_attention_enabled()
            and getattr(forward_batch, "global_num_tokens_cpu", None) is not None
            and (
                (moe_ep_size > 1 and dispatcher_backend == "naive")
                or (moe_tp_size > 1 and dispatcher_backend != "deepep")
            )
        )

    @staticmethod
    def _gather_dp_moe_inputs(
        x_sbd: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        s, b, d = x_sbd.shape
        local_x = x_sbd.permute(1, 0, 2).reshape(s * b, d).contiguous()
        local_ids = input_ids.reshape(-1).contiguous()
        if local_ids.shape[0] != local_x.shape[0]:
            raise RuntimeError(
                "DeepSeek V4 DP-MoE gather expects flat input_ids and hidden tokens "
                f"to match, got {tuple(local_ids.shape)=} and "
                f"{tuple(local_x.shape)=}."
            )

        global_x = get_global_dp_buffer()
        dp_gather_partial(global_x, local_x, forward_batch)
        global_ids = torch.empty(
            global_x.shape[0], dtype=input_ids.dtype, device=input_ids.device
        )
        dp_gather_partial(global_ids, local_ids, forward_batch)
        return global_x.view(-1, 1, d), global_ids.view(-1, 1), (s, b, d)

    @staticmethod
    def _scatter_dp_moe_output(
        x_bsd: torch.Tensor,
        local_shape: tuple[int, int, int],
        forward_batch: Any,
    ) -> torch.Tensor:
        s, b, d = local_shape
        global_x = x_bsd.reshape(-1, d).contiguous()
        local_x = global_x.new_empty(s * b, d)
        dp_scatter(local_x, global_x, forward_batch)
        return local_x.view(b, s, d).permute(1, 0, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        """Run one V4 transformer layer.

        ``hidden_states`` enters as ``[s, b, hc, d]`` (V4-native HC layout);
        the two HC rounds reduce to ``[s, b, d]`` for the inner submodule
        (attention or MoE) and re-expand back to ``[s, b, hc, d]``. The MoE
        is ``[b, s, d]``-native, so we permute around it.

        ``forward_batch`` is required and threaded through to ``self_attn`` so
        the DeepSeek V4 scheduler backend can provide request-slot metadata.
        """
        forward_batch = _deepseek_v4_require_forward_batch(
            forward_batch, "DeepSeekV4DecoderLayer.forward"
        )
        # ---- attention round ----
        residual = hidden_states
        x_sbd, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x_sbd = self.input_layernorm(x_sbd)
        if x_sbd.shape[0] != 0:
            x_sbd = self.self_attn(x_sbd, forward_batch=forward_batch)
        hidden_states = self.hc_util.layer_post(x_sbd, residual, post, comb)

        # ---- ffn round ----
        residual = hidden_states
        x_sbd, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x_sbd = self.post_attention_layernorm(x_sbd)
        # MoE is [b, s, d]-native; permute, run, permute back.
        if self._should_gather_dp_moe(forward_batch):
            x_bsd, mlp_input_ids, local_shape = self._gather_dp_moe_inputs(
                x_sbd, input_ids, forward_batch
            )
            x_bsd = self.mlp(x_bsd, mlp_input_ids, capture_dp_local=False)
            x_sbd = self._scatter_dp_moe_output(x_bsd, local_shape, forward_batch)
        else:
            x_bsd = x_sbd.permute(1, 0, 2).contiguous()
            x_bsd = self.mlp(x_bsd, input_ids, capture_dp_local=True)
            x_sbd = x_bsd.permute(1, 0, 2).contiguous()
        hidden_states = self.hc_util.layer_post(x_sbd, residual, post, comb)
        return hidden_states

    # ---- Stateful decoder layer. Same structure as `forward` but routes the
    # attention call through `self_attn.forward_stateful` and threads
    # `start_pos`. The MoE/HC paths are stateless.
    def forward_stateful(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        start_pos: int,
        req_idx: int = 0,
    ) -> torch.Tensor:
        # ---- attention round ----
        residual = hidden_states
        x_sbd, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x_sbd = self.input_layernorm(x_sbd)
        x_sbd = self.self_attn.forward_stateful(
            x_sbd, start_pos=start_pos, req_idx=req_idx
        )
        hidden_states = self.hc_util.layer_post(x_sbd, residual, post, comb)

        # ---- ffn round (identical to forward) ----
        residual = hidden_states
        x_sbd, post, comb = self.hc_util.layer_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x_sbd = self.post_attention_layernorm(x_sbd)
        x_bsd = x_sbd.permute(1, 0, 2).contiguous()
        x_bsd = self.mlp(x_bsd, input_ids)
        x_sbd = x_bsd.permute(1, 0, 2).contiguous()
        hidden_states = self.hc_util.layer_post(x_sbd, residual, post, comb)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder block (model body) and ForCausalLM wrapper.
# ---------------------------------------------------------------------------


class DeepSeekV4Model(nn.Module):
    """V4 decoder body: ``embed_tokens`` -> N decoder layers -> ``norm``.

    Audit naming: ``model.embed_tokens.weight``, ``model.layers.{i}.*``,
    ``model.norm.weight``, ``model.hc_head_params.*``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hc_util = DeepSeekV4HyperConnectionUtil(
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            hc_eps=config.hc_eps,
            norm_eps=config.norm_eps,
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.dim, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            [
                DeepSeekV4DecoderLayer(config=config, layer_id=i)
                for i in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps, dtype=torch.float32)
        self.hc_head_params = HCHeadParams(hc_mult=config.hc_mult, hidden_size=config.dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: Any,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full DeepSeek V4 decoder body.

        ``input_ids`` is ``[b, s]``. The optional ``hidden_states`` lets a
        caller inject pre-computed embeddings; when not provided we embed
        ``input_ids`` here.

        ``forward_batch`` is required and threaded through to each layer's
        attention block so the DeepSeek V4 scheduler backend can provide request-slot
        metadata.

        Returns ``[s, b, d]`` (Megatron convention) so the top-level wrapper
        can run ``lm_head`` on the last token.
        """
        forward_batch = _deepseek_v4_require_forward_batch(
            forward_batch, "DeepSeekV4Model.forward"
        )

        if hidden_states is None:
            emb = self.embed_tokens(input_ids)  # [b, s, d]
            h_sbd = emb.permute(1, 0, 2).contiguous()
        else:
            h_sbd = hidden_states

        # block_expand operates on [b, s, d] -> [b, s, hc, d]; permute around it.
        h_bsd = h_sbd.permute(1, 0, 2).contiguous()
        h_bshd = self.hc_util.block_expand(h_bsd)  # [b, s, hc, d]
        h = h_bshd.permute(1, 0, 2, 3).contiguous()  # -> [s, b, hc, d]

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, input_ids=input_ids, forward_batch=forward_batch)

        # block_head expects [b, s, hc, d] and returns [b, s, d].
        h_bshd = h.permute(1, 0, 2, 3).contiguous()
        h_bsd = self.hc_util.block_head(
            h_bshd,
            self.hc_head_params.hc_head_fn,
            self.hc_head_params.hc_head_scale,
            self.hc_head_params.hc_head_base,
        )
        h_sbd = h_bsd.permute(1, 0, 2).contiguous()
        h_sbd = self.norm(h_sbd)
        return h_sbd

    # ---- Stateful model body. Same structure as `forward` but routes each
    # layer through `forward_stateful` and threads `start_pos`. ----
    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        hidden_states: Optional[torch.Tensor] = None,
        req_idx: int = 0,
    ) -> torch.Tensor:
        if hidden_states is None and start_pos > 0 and input_ids.shape[1] > 1:
            return torch.cat(
                [
                    self.forward_stateful(
                        input_ids[:, t : t + 1],
                        start_pos=start_pos + t,
                        req_idx=req_idx,
                    )
                    for t in range(input_ids.shape[1])
                ],
                dim=0,
            )

        if hidden_states is None:
            emb = self.embed_tokens(input_ids)
            h_sbd = emb.permute(1, 0, 2).contiguous()
        else:
            h_sbd = hidden_states

        h_bsd = h_sbd.permute(1, 0, 2).contiguous()
        h_bshd = self.hc_util.block_expand(h_bsd)
        h = h_bshd.permute(1, 0, 2, 3).contiguous()

        for layer in self.layers:
            h = layer.forward_stateful(
                h, input_ids=input_ids, start_pos=start_pos, req_idx=req_idx
            )

        h_bshd = h.permute(1, 0, 2, 3).contiguous()
        h_bsd = self.hc_util.block_head(
            h_bshd,
            self.hc_head_params.hc_head_fn,
            self.hc_head_params.hc_head_scale,
            self.hc_head_params.hc_head_base,
        )
        h_sbd = h_bsd.permute(1, 0, 2).contiguous()
        h_sbd = self.norm(h_sbd)
        return h_sbd


class DeepseekV4ForCausalLM(nn.Module):
    """Top-level DeepSeek V4 causal LM wrapper.

    Construction-time guarantees (gated by the param-name smoke test):
      * Every audit-OK parameter name appears in ``named_parameters()``.
      * ``lm_head.weight`` is fp32 by construction; a forward-pre-hook casts
        the input to fp32 before the matmul.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        del quant_config, prefix  # Kept for V2/V3 signature parity.
        self.config = config

        # Read server-side overrides so V4-owned per-request buffers
        # (`Attention.kv_cache`, `Compressor.kv_state`, etc.) are sized for
        # scheduler-driven multi-request and chunk-prefill work. Overrides
        # must be set before construction.
        max_batch_size_override = getattr(config, "max_batch_size_override", None)
        ckpt_max_batch = getattr(config, "max_batch_size", 1)
        effective_max_batch = (
            max_batch_size_override
            if max_batch_size_override is not None
            else ckpt_max_batch
        )
        max_seq_len_override = getattr(config, "max_seq_len_override", None)
        ckpt_max_seq_len = getattr(config, "max_seq_len", 256)
        effective_max_seq_len = (
            max_seq_len_override if max_seq_len_override is not None else ckpt_max_seq_len
        )
        if max_batch_size_override is not None and max_batch_size_override != ckpt_max_batch:
            logger.info(
                "DeepSeek V4 max_batch_size_override=%s "
                "(ckpt config has max_batch_size=%s); using %s",
                max_batch_size_override,
                ckpt_max_batch,
                effective_max_batch,
            )
        if max_seq_len_override is not None and max_seq_len_override != ckpt_max_seq_len:
            logger.info(
                "DeepSeek V4 max_seq_len_override=%s "
                "(ckpt config has max_seq_len=%s); using %s",
                max_seq_len_override,
                ckpt_max_seq_len,
                effective_max_seq_len,
            )
        # Mutate config in-place so every downstream `getattr(config, ...)`
        # (in Attention.__init__, Compressor.__init__, Indexer.__init__) reads
        # the overrides. This avoids threading new kwargs through 4 layers of
        # constructors.
        config.max_batch_size = effective_max_batch
        config.max_seq_len = effective_max_seq_len
        self.max_batch_size = effective_max_batch
        self.max_seq_len = effective_max_seq_len

        self.model = DeepSeekV4Model(config=config)

        # Self-check: every per-layer V4 buffer is sized to max_batch_size.
        # Catches a future refactor that forgets to thread the override.
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            assert attn.kv_cache.shape[0] == self.max_batch_size, (
                f"layer {i}: Attention.kv_cache batch dim {attn.kv_cache.shape[0]} "
                f"!= max_batch_size {self.max_batch_size}"
            )
            if attn.compressor is not None:
                assert attn.compressor.kv_state.shape[0] == self.max_batch_size, (
                    f"layer {i}: Compressor.kv_state batch dim "
                    f"{attn.compressor.kv_state.shape[0]} != {self.max_batch_size}"
                )
                assert attn.compressor.score_state.shape[0] == self.max_batch_size
            if attn.indexer is not None:
                assert attn.indexer.kv_cache.shape[0] == self.max_batch_size, (
                    f"layer {i}: Indexer.kv_cache batch dim "
                    f"{attn.indexer.kv_cache.shape[0]} != {self.max_batch_size}"
                )
                assert attn.indexer.compressor.kv_state.shape[0] == self.max_batch_size
            if attn.compress_ratio:
                expected_cache_size = attn.window_size + (
                    self.max_seq_len // attn.compress_ratio
                )
                assert attn.kv_cache.shape[1] == expected_cache_size, (
                    f"layer {i}: Attention.kv_cache seq dim {attn.kv_cache.shape[1]} "
                    f"!= {expected_cache_size}"
                )
                if attn.indexer is not None:
                    assert (
                        attn.indexer.kv_cache.shape[1]
                        == self.max_seq_len // attn.indexer.compress_ratio
                    ), (
                        f"layer {i}: Indexer.kv_cache seq dim "
                        f"{attn.indexer.kv_cache.shape[1]} incompatible with "
                        f"max_seq_len {self.max_seq_len}"
                    )

        # Native ckpt stores ``head.weight`` as bf16 (audit); we upcast to fp32
        # at construction time to match official ``ParallelHead`` semantics.
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(
            self.lm_head.weight.detach().to(torch.float32)
        )
        # Sanity: the registered Parameter must actually be fp32 for the
        # construction contract to hold.
        assert self.lm_head.weight.dtype == torch.float32, (
            f"lm_head.weight must be fp32 by construction; got {self.lm_head.weight.dtype}"
        )

        def _cast_input_to_fp32(_mod, args, kwargs):
            if args:
                x = args[0]
                if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
                    args = (x.float(),) + args[1:]
            return args, kwargs

        self.lm_head.register_forward_pre_hook(_cast_input_to_fp32, with_kwargs=True)
        _maybe_wrap_deepseek_v4_lm_head_canonical(self.lm_head)

    def prepare_cuda_graph_capture(self) -> None:
        for layer in self.model.layers:
            layer.mlp.prepare_cuda_graph_capture()

    def refresh_cuda_graph_expert_weights(self) -> None:
        with torch.no_grad():
            for layer in self.model.layers:
                layer.mlp.refresh_cuda_graph_expert_weights()

    def _forward_stateful_prefix_hit_extend(
        self,
        input_ids: torch.Tensor,
        stateful_meta: Any,
        attn_backend: Optional[Any],
    ) -> torch.Tensor:
        """Replay a flat extend batch request-by-request through the full model.

        DeepSeek V4 prefix hits restore stateful attention/compressor buffers.
        If the suffix has multiple tokens, HC/MLP/layer state must advance in
        full-model order for each request; per-layer suffix replay is only valid
        for fresh prefill or single-token decode.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(
                "DeepSeek V4 prefix-hit extend replay expects flat input_ids shaped "
                f"[1, total_tokens], got {tuple(input_ids.shape)}."
            )

        outputs: list[torch.Tensor] = []
        batch_size = int(stateful_meta.batch_size)
        cu_seqlens_q = getattr(stateful_meta, "cu_seqlens_q_cpu", None)
        start_pos_cpu = getattr(stateful_meta, "start_pos_cpu", None)
        req_indices_cpu = getattr(stateful_meta, "req_indices_cpu", None)
        if cu_seqlens_q is None or start_pos_cpu is None or req_indices_cpu is None:
            raise RuntimeError(
                "DeepSeek V4 prefix-hit replay requires CPU metadata mirrors built by "
                "the attention backend."
            )

        for b in range(batch_size):
            start = cu_seqlens_q[b]
            end = cu_seqlens_q[b + 1]
            input_ids_b = input_ids[:, start:end].contiguous()
            start_pos_b = start_pos_cpu[b]
            req_idx_b = req_indices_cpu[b]

            def call_with_oft_scope(fn, token_len: int):
                if attn_backend is not None and hasattr(
                    attn_backend, "_call_with_single_request_oft_batch"
                ):
                    return attn_backend._call_with_single_request_oft_batch(
                        b, fn, token_len=token_len
                    )
                return fn()

            if start_pos_b > 0 and input_ids_b.shape[1] > 1:
                token_outputs = []
                for t in range(input_ids_b.shape[1]):
                    input_ids_t = input_ids_b[:, t : t + 1].contiguous()

                    def run_token(input_ids_t=input_ids_t, t=t):
                        return self.model.forward_stateful(
                            input_ids_t,
                            start_pos=start_pos_b + t,
                            req_idx=req_idx_b,
                        )

                    token_outputs.append(call_with_oft_scope(run_token, 1))
                out_b = torch.cat(token_outputs, dim=0)
            else:
                def run_request(input_ids_b=input_ids_b):
                    return self.model.forward_stateful(
                        input_ids_b,
                        start_pos=start_pos_b,
                        req_idx=req_idx_b,
                    )

                out_b = call_with_oft_scope(run_request, input_ids_b.shape[1])
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Any = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the V4 decoder + fp32 lm_head.

        ``positions`` is accepted for SGLang signature compat but unused
        (V4 derives positions internally from token offsets).
        ``forward_batch`` is required and threaded down to each layer's
        attention block so the DeepSeek V4 scheduler backend can provide request-slot
        metadata.

        Returns last-token logits ``[batch, vocab]`` in fp32. The fp32 head
        matmul is gated by the construction-time upcast + forward-pre-hook
        on ``self.lm_head``.
        """
        forward_batch = _deepseek_v4_require_forward_batch(
            forward_batch, "DeepseekV4ForCausalLM.forward"
        )
        del positions, kwargs  # SGLang signature compat only.

        # SGLang's scheduler passes ``input_ids`` as 1D ``[total_tokens]``
        # (flat-token batching). Some tests pass 2D ``[b, s]``. Normalise to
        # 2D before the decoder body so embedding + permute stays unchanged.
        if input_ids.dim() == 1:
            if (
                getattr(forward_batch, "forward_mode", None) is not None
                and forward_batch.forward_mode.is_decode()
            ):
                # Decode: one new token per request, total_tokens == batch.
                input_ids = input_ids.unsqueeze(1)  # [b, 1]
            else:
                # Extend: prefill tokens arrive as one flat stream. The DeepSeek
                # V4 attention backend splits multi-request batches with the
                # scheduler-provided cu_seqlens metadata.
                input_ids = input_ids.unsqueeze(0)  # [1, total_tokens]

        # V4's fp8/fp4 kernels (in inference/kernel.py) allocate output
        # buffers via ``torch.get_default_dtype()``. The test harness sets
        # bf16 globally; the SGLang server runs forward with the process-
        # default fp32, which makes the kernel reject the C tensor. Wrap
        # forward in a default-dtype context so kernels see bf16.
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        with set_default_torch_dtype(torch.bfloat16):
            # Run the decoder body. Returns ``[s, b, d]`` (Megatron convention).
            attn_backend = getattr(forward_batch, "attn_backend", None)
            stateful_meta = getattr(attn_backend, "_cached_metadata", None)
            if (
                stateful_meta is not None
                and not getattr(stateful_meta, "is_decode", False)
                and input_ids.dim() == 2
                and input_ids.shape[0] == 1
                and input_ids.shape[1] > 1
                and getattr(stateful_meta, "has_prefix_hit", False)
            ):
                h_sbd = self._forward_stateful_prefix_hit_extend(
                    input_ids, stateful_meta, attn_backend
                )
            else:
                h_sbd = self.model(input_ids, forward_batch=forward_batch)

        # When called from the SGLang server (real ForwardBatch), defer to
        # SGLang's stock LogitsProcessor so input_token_logprobs and the rest
        # of the LogitsProcessorOutput fields get populated for GRPO/eval.
        # The hidden_states need to be flat ``[total_tokens, dim]`` to match
        # SGLang's contract.
        if hasattr(forward_batch, "sampling_info"):
            # h_sbd is [s, b, d] (Megatron); SGLang expects flat [total, d].
            s, b, d = h_sbd.shape
            hidden_states_flat = h_sbd.permute(1, 0, 2).reshape(s * b, d)
            from sglang.srt.layers.logits_processor import LogitsProcessor
            if not hasattr(self, "_dsv4_logits_processor"):
                # Lazily build a LogitsProcessor. ``self.config`` is the V4
                # config; LogitsProcessor reads ``vocab_size`` and a few flags.
                self._dsv4_logits_processor = LogitsProcessor(
                    self.config, skip_all_gather=True
                ).to(self.lm_head.weight.device)
            return self._dsv4_logits_processor(
                input_ids,
                hidden_states_flat,
                self.lm_head,
                forward_batch,
            )

        # Test-harness path: return last-token logits for SimpleNamespace-style
        # ForwardBatches.
        with set_default_torch_dtype(torch.bfloat16):
            last_bd = h_sbd[-1]
            logits = self.lm_head(last_bd)  # [b, vocab]
        return logits

    # ---- Stateful top-level entry. Drives the stateful prefill+decode path
    # that owns its own per-layer kv_cache + compressor state buffers (NOT
    # SGLang's paged KV pool). Produces last-token logits ``[batch, vocab]``.
    #
    # Caller contract: invoke once with `start_pos=0` for the prefill, then
    # once per decode step with `start_pos = prompt_len + k - 1` for k = 1..K.
    # The model's per-layer state is reset only by re-instantiating the
    # model (or by calling `reset_inference_state()` if added later).
    def forward_inference(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        req_idx: int = 0,
    ) -> torch.Tensor:
        h_sbd = self.model.forward_stateful(
            input_ids, start_pos=start_pos, req_idx=req_idx
        )
        last_bd = h_sbd[-1]
        logits = self.lm_head(last_bd)
        return logits

    def reset_inference_state(self, req_idx: int) -> None:
        """Zero out per-request V4 state for slot ``req_idx``.

        Called by ``DeepSeekV4AttentionBackend.init_forward_metadata`` on
        every fresh extend (`extend_prefix_lens[i] == 0`) before any
        layer touches the slot. This prevents stale state from a prior
        tenant of the same `req_pool_idx` leaking into the next request.

        Idempotent. Safe to call any number of times. Hot-path during
        scheduler dispatch but only fires on fresh extend, not decode.
        """
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.kv_cache[req_idx].zero_()
            if attn.compressor is not None:
                attn.compressor.kv_state[req_idx].zero_()
                attn.compressor.score_state[req_idx].fill_(float("-inf"))
                # `compressor.kv_cache` is a view into
                # `Attention.kv_cache[:, win:]`, already zeroed above.
            if attn.indexer is not None:
                attn.indexer.kv_cache[req_idx].zero_()
                attn.indexer.compressor.kv_state[req_idx].zero_()
                attn.indexer.compressor.score_state[req_idx].fill_(
                    float("-inf")
                )

    def export_prefix_state(self, req_idx: int, prefix_len: int) -> dict[str, Any]:
        """Clone V4-owned per-request state for radix-prefix reuse.

        The scheduler KV pool only describes vanilla token slots. DeepSeek V4
        decode also depends on each layer's stateful window/compressed/indexer
        buffers, so a radix hit must restore these tensors before computing
        the suffix. This method intentionally only snapshots state; cache
        insertion/eviction policy stays outside the model.
        """
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")

        layers = []
        for layer in self.model.layers:
            attn = layer.self_attn
            layer_state: dict[str, Any] = {
                "prefix_len": int(prefix_len),
                "window_size": int(attn.window_size),
                "compress_ratio": int(attn.compress_ratio or 0),
                "vanilla_history_len": min(int(prefix_len), int(attn.window_size)),
                "attn_kv_cache": attn.kv_cache[req_idx].detach().clone(),
            }
            if attn.compressor is not None:
                ratio = int(attn.compress_ratio)
                layer_state["compressed_history_len"] = int(prefix_len) // ratio
                layer_state["compressor_partial_len"] = int(prefix_len) % ratio
                layer_state["compressor_kv_state"] = (
                    attn.compressor.kv_state[req_idx].detach().clone()
                )
                layer_state["compressor_score_state"] = (
                    attn.compressor.score_state[req_idx].detach().clone()
                )
            else:
                layer_state["compressed_history_len"] = 0
                layer_state["compressor_partial_len"] = 0

            if attn.indexer is not None:
                layer_state["indexer_history_len"] = int(prefix_len) // int(
                    attn.indexer.compress_ratio
                )
                layer_state["indexer_kv_cache"] = (
                    attn.indexer.kv_cache[req_idx].detach().clone()
                )
                layer_state["indexer_compressor_kv_state"] = (
                    attn.indexer.compressor.kv_state[req_idx].detach().clone()
                )
                layer_state["indexer_compressor_score_state"] = (
                    attn.indexer.compressor.score_state[req_idx].detach().clone()
                )
            else:
                layer_state["indexer_history_len"] = 0
            layers.append(layer_state)

        return {
            "prefix_len": int(prefix_len),
            "num_layers": len(layers),
            "layers": layers,
        }

    def restore_prefix_state(self, req_idx: int, prefix_state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`export_prefix_state` into a slot."""
        layers = prefix_state.get("layers")
        if not isinstance(layers, list) or len(layers) != len(self.model.layers):
            raise ValueError(
                "Invalid DeepSeek V4 prefix state: layer count does not match model."
            )

        self.reset_inference_state(req_idx)
        for i, (layer, layer_state) in enumerate(zip(self.model.layers, layers)):
            attn = layer.self_attn

            def _copy_slot(name: str, dst: torch.Tensor, src: torch.Tensor) -> None:
                if tuple(dst.shape) != tuple(src.shape):
                    raise ValueError(
                        f"Invalid DeepSeek V4 prefix state for layer {i} {name}: "
                        f"expected shape {tuple(dst.shape)}, got {tuple(src.shape)}"
                    )
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

            _copy_slot(
                "attn_kv_cache",
                attn.kv_cache[req_idx],
                layer_state["attn_kv_cache"],
            )
            if attn.compressor is not None:
                _copy_slot(
                    "compressor_kv_state",
                    attn.compressor.kv_state[req_idx],
                    layer_state["compressor_kv_state"],
                )
                _copy_slot(
                    "compressor_score_state",
                    attn.compressor.score_state[req_idx],
                    layer_state["compressor_score_state"],
                )
            if attn.indexer is not None:
                _copy_slot(
                    "indexer_kv_cache",
                    attn.indexer.kv_cache[req_idx],
                    layer_state["indexer_kv_cache"],
                )
                _copy_slot(
                    "indexer_compressor_kv_state",
                    attn.indexer.compressor.kv_state[req_idx],
                    layer_state["indexer_compressor_kv_state"],
                )
                _copy_slot(
                    "indexer_compressor_score_state",
                    attn.indexer.compressor.score_state[req_idx],
                    layer_state["indexer_compressor_score_state"],
                )

    # ---- OFT integration hooks. Both methods are called by
    # ``OFTManager`` during adapter loading (cf. oft_manager.py:735-740 for
    # ``should_apply_oft`` and oft_manager.py:53-63 for
    # ``validate_oft_target_modules``). Attention OFT adapts the five direct
    # ``DeepSeekV4Linear`` sublayers (``wq_a/wq_b/wkv/wo_a/wo_b``); the V4
    # Compressor (``self_attn.compressor.wkv``, ``self_attn.compressor.wgate``)
    # and Indexer (``self_attn.indexer.linear_wq_b``, etc.) reuse the same
    # short attribute names but live one level deeper, so a bare-suffix
    # match in OFTManager.init_oft_modules would over-wrap. ``should_apply_oft``
    # rejects those depth-2 cases. Routed expert OFT is handled on
    # ``DeepSeekV4MoE`` itself, like SGLang's FusedMoE expert OFT path, so
    # ``w1/w2/w3`` are allowed but intentionally not generically wrapped. ----
    _DSV4_ATTENTION_OFT_SUFFIXES = frozenset(
        {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}
    )
    _DSV4_EXPERT_OFT_SUFFIXES = frozenset({"w1", "w2", "w3"})
    _DSV4_ALLOWED_OFT_SUFFIXES = (
        _DSV4_ATTENTION_OFT_SUFFIXES | _DSV4_EXPERT_OFT_SUFFIXES
    )

    def should_apply_oft(self, module_name: str) -> bool:
        """Reject Compressor / Indexer modules that share attention suffix
        names but are out of the attention OFT scope (``self_attn.<name>`` only)."""
        parts = module_name.split(".")
        if len(parts) < 2:
            return False
        suffix = parts[-1]
        if suffix in self._DSV4_EXPERT_OFT_SUFFIXES:
            return False
        if suffix not in self._DSV4_ATTENTION_OFT_SUFFIXES:
            # Other modules — let OFTManager's per-suffix gate decide. We
            # only have an opinion about the DeepSeek V4 native OFT suffix set.
            return True
        # The five attention-only suffixes must be DIRECTLY under
        # ``self_attn`` to count as attention OFT scope. Anything one level deeper
        # (``self_attn.compressor.wkv``, ``self_attn.indexer.linear_wq_b``)
        # is rejected here.
        parent = parts[-2]
        return parent == "self_attn"

    def get_oft_external_target_modules(self) -> set[str]:
        return set(self._DSV4_EXPERT_OFT_SUFFIXES)

    def validate_oft_target_modules(self, target_modules: set) -> None:
        """Reject adapter ``target_modules`` outside the DeepSeek V4 OFT set.

        Called by ``OFTManager.validate_new_adapter`` via
        ``validate_model_oft_target_modules``.
        """
        unsupported = set(target_modules) - self._DSV4_ALLOWED_OFT_SUFFIXES
        if unsupported:
            raise ValueError(
                "DeepSeek V4 OFT supports attention sublayers "
                f"{sorted(self._DSV4_ATTENTION_OFT_SUFFIXES)} and routed "
                f"expert sublayers {sorted(self._DSV4_EXPERT_OFT_SUFFIXES)}; got "
                f"unsupported target modules {sorted(unsupported)}. "
                "V4 has no fused qkv_proj or gate_up_proj layers."
            )

    # ---- SGLang DefaultModelLoader hook. ``load_model`` calls
    # ``model.load_weights(weights_iter)`` with `(name, tensor)` pairs from
    # the safetensors iterator. ----
    def load_weights(self, weights) -> None:
        name_to_param = dict(self.named_parameters())
        for name, tensor in weights:
            if _is_ignore(name):
                continue
            target = _rewrite_native_key(name)
            if target is None:
                continue
            if _try_load_packed_routed_expert_param(self, target, tensor):
                continue
            param = name_to_param.get(target)
            if param is None:
                continue
            # Copy into the module's construction dtype. Official fp32
            # in-flight parameters load bf16 checkpoint tensors once here.
            _copy_deepseek_v4_param_data(param, tensor)
        self.refresh_cuda_graph_expert_weights()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward-compatible helper aliases.
# ---------------------------------------------------------------------------

_dsv4_update_cache_slots_ = _deepseek_v4_update_cache_slots_
_dsv4_require_metadata_tensor = _deepseek_v4_require_metadata_tensor
_dsv4_require_forward_batch = _deepseek_v4_require_forward_batch
_dsv4_env_impl = _deepseek_v4_env_impl
_dsv4_canonical_linear = _deepseek_v4_canonical_linear
_dsv4_best_linear = _deepseek_v4_best_linear
_maybe_wrap_dsv4_lm_head_canonical = _maybe_wrap_deepseek_v4_lm_head_canonical
_dsv4_use_tilekernels = _deepseek_v4_use_tilekernels
_dsv4_fast_mhc_pre_enabled = _deepseek_v4_fast_mhc_pre_enabled
_dsv4_tile_mhc_ops = _deepseek_v4_tile_mhc_ops
_dsv4_tile_router_op = _deepseek_v4_tile_router_op
_dsv4_linear_storage_shapes = _deepseek_v4_linear_storage_shapes
_mark_dsv4_tp_shard = _mark_deepseek_v4_tp_shard
_copy_dsv4_param_data = _copy_deepseek_v4_param_data
_dsv4_apply_expert_oft_r = _deepseek_v4_apply_expert_oft_r
_dsv4_moe_ep_info = _deepseek_v4_moe_ep_info
_dsv4_moe_tp_info = _deepseek_v4_moe_tp_info
_slice_dsv4_routed_expert_param_for_tp = (
    _slice_deepseek_v4_routed_expert_param_for_tp
)


# ---------------------------------------------------------------------------
# SGLang ModelRegistry hook. The registry scans this module for ``EntryClass``
# at import time (see ``sglang.srt.models.registry.import_model_classes``).
# ---------------------------------------------------------------------------

EntryClass = DeepseekV4ForCausalLM

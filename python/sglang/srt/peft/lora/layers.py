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
"""Single-active LoRA layer wrappers (own implementation, plain torch).

Unlike srt/lora (multi-adapter serving: per-request ``weight_indices``,
segment-GEMM ``lora_backend``), this module applies ONE active adapter's LoRA
delta to the WHOLE batch with plain torch ops:

    delta = (x @ A.T) @ B.T * scaling   # x:(...,in) -> (...,r) -> (...,out)
    out = base_output + delta

Fused base linears (QKV, gate_up) are a single ``nn.Linear`` whose output is a
concatenation of sub-projection slices. LoRA is applied PER SUB-PROJECTION to
the matching output slice, rather than replicating srt/lora's fused
``run_qkv_lora``/``run_gate_up_lora`` kernels.
"""

import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.lora.layers import FusedMoEWithLoRA


class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass

    def get_local_tp_rank(self) -> int:
        return getattr(self.base_layer, "tp_rank", 0)

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def set_lora_info(self, A: torch.Tensor, B: torch.Tensor, scaling: float):
        self.set_lora = True
        # self.A / self.B are VIEWS into the pool's ACTIVE slot, bound once by
        # LoRAManager._bind_dense_lora_views before this is ever called (both
        # for a disk-loaded adapter and for bind_zero_lora's identity boot) --
        # copy_ in place, never reassign, so a captured CUDA graph (which pins
        # data pointers) stays valid. Scaling is folded into B (data, not a
        # baked constant), so a scaling change is CUDA-graph-safe.
        self.A.copy_(A)
        self.B.copy_(B * scaling)
        # Kept for compat/logging only; the math no longer reads it.
        self.scaling = scaling

    def update_lora_info(self, A: torch.Tensor, B: torch.Tensor, scaling: float):
        # In-place streamed update: copy_ into the already-bound tensors so a
        # captured CUDA graph (which pins data pointers) stays valid. Scaling is
        # folded into B (data, not a baked constant), so a scaling change is
        # CUDA-graph-safe. Never reassign self.A / self.B.
        self.A.copy_(A)
        self.B.copy_(B * scaling)

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return base_output + (x @ self.A.T) @ self.B.T

    def dense_lora_shapes(self, rank: int):
        """((A_shape, B_shape),) for this wrapper's one dense sub-projection --
        used by LoRAMemoryPool._declare_groups to size the pool's "A:<sub>"/
        "B:<sub>" groups before any adapter is loaded. Same in/out formulas
        bind_zero_lora below uses to allocate."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        out = self.base_layer.output_partition_sizes[0]
        return (((rank, in_features), (out, rank)),)

    def bind_zero_lora(self, rank: int, dtype, device):
        """Bind ZERO LoRA buffers (set_lora=True) so an identity boot captures the
        LoRA path and a later streamed update fills the pinned buffers in place."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        out = self.base_layer.output_partition_sizes[0]
        A = torch.zeros(rank, in_features, dtype=dtype, device=device)
        B = torch.zeros(out, rank, dtype=dtype, device=device)
        self.set_lora_info(A, B, 1.0)

    def forward(self, input_: torch.Tensor, forward_batch=None):
        # duplicate the logic in ColumnParallelLinear. Accept forward_batch to
        # stay a drop-in for the base linear (v0.5.14 models pass it through to
        # some linears, e.g. Qwen2 MLP -> down_proj); it only affects the TP>1
        # quant-comm all-reduce path, which the tp=1 rollout does not hit.
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[start_idx:end_idx, :]


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def set_lora_info(
        self,
        A_gate: torch.Tensor,
        B_gate: torch.Tensor,
        A_up: torch.Tensor,
        B_up: torch.Tensor,
        scaling: float,
    ):
        self.set_lora = True
        # self.A_gate/self.B_gate/self.A_up/self.B_up are VIEWS into the
        # pool's ACTIVE slot, bound once by LoRAManager._bind_dense_lora_views
        # before this is ever called -- copy_ in place, never reassign (see
        # ColumnParallelLinearWithLoRA.set_lora_info). Scaling is folded into
        # each B (data, not a baked constant).
        self.A_gate.copy_(A_gate)
        self.B_gate.copy_(B_gate * scaling)
        self.A_up.copy_(A_up)
        self.B_up.copy_(B_up * scaling)
        # Kept for compat/logging only; the math no longer reads it.
        self.scaling = scaling

    def update_lora_info(
        self,
        A_gate: torch.Tensor,
        B_gate: torch.Tensor,
        A_up: torch.Tensor,
        B_up: torch.Tensor,
        scaling: float,
    ):
        # In-place streamed update: copy_ into the already-bound tensors so a
        # captured CUDA graph (which pins data pointers) stays valid. Scaling is
        # folded into each B (data, not a baked constant), so a scaling change is
        # CUDA-graph-safe. Never reassign self.A_* / self.B_*.
        self.A_gate.copy_(A_gate)
        self.B_gate.copy_(B_gate * scaling)
        self.A_up.copy_(A_up)
        self.B_up.copy_(B_up * scaling)

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # gate_up is one fused nn.Linear; write each sub-projection's delta
        # into its own output slice rather than a fused kernel.
        gate_size = self.base_layer.output_partition_sizes[0]
        base_output[..., :gate_size] += (x @ self.A_gate.T) @ self.B_gate.T
        base_output[..., gate_size:] += (x @ self.A_up.T) @ self.B_up.T
        return base_output

    def dense_lora_shapes(self, rank: int):
        """((A_gate_shape, B_gate_shape), (A_up_shape, B_up_shape)) -- see
        ColumnParallelLinearWithLoRA.dense_lora_shapes. Order matches
        mem_pool.DENSE_SUB_NAMES["gate_up_proj"] = ("gate", "up")."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        gate = self.base_layer.output_partition_sizes[0]
        up = self.base_layer.output_partition_sizes[1]
        return (
            ((rank, in_features), (gate, rank)),
            ((rank, in_features), (up, rank)),
        )

    def bind_zero_lora(self, rank: int, dtype, device):
        """Bind ZERO LoRA buffers (set_lora=True) so an identity boot captures the
        LoRA path and a later streamed update fills the pinned buffers in place."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        gate = self.base_layer.output_partition_sizes[0]
        up = self.base_layer.output_partition_sizes[1]
        A_gate = torch.zeros(rank, in_features, dtype=dtype, device=device)
        A_up = torch.zeros(rank, in_features, dtype=dtype, device=device)
        B_gate = torch.zeros(gate, rank, dtype=dtype, device=device)
        B_up = torch.zeros(up, rank, dtype=dtype, device=device)
        self.set_lora_info(A_gate, B_gate, A_up, B_up, 1.0)

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # B is the fused (gate;up) weight, unsharded. Since gate/up output
        # sizes are identical, one shard_size is reused (mirrors srt/lora).
        shard_size = self.base_layer.output_partition_sizes[0]
        gate_size = self.base_layer.output_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return torch.cat(
            (
                B[start_idx:end_idx, :],
                B[gate_size + start_idx : gate_size + end_idx, :],
            ),
            dim=0,
        )

    def slice_streamed_b_weights(
        self, B_gate: torch.Tensor, B_up: torch.Tensor, tp_rank: int
    ):
        """Slice the FULL per-sub-proj gate/up lora_B to this rank's TP shard, on
        the UN-fused per-sub-proj tensors that _resolve_module_args produces
        (set_lora_info takes gate/up separately). lora_A is not sliced (column
        input replicated). gate/up share the same per-rank output size."""
        gate = self.base_layer.output_partition_sizes[0]
        up = self.base_layer.output_partition_sizes[1]
        return (
            B_gate[gate * tp_rank : gate * (tp_rank + 1), :],
            B_up[up * tp_rank : up * (tp_rank + 1), :],
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)

    def set_lora_info(
        self,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        A_k: torch.Tensor,
        B_k: torch.Tensor,
        A_v: torch.Tensor,
        B_v: torch.Tensor,
        scaling: float,
    ):
        self.set_lora = True
        # self.A_q/self.B_q/self.A_k/self.B_k/self.A_v/self.B_v are VIEWS into
        # the pool's ACTIVE slot, bound once by LoRAManager._bind_dense_lora_
        # views before this is ever called -- copy_ in place, never reassign
        # (see ColumnParallelLinearWithLoRA.set_lora_info). Scaling is folded
        # into each B (data, not a baked constant).
        self.A_q.copy_(A_q)
        self.B_q.copy_(B_q * scaling)
        self.A_k.copy_(A_k)
        self.B_k.copy_(B_k * scaling)
        self.A_v.copy_(A_v)
        self.B_v.copy_(B_v * scaling)
        # Kept for compat/logging only; the math no longer reads it.
        self.scaling = scaling

    def update_lora_info(
        self,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        A_k: torch.Tensor,
        B_k: torch.Tensor,
        A_v: torch.Tensor,
        B_v: torch.Tensor,
        scaling: float,
    ):
        # In-place streamed update: copy_ into the already-bound tensors so a
        # captured CUDA graph (which pins data pointers) stays valid. Scaling is
        # folded into each B (data, not a baked constant), so a scaling change is
        # CUDA-graph-safe. Never reassign self.A_* / self.B_*.
        self.A_q.copy_(A_q)
        self.B_q.copy_(B_q * scaling)
        self.A_k.copy_(A_k)
        self.B_k.copy_(B_k * scaling)
        self.A_v.copy_(A_v)
        self.B_v.copy_(B_v * scaling)

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # q/k/v is one fused nn.Linear; write each sub-projection's delta into
        # its own output slice. GQA: k/v slices are sized by kv_proj_shard_size
        # (already head-count-correct for GQA), not q_proj_shard_size.
        q_size = self.base_layer.q_proj_shard_size
        kv_size = self.base_layer.kv_proj_shard_size
        base_output[..., :q_size] += (x @ self.A_q.T) @ self.B_q.T
        base_output[..., q_size : q_size + kv_size] += (x @ self.A_k.T) @ self.B_k.T
        base_output[..., q_size + kv_size : q_size + 2 * kv_size] += (
            x @ self.A_v.T
        ) @ self.B_v.T
        return base_output

    def dense_lora_shapes(self, rank: int):
        """((A_q_shape, B_q_shape), (A_k...), (A_v...)) -- see
        ColumnParallelLinearWithLoRA.dense_lora_shapes. Order matches
        mem_pool.DENSE_SUB_NAMES["qkv_proj"] = ("q", "k", "v"). k/v share
        kv_proj_shard_size (GQA-aware, already TP-partitioned by the base
        QKVParallelLinear)."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        q = self.base_layer.q_proj_shard_size
        kv = self.base_layer.kv_proj_shard_size
        return (
            ((rank, in_features), (q, rank)),
            ((rank, in_features), (kv, rank)),
            ((rank, in_features), (kv, rank)),
        )

    def bind_zero_lora(self, rank: int, dtype, device):
        """Bind ZERO LoRA buffers (set_lora=True) so an identity boot captures the
        LoRA path and a later streamed update fills the pinned buffers in place."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        q = self.base_layer.q_proj_shard_size
        kv = self.base_layer.kv_proj_shard_size
        A_q = torch.zeros(rank, in_features, dtype=dtype, device=device)
        A_k = torch.zeros(rank, in_features, dtype=dtype, device=device)
        A_v = torch.zeros(rank, in_features, dtype=dtype, device=device)
        B_q = torch.zeros(q, rank, dtype=dtype, device=device)
        B_k = torch.zeros(kv, rank, dtype=dtype, device=device)
        B_v = torch.zeros(kv, rank, dtype=dtype, device=device)
        self.set_lora_info(A_q, B_q, A_k, B_k, A_v, B_v, 1.0)

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int) -> torch.Tensor:
        # B is the fused (q;k;v) weight, unsharded. Mirrors srt/lora's
        # GQA-aware slicing: k/v shard by kv_proj_shard_size and replicate
        # across num_kv_head_replicas TP ranks.
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        q_size = base_layer.total_num_heads * base_layer.head_size
        k_size = base_layer.total_num_kv_heads * base_layer.head_size
        B_q_shard = B[q_start_idx:q_end_idx, :]
        B_k_shard = B[q_size + kv_start_idx : q_size + kv_end_idx, :]
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]

        return torch.cat((B_q_shard, B_k_shard, B_v_shard), dim=0)

    def slice_streamed_b_weights(
        self, B_q: torch.Tensor, B_k: torch.Tensor, B_v: torch.Tensor, tp_rank: int
    ):
        """Slice the FULL per-sub-proj q/k/v lora_B to this rank's TP shard
        (GQA-aware). Same shard math as slice_lora_b_weights, but on the UN-fused
        per-sub-proj tensors that _resolve_module_args produces (set_lora_info
        takes q/k/v separately). lora_A is not sliced (column input replicated)."""
        q = self.base_layer.q_proj_shard_size
        kv = self.base_layer.kv_proj_shard_size
        kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        return (
            B_q[q * tp_rank : q * (tp_rank + 1), :],
            B_k[kv * kv_shard_id : kv * (kv_shard_id + 1), :],
            B_v[kv * kv_shard_id : kv * (kv_shard_id + 1), :],
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

    def set_lora_info(self, A: torch.Tensor, B: torch.Tensor, scaling: float):
        self.set_lora = True
        # self.A / self.B are VIEWS into the pool's ACTIVE slot, bound once by
        # LoRAManager._bind_dense_lora_views before this is ever called -- copy_
        # in place, never reassign (see ColumnParallelLinearWithLoRA.set_lora_
        # info). Scaling is folded into B (data, not a baked constant).
        self.A.copy_(A)
        self.B.copy_(B * scaling)
        # Kept for compat/logging only; the math no longer reads it.
        self.scaling = scaling

    def update_lora_info(self, A: torch.Tensor, B: torch.Tensor, scaling: float):
        # In-place streamed update: copy_ into the already-bound tensors so a
        # captured CUDA graph (which pins data pointers) stays valid. Scaling is
        # folded into B (data, not a baked constant), so a scaling change is
        # CUDA-graph-safe. Never reassign self.A / self.B.
        self.A.copy_(A)
        self.B.copy_(B * scaling)

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return base_output + (x @ self.A.T) @ self.B.T

    def dense_lora_shapes(self, rank: int):
        """((A_shape, B_shape),) for this wrapper's one dense sub-projection --
        see ColumnParallelLinearWithLoRA.dense_lora_shapes. A is sized to the
        TP-LOCAL input (input_size_per_partition), matching bind_zero_lora
        below and slice_lora_a_weights."""
        in_features = self.base_layer.input_size_per_partition
        out = self.base_layer.output_size
        return (((rank, in_features), (out, rank)),)

    def bind_zero_lora(self, rank: int, dtype, device):
        """Bind ZERO LoRA buffers (set_lora=True) so an identity boot captures the
        LoRA path and a later streamed update fills the pinned buffers in place.

        A is sized to the TP-LOCAL input (input_size_per_partition): a
        RowParallel base layer shards its input across TP ranks, so apply_lora
        feeds LoRA the per-rank input shard -- the delta (x_shard @ A.T) @ B.T
        requires a per-rank A. This matches slice_lora_a_weights (which slices
        the streamed full A to input_size_per_partition). At tp=1,
        input_size_per_partition == input_size, so this is a no-op. B is
        unsharded (row-parallel output is replicated pre-all-reduce)."""
        in_features = self.base_layer.input_size_per_partition
        out = self.base_layer.output_size
        A = torch.zeros(rank, in_features, dtype=dtype, device=device)
        B = torch.zeros(out, rank, dtype=dtype, device=device)
        self.set_lora_info(A, B, 1.0)

    def forward(self, input_: torch.Tensor, skip_all_reduce: bool = False, forward_batch=None):
        # duplicate the logic in RowParallelLinear. Accept forward_batch to match
        # the base RowParallelLinear.forward(input_, skip_all_reduce, forward_batch)
        # (v0.5.14 Qwen2 MLP passes forward_batch to down_proj); it only selects
        # the TP>1 quant-comm all-reduce, not exercised at tp=1.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            # Apply LoRA to the partial (pre-all-reduce) output — matches
            # srt/lora's ordering: A is sliced per-rank's input shard, so the
            # subsequent all-reduce sums each rank's partial delta into the
            # full delta, same as it sums the base partial outputs.
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        ):
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return A[:, start_idx:end_idx].contiguous()

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B


class ReplicatedLinearWithLoRA(BaseLayerWithLoRA):
    """Single-active LoRA for a ReplicatedLinear that fuses two UNEQUAL-size
    sub-projections (DeepSeek/Kimi MLA fused_qkv_a_proj_with_mqa = q_a + kv_a).
    No TP sharding (replicated). LoRA is applied per sub-projection to its
    output slice; the q_a/kv_a boundary is the q_a lora_B output dim."""

    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__(base_layer)

    def set_lora_info(self, A_q_a, B_q_a, A_kv_a, B_kv_a, scaling: float):
        self.set_lora = True
        # VIEWS into the pool's ACTIVE slot, bound once by the manager -- copy_
        # in place, never reassign (CUDA-graph safe). Scaling folded into each B.
        self.A_q_a.copy_(A_q_a)
        self.B_q_a.copy_(B_q_a * scaling)
        self.A_kv_a.copy_(A_kv_a)
        self.B_kv_a.copy_(B_kv_a * scaling)
        self.scaling = scaling

    def update_lora_info(self, A_q_a, B_q_a, A_kv_a, B_kv_a, scaling: float):
        self.A_q_a.copy_(A_q_a)
        self.B_q_a.copy_(B_q_a * scaling)
        self.A_kv_a.copy_(A_kv_a)
        self.B_kv_a.copy_(B_kv_a * scaling)

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q_a_size = self.B_q_a.shape[0]
        base_output[..., :q_a_size] += (x @ self.A_q_a.T) @ self.B_q_a.T
        base_output[..., q_a_size:] += (x @ self.A_kv_a.T) @ self.B_kv_a.T
        return base_output

    def dense_lora_shapes(self, rank: int):
        """((A_q_a, B_q_a), (A_kv_a, B_kv_a)) -- order matches
        mem_pool.DENSE_SUB_NAMES["fused_qkv_a_proj_with_mqa"] = ("q_a", "kv_a").
        q_a/kv_a output sizes come from the base layer's output_sizes."""
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        q_a, kv_a = self.base_layer.output_sizes  # [q_lora_rank, kv_lora_rank+rope]
        return (((rank, in_features), (q_a, rank)),
                ((rank, in_features), (kv_a, rank)))

    def bind_zero_lora(self, rank: int, dtype, device):
        in_features = getattr(self.base_layer, "input_size", None)
        if in_features is None:
            in_features = self.base_layer.weight.shape[1]
        q_a, kv_a = self.base_layer.output_sizes
        self.set_lora_info(
            torch.zeros(rank, in_features, dtype=dtype, device=device),
            torch.zeros(q_a, rank, dtype=dtype, device=device),
            torch.zeros(rank, in_features, dtype=dtype, device=device),
            torch.zeros(kv_a, rank, dtype=dtype, device=device),
            1.0,
        )

    def forward(self, x: torch.Tensor):
        # Mirror ReplicatedLinear.forward: single un-sharded GEMM, no gather.
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        if self.set_lora:
            output = self.apply_lora(output, x)
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A  # replicated: no input sharding

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B  # replicated: no output sharding


def get_lora_layer(base_layer: nn.Module, backend=None) -> BaseLayerWithLoRA:
    # FusedMoE (expert LoRA) reuses upstream's own wrapper verbatim -- it
    # drives upstream's MoE forward/hooks/align/CUDA-graph path unmodified
    # (`.superpowers/sdd/phase3-integration-spec.md` §5/§6 Option A). Unlike
    # the single-active dense wrappers below, it needs a `lora_backend`
    # (`peft.lora.moe_backend.SingleActiveMoEBackend`) to read its per-batch
    # routing off of.
    if isinstance(base_layer, FusedMoE):
        assert backend is not None, (
            "get_lora_layer requires a lora backend (SingleActiveMoEBackend) "
            "to wrap a FusedMoE module."
        )
        return FusedMoEWithLoRA(base_layer, backend)

    supported_layer_types = {
        # the order matters: fused types must be checked before their base class
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
        ReplicatedLinear: ReplicatedLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(base_layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            return lora_layer_type(base_layer)
    raise Exception(f"No corresponding LoRA layer supported for {type(base_layer)}.")

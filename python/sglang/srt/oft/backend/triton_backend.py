import os

import torch

from sglang.srt.oft.backend.base_backend import BaseOFTBackend
from sglang.srt.oft.triton_ops import gemm_oft_r_fwd, sgemm_oft_r_fwd
from sglang.srt.oft.utils import OFTBatchInfo, generate_sequence_lengths
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TritonOFTBackend(BaseOFTBackend):
    name = "triton"

    def __init__(
        self,
        max_ofts_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_ofts_per_batch, device)

        # Single-adapter fast path. The kernel actually requires per-batch
        # adapter uniformity (every token in a batch shares one adapter), not
        # exactly one slot. So we enable the fast path whenever
        # max_ofts_per_batch <= 2:
        #   * max==1: one slot, one adapter — trivially uniform.
        #   * max==2: typical RL setup. Slot 0 is the auto-registered `None`
        #     placeholder (identity = base / reference model, used for KL
        #     against the base in PPO/GRPO). Slot 1 is the active OFT adapter
        #     loaded via the streamed identity loader (or a real trained adapter
        #     in production rollout). Each batch still uses one adapter; the
        #     0-d device tensors are updated only when the active slot changes.
        # The kernel itself handles slot 0 (block_size_val=0) by taking an
        # identity-passthrough branch — just memory copy, no GEMM. So the
        # ref-model forward is essentially free beyond the kernel launch.
        # When max_ofts_per_batch > 2 we assume a multi-tenant setup where
        # different requests in a single batch may target different adapters;
        # the segmented `sgemm_oft_r_fwd` kernel is correct for that case.
        # `prepare_oft_batch` enables the fast path only for uniform batches.
        self.single_adapter_mode = max_ofts_per_batch <= 2
        self._use_single_adapter_fast_path = False
        if self.single_adapter_mode:
            with torch.device(device):
                self._single_adapter_idx_t = torch.zeros((), dtype=torch.int32)
                self._single_block_size_val_t = torch.zeros((), dtype=torch.int32)
            self._single_adapter_idx = 0
            self._single_block_size_val = 0
        try:
            self.single_adapter_block_s = int(
                os.getenv("SGLANG_OFT_SINGLE_ADAPTER_BLOCK_S", "64")
            )
        except ValueError:
            self.single_adapter_block_s = 64
        if self.single_adapter_block_s <= 0:
            self.single_adapter_block_s = 64

    def run_oft_r_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if self._use_single_adapter_fast_path:
            return gemm_oft_r_fwd(
                x,
                weights,
                self._single_adapter_idx_t,
                self._single_block_size_val_t,
                num_slices=1,
                BLOCK_S=self.single_adapter_block_s,
            )
        return sgemm_oft_r_fwd(x, weights, self.batch_info, num_slices=1)

    def run_grouped_oft_r_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        *args,
        n_groups: int,
        **kwargs,
    ) -> torch.Tensor:
        if self._use_single_adapter_fast_path:
            if n_groups <= 0:
                raise ValueError(f"n_groups must be positive, got {n_groups}")
            return self.run_oft_r_sgemm(x, weights, *args, **kwargs)
        return super().run_grouped_oft_r_sgemm(
            x,
            weights,
            *args,
            n_groups=n_groups,
            **kwargs,
        )

    def run_qkv_oft(
        self,
        x: torch.Tensor,
        qkv_oft_r: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if self._use_single_adapter_fast_path:
            return gemm_oft_r_fwd(
                x,
                qkv_oft_r,
                self._single_adapter_idx_t,
                self._single_block_size_val_t,
                num_slices=3,
                BLOCK_S=self.single_adapter_block_s,
            )
        return sgemm_oft_r_fwd(x, qkv_oft_r, self.batch_info, num_slices=3)

    def run_gate_up_oft(
        self,
        x: torch.Tensor,
        gate_up_oft_r: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if self._use_single_adapter_fast_path:
            return gemm_oft_r_fwd(
                x,
                gate_up_oft_r,
                self._single_adapter_idx_t,
                self._single_block_size_val_t,
                num_slices=2,
                BLOCK_S=self.single_adapter_block_s,
            )
        return sgemm_oft_r_fwd(x, gate_up_oft_r, self.batch_info, num_slices=2)

    def run_fused_gate_up_inputs(
        self,
        x: torch.Tensor,
        gate_up_oft_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gate_up_oft_r.dim() != 4:
            raise RuntimeError(
                "run_fused_gate_up_inputs expects 4D R, "
                f"got shape {tuple(gate_up_oft_r.shape)}"
            )
        if not getattr(self, "_use_single_adapter_fast_path", False):
            raise NotImplementedError(
                "run_fused_gate_up_inputs: only single-adapter fast path is supported"
            )

        from sglang.srt.oft.triton_ops import fused_rotate_gate_up_inputs

        if not x.is_contiguous():
            x = x.contiguous()
        if not gate_up_oft_r.is_contiguous():
            gate_up_oft_r = gate_up_oft_r.contiguous()
        return fused_rotate_gate_up_inputs(
            x,
            gate_up_oft_r,
            slot_idx_t=self._single_adapter_idx_t,
            bsv_t=self._single_block_size_val_t,
        )

    def run_fused_rotate_project(
        self,
        x: torch.Tensor,
        R: torch.Tensor,
        weight: torch.Tensor,
        output_sizes: list,
        bias,
    ) -> torch.Tensor:
        """Fused rotate-and-project for canonical split OFT (bf16/fp16 dense).

        Hands the full 4D R buffer plus the persistent 0-d slot / block-size
        tensors to the kernel. The kernel reads both at runtime, so a single
        CUDA-graph capture covers all (slot, block_size_val) combinations —
        including block_size_val==0 (identity-passthrough), which the kernel
        handles by skipping the rotation matmul.
        """
        if R.dim() != 4:
            raise RuntimeError(
                f"run_fused_rotate_project expects 4D R, got shape {tuple(R.shape)}"
            )
        if not getattr(self, "_use_single_adapter_fast_path", False):
            # Non-fast-path (multi-tenant or mixed batch). Out of scope for
            # this kernel: fall back so the legacy segmented path handles it.
            raise NotImplementedError(
                "run_fused_rotate_project: only single-adapter fast path is supported"
            )
        if len(output_sizes) != 3:
            raise NotImplementedError(
                "run_fused_rotate_project: only QKV uses the fused fast path"
            )

        from sglang.srt.oft.triton_ops import (
            fused_rotate_project_qkv,
        )
        if not x.is_contiguous():
            x = x.contiguous()
        if not R.is_contiguous():
            R = R.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        slot_idx_t = self._single_adapter_idx_t
        bsv_t = self._single_block_size_val_t
        if len(output_sizes) == 3:
            return fused_rotate_project_qkv(
                x, R, weight, output_sizes, bias,
                slot_idx_t=slot_idx_t, bsv_t=bsv_t,
            )
        raise NotImplementedError(
            f"run_fused_rotate_project: unsupported slice count {len(output_sizes)}"
        )

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        with torch.device("cuda"):
            self.cuda_graph_batch_info = OFTBatchInfo(
                use_cuda_graph=True,
                bs=max_bs_in_cuda_graph,
                num_segments=self.max_ofts_per_batch,
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
                ),
                seg_indptr=torch.zeros(
                    max_bs_in_cuda_graph + 1, dtype=torch.int32
                ),
                weight_indices=torch.zeros(
                    max_bs_in_cuda_graph, dtype=torch.int32
                ),
                oft_block_sizes=torch.zeros(
                    self.max_ofts_per_batch, dtype=torch.int32
                ),
                permutation=None,
                max_len=num_tokens_per_bs,
            )

            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[
                    1 : max_bs_in_cuda_graph + 1
                ],
            )

    def prepare_oft_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        oft_block_sizes: list[int],
        use_cuda_graph: bool,
    ):
        import os as _os, sys as _sys
        _trace_enabled = _os.environ.get("SGLANG_OFT_PREPARE_TRACE", "0").strip() not in ("0", "false", "False")
        if _trace_enabled:
            _counter = getattr(self, "_prepare_trace_counter", 0) + 1
            self._prepare_trace_counter = _counter
            _every = int(_os.environ.get("SGLANG_OFT_PREPARE_TRACE_EVERY", "100"))
            if _every > 0 and _counter % _every == 0:
                try:
                    _mode = forward_batch.forward_mode
                    _mode_str = (
                        f"decode={_mode.is_decode()} extend={_mode.is_extend()} "
                        f"cg={_mode.is_cuda_graph()}"
                    )
                except Exception as _e:
                    _mode_str = f"forward_mode_err={_e!r}"
                _sys.stderr.write(
                    f"[prepare-trace] call={_counter} use_cg={use_cuda_graph} "
                    f"bs={forward_batch.batch_size} "
                    f"{_mode_str} "
                    f"single_mode={self.single_adapter_mode} "
                    f"wi_len={len(weight_indices)} wi_head={weight_indices[:8]} "
                    f"obs={oft_block_sizes} "
                    f"prev_idx={self._single_adapter_idx if self.single_adapter_mode else 'n/a'} "
                    f"prev_bsv={self._single_block_size_val if self.single_adapter_mode else 'n/a'}\n"
                )
                _sys.stderr.flush()
        if self.single_adapter_mode:
            # Fast path is only valid when every row in the current batch uses
            # one adapter. Mixed base+adapter batches must keep the segmented
            # metadata path even when max_ofts_per_batch <= 2.
            if weight_indices:
                first = weight_indices[0]
                self._use_single_adapter_fast_path = all(
                    w == first for w in weight_indices
                )
            else:
                self._use_single_adapter_fast_path = True

            if self._use_single_adapter_fast_path:
                adapter_idx = int(weight_indices[0]) if weight_indices else 0
                if 0 <= adapter_idx < len(oft_block_sizes):
                    block_size_val = int(oft_block_sizes[adapter_idx])
                else:
                    block_size_val = 0

                if (
                    adapter_idx != self._single_adapter_idx
                    or block_size_val != self._single_block_size_val
                ):
                    self._single_adapter_idx_t.fill_(adapter_idx)
                    self._single_block_size_val_t.fill_(block_size_val)
                    self._single_adapter_idx = adapter_idx
                    self._single_block_size_val = block_size_val
                return
        else:
            self._use_single_adapter_fast_path = False

        original_seq_lens_cpu = generate_sequence_lengths(
            forward_batch, device="cpu"
        )
        original_weight_indices_tensor = torch.tensor(
            weight_indices, dtype=torch.int32, device="cpu"
        )

        if use_cuda_graph:
            # CUDA graph captures the Triton launch grid. Keep one segment per
            # graph row so padded replay rows remain covered even when runtime
            # adapter ids differ from the all-empty capture batch.
            seg_lens_cpu = original_seq_lens_cpu.pin_memory()
            seg_indptr_cpu = torch.zeros(
                (forward_batch.batch_size + 1,), dtype=torch.int32, pin_memory=True
            )
            seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0)
            weight_indices_tensor = original_weight_indices_tensor.pin_memory()
        else:
            # Merge consecutive same-adapter segments for efficiency.
            unique_weight_indices_tensor, inverse_weight_indices_tensor = (
                torch.unique_consecutive(
                    original_weight_indices_tensor, return_inverse=True
                )
            )

            seg_lens_cpu = (
                torch.zeros_like(
                    unique_weight_indices_tensor, dtype=torch.int32, device="cpu"
                )
                .scatter_add_(
                    0,
                    inverse_weight_indices_tensor,
                    original_seq_lens_cpu,
                )
                .pin_memory()
            )

            seg_indptr_cpu = torch.zeros(
                (len(seg_lens_cpu) + 1,), dtype=torch.int32, pin_memory=True
            )
            seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0)

            weight_indices_tensor = unique_weight_indices_tensor.pin_memory()
        oft_block_sizes_tensor = torch.tensor(
            oft_block_sizes, dtype=torch.int32, pin_memory=True, device="cpu"
        )

        num_segments = len(seg_lens_cpu)

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = num_segments
            batch_info.max_len = int(max(seg_lens_cpu))
        else:
            max_len = int(max(seg_lens_cpu))

            batch_info = OFTBatchInfo(
                bs=forward_batch.batch_size,
                num_segments=num_segments,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=torch.empty(
                    (num_segments,), dtype=torch.int32, device=self.device
                ),
                seg_indptr=torch.empty(
                    (num_segments + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (num_segments,), dtype=torch.int32, device=self.device
                ),
                oft_block_sizes=torch.empty(
                    (self.max_ofts_per_batch,),
                    dtype=torch.int32,
                    device=self.device,
                ),
                permutation=None,
            )

        # Copy to device asynchronously
        batch_info.oft_block_sizes[: self.max_ofts_per_batch].copy_(
            oft_block_sizes_tensor, non_blocking=True
        )
        batch_info.weight_indices[:num_segments].copy_(
            weight_indices_tensor, non_blocking=True
        )
        batch_info.seg_indptr[: num_segments + 1].copy_(
            seg_indptr_cpu, non_blocking=True
        )
        batch_info.seg_lens[:num_segments].copy_(
            seg_lens_cpu, non_blocking=True
        )

        self._last_prepare_cpu_tensors = (
            oft_block_sizes_tensor,
            weight_indices_tensor,
            seg_indptr_cpu,
            seg_lens_cpu,
        )
        self.batch_info = batch_info
        if use_cuda_graph:
            self.refresh_cuda_graph_grouped_batch_infos()

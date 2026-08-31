import torch

from sglang.srt.oft.backend.base_backend import BaseOFTBackend
from sglang.srt.oft.torch_ops.oft_ops import sgemm_oft_r_fwd
from sglang.srt.oft.utils import OFTBatchInfo, generate_sequence_lengths
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TorchNativeOFTBackend(BaseOFTBackend):
    name = "torch_native"

    def __init__(
        self,
        max_ofts_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_ofts_per_batch, device)

    def run_oft_r_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return sgemm_oft_r_fwd(
            inputs=x,
            weights=weights,
            weight_indices=self.batch_info.weight_indices[: self.batch_info.num_segments],
            seg_len_tensor=self.batch_info.seg_lens[: self.batch_info.num_segments],
            oft_block_sizes=self.batch_info.oft_block_sizes,
            num_slices=1,
        )

    def run_qkv_oft(
        self,
        x: torch.Tensor,
        qkv_oft_r: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Apply OFT rotation for QKV projections.

        Each of Q, K, V has its own orthogonal rotation (num_blocks each),
        stacked together as 3 * num_blocks total blocks along dim 1.

        Args:
            x: (s, input_dim)
            qkv_oft_r: (num_oft, 3 * num_blocks, block_size, block_size)

        Returns:
            rotated_x: (s, 3 * input_dim)
        """
        return sgemm_oft_r_fwd(
            inputs=x,
            weights=qkv_oft_r,
            weight_indices=self.batch_info.weight_indices[: self.batch_info.num_segments],
            seg_len_tensor=self.batch_info.seg_lens[: self.batch_info.num_segments],
            oft_block_sizes=self.batch_info.oft_block_sizes,
            num_slices=3,
        )

    def run_gate_up_oft(
        self,
        x: torch.Tensor,
        gate_up_oft_r: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Apply OFT rotation for gate_up projections.

        Each of gate and up has its own orthogonal rotation (num_blocks each),
        stacked together as 2 * num_blocks total blocks along dim 1.

        Args:
            x: (s, input_dim)
            gate_up_oft_r: (num_oft, 2 * num_blocks, block_size, block_size)

        Returns:
            rotated_x: (s, 2 * input_dim)
        """
        return sgemm_oft_r_fwd(
            inputs=x,
            weights=gate_up_oft_r,
            weight_indices=self.batch_info.weight_indices[: self.batch_info.num_segments],
            seg_len_tensor=self.batch_info.seg_lens[: self.batch_info.num_segments],
            oft_block_sizes=self.batch_info.oft_block_sizes,
            num_slices=2,
        )

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        self.cuda_graph_batch_info = OFTBatchInfo(
            use_cuda_graph=True,
            bs=max_bs_in_cuda_graph,
            num_segments=self.max_ofts_per_batch,
            seg_lens=torch.full(
                (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
            ),
            seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
            weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
            oft_block_sizes=torch.zeros(self.max_ofts_per_batch, dtype=torch.int32),
            permutation=None,
            max_len=num_tokens_per_bs,
        )

        # Initialize seg_indptr for CUDA graph as they remain constant
        # across batches.
        torch.cumsum(
            self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
            dim=0,
            out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],
        )

    def prepare_oft_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        oft_block_sizes: list[int],
        use_cuda_graph: bool,
    ):
        original_seq_lens = generate_sequence_lengths(forward_batch, device="cpu")
        original_weight_indices_tensor = torch.tensor(
            weight_indices, dtype=torch.int32, device="cpu"
        )

        unique_weight_indices_tensor, inverse_weight_indices_tensor = (
            torch.unique_consecutive(
                original_weight_indices_tensor, return_inverse=True
            )
        )

        seg_lens = (
            torch.zeros_like(
                unique_weight_indices_tensor, dtype=torch.int32, device="cpu"
            )
            .scatter_add_(
                0,
                inverse_weight_indices_tensor,
                original_seq_lens,
            )
            .pin_memory()
        )

        seg_indptr = torch.zeros(
            (len(seg_lens) + 1,), dtype=torch.int32, pin_memory=True
        )
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        # Use pinned memory to avoid synchronizations during host-to-device transfer
        weight_indices_tensor = unique_weight_indices_tensor.pin_memory()
        oft_block_sizes_tensor = torch.tensor(
            oft_block_sizes, dtype=torch.int32, pin_memory=True, device="cpu"
        )

        num_segments = len(seg_lens)

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = num_segments
            batch_info.max_len = int(max(seg_lens))
        else:
            max_len = int(max(seg_lens))

            batch_info = OFTBatchInfo(
                bs=forward_batch.batch_size,
                num_segments=num_segments,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=torch.empty((num_segments,), dtype=torch.int32),
                seg_indptr=torch.empty((num_segments + 1,), dtype=torch.int32),
                weight_indices=torch.empty((num_segments,), dtype=torch.int32),
                oft_block_sizes=torch.empty(
                    (self.max_ofts_per_batch,), dtype=torch.int32,
                ),
                permutation=None,
            )

        batch_info.oft_block_sizes[: self.max_ofts_per_batch].copy_(
            oft_block_sizes_tensor
        )
        batch_info.weight_indices[:num_segments].copy_(
            weight_indices_tensor
        )
        batch_info.seg_indptr[: len(seg_indptr)].copy_(
            seg_indptr
        )
        batch_info.seg_lens[: len(seg_lens)].copy_(
            seg_lens
        )

        self._last_prepare_cpu_tensors = (
            oft_block_sizes_tensor,
            weight_indices_tensor,
            seg_indptr,
            seg_lens,
        )
        self.batch_info = batch_info
        if use_cuda_graph:
            self.refresh_cuda_graph_grouped_batch_infos()

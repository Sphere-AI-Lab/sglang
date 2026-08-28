from dataclasses import replace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseOFTBackend:
    """Base class for different OFT backends.
       Each backend has its own implementation of OFT kernels.

       Unlike LoRA which decomposes the adapter into A and B matrices,
       OFT uses a single orthogonal transformation (oft_r) per module.
       The orthogonal matrix is parameterized as a block-diagonal matrix
       using the Cayley transform of skew-symmetric blocks.

    Args:
        max_ofts_per_batch: maximum number of different OFT weights
                             that can be applied in a single forward batch.
        device: the device where the backend runs.
    """

    def __init__(self, max_ofts_per_batch: int, device: torch.device):
        self.max_ofts_per_batch = max_ofts_per_batch
        self.device = device
        self._cuda_graph_grouped_batch_infos = {}

    def run_extra_token_embedding(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor,
        extra_embeddings: torch.Tensor,
        vocab_size: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply extra token embeddings to output in-place.

        Args:
            input_ids: (s,) token IDs
            output: (s, embed_dim) output tensor to be modified
            extra_embeddings: (num_ofts, num_extra_tokens, embed_dim) extra embeddings
            vocab_size: base vocabulary size

        Returns:
            output: modified output tensor
        """
        raise NotImplementedError

    def run_oft_r_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of OFT modules with current backend.

        Unlike LoRA which has separate A (shrink) and B (expand) segment Gemm operations,
        OFT applies a single orthogonal transformation via segment Gemm: y = x @ R,
        where R is a precomputed block-diagonal orthogonal matrix.

        R is precomputed at weight loading time via the Cayley transform of
        skew-symmetric parameters. During forward pass, only the matmul is needed.

        Args:
             x: input matrix with shape (s, dim), here s is the sum of all sequence lengths
             weights: precomputed R matrices with shape (num_oft, c * num_blocks, block_size, block_size),
                      here c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
        Returns:
             result with shape (s, dim)
        """
        pass

    def run_grouped_oft_r_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        *args,
        n_groups: int,
        **kwargs,
    ) -> torch.Tensor:
        """Run OFT rotation on grouped rows using the backend's normal kernel.

        Some model paths flatten each logical token into multiple adjacent
        group rows before applying the same adapter. The standard OFT metadata
        is token-segmented, so each segment boundary must be scaled by
        ``n_groups`` before invoking the regular segmented backend.
        """
        if n_groups <= 0:
            raise ValueError(f"n_groups must be positive, got {n_groups}")
        if n_groups == 1:
            return self.run_oft_r_sgemm(x, weights, *args, **kwargs)

        batch_info = self.batch_info
        if batch_info.use_cuda_graph:
            old_batch_info = self.batch_info
            self.batch_info = self._get_cuda_graph_grouped_batch_info(
                batch_info, n_groups
            )
            try:
                return self.run_oft_r_sgemm(x, weights, *args, **kwargs)
            finally:
                self.batch_info = old_batch_info

        updates = {
            "seg_indptr": batch_info.seg_indptr * n_groups,
            "max_len": (
                int(batch_info.max_len) * n_groups
                if batch_info.max_len is not None
                else None
            ),
        }
        if batch_info.seg_lens is not None:
            updates["seg_lens"] = batch_info.seg_lens * n_groups

        grouped_batch_info = replace(batch_info, **updates)
        old_batch_info = self.batch_info
        self.batch_info = grouped_batch_info
        try:
            return self.run_oft_r_sgemm(x, weights, *args, **kwargs)
        finally:
            self.batch_info = old_batch_info

    def _get_cuda_graph_grouped_batch_info(self, batch_info, n_groups: int):
        grouped = self._cuda_graph_grouped_batch_infos.get(n_groups)
        if grouped is None:
            grouped = replace(
                batch_info,
                seg_indptr=torch.empty_like(batch_info.seg_indptr),
                seg_lens=(
                    torch.empty_like(batch_info.seg_lens)
                    if batch_info.seg_lens is not None
                    else None
                ),
            )
            self._cuda_graph_grouped_batch_infos[n_groups] = grouped
        self._refresh_cuda_graph_grouped_batch_info(n_groups, batch_info, grouped)
        return grouped

    def _refresh_cuda_graph_grouped_batch_info(
        self, n_groups: int, batch_info, grouped
    ) -> None:
        grouped.bs = batch_info.bs
        grouped.num_segments = batch_info.num_segments
        grouped.max_len = (
            int(batch_info.max_len) * n_groups
            if batch_info.max_len is not None
            else None
        )
        grouped.weight_indices = batch_info.weight_indices
        grouped.oft_block_sizes = batch_info.oft_block_sizes
        grouped.permutation = batch_info.permutation
        torch.mul(batch_info.seg_indptr, n_groups, out=grouped.seg_indptr)
        if batch_info.seg_lens is not None and grouped.seg_lens is not None:
            torch.mul(batch_info.seg_lens, n_groups, out=grouped.seg_lens)

    def refresh_cuda_graph_grouped_batch_infos(self) -> None:
        batch_info = getattr(self, "batch_info", None)
        if batch_info is None or not getattr(batch_info, "use_cuda_graph", False):
            return
        for n_groups, grouped in self._cuda_graph_grouped_batch_infos.items():
            self._refresh_cuda_graph_grouped_batch_info(
                n_groups, batch_info, grouped
            )

    def run_qkv_oft(
        self,
        x: torch.Tensor,
        qkv_oft_r: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run OFT rotation for QKV projections via segment Gemm.

        Each of Q, K, V has its own orthogonal rotation (num_blocks each),
        stacked together as 3 * num_blocks total blocks.

        Args:
            x: input matrix with shape (s, input_dim)
            qkv_oft_r: precomputed R for qkv, shape (num_oft, 3 * num_blocks, block_size, block_size)
        Returns:
            rotated_x with shape (s, 3 * input_dim)
        """
        pass

    def run_gate_up_oft(
        self,
        x: torch.Tensor,
        gate_up_oft_r: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run OFT rotation for gate_up_proj via segment Gemm.

        Each of gate and up has its own orthogonal rotation (num_blocks each),
        stacked together as 2 * num_blocks total blocks.

        Args:
            x: input matrix with shape (s, input_dim)
            gate_up_oft_r: precomputed R for gate_up, shape (num_oft, 2 * num_blocks, block_size, block_size)
        Returns:
            rotated_x with shape (s, 2 * input_dim)
        """
        pass

    def run_fused_gate_up_inputs(
        self,
        x: torch.Tensor,
        gate_up_oft_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the gate/up OFT rotation and return separate gate/up inputs.

        This fast path is for fused FC1/gate-up projections where the large
        projection GEMMs should remain in cuBLAS. Backends that cannot satisfy
        the single-adapter contract should raise ``NotImplementedError`` so
        callers can fall back to ``run_gate_up_oft``.
        """
        raise NotImplementedError

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        """Initialize the batch info for CUDA Graph mode.

        This method provides a hook for each backend to conduct its own initialization
        logic for CUDA Graph mode.

        Args:
            max_bs_in_cuda_graph: maximum batch size for CUDA Graph mode
            num_tokens_per_bs: number of tokens per sequence (1 for decoding, >1 for target_verify)
        """
        pass

    def prepare_oft_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        oft_block_sizes: list[int],
        use_cuda_graph: bool,
    ):
        """Prepare the OFT weights and batch info for current forward batch.
        This method provides a hook for each backend to conduct its own preparation
        logic for each forward batch.

        Args:
            forward_batch: the ForwardBatch object for current forward pass
            weight_indices: list of indices of OFT weights to be applied for current batch
            oft_block_sizes: list of OFT block sizes corresponding to weight_indices
            use_cuda_graph: whether to use CUDA Graph for this batch
        """
        pass

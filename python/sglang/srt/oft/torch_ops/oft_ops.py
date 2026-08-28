from typing import Optional

import torch

try:
    from sglang.srt.oft.triton_ops.cayley_neumann import cayley_neumann_fwd as _triton_cayley_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def expand_to_skew_symmetric(
    compact_vec: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Expand compact upper-triangular vectors to full skew-symmetric matrices.

    Args:
        compact_vec: (num_blocks, n_elements) where n_elements = block_size*(block_size-1)//2
        block_size: size of each block

    Returns:
        (num_blocks, block_size, block_size) skew-symmetric matrices
    """
    rows, cols = torch.triu_indices(block_size, block_size, 1)
    num_blocks = compact_vec.shape[0]
    n_elements = block_size * (block_size - 1) // 2
    compact_vec = compact_vec[:, :n_elements]
    matrix = torch.zeros(
        num_blocks,
        block_size,
        block_size,
        device=compact_vec.device,
        dtype=compact_vec.dtype,
    )
    matrix[:, rows, cols] = compact_vec
    matrix = matrix - matrix.transpose(-1, -2)
    return matrix


def cayley_neumann(
    Q_skew: torch.Tensor,
    num_terms: int = 5,
) -> torch.Tensor:
    """Compute Cayley transform via Neumann series approximation.

    Args:
        Q_skew: (num_blocks, block_size, block_size) skew-symmetric matrices
        num_terms: number of terms in Neumann series

    Returns:
        (num_blocks, block_size, block_size) orthogonal matrices
    """
    b, block_size, _ = Q_skew.shape
    R = torch.eye(block_size, device=Q_skew.device, dtype=Q_skew.dtype).repeat(b, 1, 1)
    if num_terms > 1:
        R.add_(Q_skew, alpha=2.0)
        if num_terms > 2:
            Q_squared = torch.bmm(Q_skew, Q_skew)
            R.add_(Q_squared, alpha=2.0)
            Q_power = Q_squared
            for _ in range(3, num_terms - 1):
                Q_power = torch.bmm(Q_power, Q_skew)
                R.add_(Q_power, alpha=2.0)
            Q_power = torch.bmm(Q_power, Q_skew)
            R.add_(Q_power)
    return R


def cayley_exact(
    Q_skew: torch.Tensor,
) -> torch.Tensor:
    """Compute exact Cayley transform: R = (I - Q) @ (I + Q)^{-1}.

    Args:
        Q_skew: (num_blocks, block_size, block_size) skew-symmetric matrices

    Returns:
        (num_blocks, block_size, block_size) orthogonal matrices
    """
    b, block_size, _ = Q_skew.shape
    I = (
        torch.eye(block_size, device=Q_skew.device, dtype=Q_skew.dtype)
        .unsqueeze(0)
        .expand(b, block_size, block_size)
    )
    return torch.linalg.solve(I + Q_skew, I - Q_skew, left=False)


def precompute_oft_r(
    compact_weights: torch.Tensor,
    block_size: int,
    use_neumann: bool = True,
    num_neumann_terms: int = 5,
) -> torch.Tensor:
    """Precompute orthogonal rotation matrices from compact OFT weights.

    Called at weight loading time (not during forward pass) since inference
    weights are frozen.

    Pipeline: compact upper-tri -> skew-symmetric -> Cayley transform -> R

    Args:
        compact_weights: (num_blocks, n_elements) compact upper-triangular weights
        block_size: size of each orthogonal block
        use_neumann: use Neumann approximation vs exact solve
        num_neumann_terms: terms in Neumann series

    Returns:
        (num_blocks, block_size, block_size) orthogonal rotation matrices
    """
    skew = expand_to_skew_symmetric(compact_weights, block_size)
    if use_neumann:
        if HAS_TRITON and skew.is_cuda and num_neumann_terms == 5 and block_size >= 16:
            return _triton_cayley_fwd(skew, num_neumann_terms)
        return cayley_neumann(skew, num_neumann_terms)
    else:
        return cayley_exact(skew)


def apply_block_diag_orth(
    x: torch.Tensor,
    orth_blocks: torch.Tensor,
) -> torch.Tensor:
    """Apply block-diagonal orthogonal rotation: output = x @ R.

    Args:
        x: (s, dim) where dim = num_blocks * block_size
        orth_blocks: (num_blocks, block_size, block_size)

    Returns:
        (s, dim)
    """
    num_blocks, block_size, _ = orth_blocks.shape
    s = x.shape[0]
    x_blocked = x.reshape(s, num_blocks, block_size)
    # x @ R per block: y[s,b,c] = sum_k x[s,b,k] * R[b,k,c]
    y_blocked = torch.einsum(
        "sbk,bkc->sbc", x_blocked.to(orth_blocks.dtype), orth_blocks
    )
    return y_blocked.reshape(s, -1).to(x.dtype)


def sgemm_oft_r_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    oft_block_sizes: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:
    """Reference implementation of segmented block-diagonal OFT rotation.

    Processes precomputed R matrices per segment.

    Args:
        inputs: (total_seq_len, input_dim)
        weights: (num_ofts, num_slices * num_blocks, block_size, block_size) precomputed R
        weight_indices: (num_segments,) adapter index per segment (which adapter each segment uses (index into the adapter pool))
        seg_len_tensor: (num_segments,) length of each segment (how many tokens in each segment)
        oft_block_sizes: (num_ofts,) block size per adapter
        num_slices: number of stacked modules (e.g., 3 for qkv_proj, 2 for gate_up_proj)

    Returns:
        (total_seq_len, num_slices * input_dim)
    """
    total_seq_len, input_dim = inputs.shape
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    output = torch.zeros(
        total_seq_len, num_slices * input_dim, dtype=inputs.dtype, device=inputs.device
    )

    token_offset = 0
    for oft_idx, seq_len, block_size in zip(
        weight_indices, seg_len_tensor, oft_block_sizes[weight_indices]
    ):
        if seq_len == 0:
            continue

        if block_size == 0:
            for s in range(num_slices):
                output[token_offset : token_offset + seq_len, s * input_dim : (s + 1) * input_dim] = inputs[token_offset : token_offset + seq_len]
            token_offset += seq_len
            continue

        R_all = weights[oft_idx]  # (num_slices * num_blocks, block_size, block_size)
        num_blocks_needed = input_dim // block_size
        x_seq = inputs[token_offset : token_offset + seq_len]

        for s in range(num_slices):
            R = R_all[s * num_blocks_needed : (s + 1) * num_blocks_needed, :block_size, :block_size]
            if R.shape[0] == 1 and num_blocks_needed > 1:
                R = R.repeat(num_blocks_needed, 1, 1)
            output[token_offset : token_offset + seq_len, s * input_dim : (s + 1) * input_dim] = apply_block_diag_orth(x_seq, R)

        token_offset += seq_len

    return output

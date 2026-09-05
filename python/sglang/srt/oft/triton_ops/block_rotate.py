import torch
import triton
import triton.language as tl


@triton.jit
def _oft_block_rotate_kernel(
    # Input: (M, K)
    A_ptr,
    stride_am,
    stride_ak,
    # Output: (M_expanded, K)  — one row per token-expert pair
    A_rot_ptr,
    stride_arm,
    stride_ark,
    # R matrices: (E, num_blocks, bs, bs)
    R_ptr,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    # Token routing
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    # Dimensions
    top_k: tl.constexpr,
    K: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Apply per-expert block-diagonal OFT rotation to input features.

    Grid: (cdiv(EM, BLOCK_M), num_blocks)
    - pid_m: block of sorted tokens (all same expert via moe_align_block_size)
    - pid_blk: which OFT block (0 .. K // OFT_BLOCK_SIZE - 1)

    For each program: loads BLOCK_M tokens' input at one OFT block position,
    loads the expert's R matrix block, computes rotation via tl.dot, and
    writes the rotated result to A_rot.
    """
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)

    # Bounds check
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    # Load sorted token ids and expert for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(sorted_token_ids_ptr + offs_m)
    sorted_ids = sorted_ids.to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Original token ids (A is indexed by token, not token*top_k)
    orig_ids = sorted_ids // top_k

    # Base K offset for this OFT block
    k_base = pid_blk * OFT_BLOCK_SIZE

    # EP-dispatched non-local experts are encoded as -1. The base MoE GEMM
    # filters those blocks, but the OFT pre-rotation runs before that filter,
    # so skip here to avoid reading outside the R tensor.
    if expert < 0:
        return

    # Tiled rotation: accumulate (BLOCK_M, OFT_BLOCK_SIZE) result
    # by iterating over TILE_K chunks of the inner k dimension.
    #
    # OFT convention: rot_accum[t, c] = sum_k A[t, k_base+k] * R[k, c]
    # i.e. A_rot[:, k_base:k_base+bs] = A[:, k_base:k_base+bs] @ R[expert, pid_blk]
    # (NOT R^T — the dense OFT kernel `sgemm_oft_r.py` and `apply_block_diag_orth`
    # both apply x @ R; matching that here keeps train/inference parity).

    rot_accum = tl.zeros((BLOCK_M, OFT_BLOCK_SIZE), dtype=tl.float32)

    if OFT_BLOCK_SIZE >= 16:
        for k_off in range(0, OFT_BLOCK_SIZE, TILE_K):
            # Load A tile: (BLOCK_M, TILE_K) from A[orig_ids, k_base + k_off : ...]
            k_tile_offs = (k_base + k_off + tl.arange(0, TILE_K)).to(tl.int64)
            a_ptrs = A_ptr + orig_ids[:, None] * stride_am + k_tile_offs[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

            # Load R sub-block: R[expert, pid_blk, k_off:k_off+TILE_K, :]
            # Shape: (TILE_K, OFT_BLOCK_SIZE) — rows = k inner axis, cols = c output axis.
            r_row_offs = (k_off + tl.arange(0, TILE_K)).to(tl.int64)
            r_col_offs = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
            r_ptrs = (
                R_ptr
                + expert * stride_re
                + pid_blk * stride_rb
                + r_row_offs[:, None] * stride_ri
                + r_col_offs[None, :] * stride_rj
            )
            r_sub = tl.load(r_ptrs)  # (TILE_K, OFT_BLOCK_SIZE)

            # (BLOCK_M, TILE_K) @ (TILE_K, OFT_BLOCK_SIZE)  →  x @ R per block
            # input_precision="ieee" is a no-op for bf16 operands (Triton 3.5.1
            # only honors it for fp32×fp32) but kept defensive: if R is ever
            # promoted to fp32, this enforces ieee not tf32, matching the Bridge
            # train-side `sgemm_oft_r_single.py:71` annotation.
            rot_accum += tl.dot(a_tile, r_sub, input_precision="ieee")
    else:
        out_cols = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
        for k in range(OFT_BLOCK_SIZE):
            a_col = tl.load(
                A_ptr + orig_ids * stride_am + (k_base + k) * stride_ak,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            r_row = tl.load(
                R_ptr
                + expert * stride_re
                + pid_blk * stride_rb
                + k * stride_ri
                + out_cols * stride_rj
            ).to(tl.float32)
            rot_accum += a_col[:, None] * r_row[None, :]

    # Store rotated output: A_rot[sorted_ids, k_base : k_base + bs]
    out_k_offs = (k_base + tl.arange(0, OFT_BLOCK_SIZE)).to(tl.int64)
    out_ptrs = A_rot_ptr + sorted_ids[:, None] * stride_arm + out_k_offs[None, :] * stride_ark
    tl.store(out_ptrs, rot_accum.to(A_rot_ptr.dtype.element_ty), mask=token_mask[:, None])


def apply_oft_rotation_triton(
    A: torch.Tensor,           # (M, K)
    oft_r: torch.Tensor,       # (E, num_blocks, bs, bs)
    topk_ids: torch.Tensor,    # (M, top_k)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int = 64,
) -> torch.Tensor:
    """Apply per-expert block-diagonal OFT rotation using a Triton kernel.

    Returns A_rot of shape (M * top_k, K) where each row is rotated
    by its assigned expert's block-diagonal R matrix.
    """
    M, K = A.shape
    if oft_r.dim() != 4:
        raise ValueError(
            f"oft_r must be 4D (experts, blocks, bs, bs), got {tuple(oft_r.shape)}"
        )
    bs = oft_r.shape[-1]
    from sglang.srt.oft.utils import validate_oft_block_size

    validate_oft_block_size(bs)
    if tuple(oft_r.shape[-2:]) != (bs, bs):
        raise ValueError(f"OFT blocks must be square, got {tuple(oft_r.shape[-2:])}")
    if K % bs != 0:
        raise ValueError(f"OFT hidden size {K} must be divisible by block size {bs}")
    num_blocks = K // bs
    if oft_r.shape[1] != num_blocks:
        raise ValueError(
            f"oft_r has {oft_r.shape[1]} blocks, expected {num_blocks} for K={K}, BS={bs}"
        )

    # Output buffer: one row per token-expert pair
    A_rot = torch.empty(M * top_k, K, device=A.device, dtype=A.dtype)

    # TILE_K: chunk size for the inner dimension of the rotation matmul
    tile_k = min(64, bs)

    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, block_m), num_blocks)

    _oft_block_rotate_kernel[grid](
        A, A.stride(0), A.stride(1),
        A_rot, A_rot.stride(0), A_rot.stride(1),
        oft_r, oft_r.stride(0), oft_r.stride(1), oft_r.stride(2), oft_r.stride(3),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_ids.numel(),
        top_k=top_k,
        K=K,
        OFT_BLOCK_SIZE=bs,
        BLOCK_M=block_m,
        TILE_K=tile_k,
    )

    return A_rot


@triton.jit
def _oft_block_rotate_kernel_multi_slot(
    # Input: (M, K)
    A_ptr,
    stride_am,
    stride_ak,
    # Output: (M_expanded, K)  — one row per token-expert pair
    A_rot_ptr,
    stride_arm,
    stride_ark,
    # R matrices: (S, E, num_blocks, bs, bs) — one extra leading slot axis
    # versus the single-slot kernel.
    R_ptr,
    stride_rs,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    # Per-original-token adapter slot assignment: (M,)
    slot_ids_ptr,
    # Token routing
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    # Dimensions
    top_k: tl.constexpr,
    K: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Per-token-slot-selectable sibling of _oft_block_rotate_kernel.

    Grid and per-program responsibilities are identical to the single-slot
    kernel: (cdiv(EM, BLOCK_M), num_blocks), one (BLOCK_M, OFT_BLOCK_SIZE)
    output tile per program.

    Each BLOCK_M-sized group of sorted tokens shares one expert (per
    moe_align_block_size) but may span MULTIPLE adapter slots — so, unlike
    the single-slot kernel, this cannot batch the rotation as one tl.dot
    matmul per block (that requires one shared R for the whole tile).
    Instead it gathers each row's own slot's R row per k and accumulates
    elementwise — the same algorithm as the single-slot kernel's own
    OFT_BLOCK_SIZE < 16 fallback branch, generalized from one shared
    (OFT_BLOCK_SIZE,) R row to a per-row (BLOCK_M, OFT_BLOCK_SIZE) gather.
    Accumulation is in fp32, matching both of the single-slot kernel's
    branches.
    """
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)

    # Bounds check
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    # Load sorted token ids and expert for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(sorted_token_ids_ptr + offs_m)
    sorted_ids = sorted_ids.to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # EP-dispatched non-local experts are encoded as -1; skip exactly as the
    # single-slot kernel does, to avoid reading outside the R tensor.
    if expert < 0:
        return

    # Original token ids (A and slot_ids are indexed by token, not token*top_k)
    orig_ids = sorted_ids // top_k

    # Each row's own adapter slot. Padded rows read nothing (masked) and fall
    # back to slot 0, which is always in range, so the R gather below stays
    # in bounds for every lane.
    slot = tl.load(slot_ids_ptr + orig_ids, mask=token_mask, other=0).to(tl.int64)

    # Base K offset for this OFT block
    k_base = pid_blk * OFT_BLOCK_SIZE

    # Same OFT convention as the single-slot kernel:
    #   rot_accum[t, c] = sum_k A[t, k_base+k] * R[slot_t, expert, pid_blk, k, c]
    out_cols = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
    rot_accum = tl.zeros((BLOCK_M, OFT_BLOCK_SIZE), dtype=tl.float32)

    for k in range(OFT_BLOCK_SIZE):
        a_col = tl.load(
            A_ptr + orig_ids * stride_am + (k_base + k) * stride_ak,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        r_row = tl.load(
            R_ptr
            + slot[:, None] * stride_rs
            + expert * stride_re
            + pid_blk * stride_rb
            + k * stride_ri
            + out_cols[None, :] * stride_rj,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        rot_accum += a_col[:, None] * r_row

    # Store rotated output: A_rot[sorted_ids, k_base : k_base + bs]
    out_k_offs = (k_base + tl.arange(0, OFT_BLOCK_SIZE)).to(tl.int64)
    out_ptrs = A_rot_ptr + sorted_ids[:, None] * stride_arm + out_k_offs[None, :] * stride_ark
    tl.store(out_ptrs, rot_accum.to(A_rot_ptr.dtype.element_ty), mask=token_mask[:, None])


def apply_oft_rotation_triton_multi_slot(
    A: torch.Tensor,                # (M, K)
    oft_r_all_slots: torch.Tensor,  # (S, E, num_blocks, bs, bs)
    slot_ids: torch.Tensor,         # (M,) long, one entry per original token
    topk_ids: torch.Tensor,         # (M, top_k)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int = 64,
) -> torch.Tensor:
    """Multi-slot sibling of apply_oft_rotation_triton: selects each token's
    own adapter's R-block via ``slot_ids`` instead of applying one shared R to
    every token. See _oft_block_rotate_kernel_multi_slot's docstring for why
    this cannot reuse the single-slot kernel's tl.dot path.

    Returns A_rot of shape (M * top_k, K), same contract as the single-slot
    entry point.
    """
    M, K = A.shape
    if oft_r_all_slots.dim() != 5:
        raise ValueError(
            "oft_r_all_slots must be 5D (slots, experts, blocks, bs, bs), "
            f"got {tuple(oft_r_all_slots.shape)}"
        )
    bs = oft_r_all_slots.shape[-1]
    from sglang.srt.oft.utils import validate_oft_block_size

    validate_oft_block_size(bs)
    if tuple(oft_r_all_slots.shape[-2:]) != (bs, bs):
        raise ValueError(
            f"OFT blocks must be square, got {tuple(oft_r_all_slots.shape[-2:])}"
        )
    if K % bs != 0:
        raise ValueError(f"OFT hidden size {K} must be divisible by block size {bs}")
    num_blocks = K // bs
    if oft_r_all_slots.shape[2] != num_blocks:
        raise ValueError(
            f"oft_r_all_slots has {oft_r_all_slots.shape[2]} blocks, "
            f"expected {num_blocks} for K={K}, BS={bs}"
        )
    if slot_ids.shape[0] != M:
        raise ValueError(
            f"slot_ids has {slot_ids.shape[0]} entries, expected {M} (one per token)"
        )

    # Output buffer: one row per token-expert pair
    A_rot = torch.empty(M * top_k, K, device=A.device, dtype=A.dtype)

    EM = sorted_token_ids.shape[0]
    grid = (triton.cdiv(EM, block_m), num_blocks)

    _oft_block_rotate_kernel_multi_slot[grid](
        A, A.stride(0), A.stride(1),
        A_rot, A_rot.stride(0), A_rot.stride(1),
        oft_r_all_slots,
        oft_r_all_slots.stride(0), oft_r_all_slots.stride(1),
        oft_r_all_slots.stride(2), oft_r_all_slots.stride(3), oft_r_all_slots.stride(4),
        slot_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_ids.numel(),
        top_k=top_k,
        K=K,
        OFT_BLOCK_SIZE=bs,
        BLOCK_M=block_m,
    )

    return A_rot


@triton.jit
def _oft_block_rotate_kernel_multi_slot_dot(
    # Input: (M, K)
    A_ptr,
    stride_am,
    stride_ak,
    # Output: (M_expanded, K) — one row per token-expert pair
    A_rot_ptr,
    stride_arm,
    stride_ark,
    # R matrices: (S, E, num_blocks, bs, bs)
    R_ptr,
    stride_rs,
    stride_re,
    stride_rb,
    stride_ri,
    stride_rj,
    # Per-slot token routing (each program handles exactly one slot, so a
    # BLOCK_M tile always shares one adapter -- unlike the shared-alignment
    # multi-slot kernel above, this can batch the rotation as one tl.dot)
    sorted_token_ids_ptr,
    stride_sts,
    stride_stm,
    expert_ids_ptr,
    stride_eis,
    stride_eim,
    num_tokens_post_padded_ptr,
    stride_ntpp,
    oft_ids_ptr,
    adapter_enabled_ptr,
    num_valid_tokens,
    # Dimensions
    top_k: tl.constexpr,
    K: tl.constexpr,
    OFT_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Per-slot-aligned sibling of _oft_block_rotate_kernel_multi_slot.

    Grid: (slot, cdiv(EM, BLOCK_M), num_blocks) -- pid_slot indexes a
    PER-SLOT alignment (each slot's own moe_align_block_size-style routing,
    built by moe_lora_align_block_size), so every BLOCK_M-sized tile shares
    one adapter slot AND one expert, enabling a real tl.dot matmul per tile
    instead of the shared-alignment kernel's per-row elementwise gather.
    Trades O(1) alignment cost for O(num_configured_slots) grid/launch
    overhead -- see the module docstring for when each kernel wins.
    """
    pid_slot = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_blk = tl.program_id(2)

    slot_id = tl.load(oft_ids_ptr + pid_slot).to(tl.int64)
    if slot_id < 0:
        return
    if tl.load(adapter_enabled_ptr + slot_id) == 0:
        return

    num_tokens_post_padded = tl.load(
        num_tokens_post_padded_ptr + slot_id * stride_ntpp
    )
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    sorted_ids = tl.load(
        sorted_token_ids_ptr + slot_id * stride_sts + offs_m * stride_stm
    )
    sorted_ids = sorted_ids.to(tl.int64)
    token_mask = sorted_ids < num_valid_tokens

    expert = tl.load(
        expert_ids_ptr + slot_id * stride_eis + pid_m * stride_eim
    ).to(tl.int64)

    orig_ids = sorted_ids // top_k
    k_base = pid_blk * OFT_BLOCK_SIZE

    # EP-dispatched non-local experts are encoded as -1; skip exactly as the
    # shared-alignment kernel does, to avoid reading outside the R tensor.
    if expert < 0:
        return

    # OFT convention: rot_accum[t, c] = sum_k A[t, k_base+k] * R[slot, expert, pid_blk, k, c]
    rot_accum = tl.zeros((BLOCK_M, OFT_BLOCK_SIZE), dtype=tl.float32)

    if OFT_BLOCK_SIZE >= 16:
        for k_off in range(0, OFT_BLOCK_SIZE, TILE_K):
            k_tile_offs = (k_base + k_off + tl.arange(0, TILE_K)).to(tl.int64)
            a_ptrs = A_ptr + orig_ids[:, None] * stride_am + k_tile_offs[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)

            r_row_offs = (k_off + tl.arange(0, TILE_K)).to(tl.int64)
            r_col_offs = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
            r_ptrs = (
                R_ptr
                + slot_id * stride_rs
                + expert * stride_re
                + pid_blk * stride_rb
                + r_row_offs[:, None] * stride_ri
                + r_col_offs[None, :] * stride_rj
            )
            r_sub = tl.load(r_ptrs)

            # input_precision="ieee": a no-op for bf16 (Triton only honors it
            # for fp32xfp32) but kept defensive in case R is ever promoted.
            rot_accum += tl.dot(a_tile, r_sub, input_precision="ieee")
    else:
        out_cols = tl.arange(0, OFT_BLOCK_SIZE).to(tl.int64)
        for k in range(OFT_BLOCK_SIZE):
            a_col = tl.load(
                A_ptr + orig_ids * stride_am + (k_base + k) * stride_ak,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            r_row = tl.load(
                R_ptr
                + slot_id * stride_rs
                + expert * stride_re
                + pid_blk * stride_rb
                + k * stride_ri
                + out_cols * stride_rj
            ).to(tl.float32)
            rot_accum += a_col[:, None] * r_row[None, :]

    out_k_offs = (k_base + tl.arange(0, OFT_BLOCK_SIZE)).to(tl.int64)
    out_ptrs = A_rot_ptr + sorted_ids[:, None] * stride_arm + out_k_offs[None, :] * stride_ark
    tl.store(out_ptrs, rot_accum.to(A_rot_ptr.dtype.element_ty), mask=token_mask[:, None])


def apply_oft_rotation_triton_multi_slot_dot(
    A: torch.Tensor,               # (M, K)
    oft_r_all_slots: torch.Tensor, # (S, E, num_blocks, bs, bs)
    topk_ids: torch.Tensor,        # (M, top_k)
    sorted_token_ids: torch.Tensor,     # (S, max_num_tokens_padded)
    expert_ids: torch.Tensor,           # (S, max_num_m_blocks)
    num_tokens_post_padded: torch.Tensor,  # (S,)
    oft_ids: torch.Tensor,               # (S,)
    adapter_enabled: torch.Tensor,       # (S,)
    top_k: int,
    block_m: int = 64,
) -> torch.Tensor:
    """Per-slot-aligned tl.dot sibling of apply_oft_rotation_triton_multi_slot.

    Alignment tensors must come from a PER-SLOT alignment pass (e.g.
    moe_lora_align_block_size), not the shared moe_align_block_size used by
    the elementwise multi-slot kernel -- each row of sorted_token_ids/
    expert_ids belongs to one adapter slot's own routing.

    Returns A_rot of shape (M * top_k, K), same contract as the other
    multi-slot entry point.
    """
    M, K = A.shape
    if oft_r_all_slots.dim() != 5:
        raise ValueError(
            "oft_r_all_slots must be 5D (slots, experts, blocks, bs, bs), "
            f"got {tuple(oft_r_all_slots.shape)}"
        )
    bs = oft_r_all_slots.shape[-1]
    from sglang.srt.oft.utils import validate_oft_block_size

    validate_oft_block_size(bs)
    if tuple(oft_r_all_slots.shape[-2:]) != (bs, bs):
        raise ValueError(
            f"OFT blocks must be square, got {tuple(oft_r_all_slots.shape[-2:])}"
        )
    if K % bs != 0:
        raise ValueError(f"OFT hidden size {K} must be divisible by block size {bs}")
    num_blocks = K // bs
    if oft_r_all_slots.shape[2] != num_blocks:
        raise ValueError(
            f"oft_r_all_slots has {oft_r_all_slots.shape[2]} blocks, "
            f"expected {num_blocks} for K={K}, BS={bs}"
        )
    if sorted_token_ids.dim() != 2 or expert_ids.dim() != 2:
        raise ValueError("per-slot alignment tensors must be two-dimensional")
    if num_tokens_post_padded.dim() != 1 or oft_ids.dim() != 1:
        raise ValueError("per-slot padded counts and slot IDs must be one-dimensional")
    if sorted_token_ids.shape[0] != oft_ids.numel():
        raise ValueError("alignment slot dimension must match oft_ids")

    A_rot = A[:, None, :].expand(-1, top_k, -1).reshape(-1, K).clone()

    tile_k = min(64, bs)
    max_num_m_blocks = expert_ids.shape[1]
    grid = (oft_ids.numel(), max_num_m_blocks, num_blocks)

    _oft_block_rotate_kernel_multi_slot_dot[grid](
        A, A.stride(0), A.stride(1),
        A_rot, A_rot.stride(0), A_rot.stride(1),
        oft_r_all_slots,
        oft_r_all_slots.stride(0), oft_r_all_slots.stride(1), oft_r_all_slots.stride(2),
        oft_r_all_slots.stride(3), oft_r_all_slots.stride(4),
        sorted_token_ids, sorted_token_ids.stride(0), sorted_token_ids.stride(1),
        expert_ids, expert_ids.stride(0), expert_ids.stride(1),
        num_tokens_post_padded, num_tokens_post_padded.stride(0),
        oft_ids, adapter_enabled,
        topk_ids.numel(),
        top_k=top_k,
        K=K,
        OFT_BLOCK_SIZE=bs,
        BLOCK_M=block_m,
        TILE_K=tile_k,
    )

    return A_rot


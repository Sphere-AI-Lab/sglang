from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.oft.oft_moe_runners import (
    OFTInfo,
    _compute_oft_alignment,
    _oft_prerotate,
)
from sglang.srt.oft.triton_ops.block_rotate import (
    apply_oft_rotation_triton,
)
from sglang.srt.oft.triton_ops.grouped_moe_rotate_project import (
    fused_split_w13_oft_grouped_moe,
    packed_bmm_split_w13_oft_grouped_moe,
)
from sglang.srt.oft.utils import MoEOFTBatchInfo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

M = 8
HIDDEN = 32
HALF = 16
EXPERTS = 2
TOP_K = 1
BLOCK_M = 8
TOL = 2e-3


def _fixture(block_size: int):
    generator = torch.Generator(device="cuda").manual_seed(17 + block_size)
    hidden_states = (
        torch.randn(
            M,
            HIDDEN,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.01
    ).contiguous()
    w13 = (
        torch.randn(
            EXPERTS,
            2 * HALF,
            HIDDEN,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    ).contiguous()
    blocks = HIDDEN // block_size
    eye = torch.eye(block_size, device="cuda", dtype=torch.float32)
    noise = (
        torch.randn(
            EXPERTS,
            blocks,
            block_size,
            block_size,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    skew = noise - noise.transpose(-1, -2)
    w1_oft_r = (eye + skew).to(torch.bfloat16).contiguous()
    w3_oft_r = (eye - 0.5 * skew).to(torch.bfloat16).contiguous()
    topk_ids = torch.tensor(
        [[0], [1], [0], [1], [0], [1], [0], [1]],
        device="cuda",
        dtype=torch.int32,
    )
    routing = moe_align_block_size(topk_ids, BLOCK_M, EXPERTS)
    return hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids, routing


def _rotate_row(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    block_size = rotation.shape[-1]
    blocks = [
        x[offset : offset + block_size].float() @ rotation[block].float()
        for block, offset in enumerate(range(0, HIDDEN, block_size))
    ]
    return torch.cat(blocks).to(torch.bfloat16).float()


def _project_reference(hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids):
    rows = []
    for token in range(M):
        expert = int(topk_ids[token, 0].item())
        gate_x = _rotate_row(hidden_states[token], w1_oft_r[expert])
        up_x = _rotate_row(hidden_states[token], w3_oft_r[expert])
        gate = gate_x @ w13[expert, :HALF].float().T
        up = up_x @ w13[expert, HALF:].float().T
        rows.append(torch.cat((gate, up)))
    return torch.stack(rows).to(torch.bfloat16).unsqueeze(1)


def _assert_alignment_blocks_are_homogeneous(
    sorted_token_ids,
    expert_ids,
    oft_ids,
    num_tokens_post_padded,
    topk_ids,
    row_slots,
    block_m,
):
    flat_experts = topk_ids.reshape(-1)
    for slot_id in oft_ids:
        num_blocks = int(num_tokens_post_padded[int(slot_id)].item()) // block_m
        for block_idx in range(num_blocks):
            row_ids = sorted_token_ids[
                int(slot_id), block_idx * block_m : (block_idx + 1) * block_m
            ]
            row_ids = row_ids[row_ids < flat_experts.numel()].long()
            if row_ids.numel() == 0:
                continue
            assert torch.all(
                flat_experts[row_ids] == expert_ids[int(slot_id), block_idx]
            )
            assert torch.all(row_slots[row_ids] == slot_id)


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_grouped_moe_direct_and_packed_match_torch(block_size):
    hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids, routing = _fixture(
        block_size
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    assert int(num_tokens_post_padded.item()) > M * TOP_K

    kwargs = dict(
        hidden_states=hidden_states,
        w13=w13,
        w1_oft_r=w1_oft_r,
        w3_oft_r=w3_oft_r,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_m=BLOCK_M,
    )
    direct = fused_split_w13_oft_grouped_moe(**kwargs)
    packed = packed_bmm_split_w13_oft_grouped_moe(**kwargs)
    torch.cuda.synchronize()
    reference = _project_reference(
        hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids
    )

    assert direct.shape == (M, TOP_K, 2 * HALF)
    assert packed.shape == direct.shape
    direct_error = (direct.float() - reference.float()).abs().max().item()
    packed_error = (packed.float() - reference.float()).abs().max().item()
    assert direct_error <= TOL, f"BS={block_size} direct max_abs={direct_error:.2e}"
    assert packed_error <= TOL, f"BS={block_size} packed max_abs={packed_error:.2e}"


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_shared_grouped_rotation_matches_torch(block_size):
    hidden_states, _, w1_oft_r, _, topk_ids, _ = _fixture(block_size)
    identity = torch.eye(
        block_size, device="cuda", dtype=torch.bfloat16
    ).view(1, 1, block_size, block_size)
    rotations = torch.stack(
        (identity.expand_as(w1_oft_r), w1_oft_r), dim=0
    ).contiguous()
    request_slots = torch.ones(M, device="cuda", dtype=torch.int32)
    batch = MoEOFTBatchInfo(
        seg_indptr=torch.tensor([0, M], device="cuda", dtype=torch.int32),
        req_to_oft=torch.tensor([1], device="cuda", dtype=torch.int32),
        adapter_enabled=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        token_oft_mapping=request_slots,
        num_tokens=M,
    )
    oft_info = OFTInfo(
        w13_oft_r=rotations,
        w1_oft_r=None,
        w3_oft_r=None,
        w2_oft_r=rotations,
        batch_info=batch,
        num_experts=EXPERTS,
        max_ofts=2,
        has_active_oft=True,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded, oft_ids = (
        _compute_oft_alignment(topk_ids, oft_info)
    )
    rotated = apply_oft_rotation_triton(
        hidden_states,
        rotations,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        oft_ids,
        batch.adapter_enabled,
        top_k=TOP_K,
        block_m=64,
    )
    torch.cuda.synchronize()

    reference = torch.stack(
        [
            _rotate_row(hidden_states[token], w1_oft_r[int(topk_ids[token, 0])])
            for token in range(M)
        ]
    )
    assert rotated.shape == (M * TOP_K, HIDDEN)
    error = (rotated.float() - reference.float()).abs().max().item()
    assert error <= TOL, f"BS={block_size} shared max_abs={error:.2e}"


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_rotation_routes_base_and_two_adapters_through_both_moe_stages(
    block_size,
):
    num_tokens, top_k, num_experts, block_m = 4, 2, 3, 64
    eye = torch.eye(block_size, device="cuda", dtype=torch.bfloat16)
    rotations = eye.view(1, 1, 1, block_size, block_size).expand(
        3, num_experts, 1, block_size, block_size
    ).clone()
    rotations[1, :, 0] = torch.diag(
        torch.tensor(
            [-1 if i % 2 == 0 else 1 for i in range(block_size)],
            device="cuda",
            dtype=torch.bfloat16,
        )
    )
    rotations[2, :, 0] = torch.flip(eye, dims=[1])
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    request_slots = torch.tensor([0, 1, 2, 1], device="cuda")
    batch = MoEOFTBatchInfo(
        seg_indptr=torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int32),
        req_to_oft=request_slots.to(torch.int32),
        adapter_enabled=torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32),
        token_oft_mapping=request_slots.to(torch.int32),
        num_tokens=num_tokens,
    )
    oft_info = OFTInfo(
        w13_oft_r=rotations,
        w1_oft_r=None,
        w3_oft_r=None,
        w2_oft_r=rotations,
        batch_info=batch,
        num_experts=num_experts,
        max_ofts=3,
        has_active_oft=True,
    )

    # Gate/up starts from one row per request and expands to one row per route.
    gate_up_input = torch.arange(
        num_tokens * block_size, device="cuda", dtype=torch.bfloat16
    ).view(num_tokens, block_size)
    sorted_ids, expert_ids, padded, oft_ids = _compute_oft_alignment(
        topk_ids, oft_info, row_group_factor=1
    )
    expanded_slots = request_slots.repeat_interleave(top_k)
    _assert_alignment_blocks_are_homogeneous(
        sorted_ids,
        expert_ids,
        oft_ids,
        padded,
        topk_ids,
        expanded_slots,
        block_m,
    )
    actual_gate_up = apply_oft_rotation_triton(
        gate_up_input,
        rotations,
        topk_ids,
        sorted_ids,
        expert_ids,
        padded,
        oft_ids,
        batch.adapter_enabled,
        top_k=top_k,
        block_m=block_m,
    )
    expected_gate_up = torch.stack(
        [
            gate_up_input[token]
            if int(request_slots[token]) == 0
            else gate_up_input[token]
            @ rotations[
                int(request_slots[token]), int(topk_ids[token, route]), 0
            ]
            for token in range(num_tokens)
            for route in range(top_k)
        ]
    )
    torch.testing.assert_close(actual_gate_up, expected_gate_up, rtol=0, atol=0)

    # Down starts from M * top_k rows. Scaling segment boundaries by top_k
    # repeats each request's adapter slot for both routed expert rows.
    down_input = torch.arange(
        num_tokens * top_k * block_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).view(num_tokens * top_k, block_size)
    down_topk_ids = topk_ids.reshape(-1, 1)
    sorted_ids, expert_ids, padded, oft_ids = _compute_oft_alignment(
        down_topk_ids, oft_info, row_group_factor=top_k
    )
    _assert_alignment_blocks_are_homogeneous(
        sorted_ids,
        expert_ids,
        oft_ids,
        padded,
        down_topk_ids,
        expanded_slots,
        block_m,
    )
    actual_down = apply_oft_rotation_triton(
        down_input,
        rotations,
        down_topk_ids,
        sorted_ids,
        expert_ids,
        padded,
        oft_ids,
        batch.adapter_enabled,
        top_k=1,
        block_m=block_m,
    )
    expected_down = torch.stack(
        [
            down_input[row]
            if int(expanded_slots[row]) == 0
            else down_input[row]
            @ rotations[
                int(expanded_slots[row]), int(down_topk_ids[row, 0]), 0
            ]
            for row in range(num_tokens * top_k)
        ]
    )
    torch.testing.assert_close(actual_down, expected_down, rtol=0, atol=0)


def test_oft_prerotate_uses_alignment_block_size_for_nondefault_moe_config():
    num_tokens, top_k, num_experts = 4, 2, 3
    oft_block_size, runtime_block_m = 4, 16
    eye = torch.eye(oft_block_size, device="cuda", dtype=torch.bfloat16)
    rotations = eye.view(1, 1, 1, oft_block_size, oft_block_size).expand(
        3, num_experts, 1, oft_block_size, oft_block_size
    ).clone()
    rotations[1, :, 0] = torch.diag(
        torch.tensor([-1, 1, -1, 1], device="cuda", dtype=torch.bfloat16)
    )
    rotations[2, :, 0] = torch.flip(eye, dims=[1])
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [0, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    request_slots = torch.tensor([0, 1, 2, 1], device="cuda")
    batch = MoEOFTBatchInfo(
        seg_indptr=torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int32),
        req_to_oft=request_slots.to(torch.int32),
        adapter_enabled=torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32),
        token_oft_mapping=request_slots.to(torch.int32),
        num_tokens=num_tokens,
    )
    oft_info = OFTInfo(
        w13_oft_r=rotations,
        w1_oft_r=None,
        w3_oft_r=None,
        w2_oft_r=rotations,
        batch_info=batch,
        num_experts=num_experts,
        max_ofts=3,
        has_active_oft=True,
    )
    hidden_states = torch.arange(
        num_tokens * oft_block_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).view(num_tokens, oft_block_size)
    routing = moe_align_block_size(topk_ids, runtime_block_m, num_experts)

    actual, *_ = _oft_prerotate(
        hidden_states,
        rotations,
        oft_info,
        torch.empty(
            num_tokens * top_k, 1, device="cuda", dtype=torch.bfloat16
        ),
        torch.ones(num_tokens, top_k, device="cuda"),
        topk_ids,
        *routing,
        top_k,
        num_experts,
        runtime_block_m,
    )
    expected = torch.stack(
        [
            hidden_states[token]
            if int(request_slots[token]) == 0
            else hidden_states[token]
            @ rotations[
                int(request_slots[token]), int(topk_ids[token, route]), 0
            ]
            for token in range(num_tokens)
            for route in range(top_k)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_nonlocal_route_rows_remain_zero_with_padding(block_size):
    hidden_states, w13, w1_oft_r, w3_oft_r, topk_ids, routing = _fixture(
        block_size
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    nonlocal_experts = expert_ids.clone()
    first_block_routes = sorted_token_ids[:BLOCK_M]
    valid_routes = first_block_routes[first_block_routes < M * TOP_K].long()
    nonlocal_experts[0] = -1

    out = fused_split_w13_oft_grouped_moe(
        hidden_states=hidden_states,
        w13=w13,
        w1_oft_r=w1_oft_r,
        w3_oft_r=w3_oft_r,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=nonlocal_experts,
        num_tokens_post_padded=num_tokens_post_padded,
        block_m=BLOCK_M,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(out.reshape(M * TOP_K, -1)[valid_routes]).item() == 0

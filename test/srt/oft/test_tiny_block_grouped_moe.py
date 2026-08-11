from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    apply_oft_rotation_triton,
)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.oft.triton_ops.grouped_moe_rotate_project import (
    fused_split_w13_oft_grouped_moe,
    packed_bmm_split_w13_oft_grouped_moe,
)

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
    hidden_states, _, w1_oft_r, _, topk_ids, routing = _fixture(block_size)
    sorted_token_ids, expert_ids, num_tokens_post_padded = routing
    rotated = apply_oft_rotation_triton(
        hidden_states,
        w1_oft_r,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        TOP_K,
        block_m=BLOCK_M,
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

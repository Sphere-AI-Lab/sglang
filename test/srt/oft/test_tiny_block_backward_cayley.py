from __future__ import annotations

import pytest
import torch

from sglang.srt.peft.oft.triton_ops.cayley_neumann import (
    cayley_neumann,
    cayley_neumann_bwd,
    cayley_neumann_fwd,
)
from sglang.srt.peft.oft.triton_ops.gemm_oft_r_backward import gemm_oft_r_bwd
from sglang.srt.peft.oft.triton_ops.sgemm_oft_r_bwd import sgemm_oft_r_grad_R
from sglang.srt.peft.oft.torch_ops.oft_ops import (
    cayley_neumann as torch_cayley_neumann,
)

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
TOL = 2e-3


def _reference_backward(x, weights, grad_y, num_slices=1):
    _, _, block_size, _ = weights.shape
    total_tokens, input_dim = x.shape
    num_blocks = input_dim // block_size
    grad_x = torch.zeros_like(x, dtype=torch.float32)
    grad_R = torch.zeros_like(weights, dtype=torch.float32)
    for slice_id in range(num_slices):
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            stop = start + block_size
            weight_idx = slice_id * num_blocks + block_idx
            x_block = x[:, start:stop].float()
            gy_block = grad_y[
                :, slice_id * input_dim + start : slice_id * input_dim + stop
            ].float()
            R_block = weights[0, weight_idx].float()
            grad_x[:, start:stop] += gy_block @ R_block.transpose(0, 1)
            grad_R[0, weight_idx] = x_block.transpose(0, 1) @ gy_block
    return grad_x, grad_R


@CUDA_ONLY
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_backward_and_segmented_grad_r_match_torch(block_size):
    generator = torch.Generator(device="cuda").manual_seed(41 + block_size)
    total_tokens = 17
    input_dim = 32
    num_slices = 2
    num_blocks = input_dim // block_size
    x = (
        torch.randn(
            total_tokens,
            input_dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    ).contiguous()
    weights = (
        torch.randn(
            1,
            num_slices * num_blocks,
            block_size,
            block_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    ).contiguous()
    grad_y = (
        torch.randn(
            total_tokens,
            num_slices * input_dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    ).contiguous()
    slot = torch.tensor(0, device="cuda", dtype=torch.int32)
    bsv = torch.tensor(block_size, device="cuda", dtype=torch.int32)

    grad_x, grad_R = gemm_oft_r_bwd(
        x, weights, grad_y, slot, bsv, num_slices=num_slices
    )
    expected_grad_x, expected_grad_R = _reference_backward(
        x, weights, grad_y, num_slices
    )
    segmented_grad_R = sgemm_oft_r_grad_R(
        x, grad_y[:, :input_dim].contiguous(), num_blocks, block_size
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        grad_x.float(), expected_grad_x, atol=TOL, rtol=0
    )
    torch.testing.assert_close(
        grad_R.float(), expected_grad_R, atol=TOL, rtol=0
    )
    torch.testing.assert_close(
        segmented_grad_R.float(), expected_grad_R[0, :num_blocks], atol=TOL, rtol=0
    )


def _torch_cayley_neumann_bwd(grad_R, Q_skew):
    q_t = Q_skew.transpose(-1, -2)
    g_prev = grad_R
    acc = grad_R
    for _ in range(3):
        g_k = (2.0 * grad_R + g_prev @ q_t).to(grad_R.dtype)
        g_prev = g_k
        acc = (g_k + q_t @ acc).to(grad_R.dtype)
    return acc


@CUDA_ONLY
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_direct_cayley_forward_backward_match_torch(block_size):
    generator = torch.Generator(device="cuda").manual_seed(73 + block_size)
    raw = (
        torch.randn(
            3,
            block_size,
            block_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.01
    )
    q_skew = (raw - raw.transpose(-1, -2)).contiguous()
    grad_R = (
        torch.randn(
            q_skew.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.01
    ).contiguous()

    actual_R = cayley_neumann_fwd(q_skew)
    actual_grad = cayley_neumann_bwd(grad_R, q_skew)
    expected_R = torch_cayley_neumann(q_skew)
    expected_grad = _torch_cayley_neumann_bwd(grad_R, q_skew)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_R.float(), expected_R.float(), atol=TOL, rtol=0)
    torch.testing.assert_close(
        actual_grad.float(), expected_grad.float(), atol=TOL, rtol=0
    )


def test_tiny_public_cayley_is_cpu_gradcheck_safe():
    raw = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True) * 0.01
    q_skew = raw - raw.transpose(-1, -2)
    assert torch.autograd.gradcheck(
        lambda q: cayley_neumann(q),
        (q_skew,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


@CUDA_ONLY
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_public_cayley_cuda_autograd_matches_direct_backward(block_size):
    raw = torch.randn(
        2,
        block_size,
        block_size,
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    q_skew = (raw - raw.transpose(-1, -2)).detach().requires_grad_(True)
    grad_R = torch.randn_like(q_skew) * 0.01
    output = cayley_neumann(q_skew)
    (actual_grad,) = torch.autograd.grad(output, q_skew, grad_R)
    expected_R = torch_cayley_neumann(q_skew)
    expected_grad = _torch_cayley_neumann_bwd(grad_R, q_skew)
    torch.testing.assert_close(output, expected_R, atol=2e-5, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=0)

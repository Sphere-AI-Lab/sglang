"""Large OFT block sizes must launch, and must agree with the small ones.

The kernel stages the BS x BS rotation block in shared memory, so its footprint
is 6*BS*(BS+128) bytes against a 232,448 B limit -- exact, verified at
BS=256/512/1024. Above BS=128 it cannot launch at all. The tiled path walks the
rotation block in OFT_TILE_K sub-tiles so the footprint stops depending on BS.

Correctness is anchored two ways. Against an identity rotation the fused result
must equal a plain projection, so a mismatch is the kernel's and not a drifting
reference's. Against a real rotation it must match an fp32 torch implementation
of the same operation. Both references are fp32 while the kernel accumulates in
fp32 and casts to bf16, so ~1e-4 is rounding; the bar is 2e-3, the tolerance the
kernel file's own parity harness already uses.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.oft.triton_ops.fused_rotate_project import (
    OFT_TILE_K,
    fused_rotate_project_qkv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Llama-3.1-8B fused QKV: hidden 4096 in, (32 + 8 + 8) * 128 = 6144 out.
K, OUT = 4096, [4096, 1024, 1024]
TOL = 2e-3


def _inputs(M, BS, device="cuda", dtype=torch.bfloat16, rotate=True, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(M, K, device=device, dtype=dtype, generator=g) * 0.01).contiguous()
    W = (torch.randn(sum(OUT), K, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    blocks = 3 * (K // BS)
    eye = torch.eye(BS, device=device, dtype=dtype)
    if not rotate:
        return x, eye.expand(blocks, BS, BS).contiguous(), W
    # Identity plus a small skew: exercises every element of R, not just the
    # diagonal, while staying close enough to orthogonal to be realistic.
    noise = torch.randn(blocks, BS, BS, device=device, dtype=torch.float32, generator=g) * 0.02
    skew = noise - noise.transpose(-1, -2)
    R = (eye.float().unsqueeze(0) + skew).to(dtype).contiguous()
    return x, R, W


def _reference(x, R, W, out_sizes):
    """Rotate each block of x, then project. fp32 throughout."""
    BS = R.shape[-1]
    blocks_per_slice = R.shape[0] // len(out_sizes)
    outs = []
    offset = 0
    for s, width in enumerate(out_sizes):
        Ws = W[offset:offset + width].float()
        offset += width
        acc = torch.zeros(x.shape[0], width, device=x.device, dtype=torch.float32)
        for b in range(blocks_per_slice):
            k0 = b * BS
            xb = x[:, k0:k0 + BS].float()
            Rb = R[s * blocks_per_slice + b].float()
            acc += (xb @ Rb) @ Ws[:, k0:k0 + BS].T
        outs.append(acc)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("BS", [256, 512, 1024])
@pytest.mark.parametrize("M", [1, 64, 256])
def test_large_blocks_launch_and_are_correct(BS, M):
    """The whole point: these three block sizes cannot launch today."""
    x, R, W = _inputs(M, BS)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 64, 256])
def test_small_blocks_still_correct(BS, M):
    """The untiled path must be untouched. If this breaks, the constexpr switch
    is selecting the tiled path where it should not."""
    x, R, W = _inputs(M, BS)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [128, 256, 1024])
def test_identity_rotation_is_a_plain_projection(BS):
    """With R = I the kernel must reproduce x @ W.T. A failure here means the
    rotation matmul is wrong, independent of any reference implementation."""
    x, R, W = _inputs(64, BS, rotate=False)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - (x.float() @ W.float().T)).abs().max().item()
    assert err <= TOL, f"BS={BS} max_abs={err:.2e}"


def test_the_two_paths_agree_at_the_boundary():
    """BS=128 is the largest untiled size. Forcing the tiled path at the same BS
    must give the same answer -- this is what proves the tiled path is a
    reimplementation of the operator and not a different operator."""
    x, R, W = _inputs(64, 128)
    untiled = fused_rotate_project_qkv(x, R, W, OUT)
    tiled = fused_rotate_project_qkv(x, R, W, OUT, force_tiled=True)
    torch.cuda.synchronize()
    err = (untiled.float() - tiled.float()).abs().max().item()
    assert err <= TOL, f"paths disagree by {err:.2e}"


def test_the_tile_width_divides_every_supported_block():
    """OFT_TILE_K must divide each BS the kernel accepts, or the inner loop
    reads past the edge of the rotation block."""
    for BS in (16, 32, 64, 128, 256, 512, 1024):
        assert BS % min(OFT_TILE_K, BS) == 0, BS


def test_shared_memory_no_longer_scales_with_block_size():
    """The regression guard. If a full-BS load is ever reinstated, BS=1024 stops
    launching and this fails with the OutOfResources message."""
    x, R, W = _inputs(8, 1024)
    out = fused_rotate_project_qkv(x, R, W, OUT)  # must not raise
    torch.cuda.synchronize()
    assert out.shape == (8, sum(OUT))

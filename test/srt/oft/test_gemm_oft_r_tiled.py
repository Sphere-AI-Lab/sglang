"""`gemm_oft_r_fwd` must launch and stay correct at large OFT block sizes.

This is the un-fused rotation path -- the one `o_proj` and `down_proj` take,
since they have no sibling projection to fuse into. It had the same defect the
fused kernel had: it staged the whole ``BLOCK_SIZE x BLOCK_SIZE`` rotation block
in shared memory, so at BS=1024 it asked for

    (BLOCK_S * BS + BS * BS) * 2 = (64*1024 + 1024*1024) * 2 = 2,228,224 B

against a 232,448 B limit, and could not launch at all. Observed in a real RL
rollout, not constructed: an OFT arm with ``--target all`` reaches this kernel on
every layer, and died here after the fused kernel was already tiled.

Tiling K alone does not fix it. The accumulator is ``(BLOCK_S, BLOCK_SIZE)``
fp32 -- 64*1024*4 = 256 KB of registers per program at BS=1024, more than an
entire SM's register file. The output columns have to be tiled as well, which is
why the launcher grew a third factor on grid axis 0.

Correctness is anchored two ways, the same way the fused kernel's tests are:
against an identity rotation (where the answer must be a plain copy, a reference
that cannot itself drift) and against an fp32 torch matmul of the same
block-diagonal operation. Both references are fp32 while the kernel accumulates
in fp32 and casts to bf16, so ~1e-3 is rounding; the bar is 2e-3, matching the
tolerance the fused kernel's parity harness already uses.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.oft.triton_ops.gemm_oft_r import gemm_oft_r_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TOL = 2e-3
# Llama-3.1-8B hidden size: what o_proj and down_proj rotate in the failing run.
INPUT_DIM = 4096


def _inputs(tokens, block_size, num_slices=1, rotate=True, dim=INPUT_DIM, seed=0):
    """x, weights, and the two 0-d control tensors the launcher takes."""
    dev, dt = "cuda", torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(seed)
    num_blocks = dim // block_size
    x = (torch.randn(tokens, dim, device=dev, dtype=dt, generator=g) * 0.01).contiguous()

    eye = torch.eye(block_size, device=dev, dtype=dt)
    total = num_slices * num_blocks
    if rotate:
        # Identity plus a small skew: touches every element of R rather than
        # just the diagonal, while staying close to orthogonal as a real
        # Cayley-constructed R would be.
        noise = torch.randn(
            total, block_size, block_size, device=dev, dtype=torch.float32, generator=g
        ) * 0.02
        R = (eye.float().unsqueeze(0) + (noise - noise.transpose(-1, -2))).to(dt)
    else:
        R = eye.expand(total, block_size, block_size).clone()
    weights = R.unsqueeze(0).contiguous()  # (num_ofts=1, total_blocks, BS, BS)

    slot = torch.zeros((), device=dev, dtype=torch.int32)
    bsv = torch.tensor(block_size, device=dev, dtype=torch.int32)
    return x, weights, slot, bsv


def _reference(x, weights, block_size, num_slices):
    """Block-diagonal x @ R per slice, fp32, concatenated along columns."""
    dim = x.shape[1]
    num_blocks = dim // block_size
    R = weights[0].float()
    outs = []
    for s in range(num_slices):
        acc = torch.empty(x.shape[0], dim, device=x.device, dtype=torch.float32)
        for b in range(num_blocks):
            k0 = b * block_size
            acc[:, k0:k0 + block_size] = (
                x[:, k0:k0 + block_size].float() @ R[s * num_blocks + b]
            )
        outs.append(acc)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("block_size", [256, 512, 1024])
@pytest.mark.parametrize("tokens", [1, 64, 256])
def test_large_blocks_launch_and_are_correct(block_size, tokens):
    """The whole point: these three block sizes cannot launch before the fix."""
    x, w, slot, bsv = _inputs(tokens, block_size)
    out = gemm_oft_r_fwd(x, w, slot, bsv, num_slices=1)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, w, block_size, 1)).abs().max().item()
    assert err <= TOL, f"BS={block_size} tokens={tokens} max_abs={err:.2e}"


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("tokens", [1, 64, 256])
def test_small_blocks_are_unchanged(block_size, tokens):
    """At BS <= 128 the tiles degenerate to one iteration, so this path must
    behave exactly as it did. If this breaks, the tile picker is splitting a
    block size that was always fine."""
    x, w, slot, bsv = _inputs(tokens, block_size)
    out = gemm_oft_r_fwd(x, w, slot, bsv, num_slices=1)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, w, block_size, 1)).abs().max().item()
    assert err <= TOL, f"BS={block_size} tokens={tokens} max_abs={err:.2e}"


@pytest.mark.parametrize("block_size", [8, 4])
def test_tiny_blocks_keep_the_elementwise_fallback(block_size):
    """Below 16 the kernel takes an element-wise loop instead of tl.dot. Column
    tiling must not disturb it."""
    x, w, slot, bsv = _inputs(64, block_size)
    out = gemm_oft_r_fwd(x, w, slot, bsv, num_slices=1)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, w, block_size, 1)).abs().max().item()
    assert err <= TOL, f"BS={block_size} max_abs={err:.2e}"


@pytest.mark.parametrize("block_size", [128, 512, 1024])
@pytest.mark.parametrize("num_slices", [2, 3])
def test_fused_slice_counts(block_size, num_slices):
    """num_slices 2 and 3 are the gate_up and qkv layouts. They multiply the
    grid, so they exercise the block/token/column index decomposition that the
    third grid factor changed."""
    x, w, slot, bsv = _inputs(64, block_size, num_slices=num_slices)
    out = gemm_oft_r_fwd(x, w, slot, bsv, num_slices=num_slices)
    torch.cuda.synchronize()
    assert out.shape == (64, num_slices * INPUT_DIM)
    err = (out.float() - _reference(x, w, block_size, num_slices)).abs().max().item()
    assert err <= TOL, f"BS={block_size} slices={num_slices} max_abs={err:.2e}"


@pytest.mark.parametrize("block_size", [128, 1024])
def test_identity_rotation_is_a_copy(block_size):
    """With R = I the output must equal the input. A failure here is the
    rotation matmul's, independent of any reference implementation."""
    x, w, slot, bsv = _inputs(64, block_size, rotate=False)
    out = gemm_oft_r_fwd(x, w, slot, bsv, num_slices=1)
    torch.cuda.synchronize()
    err = (out.float() - x.float()).abs().max().item()
    assert err <= TOL, f"BS={block_size} max_abs={err:.2e}"


@pytest.mark.parametrize("block_size", [128, 1024])
def test_zero_block_size_is_an_identity_passthrough(block_size):
    """block_size_val == 0 means "no adapter". It is a RUNTIME value, so this
    branch compiles alongside the dot branch and its own shared-memory
    footprint counts against the same budget -- the exact trap that made the
    first pass at the fused kernel still fail at BS=512."""
    x, w, slot, _ = _inputs(64, block_size)
    bsv0 = torch.zeros((), device=x.device, dtype=torch.int32)
    out = gemm_oft_r_fwd(x, w, slot, bsv0, num_slices=1)
    torch.cuda.synchronize()
    assert torch.equal(out, x)


def test_the_tile_picker_respects_the_shared_memory_budget():
    """The regression guard, checkable without a GPU-side launch: whatever the
    picker returns must fit, and must divide the block size or the inner loop
    reads past the edge of the rotation block."""
    from sglang.srt.oft.triton_ops.gemm_oft_r import (
        OFT_SMEM_BUDGET,
        _pick_tiles,
        _tiled_smem_bytes,
    )

    for block_size in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        for block_s in (16, 64):
            tile_k, tile_n = _pick_tiles(block_size, block_s)
            assert block_size % tile_k == 0, (block_size, tile_k)
            assert block_size % tile_n == 0, (block_size, tile_n)
            assert _tiled_smem_bytes(block_s, tile_k, tile_n) <= OFT_SMEM_BUDGET, (
                block_size, block_s, tile_k, tile_n
            )


def test_small_block_sizes_are_not_tiled_at_all():
    """The no-regression contract stated as an assertion: at or below 128 the
    picker must return the block size itself, so both loops run exactly once
    and the generated code matches what shipped before."""
    from sglang.srt.oft.triton_ops.gemm_oft_r import _pick_tiles

    for block_size in (4, 8, 16, 32, 64, 128):
        assert _pick_tiles(block_size, 64) == (block_size, block_size), block_size

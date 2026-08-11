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


@pytest.mark.parametrize("BS", [4, 8])
@pytest.mark.parametrize("M", [1, 8, 64])
def test_tiny_qkv_blocks_are_correct(BS, M):
    x, R, W = _inputs(M, BS)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [4, 8, 128, 256, 1024])
def test_identity_rotation_is_a_plain_projection(BS):
    """With R = I the kernel must reproduce x @ W.T. A failure here means the
    rotation matmul is wrong, independent of any reference implementation."""
    x, R, W = _inputs(64, BS, rotate=False)
    out = fused_rotate_project_qkv(x, R, W, OUT)
    torch.cuda.synchronize()
    err = (out.float() - (x.float() @ W.float().T)).abs().max().item()
    assert err <= TOL, f"BS={BS} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [4, 8])
def test_tiny_qkv_runtime_identity_slot(BS):
    """The runtime zero sentinel must bypass R for captured adapter slots."""
    x, R, W = _inputs(32, BS, rotate=True)
    R4 = torch.stack([R, R], dim=0).contiguous()
    slot = torch.tensor(1, device="cuda", dtype=torch.int32)
    bsv = torch.tensor(0, device="cuda", dtype=torch.int32)
    out = fused_rotate_project_qkv(
        x, R4, W, OUT, slot_idx_t=slot, bsv_t=bsv
    )
    torch.cuda.synchronize()
    err = (out.float() - (x.float() @ W.float().T)).abs().max().item()
    assert err <= TOL, f"BS={BS} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [4, 8])
def test_tiny_qkv_cuda_graph_replays_identity_then_rotation(BS):
    x, R, W = _inputs(8, BS, rotate=True)
    R4 = torch.stack([R, R], dim=0).contiguous()
    slot = torch.tensor(0, device="cuda", dtype=torch.int32)
    bsv = torch.tensor(0, device="cuda", dtype=torch.int32)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        fused_rotate_project_qkv(
            x, R4, W, OUT, slot_idx_t=slot, bsv_t=bsv
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = fused_rotate_project_qkv(
            x, R4, W, OUT, slot_idx_t=slot, bsv_t=bsv
        )
    output_ptr = captured_out.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    identity_error = (
        captured_out.float() - (x.float() @ W.float().T)
    ).abs().max().item()
    assert identity_error <= TOL, f"BS={BS} identity max_abs={identity_error:.2e}"

    slot.fill_(1)
    bsv.fill_(BS)
    graph.replay()
    torch.cuda.synchronize()
    rotation_error = (
        captured_out.float() - _reference(x, R, W, OUT)
    ).abs().max().item()
    assert captured_out.data_ptr() == output_ptr
    assert rotation_error <= TOL, f"BS={BS} rotation max_abs={rotation_error:.2e}"


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
    for BS in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        assert BS % min(OFT_TILE_K, BS) == 0, BS


def test_shared_memory_no_longer_scales_with_block_size():
    """The regression guard. If a full-BS load is ever reinstated, BS=1024 stops
    launching and this fails with the OutOfResources message."""
    x, R, W = _inputs(8, 1024)
    out = fused_rotate_project_qkv(x, R, W, OUT)  # must not raise
    torch.cuda.synchronize()
    assert out.shape == (8, sum(OUT))


from sglang.srt.oft.triton_ops.fused_rotate_project import (  # noqa: E402
    fused_rotate_gate_up_inputs,
    fused_rotate_project_gate_up,
)

# Llama-3.1-8B FC1: hidden 4096 in, gate and up of 14336 each.
FC1_K, FC1_OUT = 4096, [14336, 14336]


def _fc1_inputs(M, BS, device="cuda", dtype=torch.bfloat16, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(M, FC1_K, device=device, dtype=dtype, generator=g) * 0.01).contiguous()
    W = (torch.randn(sum(FC1_OUT), FC1_K, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    blocks = 2 * (FC1_K // BS)
    eye = torch.eye(BS, device=device, dtype=dtype)
    noise = torch.randn(blocks, BS, BS, device=device, dtype=torch.float32, generator=g) * 0.02
    R = (eye.float().unsqueeze(0) + (noise - noise.transpose(-1, -2))).to(dtype).contiguous()
    return x, R, W


@pytest.mark.parametrize("BS", [128, 256, 512, 1024])
@pytest.mark.parametrize("M", [1, 64])
def test_gate_up_projection_at_large_blocks(BS, M):
    """gate_up shares the inner routine with QKV but picks different tiles --
    GROUP_N up to 4 -- so its shared-memory budget is tighter and it needs its
    own coverage."""
    x, R, W = _fc1_inputs(M, BS)
    out = fused_rotate_project_gate_up(x, R, W, FC1_OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, FC1_OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [4, 8])
@pytest.mark.parametrize("M", [1, 8, 64])
def test_tiny_gate_up_projection(BS, M):
    x, R, W = _fc1_inputs(M, BS)
    out = fused_rotate_project_gate_up(x, R, W, FC1_OUT)
    torch.cuda.synchronize()
    err = (out.float() - _reference(x, R, W, FC1_OUT)).abs().max().item()
    assert err <= TOL, f"BS={BS} M={M} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [128, 256, 512, 1024])
def test_gate_up_inputs_at_large_blocks(BS):
    """No projection here -- it returns the two rotated inputs, so the reference
    is the rotation alone, and it is a separate kernel from the two above."""
    x, R, _ = _fc1_inputs(64, BS)
    x_gate, x_up = fused_rotate_gate_up_inputs(x, R)
    torch.cuda.synchronize()
    blocks_per_slice = R.shape[0] // 2
    for idx, got in enumerate((x_gate, x_up)):
        expect = torch.empty_like(got, dtype=torch.float32)
        for b in range(blocks_per_slice):
            k0 = b * BS
            expect[:, k0:k0 + BS] = (
                x[:, k0:k0 + BS].float() @ R[idx * blocks_per_slice + b].float()
            )
        err = (got.float() - expect).abs().max().item()
        assert err <= TOL, f"BS={BS} slice={idx} max_abs={err:.2e}"


@pytest.mark.parametrize("BS", [4, 8])
@pytest.mark.parametrize("M", [1, 8, 64])
def test_tiny_gate_up_inputs(BS, M):
    x, R, _ = _fc1_inputs(M, BS)
    x_gate, x_up = fused_rotate_gate_up_inputs(x, R)
    torch.cuda.synchronize()
    blocks_per_slice = R.shape[0] // 2
    for idx, got in enumerate((x_gate, x_up)):
        expect = torch.empty_like(got, dtype=torch.float32)
        for b in range(blocks_per_slice):
            k0 = b * BS
            expect[:, k0:k0 + BS] = (
                x[:, k0:k0 + BS].float()
                @ R[idx * blocks_per_slice + b].float()
            )
        err = (got.float() - expect).abs().max().item()
        assert err <= TOL, f"BS={BS} M={M} slice={idx} max_abs={err:.2e}"


def test_the_tile_picker_respects_the_shared_memory_budget():
    """GROUP_N >= 2 cannot afford a 128-wide tile: 245,760 B against 232,448.
    The picker must reduce it rather than let the launch fail the way BS > 128
    used to."""
    from sglang.srt.oft.triton_ops.fused_rotate_project import (
        OFT_SMEM_BUDGET,
        _pick_tile_k,
        _tiled_smem_bytes,
    )

    for BS in (4, 8, 128, 256, 512, 1024):
        for block_m, block_n, group_n in ((64, 64, 1), (64, 64, 2), (16, 64, 4), (32, 64, 8)):
            tk = _pick_tile_k(BS, block_m, block_n, group_n)
            assert tk >= min(16, BS), (BS, group_n, tk)
            assert BS % tk == 0, (BS, tk)
            assert _tiled_smem_bytes(tk, block_m, block_n, group_n) <= OFT_SMEM_BUDGET

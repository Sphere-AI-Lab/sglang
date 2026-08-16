import pytest
import torch

from sglang.srt.peft.oft.layers import split_dense_merged_projection


_K = 16
_M = 2
_DTYPE = torch.bfloat16


class _SentinelBackend:
    def run_fused_rotate_project(self, x, R, weight, output_sizes, bias):
        return torch.full(
            (x.shape[0], sum(output_sizes)),
            9,
            dtype=x.dtype,
            device=x.device,
        )

    def run_qkv_oft(self, x, R):
        return torch.cat(
            [
                torch.ones_like(x),
                torch.full_like(x, 2),
                torch.full_like(x, 3),
            ],
            dim=-1,
        )

    def run_fused_gate_up_inputs(self, x, R):
        return torch.full_like(x, 11), torch.full_like(x, 13)

    def run_gate_up_oft(self, x, R):
        return torch.cat(
            [torch.ones_like(x), torch.full_like(x, 2)], dim=-1
        )


@pytest.fixture(autouse=True)
def _clear_global_fused_disable(monkeypatch):
    monkeypatch.delenv(
        "SGLANG_OFT_DISABLE_FUSED_ROTATE_PROJECT", raising=False
    )


def _x():
    return torch.zeros((_M, _K), dtype=_DTYPE)


def _weight(num_slices):
    return torch.cat(
        [torch.eye(_K, dtype=_DTYPE) for _ in range(num_slices)], dim=0
    ).contiguous()


def _r_buffer(num_slices, block_size):
    blocks_per_slice = _K // block_size
    return torch.zeros(
        (1, num_slices * blocks_per_slice, block_size, block_size),
        dtype=_DTYPE,
    )


@pytest.mark.parametrize("block_size", [4, 8])
def test_tiny_qkv_pool_defaults_to_unfused_output(block_size):
    x = _x()
    got = split_dense_merged_projection(
        x,
        _weight(3),
        None,
        [_K, _K, _K],
        _r_buffer(3, block_size),
        _SentinelBackend(),
    )
    expected = torch.cat(
        [
            torch.ones_like(x),
            torch.full_like(x, 2),
            torch.full_like(x, 3),
        ],
        dim=-1,
    )
    assert torch.equal(got, expected)


def test_qkv_pool_at_width_16_keeps_fused_output():
    x = _x()
    got = split_dense_merged_projection(
        x,
        _weight(3),
        None,
        [_K, _K, _K],
        _r_buffer(3, 16),
        _SentinelBackend(),
    )
    expected = torch.full((_M, 3 * _K), 9, dtype=_DTYPE)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("block_size", [4, 8])
def test_tiny_gate_up_pool_keeps_fused_output(block_size):
    x = _x()
    got = split_dense_merged_projection(
        x,
        _weight(2),
        None,
        [_K, _K],
        _r_buffer(2, block_size),
        _SentinelBackend(),
    )
    expected = torch.cat(
        [torch.full_like(x, 11), torch.full_like(x, 13)], dim=-1
    )
    assert torch.equal(got, expected)

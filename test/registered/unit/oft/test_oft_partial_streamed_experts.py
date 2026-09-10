"""Canonical expert rotations can arrive in independently partitioned chunks."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager


@pytest.mark.parametrize("order", list(itertools.permutations(("gate", "up", "down"))))
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_streamed_expert_projection_chunks_preserve_prior_rotations(order, tp_rank):
    pool = object.__new__(OFTMemoryPool)
    pool._groups = {
        name: {0: torch.full((3, 3, 1, 4, 4), -7.0)}
        for name in ("w1_oft_r", "w3_oft_r", "w2_oft_r")
    }
    for layers in pool._groups.values():
        layers[0][2] = torch.eye(4)
    pointers = {name: layers[0].data_ptr() for name, layers in pool._groups.items()}
    manager = object.__new__(OFTManager)
    manager.oft_type = "canonical_oft"
    manager.oft_r_dtype = torch.float32
    manager.memory_pool = pool
    moe = SimpleNamespace(
        num_local_experts=3,
        moe_tp_rank=tp_rank,
        moe_tp_size=2,
        w13_weight=torch.empty(0),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id - 2,
    )
    manager._find_fused_moe_modules = lambda: {0: moe}
    compact = torch.tensor([[0.1, 0, 0, 0, 0, 0]])
    # Hand-computed five-term Neumann result, independent of loader helpers.
    rotation = torch.tensor(
        [
            [0.9801, 0.198, 0, 0],
            [-0.198, 0.9801, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    targets = {"gate": ("w1_oft_r", 0), "up": ("w3_oft_r", 1), "down": ("w2_oft_r", 0)}
    written = set()

    for projection in order:
        group, local_id = targets[projection]
        weights = (
            torch.cat([torch.zeros_like(compact), compact])
            if projection == "down"
            else compact
        )
        manager.apply_streamed_expert_oft(
            {0: {local_id + 2: {projection + "_proj.oft_R": weights}}},
            block_size=4,
            slot_idx=2,
        )
        written.add((group, local_id))

        for name, layers in pool._groups.items():
            assert layers[0].data_ptr() == pointers[name]
            torch.testing.assert_close(
                layers[0][:2], torch.full_like(layers[0][:2], -7)
            )
            for expert_id in range(3):
                has_rotation = (name, expert_id) in written and not (
                    name == "w2_oft_r" and tp_rank == 0
                )
                expected = rotation if has_rotation else torch.eye(4)
                torch.testing.assert_close(layers[0][2, expert_id, 0], expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

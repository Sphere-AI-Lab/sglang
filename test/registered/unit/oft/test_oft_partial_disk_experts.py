"""Disk expert adapters leave omitted expert/projection rotations neutral."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager


def _manager(oft_type):
    gate_groups = (
        ("w1_oft_r", "w3_oft_r") if oft_type == "canonical_oft" else ("w13_oft_r",)
    )
    pool = object.__new__(OFTMemoryPool)
    pool._groups = {
        name: {0: torch.full((3, 3, 1, 4, 4), -7.0)}
        for name in (*gate_groups, "w2_oft_r")
    }
    for layers in pool._groups.values():
        layers[0][2] = torch.eye(4)
    manager = object.__new__(OFTManager)
    manager.memory_pool = pool
    manager.oft_type = oft_type
    manager.oft_r_dtype = torch.float32
    return manager


def _compact(value):
    return torch.tensor([[value, 0, 0, 0, 0, 0]], dtype=torch.float64)


def _rotation(value):
    # Hand-computed five-term Neumann rotations for the compact fixtures.
    diagonal, off_diagonal = {
        0.05: (0.99500625, 0.09975),
        0.1: (0.9801, 0.198),
    }[value]
    return torch.tensor(
        [
            [diagonal, off_diagonal, 0, 0],
            [-off_diagonal, diagonal, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize(
    "oft_type,projections",
    [
        ("canonical_oft", ("gate_proj", "up_proj", "down_proj")),
        ("oft", ("gate_proj", "down_proj")),
        ("canonical_oft", ("down_proj",)),
        ("canonical_oft", ("gate_proj",)),
        ("canonical_oft", ("up_proj",)),
        ("canonical_oft", ("up_proj", "down_proj")),
    ],
)
def test_partial_disk_experts_preserve_omitted_rotations(
    oft_type, projections, tp_rank
):
    manager = _manager(oft_type)
    moe = SimpleNamespace(
        num_local_experts=3,
        moe_tp_rank=tp_rank,
        moe_tp_size=2,
        w13_weight=torch.empty(0),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id - 2,
    )
    weights = {
        projection
        + ".oft_R": (
            torch.cat([_compact(0.05), _compact(0.1)])
            if projection == "down_proj"
            else _compact(0.1)
        )
        for projection in projections
    }
    # Expert 2 belongs to this EP rank. Experts 0 and 9 do not; neither may
    # index a local slot, and local experts 3 and 4 are omitted entirely.
    manager._apply_expert_oft_to_module(
        moe, {0: weights, 2: weights, 9: weights}, 4, layer_id=0, slot_idx=2
    )

    expected_groups = {
        "gate_proj": "w1_oft_r" if oft_type == "canonical_oft" else "w13_oft_r",
        "up_proj": "w3_oft_r",
        "down_proj": "w2_oft_r",
    }
    supplied_groups = {expected_groups[projection] for projection in projections}
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for group, layers in manager.memory_pool._groups.items():
        buffers = layers[0]
        torch.testing.assert_close(buffers[:2], torch.full_like(buffers[:2], -7))
        for local_id in range(3):
            if group in supplied_groups and local_id == 0:
                value = 0.05 if group == "w2_oft_r" and tp_rank == 0 else 0.1
                expected = _rotation(value)
            else:
                expected = torch.eye(4)
            actual = buffers[2, local_id, 0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(x @ actual, x @ expected)


def test_partial_disk_experts_keep_independently_omitted_projections_neutral():
    manager = _manager("canonical_oft")
    moe = SimpleNamespace(
        num_local_experts=3,
        moe_tp_rank=0,
        moe_tp_size=1,
        w13_weight=torch.empty(0),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
    )
    manager._apply_expert_oft_to_module(
        moe,
        {
            0: {"gate_proj.oft_R": _compact(0.1)},
            1: {"up_proj.oft_R": _compact(0.05)},
        },
        4,
        layer_id=0,
        slot_idx=2,
    )

    for group, supplied_id, value in (("w1_oft_r", 0, 0.1), ("w3_oft_r", 1, 0.05)):
        rotations = manager.memory_pool.slot(group, 0, 2)
        for local_id in range(3):
            expected = _rotation(value) if local_id == supplied_id else torch.eye(4)
            torch.testing.assert_close(rotations[local_id, 0], expected)


@pytest.mark.parametrize("include_gate", [False, True])
def test_shared_disk_experts_reject_independent_up_rotation(include_gate):
    manager = _manager("oft")
    moe = SimpleNamespace(
        num_local_experts=3,
        moe_tp_rank=0,
        moe_tp_size=1,
        w13_weight=torch.empty(0),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
    )
    weights = {"up_proj.oft_R": _compact(0.1)}
    if include_gate:
        weights["gate_proj.oft_R"] = _compact(0.05)
    before = {
        name: layers[0].clone() for name, layers in manager.memory_pool._groups.items()
    }

    with pytest.raises(ValueError, match="up_proj.*canonical_oft"):
        manager._apply_expert_oft_to_module(moe, {0: weights}, 4, 0, 2)

    for name, layers in manager.memory_pool._groups.items():
        torch.testing.assert_close(layers[0], before[name])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

"""Streamed expert compacts retain compute precision in fixed resident slots."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager


def _manager(oft_type, storage_dtype, tp_rank):
    groups = (
        ("w1_oft_r", "w3_oft_r", "w2_oft_r")
        if oft_type == "canonical_oft"
        else ("w13_oft_r", "w2_oft_r")
    )
    pool = object.__new__(OFTMemoryPool)
    pool._groups = {
        name: {0: torch.full((3, 3, 1, 4, 4), -7, dtype=storage_dtype)}
        for name in groups
    }
    for layers in pool._groups.values():
        layers[0][2] = torch.eye(4, dtype=storage_dtype)
    manager = object.__new__(OFTManager)
    manager.oft_type = oft_type
    manager.oft_r_dtype = storage_dtype
    manager.memory_pool = pool
    moe = SimpleNamespace(
        num_local_experts=3,
        moe_tp_rank=tp_rank,
        moe_tp_size=2,
        w13_weight=torch.empty(0, dtype=storage_dtype),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id - 2,
    )
    manager._find_fused_moe_modules = lambda: {0: moe}
    return manager


def _expected_rotation(compact_dtype, storage_dtype):
    # For q=0.024, the five-term Neumann result has diagonal
    # 1-2*q^2+q^4 and off-diagonal 2*q-2*q^3. Computing in FP32 then
    # casting gives a different BF16 off-diagonal from rounding q first.
    if compact_dtype == torch.bfloat16:
        diagonal, off_diagonal = 1.0, 0.048095703125
    elif storage_dtype == torch.bfloat16:
        diagonal, off_diagonal = 1.0, 0.0478515625
    else:
        diagonal, off_diagonal = 0.998848331776, 0.047972352
    return torch.tensor(
        [
            [diagonal, off_diagonal, 0, 0],
            [-off_diagonal, diagonal, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=storage_dtype,
    )


@pytest.mark.parametrize("oft_type", ["oft", "canonical_oft"])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "compact_dtype,storage_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_streamed_experts_compute_before_casting_into_resident_slots(
    oft_type, tp_rank, reverse, compact_dtype, storage_dtype
):
    manager = _manager(oft_type, storage_dtype, tp_rank)
    pool = manager.memory_pool
    pointers = {name: layers[0].data_ptr() for name, layers in pool._groups.items()}
    compact = torch.tensor([[0.024, 0, 0, 0, 0, 0]], dtype=compact_dtype)
    before_compact = compact.clone()
    gate = "w1_oft_r" if oft_type == "canonical_oft" else "w13_oft_r"
    chunks = [("gate_proj", gate, 0), ("down_proj", "w2_oft_r", 0)]
    if oft_type == "canonical_oft":
        chunks.insert(1, ("up_proj", "w3_oft_r", 1))
    if reverse:
        chunks.reverse()
    written = set()
    rotation = _expected_rotation(compact_dtype, storage_dtype)
    identity = torch.eye(4, dtype=storage_dtype)

    for projection, group, local_id in chunks:
        weights = (
            torch.cat([torch.zeros_like(compact), compact])
            if projection == "down_proj"
            else compact
        )
        manager.apply_streamed_expert_oft(
            {0: {local_id + 2: {projection + ".oft_R": weights}}},
            block_size=4,
            slot_idx=2,
        )
        written.add((group, local_id))

        for name, layers in pool._groups.items():
            actual = layers[0]
            assert actual.data_ptr() == pointers[name]
            assert actual.dtype == storage_dtype
            torch.testing.assert_close(actual[:2], torch.full_like(actual[:2], -7))
            for expert_id in range(3):
                changed = (name, expert_id) in written and not (
                    name == "w2_oft_r" and tp_rank == 0
                )
                expected = rotation if changed else identity
                # BF16 must match exactly: its default tolerance would hide
                # rounding the input compact before the Cayley computation.
                tolerances = (
                    {"rtol": 0, "atol": 0}
                    if storage_dtype == torch.bfloat16
                    else {"rtol": 1e-6, "atol": 1e-8}
                )
                torch.testing.assert_close(
                    actual[2, expert_id, 0], expected, **tolerances
                )
        torch.testing.assert_close(compact, before_compact, rtol=0, atol=0)


def test_streamed_experts_reject_resident_dtype_that_violates_pool_contract():
    manager = _manager("oft", torch.float32, tp_rank=0)
    manager.oft_r_dtype = torch.bfloat16
    before = manager.memory_pool.slot("w13_oft_r", 0, 2).clone()
    compact = torch.tensor([[0.024, 0, 0, 0, 0, 0]])

    with pytest.raises(RuntimeError, match="captured.*R buffer"):
        manager.apply_streamed_expert_oft(
            {0: {2: {"gate_proj.oft_R": compact}}}, block_size=4, slot_idx=2
        )

    torch.testing.assert_close(manager.memory_pool.slot("w13_oft_r", 0, 2), before)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

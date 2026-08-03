"""The streamed-OFT chunk limit must bound the scratch that actually allocates.

Observed in a real RL smoke, not constructed: an OFT b1024 arm colocated with a
paused ~20 GB KV-cache arena consumed 7-9 GB of transient GPU memory on EVERY
adapter update, and the arena resume that runs immediately after the update
OOM'd inside torch_memory_saver's cuMemCreate on the fourth rollout (free
memory measured at the resume threshold: 23.7 GB, succeeded once at that value
and failed the next time).

Two defects produced that, and each gets its own pin here:

1. The limit was accounted in COMPACT bytes -- the upper-triangular payload --
   while the allocations it exists to bound are the expanded full blocks and
   cayley_neumann's intermediates, ~10x larger. A "512 MB" chunk expanded to
   ~4-5 GB of scratch.
2. Row-parallel groups were flushed WHOLE, bypassing the limiter entirely --
   and fc2/down_proj, the row-parallel module, carries the most blocks of any
   module in the model, so the biggest group was exactly the one never chunked.

Correctness of the chunked flush is anchored against the unchunked flush: same
groups, same writes, byte-identical rotation matrices per (target, layer).
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.oft.streamed_weight_loader import (
    _CAYLEY_LIVE_FULL_TENSORS,
    _flush_oft_group_in_chunks,
    _oft_working_set_bytes,
)

BLOCK_SIZE = 8


class RecordingPool:
    """Captures every _write_precomputed_oft_r call, in order."""

    def __init__(self):
        self.writes = []

    def _write_precomputed_oft_r(
        self, buffer_id, fused_target, layer_id, packed_r, block_size,
        slice_index=None, split_count=1,
    ):
        self.writes.append(
            (fused_target, layer_id, packed_r.clone(), slice_index, split_count)
        )


def _item(layer_id: int, num_blocks: int = 2, target: str = "qkv_proj", seed: int = 0):
    generator = torch.Generator().manual_seed(seed + layer_id)
    compact = torch.randn(
        num_blocks, BLOCK_SIZE * (BLOCK_SIZE - 1) // 2,
        generator=generator, dtype=torch.float32,
    )
    return (layer_id, target, compact, None, 1)


class TestWorkingSetAccounting:
    def test_working_set_counts_full_blocks_times_live_tensors(self):
        item = _item(0, num_blocks=3)
        expected = 3 * BLOCK_SIZE * BLOCK_SIZE * 4 * _CAYLEY_LIVE_FULL_TENSORS
        assert _oft_working_set_bytes(item[2], BLOCK_SIZE) == expected

    def test_the_working_set_dwarfs_the_compact_bytes(self):
        """The heart of defect (1): if these were close, compact accounting
        would have been fine. The ratio is ~2x from triangle-to-full alone and
        ~10x with the live intermediates, which is why a compact-bytes limit
        under-provisioned by an order of magnitude."""
        item = _item(0, num_blocks=1)
        compact_bytes = item[2].numel() * item[2].element_size()
        assert _oft_working_set_bytes(item[2], BLOCK_SIZE) > 9 * compact_bytes


class TestChunkedFlush:
    def _flush(self, items, limit_bytes):
        pool = RecordingPool()
        _flush_oft_group_in_chunks(
            pool, 0, BLOCK_SIZE, torch.device("cpu"), items, limit_bytes,
        )
        return pool.writes

    def test_a_tight_limit_splits_the_group_without_changing_the_result(self):
        """Chunked and unchunked flushes must write byte-identical rotations:
        the limit is a memory knob, never a numerics knob."""
        items = [_item(layer) for layer in range(6)]
        one_item = _oft_working_set_bytes(items[0][2], BLOCK_SIZE)

        unchunked = self._flush(items, 0)
        chunked = self._flush(items, one_item)  # forces one item per chunk

        assert len(unchunked) == len(chunked) == 6
        for (t_a, l_a, r_a, *_), (t_b, l_b, r_b, *_) in zip(unchunked, chunked):
            assert (t_a, l_a) == (t_b, l_b)
            assert torch.equal(r_a, r_b)

    def test_zero_limit_disables_chunking(self):
        items = [_item(layer) for layer in range(4)]
        writes = self._flush(items, 0)
        # One flush call packs all items into one precompute; per-item writes
        # still happen, so the observable is the ORDER, which must match input.
        assert [layer for _, layer, *_ in writes] == [0, 1, 2, 3]

    def test_row_parallel_items_go_through_the_same_limiter(self):
        """Defect (2) pinned at the call-signature level: the row-parallel loop
        now routes through _flush_oft_group_in_chunks, so a sliced item (a
        row-parallel shard, split_count > 1) must both split under a tight
        limit and carry its slice metadata through to the write."""
        items = [
            (layer, "down_proj", _item(layer)[2], 0, 2) for layer in range(4)
        ]
        one_item = _oft_working_set_bytes(items[0][2], BLOCK_SIZE)
        writes = self._flush(items, one_item)
        assert len(writes) == 4
        for _, _, _, slice_index, split_count in writes:
            assert (slice_index, split_count) == (0, 2)

    def test_the_rotations_are_orthogonal(self):
        """Sanity anchor with a reference that cannot drift: whatever the
        chunking does, precompute_oft_r output must stay a rotation.

        The skew entries are scaled to ~1e-2, the regime the 5-term Neumann
        approximation of the Cayley transform is built for -- and the regime
        production sees, since OFT parameters start at zero (R = I) and move
        by learning-rate-sized steps. At N(0,1) entries the series diverges,
        which is an approximation property, not a chunking defect."""
        layer_id, target, compact, slice_index, split_count = _item(0, num_blocks=2)
        items = [(layer_id, target, compact * 0.01, slice_index, split_count)]
        (_, _, packed_r, *_), = self._flush(items, 0)
        eye = torch.eye(BLOCK_SIZE)
        for block in packed_r:
            product = block @ block.transpose(-1, -2)
            assert torch.allclose(product, eye, atol=1e-3), product

"""GPU regression coverage for Task 4b: the expert-OFT streamed-weight write
path must write into the ADAPTER'S OWN assigned buffer slot (not always
`active_idx`), and every slot's expert-OFT buffers must default to identity
before -- or absent -- real adapter weights.

This drives the REAL production functions on the exact call chain the bug
lived in (`OFTMemoryPool.reset_buffer_slot_to_identity`,
`streamed_weight_loader._resolve_streamed_oft_tensor_groups` /
`_commit_streamed_oft_tensor_groups`, `OFTManager.apply_streamed_expert_oft`)
against a REAL `OFTMemoryPool` and REAL `FusedMoE` layers (constructed
directly, mirroring test_nvfp4_moe_backends.py's single-process-dist
pattern), rather than calling any of them in isolation or against hand-mocked
buffers -- the bug was in how these are WIRED TOGETHER in production
(`_commit_streamed_oft_tensor_groups` never passed `slot_idx`, and
`reset_buffer_slot_to_identity` never touched the expert groups), not in any
one function's own logic. A full server/tokenizer-manager boot is not
needed: the bug and its fix live entirely inside
OFTManager/OFTMemoryPool/streamed_weight_loader, so this drives those
directly at the same fidelity a real native-RPC adapter load
(`load_adapter_from_tensors`) would, without multiprocess Engine overhead --
and it lets the assertions read the pool's slot-indexed storage directly,
which a real multiprocess Engine's separate scheduler process would not
allow from the test process.
"""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.streamed_weight_loader import (
    _commit_streamed_oft_tensor_groups,
    _resolve_streamed_oft_tensor_groups,
)
from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

BLOCK_SIZE = 32
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 64
NUM_EXPERTS = 4
NUM_LAYERS = 2

_PROJ_TO_GROUP = {
    "gate_proj": "w1_oft_r",
    "up_proj": "w3_oft_r",
    "down_proj": "w2_oft_r",
}


class _MoEBlock(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.mlp = moe


class _TinyMoEModel(torch.nn.Module):
    """Minimal real nn.Module wrapping real FusedMoE layers at
    `layers.{layer_id}.mlp`, matching the module-path shape
    `get_layer_id`/`_find_fused_moe_layers` need (a `layers.<N>.` segment in
    the qualified module name). No attention/embedding -- this test targets
    MoE expert OFT only.
    """

    def __init__(self, moe_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(_MoEBlock(m) for m in moe_layers)

    def get_hidden_dim(self, module_name, layer_idx):
        # OFTMemoryPool._declare_groups also registers a DENSE R:{module}
        # group for every target_modules entry, even though gate_proj/
        # up_proj/down_proj are MoE-only in this synthetic model (no dense
        # Linear of that name exists) -- those dense groups get allocated
        # and identity-filled but are never written to by this test's
        # expert-only payloads. Returning a trivial shape here avoids
        # needing real per-architecture hidden-dim knowledge for a group
        # this test never exercises.
        return (HIDDEN_SIZE, HIDDEN_SIZE)


def _make_fused_moe(layer_id: int) -> torch.nn.Module:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    return FusedMoE(
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        layer_id=layer_id,
        top_k=2,
        params_dtype=torch.bfloat16,
    ).cuda()


def _expert_oft_payload(seed: int, layer_ids=range(NUM_LAYERS)):
    """One adapter's compact expert-OFT payload: real checkpoint-name
    tensors (same naming `_partition_expert_oft_tensors` matches), plus the
    expected precomputed R per (layer, expert, proj) -- computed with the
    SAME `precompute_oft_r` helper the production write path itself calls,
    since this test checks WHERE the values land (which buffer slot), not
    the Cayley math (covered elsewhere).
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden_blocks = HIDDEN_SIZE // BLOCK_SIZE
    inter_blocks = INTERMEDIATE_SIZE // BLOCK_SIZE
    n_elem = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2

    named_tensors = {}
    expected = {}
    for layer_id in layer_ids:
        for expert_id in range(NUM_EXPERTS):
            for proj, num_blocks in (
                ("gate_proj", hidden_blocks),
                ("up_proj", hidden_blocks),
                ("down_proj", inter_blocks),
            ):
                compact = (torch.randn((num_blocks, n_elem), generator=gen) * 0.02).to(
                    dtype=torch.bfloat16, device="cuda"
                )
                name = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj}.oft_R"
                named_tensors[name] = compact
                expected[(layer_id, expert_id, proj)] = (
                    precompute_oft_r(compact, BLOCK_SIZE).detach().clone()
                )
    return named_tensors, expected


class TestExpertOFTWritePath(CustomTestCase):
    """Task 4b regression coverage: fix 1 (thread the adapter's real
    buffer_id into apply_streamed_expert_oft instead of always writing
    active_idx) and fix 2 (identity-fill every slot of the expert-OFT groups
    on reset_buffer_slot_to_identity, not just active_idx at boot)."""

    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29653)
        # set_default_device is process-global (not scoped to this
        # TestCase), so it must be restored in tearDownClass -- otherwise it
        # leaks into whatever test file/class runs next in the same pytest
        # process (observed: it broke test_oft_moe_multi_tenancy.py's
        # CPU-tensor-device assertions when both files ran in one session).
        cls._prev_default_device = torch.get_default_device()
        torch.set_default_device("cuda")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device(cls._prev_default_device)

    def _make_pool_and_manager_double(self):
        with get_context().override_server_args(
            model_path="dummy"
        ), get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ):
            moe0, moe1 = _make_fused_moe(0), _make_fused_moe(1)
            model = _TinyMoEModel([moe0, moe1])
            pool = OFTMemoryPool(
                base_hf_config=SimpleNamespace(
                    num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE
                ),
                max_ofts_per_batch=3,
                dtype=torch.bfloat16,
                tp_size=1,
                tp_rank=0,
                max_oft_block_size=BLOCK_SIZE,
                target_modules={"gate_proj", "up_proj", "down_proj"},
                base_model=model,
                eviction_policy="lru",
                oft_added_tokens_size=0,
                oft_type="canonical_oft",
            )

        # Lightweight OFTManager stand-in, following this file's established
        # pattern (test_oft_moe_multi_tenancy.py): bind REAL production
        # methods onto a double carrying a REAL memory_pool, rather than
        # reimplementing their logic.
        tm = SimpleNamespace(
            memory_pool=pool,
            oft_modules=None,
            oft_type="canonical_oft",
            target_modules={"gate_proj", "up_proj", "down_proj"},
            max_oft_block_size=BLOCK_SIZE,
            _find_fused_moe_modules=lambda: {0: moe0, 1: moe1},
        )
        tm.apply_streamed_expert_oft = MethodType(
            OFTManager.apply_streamed_expert_oft, tm
        )
        # Real server boot always binds moe.w1_oft_r/w3_oft_r/w2_oft_r to the
        # pool's active_idx view before any streamed load can happen (see
        # OFTManager._init_identity_expert_oft_for_cuda_graph, called once
        # from init_state at boot) -- reproduce that here so
        # apply_streamed_expert_oft's slot_idx=None legacy branch resolves a
        # real (bound, non-None) module attribute instead of module-attribute
        # None. Without this step, the pre-fix bug (writing to active_idx
        # regardless of buffer_id) would raise the CUDA-graph-buffer-mismatch
        # guard instead of silently landing at the wrong (but real) slot --
        # a real failure either way, but not the actual production symptom.
        tm._init_identity_expert_oft_for_cuda_graph = MethodType(
            OFTManager._init_identity_expert_oft_for_cuda_graph, tm
        )
        tm._init_identity_expert_oft_for_cuda_graph()
        return tm, pool

    def _load_into_slot(self, tm, buffer_id, named_tensors):
        """Mirrors OFTManager.load_adapter_from_tensors's real call order:
        resolve+validate the payload, then (as allocate_buffer_slot_with_
        eviction's caller does on every dynamic load) reset the target slot
        to identity, then commit."""
        items = list(named_tensors.items())
        plan, err = _resolve_streamed_oft_tensor_groups(tm, items, BLOCK_SIZE)
        self.assertIsNotNone(plan, err)
        tm.memory_pool.reset_buffer_slot_to_identity(buffer_id)
        ok, msg = _commit_streamed_oft_tensor_groups(
            tm,
            items,
            plan,
            buffer_id,
            BLOCK_SIZE,
            f"adapter-{buffer_id}",
            f"adapter-{buffer_id}",
        )
        self.assertTrue(ok, msg)

    def _assert_slot_matches(self, pool, buffer_id, expected):
        for (layer_id, expert_id, proj), expected_r in expected.items():
            group = _PROJ_TO_GROUP[proj]
            actual = pool._groups[group][layer_id][buffer_id, expert_id]
            torch.testing.assert_close(
                actual,
                expected_r,
                msg=lambda m, g=group, l=layer_id, e=expert_id: (
                    f"{g} layer={l} expert={e}: {m}"
                ),
            )

    def _assert_slot_is_identity(self, pool, buffer_id, layer_id):
        for group in ("w1_oft_r", "w3_oft_r", "w2_oft_r"):
            tensor = pool._groups[group][layer_id][buffer_id]  # (E, num_blocks, bs, bs)
            eye = torch.eye(BLOCK_SIZE, dtype=tensor.dtype, device=tensor.device)
            self.assertTrue(
                torch.equal(tensor, eye.expand_as(tensor)),
                f"{group} layer={layer_id} slot={buffer_id} is not identity",
            )

    def test_two_adapters_land_in_their_own_slots(self):
        """Fix 1 regression: two adapters dynamically loaded into DIFFERENT
        buffer slots must each read back from THEIR OWN slot. Before the
        fix, `_commit_streamed_oft_tensor_groups` called
        `apply_streamed_expert_oft(fused_expert_chunk, block_size)` with no
        `slot_idx`, so every adapter's real expert-OFT weights landed at
        `active_idx` (slot 0) regardless of the buffer_id it was actually
        assigned -- this test drives buffer_id=1 and buffer_id=2 (both !=
        active_idx=0) and reads the pool's slot-indexed storage directly."""
        tm, pool = self._make_pool_and_manager_double()
        payload_a, expected_a = _expert_oft_payload(seed=1)
        payload_b, expected_b = _expert_oft_payload(seed=2)

        self._load_into_slot(tm, 1, payload_a)
        self._load_into_slot(tm, 2, payload_b)

        self._assert_slot_matches(pool, 1, expected_a)
        self._assert_slot_matches(pool, 2, expected_b)

        # Fixture sanity: the two adapters' real weights must actually
        # differ, or a "both landed at the same slot" bug could still pass
        # a same-value per-slot check by coincidence.
        for key in expected_a:
            self.assertFalse(
                torch.equal(expected_a[key], expected_b[key]),
                f"test fixture bug: adapter A and B produced identical R for {key}",
            )

    def test_slot_starts_identity_and_stays_identity_without_moe_weights(self):
        """Fix 2 regression: reset_buffer_slot_to_identity must identity-fill
        the expert-OFT groups (w1/w3/w2_oft_r) too, not just the dense
        R_buffer/embedding_R_buffer/lm_head_R_buffer groups. Proves: (a) a
        freshly reset slot reads identity before any real weights are
        written (previously: uninitialized torch.empty garbage, since only
        active_idx ever got identity-filled, at boot); (b) loading an
        adapter with no expert-OFT weights for one layer leaves that layer's
        slot buffers identity; (c) a slot reused after a real occupant's
        weights are written, then reset and given a new occupant that
        doesn't touch that layer, does not leak the previous occupant's
        rotation."""
        tm, pool = self._make_pool_and_manager_double()

        # (a) fresh slot, before any write.
        pool.reset_buffer_slot_to_identity(1)
        self._assert_slot_is_identity(pool, 1, layer_id=0)
        self._assert_slot_is_identity(pool, 1, layer_id=1)

        # (b) adapter with expert-OFT weights for layer 0 ONLY -- layer 1
        # must stay identity (no MoE-target weights for that layer).
        payload_layer0_only, expected_layer0_only = _expert_oft_payload(
            seed=3, layer_ids=[0]
        )
        self._load_into_slot(tm, 1, payload_layer0_only)
        self._assert_slot_matches(pool, 1, expected_layer0_only)
        self._assert_slot_is_identity(pool, 1, layer_id=1)

        # (c) evict-and-reuse: slot 1 currently holds a real (non-identity)
        # layer-0 rotation from the adapter above. Reset it (mirroring
        # allocate_buffer_slot_with_eviction -> reset_buffer_slot_to_identity
        # on every dynamic load) and load a DIFFERENT adapter that targets
        # layer 1 ONLY into the SAME slot -- the previous occupant's layer-0
        # rotation must not leak through, and the new occupant's layer-1
        # rotation must land correctly.
        pool.reset_buffer_slot_to_identity(1)
        payload_layer1_only, expected_layer1_only = _expert_oft_payload(
            seed=4, layer_ids=[1]
        )
        self._load_into_slot(tm, 1, payload_layer1_only)
        self._assert_slot_is_identity(pool, 1, layer_id=0)
        self._assert_slot_matches(pool, 1, expected_layer1_only)


if __name__ == "__main__":
    unittest.main()

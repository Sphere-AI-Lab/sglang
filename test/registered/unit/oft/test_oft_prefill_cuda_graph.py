"""OFT prefill CUDA-graph protocol, mirrored from srt/lora: a static
prefill batch info refreshed in place (backend), an eligibility gate shared
by prepare_oft_batch and the runner (manager), and the routing between the
two."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.oft.backend.triton_backend import (
    PREFILL_CUDA_GRAPH_OFT_SEGMENTS,
    TritonOFTBackend,
)
from sglang.srt.oft.mem_pool import OFTMemoryPool
from sglang.srt.oft.oft_manager import OFTManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _extend_batch(extend_seq_lens, oft_ids=None, device="cpu"):
    num_tokens = sum(extend_seq_lens)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(extend_seq_lens),
        input_ids=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=list(extend_seq_lens),
        extend_num_tokens=num_tokens,
        spec_info=None,
        oft_ids=oft_ids if oft_ids is not None else [None] * len(extend_seq_lens),
    )


@unittest.skipUnless(torch.cuda.is_available(), "backend prepare pins host memory")
class TestTritonBackendStaticPrefillBatchInfo(unittest.TestCase):
    def _backend(self):
        # Three addressable slots -> segmented kernels (single_adapter_mode off).
        backend = TritonOFTBackend(
            max_ofts_per_batch=3,
            device=torch.device("cuda"),
            server_args=SimpleNamespace(oft_double_buffer=False),
        )
        backend.init_prefill_cuda_graph_batch_info(max_num_tokens=4096)
        return backend

    def test_init_pins_segment_slots_and_limits(self):
        backend = self._backend()
        info = backend.prefill_cuda_graph_batch_info
        self.assertEqual(info.bs, PREFILL_CUDA_GRAPH_OFT_SEGMENTS)
        self.assertEqual(info.num_segments, PREFILL_CUDA_GRAPH_OFT_SEGMENTS)
        self.assertEqual(info.seg_lens.shape[0], PREFILL_CUDA_GRAPH_OFT_SEGMENTS)
        self.assertEqual(info.seg_indptr.shape[0], PREFILL_CUDA_GRAPH_OFT_SEGMENTS + 1)
        self.assertEqual(
            backend.prefill_cuda_graph_max_bs, PREFILL_CUDA_GRAPH_OFT_SEGMENTS
        )
        self.assertEqual(backend.prefill_cuda_graph_max_tokens, 4096)

    def test_prefill_mode_refreshes_the_static_object_in_place(self):
        backend = self._backend()
        static = backend.prefill_cuda_graph_batch_info
        backend.prepare_oft_batch(
            forward_batch=_extend_batch([4, 1, 3], device="cuda"),
            weight_indices=[1, 2, 1],
            oft_block_sizes=[0, 32, 32],
            use_cuda_graph=False,
            use_prefill_cuda_graph=True,
        )
        torch.cuda.synchronize()
        self.assertIs(backend.batch_info, static)
        n = PREFILL_CUDA_GRAPH_OFT_SEGMENTS
        self.assertEqual(static.num_segments, n)  # pinned for the captured grid
        self.assertEqual(static.max_len, 4)
        self.assertEqual(static.seg_lens[:3].tolist(), [4, 1, 3])
        self.assertEqual(static.seg_lens[3:].tolist(), [0] * (n - 3))
        self.assertEqual(static.seg_indptr[:4].tolist(), [0, 4, 5, 8])
        self.assertEqual(static.seg_indptr[4:].tolist(), [8] * (n - 3))
        self.assertEqual(static.weight_indices[:3].tolist(), [1, 2, 1])
        self.assertEqual(static.weight_indices[3:].tolist(), [0] * (n - 3))

        # A smaller batch reuses the same object and clears the old tail.
        backend.prepare_oft_batch(
            forward_batch=_extend_batch([5], device="cuda"),
            weight_indices=[2],
            oft_block_sizes=[0, 32, 32],
            use_cuda_graph=False,
            use_prefill_cuda_graph=True,
        )
        torch.cuda.synchronize()
        self.assertIs(backend.batch_info, static)
        self.assertEqual(static.seg_lens[:2].tolist(), [5, 0])
        self.assertEqual(static.seg_indptr[1:].tolist(), [5] * n)
        self.assertEqual(static.weight_indices[:2].tolist(), [2, 0])

    def test_eager_mode_keeps_building_fresh_batch_info(self):
        backend = self._backend()
        backend.prepare_oft_batch(
            forward_batch=_extend_batch([4, 1, 3], device="cuda"),
            weight_indices=[1, 2, 1],
            oft_block_sizes=[0, 32, 32],
            use_cuda_graph=False,
            use_prefill_cuda_graph=False,
        )
        self.assertIsNot(backend.batch_info, backend.prefill_cuda_graph_batch_info)
        self.assertEqual(backend.batch_info.num_segments, 3)

    def test_base_backend_default_is_unsupported(self):
        from sglang.srt.oft.backend.base_backend import BaseOFTBackend

        base = BaseOFTBackend(max_ofts_per_batch=2, device=torch.device("cpu"))
        self.assertFalse(base.supports_prefill_cuda_graph)
        self.assertIsNone(base.prefill_cuda_graph_max_bs)
        with self.assertRaises(NotImplementedError):
            base.init_prefill_cuda_graph_batch_info(max_num_tokens=16)


class TestManagerEligibilityGate(unittest.TestCase):
    def _tm(self, *, initialized=True, single_adapter_mode=True, dp_attention=False):
        tm = SimpleNamespace()
        tm.enable_dp_attention = dp_attention
        tm.oft_backend = SimpleNamespace(
            prefill_cuda_graph_max_bs=32 if initialized else None,
            prefill_cuda_graph_max_tokens=4096 if initialized else None,
            single_adapter_mode=single_adapter_mode,
        )
        tm.can_use_prefill_cuda_graph = MethodType(
            OFTManager.can_use_prefill_cuda_graph, tm
        )
        return tm

    def test_uniform_extend_batch_is_eligible(self):
        tm = self._tm()
        self.assertTrue(
            tm.can_use_prefill_cuda_graph(_extend_batch([4, 3], ["a", "a"]))
        )
        self.assertTrue(
            tm.can_use_prefill_cuda_graph(_extend_batch([4, 3], [None, None]))
        )

    def test_not_initialized_or_dp_attention_is_ineligible(self):
        self.assertFalse(
            self._tm(initialized=False).can_use_prefill_cuda_graph(_extend_batch([4]))
        )
        self.assertFalse(
            self._tm(dp_attention=True).can_use_prefill_cuda_graph(_extend_batch([4]))
        )

    def test_decode_and_target_verify_belong_to_the_decode_graph(self):
        tm = self._tm()
        fb = _extend_batch([1, 1])
        fb.forward_mode = ForwardMode.DECODE
        self.assertFalse(tm.can_use_prefill_cuda_graph(fb))
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        self.assertFalse(tm.can_use_prefill_cuda_graph(fb))

    def test_mixed_batch_is_eager_on_the_single_adapter_fast_path_only(self):
        mixed = _extend_batch([4, 3], ["a", None])
        self.assertFalse(
            self._tm(single_adapter_mode=True).can_use_prefill_cuda_graph(mixed)
        )
        self.assertTrue(
            self._tm(single_adapter_mode=False).can_use_prefill_cuda_graph(mixed)
        )

    def test_limits_and_missing_fields(self):
        tm = self._tm()
        self.assertFalse(tm.can_use_prefill_cuda_graph(_extend_batch([1] * 33)))
        self.assertFalse(tm.can_use_prefill_cuda_graph(_extend_batch([4097])))
        fb = _extend_batch([4])
        fb.oft_ids = None
        self.assertFalse(tm.can_use_prefill_cuda_graph(fb))
        fb = _extend_batch([4])
        fb.extend_num_tokens = None
        self.assertFalse(tm.can_use_prefill_cuda_graph(fb))


class TestManagerSupportAndRouting(unittest.TestCase):
    def _support_tm(self, *, backend_supports=True, dp_attention=False, expert_oft=False):
        tm = SimpleNamespace()
        tm.enable_dp_attention = dp_attention
        tm.oft_backend = SimpleNamespace(supports_prefill_cuda_graph=backend_supports)
        tm.memory_pool = SimpleNamespace(has_expert_oft_groups=lambda: expert_oft)
        tm._has_moe_expert_oft_buffers = MethodType(
            OFTManager._has_moe_expert_oft_buffers, tm
        )
        return OFTManager.supports_prefill_cuda_graph.fget(tm)

    def test_supports_dense_and_attention_only_moe(self):
        # Attention-only MoE OFT declares no expert groups in the pool.
        self.assertTrue(self._support_tm(expert_oft=False))

    def test_excludes_backend_without_support_dp_attention_and_expert_oft(self):
        self.assertFalse(self._support_tm(backend_supports=False))
        self.assertFalse(self._support_tm(dp_attention=True))
        self.assertFalse(self._support_tm(expert_oft=True))

    def test_pool_reports_declared_expert_groups(self):
        pool = OFTMemoryPool.__new__(OFTMemoryPool)
        pool._groups = {"R:q_proj": {}}
        self.assertFalse(pool.has_expert_oft_groups())
        pool._groups["w2_oft_r"] = {}
        self.assertTrue(pool.has_expert_oft_groups())

    def _routing_tm(self, eligible):
        calls = []
        tm = SimpleNamespace()
        tm.max_ofts_per_batch = 2
        tm.max_bs_in_cuda_graph = 8
        tm._moe_cg_slot_ids_buffer = None
        slots = {None: 0, "a": 1}
        tm.memory_pool = SimpleNamespace(
            active_idx=1, uid_to_buffer_id=slots, get_buffer_id=lambda uid: slots[uid]
        )
        tm.adapters = {}
        tm.configs = {"a": SimpleNamespace(block_size=32)}
        tm._find_fused_moe_modules = lambda: {}
        tm.can_use_prefill_cuda_graph = lambda forward_batch: eligible
        tm.oft_backend = SimpleNamespace(
            prepare_oft_batch=lambda **kw: calls.append(
                (kw["use_cuda_graph"], kw["use_prefill_cuda_graph"])
            )
        )
        tm.prepare_oft_batch = MethodType(OFTManager.prepare_oft_batch, tm)
        return tm, calls

    def test_extend_batch_routes_to_the_prefill_graph_when_eligible(self):
        tm, calls = self._routing_tm(eligible=True)
        tm.prepare_oft_batch(_extend_batch([4, 3], ["a", "a"]))
        self.assertEqual(calls, [(False, True)])

    def test_extend_batch_stays_eager_when_ineligible(self):
        tm, calls = self._routing_tm(eligible=False)
        tm.prepare_oft_batch(_extend_batch([4, 3], ["a", None]))
        self.assertEqual(calls, [(False, False)])

    def test_decode_graph_batch_never_uses_the_prefill_path(self):
        tm, calls = self._routing_tm(eligible=True)
        fb = _extend_batch([1, 1], ["a", "a"])
        fb.forward_mode = ForwardMode.DECODE
        tm.prepare_oft_batch(fb)
        self.assertEqual(calls, [(True, False)])

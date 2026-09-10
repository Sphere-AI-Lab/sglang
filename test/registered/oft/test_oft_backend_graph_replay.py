"""Replay one OFT graph across base, uniform-adapter and mixed routing."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.oft.backend.triton_backend import TritonOFTBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")


class TestOFTBackendCudaGraph(CustomTestCase):
    def test_moe_graph_replays_changed_base_slot(self):
        backend = TritonOFTBackend(2, torch.device("cuda"))
        backend.is_moe_oft = True
        backend._moe_num_experts = 2
        backend._moe_top_k = 1
        backend.init_cuda_graph_batch_info(4, 1)
        batch = SimpleNamespace(batch_size=4, forward_mode=ForwardMode.DECODE)
        backend.prepare_oft_batch(batch, [0, 1, 0, 1], [0, 4], use_cuda_graph=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            enabled = backend.batch_info.moe_oft_info.adapter_enabled.clone()
        # Base/A -> B/A -> B/base -> base-only; the graph keeps its addresses.
        for indices, sizes, expected in (
            ([0, 1, 0, 1], [0, 4], [0, 1]),
            ([0, 1, 0, 1], [4, 4], [1, 1]),
            ([0, 1, 0, 1], [4, 0], [1, 0]),
            ([1, 1, 1, 1], [4, 0], [0, 0]),
            ([0, 0, 0, 0], [4, 0], [1, 0]),
        ):
            backend.prepare_oft_batch(batch, indices, sizes, use_cuda_graph=True)
            graph.replay()
            self.assertEqual(enabled.tolist(), expected)
            self.assertEqual(backend.batch_info.has_active_oft, bool(any(expected)))

    def test_uniform_capture_replays_mixed_and_changed_adapter_weights(self):
        device = torch.device("cuda")
        batch = SimpleNamespace(batch_size=4, forward_mode=ForwardMode.DECODE)
        for slices, method in (
            (1, "run_oft_r_sgemm"),
            (2, "run_gate_up_oft"),
            (3, "run_qkv_oft"),
        ):
            with self.subTest(slices=slices):
                backend = TritonOFTBackend(2, device)
                backend.init_cuda_graph_batch_info(4, 1)
                inputs = torch.arange(64, device=device, dtype=torch.float16).reshape(
                    4, 16
                )
                weights = torch.eye(16, device=device, dtype=torch.float16).repeat(
                    2, slices, 1, 1
                )
                weights[1].neg_()
                backend.prepare_oft_batch(batch, [0] * 4, [0, 16], use_cuda_graph=True)
                operation = getattr(backend, method)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        operation(inputs, weights)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = operation(inputs, weights)
                for indices in ([0, 1, 0, 1], [1] * 4, [0] * 4):
                    backend.prepare_oft_batch(
                        batch, indices, [0, 16], use_cuda_graph=True
                    )
                    graph.replay()
                    expected = inputs.repeat(1, slices)
                    mask = torch.tensor(indices, device=device, dtype=torch.bool)
                    expected[mask] = -inputs.repeat(1, slices)[mask]
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                weights[1].copy_(
                    torch.eye(16, device=device, dtype=torch.float16).repeat(
                        slices, 1, 1
                    )
                )
                backend.prepare_oft_batch(batch, [1] * 4, [0, 16], use_cuda_graph=True)
                graph.replay()
                torch.testing.assert_close(
                    output, inputs.repeat(1, slices), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()

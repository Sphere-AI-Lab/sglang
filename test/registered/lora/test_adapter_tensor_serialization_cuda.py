"""Mixed CPU/CUDA adapter payloads survive independent rank consumption."""

import gc
import multiprocessing as mp
import traceback
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")

_CYCLES = 3
_RANKS = 2


def _check_received(tensors, cycle, rank):
    cpu = torch.arange(24, dtype=torch.float32) + 100 * cycle
    expected = {
        "cpu_base": cpu,
        "cpu_matrix": cpu.reshape(4, 6).t(),
        "cpu_offset": cpu[3:18:2],
        "cpu_bfloat16": torch.tensor([cycle, -2, 3], dtype=torch.bfloat16),
        "cpu_int64": torch.tensor([cycle, 2**40], dtype=torch.int64),
        "cpu_empty": torch.empty(0, dtype=torch.bfloat16),
    }
    for name, value in expected.items():
        torch.testing.assert_close(tensors[name], value, rtol=0, atol=0)
        assert tensors[name].stride() == value.stride(), name
        assert tensors[name].storage_offset() == value.storage_offset(), name

    # Mutating one received view must update its aliases, without modifying
    # either the producer's CPU weights or the other rank's snapshot.
    replacement = -1000 - rank
    tensors["cpu_base"][3] = replacement
    assert tensors["cpu_offset"][0].item() == replacement
    assert tensors["cpu_matrix"][3, 0].item() == replacement

    cuda = torch.arange(32, dtype=torch.float32) + 1000 * cycle
    for name, value in (
        ("cuda_base", cuda),
        ("cuda_matrix", cuda.reshape(4, 8).t()),
        ("cuda_offset", cuda[3:19:2]),
    ):
        assert tensors[name].is_cuda, name
        torch.testing.assert_close(tensors[name].cpu(), value, rtol=0, atol=0)
        assert tensors[name].stride() == value.stride(), name
        assert tensors[name].storage_offset() == value.storage_offset(), name


def _consume_rank(connection, rank):
    try:
        torch.cuda.set_device(0)
        monkey_patch_torch_reductions()
        for cycle in range(_CYCLES):
            # Each process receives only the payload serialized for its rank.
            tensors = MultiprocessingSerializer.deserialize(connection.recv())
            _check_received(tensors, cycle, rank)
            torch.cuda.synchronize()
            del tensors
            gc.collect()
            connection.send((cycle, None))
    except BaseException:
        connection.send((None, traceback.format_exc()))
        raise
    finally:
        connection.close()


class TestAdapterTensorSerializationCuda(CustomTestCase):
    def test_mixed_payloads_across_independent_consumers(self):
        from sglang.srt.entrypoints.engine import Engine

        self.assertTrue(torch.cuda.is_available())
        torch.cuda.set_device(0)
        owner = SimpleNamespace(server_args=SimpleNamespace(tp_size=_RANKS))
        context = mp.get_context("spawn")
        consumers = []
        tensors = None
        cuda_base = torch.empty(32, dtype=torch.float32, device="cuda")
        try:
            for rank in range(_RANKS):
                producer, consumer = context.Pipe()
                process = context.Process(target=_consume_rank, args=(consumer, rank))
                process.start()
                consumer.close()
                consumers.append((process, producer))

            for cycle in range(_CYCLES):
                cpu = torch.arange(24, dtype=torch.float32) + 100 * cycle
                # Reuse the producer allocation to exercise CUDA IPC cache
                # and reference-count handling on subsequent exports.
                cuda_base.copy_(
                    torch.arange(32, dtype=torch.float32, device="cuda") + 1000 * cycle
                )
                torch.cuda.synchronize()
                tensors = {
                    "cpu_base": cpu,
                    "cpu_matrix": cpu.reshape(4, 6).t(),
                    "cpu_offset": cpu[3:18:2],
                    "cpu_bfloat16": torch.tensor([cycle, -2, 3], dtype=torch.bfloat16),
                    "cpu_int64": torch.tensor([cycle, 2**40], dtype=torch.int64),
                    "cpu_empty": torch.empty(0, dtype=torch.bfloat16),
                    "cuda_base": cuda_base,
                    "cuda_matrix": cuda_base.reshape(4, 8).t(),
                    "cuda_offset": cuda_base[3:19:2],
                }
                payloads = Engine._serialize_tensors_per_rank(owner, tensors, None)
                self.assertEqual(len(payloads), _RANKS)
                for (_, connection), payload in zip(consumers, payloads):
                    connection.send(payload)
                for rank, (_, connection) in enumerate(consumers):
                    self.assertTrue(
                        connection.poll(90), f"rank {rank} timed out in cycle {cycle}"
                    )
                    completed_cycle, error = connection.recv()
                    self.assertIsNone(error, f"rank {rank}: {error}")
                    self.assertEqual(completed_cycle, cycle)
                self.assertEqual(cpu[3].item(), 3 + 100 * cycle)

            for rank, (process, _) in enumerate(consumers):
                process.join(timeout=15)
                self.assertEqual(
                    process.exitcode, 0, f"rank {rank} did not exit cleanly"
                )
        finally:
            # The producer must outlive all consumers of its CUDA IPC storage,
            # including when a consumer fails an assertion or stops responding.
            for process, connection in consumers:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                connection.close()
            del tensors, cuda_base
            gc.collect()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    unittest.main()

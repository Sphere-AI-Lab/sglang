"""Adapter CPU payloads must not consume one shared-memory FD per tensor."""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()
with patch.dict(sys.modules, {"megatron": None}):
    from sglang.srt.entrypoints.engine import Engine

from sglang.srt.utils import MultiprocessingSerializer


def _serialize(tensors, ranks=4):
    owner = SimpleNamespace(server_args=SimpleNamespace(tp_size=ranks))
    return Engine._serialize_tensors_per_rank(owner, tensors, None)


def _descriptor_probe(reject):
    import resource
    from multiprocessing import resource_sharer

    initial_fds = len(os.listdir("/proc/self/fd"))
    # Importing Engine can already initialize CUDA on a GPU test host. CPU
    # serialization itself must not change that state.
    initial_cuda_initialized = torch.cuda.is_initialized()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(512, hard), hard))
    # 400 storages fit as values but exceed this limit as shared handles.
    for _ in range(3):
        tensors = {str(index): torch.tensor([index]) for index in range(400)}
        if reject:
            tensors["invalid"] = lambda: None
            with pytest.raises((AttributeError, TypeError)):
                _serialize(tensors)
        else:
            payloads = _serialize(tensors)
            for payload in payloads:
                values = MultiprocessingSerializer.deserialize(payload)
                assert values["399"].item() == 399
                del values
            del payload, payloads
        del tensors
        gc.collect()
        assert not resource_sharer._resource_sharer._cache
        assert len(os.listdir("/proc/self/fd")) <= initial_fds + 4
    assert torch.cuda.is_initialized() == initial_cuda_initialized
    print(json.dumps({"cycles": 3, "rejected": reject}))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux FD accounting")
@pytest.mark.parametrize("reject", [False, True], ids=["roundtrip", "rejected-payload"])
def test_cpu_payloads_do_not_exhaust_or_leak_descriptors(reject):
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--fd-probe", str(int(reject))],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["cycles"] == 3


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool]
)
def test_cpu_payload_preserves_values_dtype_and_empty_tensors(dtype):
    source = {
        "values": torch.tensor([0, 1, 1, 0], dtype=dtype),
        "empty": torch.empty(0, dtype=dtype),
    }
    for payload in _serialize(source, ranks=2):
        restored = MultiprocessingSerializer.deserialize(payload)
        for key in source:
            assert restored[key].dtype == dtype
            assert torch.equal(restored[key], source[key])


def test_cpu_payload_preserves_shared_storage_views_and_strides():
    base = torch.arange(24, dtype=torch.float32)
    tensors = {"matrix": base.reshape(4, 6).t(), "offset": base[3:18:2], "same": base}
    payloads = _serialize(tensors, ranks=2)
    for payload in payloads:
        restored = MultiprocessingSerializer.deserialize(payload)
        for key in tensors:
            assert torch.equal(restored[key], tensors[key])
            assert restored[key].stride() == tensors[key].stride()
            assert restored[key].storage_offset() == tensors[key].storage_offset()
        restored["same"][3] = 123
        assert restored["offset"][0].item() == 123
        assert restored["matrix"][3, 0].item() == 123
        # Serialized CPU weights are snapshots, not mutable producer storage.
        assert base[3].item() == 3


def test_adapter_cpu_serialization_does_not_change_default_sharing_policy():
    from multiprocessing.reduction import ForkingPickler

    from torch import multiprocessing

    before = ForkingPickler._extra_reducers.copy()
    strategy = multiprocessing.get_sharing_strategy()
    tensors = {"weight": torch.ones(2)}
    payloads = _serialize(tensors, ranks=2)
    for payload in payloads:
        MultiprocessingSerializer.deserialize(payload)
    assert ForkingPickler._extra_reducers == before
    assert multiprocessing.get_sharing_strategy() == strategy


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fd-probe"]:
        try:
            _descriptor_probe(bool(int(sys.argv[2])))
        except BaseException:
            import traceback

            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
    else:
        args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
        raise SystemExit(pytest.main([__file__, *args]))

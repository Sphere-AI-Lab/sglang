"""Host dispatch contracts; actual CUDA capture is covered by the GPU test."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _method(filename, class_name, name, **globals_):
    path = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/oft/backend" / filename
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = dict(globals_)
    exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
    return namespace[name]


class _ReachedSegmentedMetadata(Exception):
    pass


def _segmented_metadata(*args, **kwargs):
    # Stop at the real metadata boundary: this test proves host dispatch only.
    raise _ReachedSegmentedMetadata


@pytest.mark.parametrize("indices", ([0, 0], [1, 1], [0, 1]))
def test_triton_graph_always_prepares_segmented_metadata(indices, monkeypatch):
    monkeypatch.delenv("SGLANG_OFT_PREPARE_TRACE", raising=False)
    prepare = _method(
        "triton_backend.py",
        "TritonOFTBackend",
        "prepare_oft_batch",
        generate_sequence_lengths=_segmented_metadata,
    )
    backend = SimpleNamespace(
        single_adapter_mode=True,
        is_moe_oft=False,
        _use_single_adapter_fast_path=True,
        _single_adapter_idx=indices[0],
        _single_block_size_val=0 if indices[0] == 0 else 16,
    )
    with pytest.raises(_ReachedSegmentedMetadata):
        prepare(backend, SimpleNamespace(), indices, [0, 16], use_cuda_graph=True)
    assert backend._use_single_adapter_fast_path is False


def test_triton_uniform_eager_batch_keeps_fast_path(monkeypatch):
    monkeypatch.delenv("SGLANG_OFT_PREPARE_TRACE", raising=False)
    prepare = _method(
        "triton_backend.py",
        "TritonOFTBackend",
        "prepare_oft_batch",
        generate_sequence_lengths=_segmented_metadata,
    )
    backend = SimpleNamespace(
        single_adapter_mode=True,
        is_moe_oft=False,
        _use_single_adapter_fast_path=False,
        _single_adapter_idx=1,
        _single_block_size_val=16,
    )
    prepare(backend, SimpleNamespace(), [1, 1], [0, 16], use_cuda_graph=False)
    assert backend._use_single_adapter_fast_path is True


def test_torch_native_rejects_cuda_graph_before_capture():
    def allocate(*args, **kwargs):
        pytest.fail("torch_native accepted graph initialization and allocated metadata")

    initialize = _method(
        "torch_backend.py",
        "TorchNativeOFTBackend",
        "init_cuda_graph_batch_info",
        OFTBatchInfo=SimpleNamespace,
        torch=SimpleNamespace(full=allocate, int32=object()),
    )
    with pytest.raises(ValueError, match="torch_native.*--disable-cuda-graph"):
        initialize(SimpleNamespace(max_ofts_per_batch=2), 2, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

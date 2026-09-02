import asyncio
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.oft.io_types import (
    LoadOFTAdapterFromDistributedReqInput,
    LoadOFTAdapterFromTensorsReqInput,
    OFTUpdateOutput,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _TokenizerRecorder:
    def __init__(self, result=None):
        self.result = result or OFTUpdateOutput(success=True)
        self.tensor_request = None
        self.distributed_request = None

    async def load_oft_adapter_from_tensors(self, request, http_request):
        self.tensor_request = request
        return self.result

    async def load_oft_adapter_from_distributed(self, request, http_request):
        self.distributed_request = request
        return self.result


def _engine(recorder):
    engine = Engine.__new__(Engine)
    engine.loop = asyncio.new_event_loop()
    engine.tokenizer_manager = recorder
    engine._serialize_tensors_per_rank = lambda tensors, load_format: [b"rank-0"]
    return engine


def test_engine_tensor_load_preserves_native_admission_options():
    recorder = _TokenizerRecorder()
    engine = _engine(recorder)
    try:
        result = engine.load_oft_adapter_from_tensors(
            "adapter-a",
            {"weight": "tensor"},
            {"oft_block_size": 4},
            load_format="flattened_bucket",
            pinned=True,
            upsert=True,
        )
    finally:
        engine.loop.close()

    assert result.success
    request = recorder.tensor_request
    assert isinstance(request, LoadOFTAdapterFromTensorsReqInput)
    assert request.adapter_name == "adapter-a"
    assert request.serialized_named_tensors == [b"rank-0"]
    assert request.load_format == "flattened_bucket"
    assert request.pinned is True
    assert request.upsert is True


def test_engine_distributed_load_preserves_native_admission_options():
    recorder = _TokenizerRecorder()
    engine = _engine(recorder)
    try:
        result = engine.load_oft_adapter_from_distributed(
            "adapter-b",
            {"oft_block_size": 8},
            names=["layer.oft_R"],
            dtypes=["float16"],
            shapes=[[2, 2]],
            group_name="adapter-group",
            pinned=True,
            upsert=True,
        )
    finally:
        engine.loop.close()

    assert result.success
    request = recorder.distributed_request
    assert isinstance(request, LoadOFTAdapterFromDistributedReqInput)
    assert request.adapter_name == "adapter-b"
    assert request.names == ["layer.oft_R"]
    assert request.dtypes == ["float16"]
    assert request.shapes == [[2, 2]]
    assert request.group_name == "adapter-group"
    assert request.pinned is True
    assert request.upsert is True


def test_http_server_registers_native_oft_routes():
    route_methods = {
        route.path: set(route.methods or [])
        for route in http_server.app.routes
        if hasattr(route, "methods")
    }

    assert route_methods["/load_oft_adapter_from_tensors"] == {"POST"}
    assert route_methods["/load_oft_adapter_from_distributed"] == {"POST"}
    assert route_methods["/unload_oft_adapter"] == {"POST"}


def test_http_tensor_load_maps_backend_failure_to_bad_request():
    recorder = _TokenizerRecorder(
        OFTUpdateOutput(success=False, error_message="rejected")
    )
    previous_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=recorder))
    request = LoadOFTAdapterFromTensorsReqInput(
        adapter_name="adapter-a",
        config_dict={"oft_block_size": 4},
        serialized_named_tensors=[],
    )
    try:
        response = asyncio.run(
            http_server.load_oft_adapter_from_tensors(request, None)
        )
    finally:
        http_server.set_global_state(previous_state)

    assert response.status_code == 400
    assert recorder.tensor_request is request

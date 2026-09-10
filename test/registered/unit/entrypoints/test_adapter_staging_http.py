"""HTTP error mapping for the shared native adapter staging endpoints."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    UpdateAdapterFromDistributedReqInput,
)


class _RejectingTokenizerManager:
    async def update_adapter_from_distributed(self, obj, request):
        raise ValueError("stale adapter version")

    async def activate_adapter_version(self, obj, request):
        raise ValueError("staged adapter identity mismatch")


def _call_with_rejecting_manager(handler, request):
    prior_state = http_server.get_global_state()
    http_server.set_global_state(
        SimpleNamespace(tokenizer_manager=_RejectingTokenizerManager())
    )
    try:
        return asyncio.run(handler(request, None))
    finally:
        http_server._global_state = prior_state


def test_stage_validation_error_returns_bad_request_for_lora_and_oft():
    for load_format in ("lora_adapter", "oft_adapter"):
        request = UpdateAdapterFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            load_format=load_format,
            adapter_name="policy",
            adapter_version="2",
            double_buffer=True,
        )

        response = _call_with_rejecting_manager(
            http_server.update_adapter_from_distributed, request
        )

        assert response.status_code == 400
        assert json.loads(response.body) == {
            "success": False,
            "message": "stale adapter version",
            "error": "stale adapter version",
        }


def test_activation_validation_error_returns_bad_request_for_lora_and_oft():
    for load_format in ("lora_adapter", "oft_adapter"):
        request = ActivateAdapterVersionReqInput(
            adapter_name="policy",
            adapter_version="2",
            load_format=load_format,
        )

        response = _call_with_rejecting_manager(
            http_server.activate_adapter_version, request
        )

        assert response.status_code == 400
        assert json.loads(response.body) == {
            "success": False,
            "message": "staged adapter identity mismatch",
            "error": "staged adapter identity mismatch",
        }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

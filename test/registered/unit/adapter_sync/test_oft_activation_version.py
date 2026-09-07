import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    ActivateAdapterVersionReqOutput,
    UpdateAdapterFromDistributedReqInput,
    UpdateAdapterFromDistributedReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin


@pytest.mark.parametrize("success", [True, False])
def test_sibling_version_is_published_at_activation_only(success):
    async def check():
        tm = TokenizerControlMixin()
        tm.server_args = SimpleNamespace(dp_size=1, enable_dp_attention=False)
        tm.auto_create_handle_loop = lambda: None
        tm._staging_backend_for = lambda obj: None
        tm.register_oft_ref = AsyncMock()
        tm.bump_oft_version = AsyncMock(return_value="")
        tm.oft_registry = object()
        tm.oft_ref_cache = {"orbit_oft": SimpleNamespace(oft_id="adapter-id")}
        tm.is_pause_cond = asyncio.Condition()
        tm.is_pause = True
        tm.update_adapter_from_distributed_communicator = AsyncMock(
            return_value=[
                UpdateAdapterFromDistributedReqOutput(success=True, message="staged")
            ]
        )
        tm.activate_adapter_version_communicator = AsyncMock(
            return_value=[
                ActivateAdapterVersionReqOutput(success=success, message="activation")
            ]
        )
        stage = UpdateAdapterFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            load_format="oft_adapter",
            adapter_name="orbit_oft",
            adapter_version="7",
            double_buffer=True,
        )
        await tm.update_adapter_from_distributed(stage)
        tm.bump_oft_version.assert_not_awaited()
        activate = ActivateAdapterVersionReqInput(
            adapter_name="orbit_oft", adapter_version="7", load_format="oft_adapter"
        )
        await tm.activate_adapter_version(activate)
        if success:
            tm.bump_oft_version.assert_awaited_once_with(activate, True)
            assert activate.adapter_id == "adapter-id"
        else:
            tm.bump_oft_version.assert_not_awaited()

    asyncio.run(check())

from threading import Event
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.io_struct import UpdateAdapterFromDistributedReqInput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream staging test")
@pytest.mark.parametrize("success", [True, False])
def test_oft_stage_returns_to_scheduler_before_receive_finishes(success):
    started, release = Event(), Event()

    def receive(req):
        started.set()
        assert release.wait(5)
        return success, "staged" if success else "receive failed"

    scheduler = SimpleNamespace(
        device="cuda",
        gpu_id=torch.cuda.current_device(),
        forward_ct=10,
        server_args=SimpleNamespace(oft_impl="sibling"),
    )
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_adapter_from_distributed=receive),
        draft_worker=None,
        tp_cpu_group=None,
        memory_saver_adapter=None,
        flush_cache=lambda **kw: True,
        is_fully_idle=lambda **kw: False,
        scheduler=scheduler,
    )
    req = UpdateAdapterFromDistributedReqInput(
        names=[],
        dtypes=[],
        shapes=[],
        load_format="oft_adapter",
        adapter_name="orbit_oft",
        adapter_version="7",
        double_buffer=True,
    )
    try:
        response = manager.update_adapter_from_distributed(req)
        assert (
            response is None
        ), "staging must defer its response instead of blocking the scheduler"
        assert started.wait(5)
        assert manager.poll_adapter_stage() is None
        scheduler.forward_ct += 3
    finally:
        release.set()
    manager._async_adapter_stage.future.result(timeout=5)
    original_req, response = manager.poll_adapter_stage()
    assert original_req is req
    assert response.success is success
    assert response.staged_adapter_version == ("7" if success else None)
    assert manager.poll_adapter_stage() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream staging test")
def test_stage_reports_peer_failure_before_acknowledging(monkeypatch):
    from sglang.srt.managers.scheduler_components import async_adapter_stage as staging
    from sglang.srt.managers.io_struct import UpdateAdapterFromDistributedReqOutput

    stage = staging.AsyncAdapterStage(torch.cuda.current_device())
    stage.completion_group = object()
    monkeypatch.setattr(staging.dist, "get_world_size", lambda group: 2)

    def gather(outcomes, local, *, group):
        outcomes[:] = [local, (False, "TP1 stage failed")]

    monkeypatch.setattr(staging.dist, "all_gather_object", gather)
    req = UpdateAdapterFromDistributedReqInput(
        names=[],
        dtypes=[],
        shapes=[],
        adapter_version="7",
        double_buffer=True,
    )
    stage.submit(
        req,
        lambda req: UpdateAdapterFromDistributedReqOutput(
            success=True, message="local ready", staged_adapter_version="7"
        ),
        0,
    )
    stage.future.result(timeout=5)
    _, response = stage.poll()
    assert not response.success
    assert response.staged_adapter_version is None
    assert "TP1 stage failed" in response.message

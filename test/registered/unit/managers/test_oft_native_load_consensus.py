"""Native load handlers must not hide non-sender rank failures."""

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = _module(ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclass
class Output:
    success: bool
    error_message: str = ""
    loaded_adapters: object = None
    previous_adapter_preserved: bool = False
    inconsistent_update: bool = False


class Distributed:
    def __init__(self, replies):
        self.replies = replies
        self.seen = []

    def get_world_size(self, *, group):
        return 2

    def all_gather_object(self, outputs, local, *, group):
        self.seen.append((group, local))
        outputs[:] = [local, self.replies[group]]


def _scheduler(distributed, local, installed=None, groups=("tp",)):
    consensus = _module(SRT / "managers/scheduler_components/tp_update_consensus.py")
    scope = dict(
        torch=NS(distributed=distributed),
        OFTUpdateOutput=Output,
        run_tp_oft_load=getattr(consensus, "run_tp_oft_load", None),
    )
    source = ast.parse((SRT / "managers/scheduler.py").read_text())
    cls = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if getattr(n, "name", "")
        in {
            "_load_oft_adapter_with_consensus",
            "load_oft_adapter",
            "load_oft_adapter_from_tensors",
            "load_oft_adapter_from_distributed",
        }
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), "scheduler.py", "exec"), scope)
    scheduler = scope["Scheduler"]()

    def load(_):
        if isinstance(local, Exception):
            raise local
        return local

    scheduler.tp_worker = NS(
        load_oft_adapter=load,
        load_oft_adapter_from_tensors=load,
        load_oft_adapter_from_distributed=load,
        model_runner=NS(
            oft_manager=NS(
                refs={
                    "uid": installed
                    or NS(adapter_name="A", adapter_id="uid", adapter_version=2)
                }
            )
        ),
    )
    scheduler._adapter_unload_cpu_groups = lambda: groups
    return scheduler


def _request():
    ref = NS(adapter_name="A", adapter_id="uid", adapter_version=2)
    return NS(adapter_name="A", adapter_id="uid", to_ref=lambda: ref)


@pytest.mark.parametrize(
    "method",
    [
        "load_oft_adapter",
        "load_oft_adapter_from_tensors",
        "load_oft_adapter_from_distributed",
    ],
)
def test_non_sender_failure_is_reported_and_marks_partial_update(method):
    distributed = Distributed({"tp": (False, "bad payload", True, False)})
    scheduler = _scheduler(distributed, Output(success=True))
    result = getattr(scheduler, method)(_request())
    assert not result.success
    assert "TP rank 1" in result.error_message
    assert result.inconsistent_update
    assert not result.previous_adapter_preserved


@pytest.mark.parametrize("version", [1, 3])
def test_success_with_wrong_installed_version_is_quarantined(version):
    distributed = Distributed({"tp": (True, "", False, False)})
    scheduler = _scheduler(
        distributed,
        Output(success=True),
        NS(adapter_name="A", adapter_id="uid", adapter_version=version),
    )
    result = scheduler.load_oft_adapter_from_tensors(_request())
    assert not result.success
    assert result.inconsistent_update


def test_unanimous_safe_rejection_preserves_previous_adapter():
    distributed = Distributed({"tp": (False, "rejected", True, False)})
    scheduler = _scheduler(
        distributed, Output(success=False, previous_adapter_preserved=True)
    )
    result = scheduler.load_oft_adapter_from_tensors(_request())
    assert not result.success
    assert result.previous_adapter_preserved
    assert not result.inconsistent_update
    assert len(distributed.seen) == 1


def test_exception_still_participates_in_all_control_groups():
    distributed = Distributed(
        {"tp": (True, "", False, False), "cp": (True, "", False, False)}
    )
    scheduler = _scheduler(
        distributed, ValueError("decode failed"), groups=("tp", "cp")
    )
    result = scheduler.load_oft_adapter_from_tensors(_request())
    assert not result.success
    assert result.inconsistent_update
    assert [group for group, _ in distributed.seen] == ["tp", "cp"]


def test_unanimous_success_is_published():
    distributed = Distributed({"tp": (True, "", False, False)})
    scheduler = _scheduler(distributed, Output(success=True))
    result = scheduler.load_oft_adapter_from_tensors(_request())
    assert result.success
    assert not result.inconsistent_update
    assert len(distributed.seen) == 1


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_unavailable_consensus_fails_closed(missing):
    distributed = Distributed({"tp": None})
    if not missing:

        def fail(*args, **kwargs):
            raise RuntimeError("collective failed")

        distributed.all_gather_object = fail
    result = _scheduler(distributed, Output(success=True)).load_oft_adapter(_request())
    assert not result.success
    assert result.inconsistent_update


def test_second_control_group_failure_is_not_hidden():
    distributed = Distributed(
        {"tp": (True, "", False, False), "cp": (False, "rejected", True, False)}
    )
    result = _scheduler(
        distributed, Output(success=True), groups=("tp", "cp")
    ).load_oft_adapter(_request())
    assert not result.success
    assert result.inconsistent_update
    assert not result.previous_adapter_preserved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))

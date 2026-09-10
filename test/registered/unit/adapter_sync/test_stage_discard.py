"""Execute rollback orchestration without importing GPU runtime dependencies.

AST loading keeps the production methods intact; only IPC, reference records,
and allocation are replaced at their external boundaries.
"""

import ast
import asyncio
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Register without importing sglang's GPU-dependent public package initializer.
register_cpu_ci = load_module(
    ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_class(path, name, methods=None, **scope):
    tree = ast.parse((SRT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.body = [n for n in node.body if getattr(n, "name", None) in methods] or [
            ast.Pass()
        ]
        node.bases = []
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SRT / path), "exec"), scope)
    return scope[name]


@dataclass(frozen=True)
class LoRARef:
    lora_name: str
    lora_path: str
    pinned: bool = True
    reloadable: bool = False
    version: int = 3
    lora_id: str = "id-a"


@dataclass(frozen=True)
class OFTRef:
    adapter_name: str
    adapter_path: str
    pinned: bool = True
    reloadable: bool = False
    adapter_version: int = 3
    adapter_id: str = "id-a"


@pytest.fixture(params=["lora", "oft"])
def kind(request):
    return request.param


def make_manager(kind, state="exact"):
    label = "LoRA" if kind == "lora" else "OFT"
    cls = load_class(
        f"{kind}/staged_manager.py", f"Staged{label}Manager", {"discard_adapter_stage"}
    )
    manager = cls()
    pool_cls = load_class(
        f"{kind}/staged_manager.py",
        f"Staged{label}MemoryPool",
        {"staged_identity", "_require_staged_identity", "discard_stage"},
    )
    pool = pool_cls()
    pool._staged_uid = None if state in ("absent", "pending_only") else "id-a"
    pool._staged_version = None if pool._staged_uid is None else 4
    pool._staged_name = "policy-a" if pool._staged_uid else None
    pool.uid_to_buffer_id = {"id-a": 0}
    pool.buffer_id_to_uid = ["id-a"]
    pool._active_versions = {"id-a": 3}
    manager.memory_pool = pool
    old_ref = (
        LoRARef("policy-a", "/old") if kind == "lora" else OFTRef("policy-a", "/old")
    )
    ref = replace(
        old_ref, **({"version": 4} if kind == "lora" else {"adapter_version": 4})
    )
    pending = NS(ref=ref, uid="id-a", version=4, name="policy-a")
    setattr(
        manager,
        f"_pending_{kind}_stage",
        None if state in ("absent", "leaked") else pending,
    )
    setattr(manager, "lora_refs" if kind == "lora" else "refs", {"id-a": old_ref})
    manager.configs = {"id-a": object()}
    setattr(manager, "loras" if kind == "lora" else "adapters", {"id-a": object()})
    setattr(manager, "num_pinned_loras" if kind == "lora" else "num_pinned", 1)
    setattr(manager, f"create_{kind}_update_result", lambda **kw: NS(**kw))
    return manager


@pytest.mark.parametrize("state", ["exact", "absent", "leaked"])
def test_discard_preserves_active_state_and_is_idempotent(kind, state):
    manager = make_manager(kind, state)
    before = {
        k: v.copy() if isinstance(v, dict) else v
        for k, v in vars(manager).items()
        if not k.startswith("_pending")
    }
    pool_before = {
        k: v.copy() if isinstance(v, dict) else v
        for k, v in vars(manager.memory_pool).items()
        if not k.startswith("_staged")
    }
    assert hasattr(manager, "discard_adapter_stage"), "stage-only rollback is missing"
    for _ in range(2):
        assert manager.discard_adapter_stage("policy-a", "4", adapter_id="id-a").success
        assert getattr(manager, f"_pending_{kind}_stage") is None
        assert manager.memory_pool.staged_identity() is None
        for key, value in before.items():
            assert getattr(manager, key) == value
        for key, value in pool_before.items():
            assert getattr(manager.memory_pool, key) == value


@pytest.mark.parametrize(
    "state,name,uid,version",
    [
        ("exact", "wrong", "id-a", "4"),
        ("exact", "policy-a", "wrong", "4"),
        ("exact", "policy-a", "id-a", "5"),
        ("exact", "policy-a", "id-a", 4.5),
        ("exact", "policy-a", "id-a", True),
        ("exact", "", "id-a", "4"),
        ("exact", "policy-a", None, "4"),
        ("pending_only", "policy-a", "id-a", "4"),
        ("leaked", "wrong", "id-a", "4"),
        ("leaked", "policy-a", "wrong", "4"),
        ("leaked", "policy-a", "id-a", "5"),
    ],
)
def test_discard_rejects_nonexact_identity_without_mutation(
    kind, state, name, uid, version
):
    manager = make_manager(kind, state)
    pending = getattr(manager, f"_pending_{kind}_stage")
    pool_before = vars(manager.memory_pool).copy()
    assert hasattr(manager, "discard_adapter_stage"), "stage-only rollback is missing"
    assert not manager.discard_adapter_stage(name, version, adapter_id=uid).success
    assert getattr(manager, f"_pending_{kind}_stage") is pending
    assert vars(manager.memory_pool) == pool_before


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_cleanup_runs_locally_before_consensus_on_every_rank(failing_rank):
    consensus = load_module(
        SRT / "managers/scheduler_components/tp_update_consensus.py"
    )
    assert hasattr(
        consensus, "run_tp_adapter_stage_discard"
    ), "cleanup consensus is missing"
    results = [
        (
            rank != failing_rank,
            (
                "Failed to discard adapter stage: injected"
                if rank == failing_rank
                else "discarded"
            ),
            None,
        )
        for rank in range(2)
    ]
    for rank in range(2):
        events = []

        def discard():
            events.append("discard")
            if rank == failing_rank:
                raise RuntimeError("injected")
            return True, "discarded"

        def gather(outputs, local, *, group):
            assert events == ["discard"]
            assert local == results[rank]
            events.append("consensus")
            outputs[:] = results

        result = consensus.run_tp_adapter_stage_discard(
            distributed=NS(get_world_size=lambda **kw: 2, all_gather_object=gather),
            group="cpu",
            discard=discard,
        )
        assert result[0] is False
        assert f"TP rank {failing_rank}" in result[1]
        assert events == ["discard", "consensus"]


def test_discard_routes_identity_only_to_the_selected_manager(kind):
    manager = make_manager(kind)
    consensus = load_module(
        SRT / "managers/scheduler_components/tp_update_consensus.py"
    )
    assert hasattr(
        consensus, "run_tp_adapter_stage_discard"
    ), "cleanup routing is missing"
    runner_cls = load_class(
        "model_executor/model_runner_components/weight_updater.py",
        "WeightUpdater",
        {"discard_adapter_stage"},
    )
    updater = runner_cls()
    runner = NS(
        server_args=NS(
            enable_lora_staging=kind == "lora",
            peft_method="oft" if kind == "oft" else None,
        )
    )
    setattr(runner, kind + "_manager", manager)
    updater.get_model_runner = lambda: runner
    worker_cls = load_class(
        "managers/tp_worker.py", "BaseTpWorker", {"discard_adapter_stage"}
    )
    worker = worker_cls()
    worker.model_runner = NS(weight_updater=updater)
    calls = []

    def gather(outputs, local, *, group):
        calls.append((local, group))
        outputs[:] = [local]

    scheduler_cls = load_class(
        "managers/scheduler_components/weight_updater.py",
        "SchedulerWeightUpdaterManager",
        {"discard_adapter_stage"},
        torch=NS(
            distributed=NS(get_world_size=lambda **kw: 1, all_gather_object=gather)
        ),
        run_tp_adapter_stage_discard=consensus.run_tp_adapter_stage_discard,
        DiscardAdapterStageReqOutput=NS,
    )
    scheduler = scheduler_cls()
    scheduler.tp_worker = worker
    scheduler.tp_cpu_group = "cpu"
    req = NS(
        load_format=kind + "_adapter",
        adapter_name="policy-a",
        adapter_id="id-a",
        adapter_version="4",
    )
    result = scheduler.discard_adapter_stage(req)
    assert result.success, result.message
    assert getattr(manager, f"_pending_{kind}_stage") is None
    assert calls[0][1] == "cpu"


def test_pending_and_pool_disagreement_is_not_discarded(kind):
    manager = make_manager(kind)
    manager.memory_pool._staged_version = 5
    pending = getattr(manager, f"_pending_{kind}_stage")
    assert hasattr(manager, "discard_adapter_stage"), "stage-only rollback is missing"
    assert not manager.discard_adapter_stage("policy-a", "4", adapter_id="id-a").success
    assert manager.memory_pool.staged_identity() == ("id-a", 5)
    assert getattr(manager, f"_pending_{kind}_stage") is pending


def test_oft_inconsistent_pending_reference_is_not_discarded():
    manager = make_manager("oft")
    pending = manager._pending_oft_stage
    pending.ref = replace(pending.ref, adapter_name="other")
    assert not manager.discard_adapter_stage("policy-a", "4", adapter_id="id-a").success
    assert manager._pending_oft_stage is pending
    assert manager.memory_pool.staged_identity() == ("id-a", 4)


@pytest.mark.parametrize("fault", ["raises", "missing"])
def test_cleanup_consensus_errors_fail_closed(fault):
    consensus = load_module(
        SRT / "managers/scheduler_components/tp_update_consensus.py"
    )

    def gather(outputs, local, *, group):
        if fault == "raises":
            raise RuntimeError("CPU collective failed")
        outputs[:] = [local, None]

    result = consensus.run_tp_adapter_stage_discard(
        distributed=NS(get_world_size=lambda **kw: 2, all_gather_object=gather),
        group="cpu",
        discard=lambda: (True, "discarded"),
    )
    assert result[0] is False
    assert result[2] is None


@pytest.fixture
def runtime(monkeypatch):
    backend = load_module(SRT / "adapter_sync/tokenizer_backend.py")
    comm = load_module(SRT / "managers/communicator.py")
    for name in (
        "sglang.srt.oft",
        "sglang.srt.adapter_sync.tokenizer_backend",
        "sglang.srt.managers.communicator",
        "sglang.srt.managers.io_struct",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["sglang.srt.oft"].tokenizer_hooks = NS()
    monkeypatch.setitem(
        sys.modules, "sglang.srt.adapter_sync.tokenizer_backend", backend
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.communicator", comm)
    sys.modules["sglang.srt.managers.io_struct"].DiscardAdapterStageReqInput = NS
    return backend, comm


def make_tokenizer(kind, runtime):
    backend_module, comm = runtime
    label = "LoRA" if kind == "lora" else "OFT"
    backend_cls = load_class(
        f"{kind}/staged_manager.py",
        f"{label}StagingBackend",
        AdapterStagingBackend=backend_module.AdapterStagingBackend,
        LoRARef=LoRARef,
        OFTRef=OFTRef,
        replace=replace,
    )
    control_cls = load_class(
        "managers/tokenizer_control_mixin.py",
        "TokenizerControlMixin",
        {
            "update_adapter_from_distributed",
            "_rollback_failed_adapter_stage",
            "_run_adapter_activation_safely",
        },
        FanOutCommunicator=comm.FanOutCommunicator,
        DiscardAdapterStageReqInput=NS,
        asyncio=asyncio,
    )
    tm = control_cls()
    tm.server_args = NS(dp_size=1, enable_dp_attention=False, tokenizer_worker_num=1)
    tm.auto_create_handle_loop = lambda: None

    def available(name):
        if name in tm.failed_lora_activations:
            raise ValueError("restart required")

    tm._assert_native_lora_available = available
    ref = LoRARef("policy-a", "/old") if kind == "lora" else OFTRef("policy-a", "/old")
    registry_cls = load_class(
        "lora/lora_registry.py", "LoRARegistry", {"register_or_reuse"}, replace=replace
    )
    registry = registry_cls()
    registry._registry = {"policy-a": ref}
    registry._registry_lock = NS(reader_lock=asyncio.Lock())
    registry.get_all_adapters = lambda: registry._registry.copy()
    prefix = "lora" if kind == "lora" else "peft"
    setattr(tm, prefix + "_registry", registry)
    setattr(tm, prefix + "_ref_cache", {"policy-a": ref})
    setattr(tm, prefix + "_update_lock", asyncio.Lock())
    setattr(tm, "pending_" + kind + "_stage", None)
    setattr(tm, "failed_" + kind + "_activations", {})
    backend = backend_cls(tm)
    tm._staging_backend_for = lambda req: backend
    return tm, backend, ref


def request_for(kind, version="4", double_buffer=True):
    return NS(
        load_format=kind + "_adapter",
        adapter_name="policy-a",
        adapter_id="id-a",
        adapter_version=version,
        double_buffer=double_buffer,
    )


@pytest.mark.parametrize(
    "field,value",
    [("adapter_id", None), ("adapter_version", 4.5), ("load_format", "wrong")],
)
def test_reservation_clear_requires_exact_identity(kind, runtime, field, value):
    async def run():
        tm, backend, _ = make_tokenizer(kind, runtime)
        req = request_for(kind)
        await backend.reserve_stage(req)
        pending = getattr(tm, f"pending_{kind}_stage")
        setattr(req, field, value)
        with pytest.raises(ValueError):
            backend.clear_stage_reservation(req)
        assert getattr(tm, f"pending_{kind}_stage") is pending

    asyncio.run(run())


@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "dispatch",
        "stage_merge",
        "empty",
        "cleanup",
        "cleanup_dispatch",
        "cleanup_merge",
        "clear",
    ],
)
def test_failed_stage_rolls_back_or_quarantines(kind, runtime, failing_rank, fault):
    async def run():
        tm, backend, old_ref = make_tokenizer(kind, runtime)
        dispatches = []
        rollbacks = []

        async def stage(req):
            dispatches.append(req.adapter_version)
            if fault == "dispatch":
                raise RuntimeError("stage dispatch failed")
            if fault == "stage_merge":
                return [NS(success=False)]
            return [
                NS(
                    success=rank != failing_rank,
                    message="stage failed",
                    staged_adapter_version=None,
                    active_adapter_version=None,
                )
                for rank in range(2)
            ]

        async def rollback(req):
            rollbacks.append(vars(req))
            assert backend.lifecycle_lock.locked()
            if fault == "cleanup_dispatch":
                raise RuntimeError("cleanup dispatch failed")
            if fault == "empty":
                return []
            if fault == "cleanup_merge":
                return [NS(success=True)]
            return [
                NS(
                    success=not (fault == "cleanup" and rank == failing_rank),
                    message="cleanup result",
                )
                for rank in range(2)
            ]

        if fault == "clear":

            def fail_clear(req):
                raise RuntimeError("reservation mismatch")

            backend.clear_stage_reservation = fail_clear
        tm.update_adapter_from_distributed_communicator = stage
        tm.discard_adapter_stage_communicator = rollback
        success, message = await tm.update_adapter_from_distributed(request_for(kind))
        assert not success
        assert rollbacks == [
            {
                "load_format": kind + "_adapter",
                "adapter_name": "policy-a",
                "adapter_id": "id-a",
                "adapter_version": "4",
            }
        ]
        prefix = "lora" if kind == "lora" else "peft"
        assert (
            getattr(tm, prefix + "_registry").get_all_adapters()["policy-a"] is old_ref
        )
        assert getattr(tm, prefix + "_ref_cache")["policy-a"] is old_ref
        if fault in ("empty", "cleanup", "cleanup_dispatch", "cleanup_merge", "clear"):
            assert getattr(tm, "pending_" + kind + "_stage") is not None
            assert "policy-a" in getattr(tm, "failed_" + kind + "_activations")
            assert message.endswith("restart required")
            with pytest.raises(ValueError):
                await tm.update_adapter_from_distributed(request_for(kind, "5"))
            assert dispatches == ["4"]
        else:
            assert getattr(tm, "pending_" + kind + "_stage") is None
            assert "rollback" in message.lower()
            await tm.update_adapter_from_distributed(request_for(kind, "5"))
            assert dispatches == ["4", "5"]

    asyncio.run(run())


def test_cancellation_holds_reservation_and_lock_until_rollback_finishes(kind, runtime):
    async def run():
        tm, backend, _ = make_tokenizer(kind, runtime)
        entered, release = asyncio.Event(), asyncio.Event()

        async def stage(req):
            return [NS(success=False, message="stage failed")]

        async def rollback(req):
            entered.set()
            await release.wait()
            return [NS(success=True, message="discarded")]

        tm.update_adapter_from_distributed_communicator = stage
        tm.discard_adapter_stage_communicator = rollback
        task = asyncio.create_task(
            tm.update_adapter_from_distributed(request_for(kind))
        )
        for _ in range(20):
            if entered.is_set() or task.done():
                break
            await asyncio.sleep(0)
        assert entered.is_set(), "failed stage never dispatched cleanup"
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert backend.lifecycle_lock.locked()
        assert getattr(tm, "pending_" + kind + "_stage") is not None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not backend.lifecycle_lock.locked()
        assert getattr(tm, "pending_" + kind + "_stage") is None

    asyncio.run(run())


@pytest.mark.parametrize("activation_attempted", [False, True])
def test_synchronous_lora_only_discards_unanimous_pre_activation_failure(
    runtime, activation_attempted
):
    async def run():
        tm, backend, _ = make_tokenizer("lora", runtime)

        async def safely(backend, obj, operation):
            async with backend.lifecycle_lock:
                return await operation()

        tm._run_adapter_activation_safely = safely
        rollbacks = []

        async def stage(req):
            return [
                NS(
                    success=False,
                    message="stage failed",
                    staged_adapter_version=None,
                    active_adapter_version=None,
                ),
                NS(
                    success=False,
                    message="failed",
                    staged_adapter_version="4" if activation_attempted else None,
                    active_adapter_version=None,
                ),
            ]

        async def rollback(req):
            rollbacks.append(req)
            return [NS(success=True, message="discarded")]

        tm.update_adapter_from_distributed_communicator = stage
        tm.discard_adapter_stage_communicator = rollback
        success, message = await tm.update_adapter_from_distributed(
            request_for("lora", double_buffer=False)
        )
        assert not success
        assert bool(rollbacks) is not activation_attempted
        assert (tm.pending_lora_stage is not None) is activation_attempted
        assert bool(tm.failed_lora_activations) is activation_attempted

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

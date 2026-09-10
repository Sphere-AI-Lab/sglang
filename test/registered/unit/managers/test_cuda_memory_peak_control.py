"""Exercise allocator control methods without loading the GPU runtime.

Only CUDA/collectives and IPC are faked; source-extracted production handlers
perform the state transitions and evidence validation.
"""

import ast
import asyncio
import importlib.util
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANAGERS = ROOT / "python/sglang/srt/managers"
spec = importlib.util.spec_from_file_location(
    "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def extract(filename, names, scope, *, methods=None, structs=False):
    tree = ast.parse((MANAGERS / filename).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names), "CUDA peak control is missing"
    for node in nodes:
        if structs:
            node.bases = []
            node.keywords = []
            node.decorator_list = [
                ast.Call(
                    func=ast.Name(id="dataclass", ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg="kw_only", value=ast.Constant(True))],
                )
            ]
        elif methods is not None:
            node.bases = []
            node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + nodes
    exec(
        compile(ast.fix_missing_locations(tree), str(MANAGERS / filename), "exec"),
        scope,
    )


@pytest.fixture
def api():
    scope = {"dataclass": dataclass}
    extract(
        "io_struct.py",
        {
            "CudaMemoryPeakRankResult",
            "ResetCudaMemoryPeakReqInput",
            "ResetCudaMemoryPeakReqOutput",
            "ReadCudaMemoryPeakReqInput",
            "ReadCudaMemoryPeakReqOutput",
        },
        scope,
        structs=True,
    )
    extract(
        "scheduler.py",
        {"Scheduler"},
        scope,
        methods={
            "reset_cuda_memory_peak",
            "read_cuda_memory_peak",
            "_cuda_memory_peak_control",
        },
    )
    extract(
        "tokenizer_control_mixin.py",
        {"TokenizerControlMixin"},
        scope,
        methods={
            "reset_cuda_memory_peak",
            "read_cuda_memory_peak",
            "_cuda_memory_peak_control",
        },
    )
    return scope


def evidence(api, size, operation="reset", **overrides):
    return [
        api["CudaMemoryPeakRankResult"](
            **dict(
                dict(
                    rank=rank,
                    sample_id="sample-a",
                    operation=operation,
                    success=True,
                    message="",
                    allocated_bytes=100 * (rank + 1) if operation == "read" else None,
                    reserved_bytes=200 * (rank + 1) if operation == "read" else None,
                ),
                **overrides,
            )
        )
        for rank in range(size)
    ]


def scheduler(api, rank, size, records, *, fail=None, stats=(100, 200)):
    events = []
    obj = api["Scheduler"]()
    obj.ps = NS(tp_rank=rank, tp_size=size, pp_size=1, dp_size=1)
    obj.tp_cpu_group = "tp-cpu"

    def cuda_call(name, value=None):
        def call(device):
            assert device == 7
            events.append(name)
            if name == fail:
                raise RuntimeError("injected CUDA error")
            return value

        return call

    def gather(output, local, *, group):
        assert group == "tp-cpu"
        assert local.rank == rank
        events.append("gather")
        actual = records(local) if callable(records) else list(records)
        output[:] = actual

    api["torch"] = NS(
        cuda=NS(
            current_device=lambda: 7,
            synchronize=cuda_call("sync"),
            reset_peak_memory_stats=cuda_call("reset"),
            max_memory_allocated=cuda_call("allocated", stats[0]),
            max_memory_reserved=cuda_call("reserved", stats[1]),
        ),
        distributed=NS(all_gather_object=gather, get_world_size=lambda **kw: size),
    )
    api["get_parallel"] = lambda: NS(dp_size=1, pp_size=1)
    return obj, events


def call(api, obj, operation, sample_id="sample-a"):
    request = api[f"{operation.title()}CudaMemoryPeakReqInput"](sample_id=sample_id)
    return getattr(obj, f"{operation}_cuda_memory_peak")(request)


def test_cpu_boundary_reset_then_read_returns_allocator_values():
    scope = api.__wrapped__()
    obj, _ = scheduler(scope, 0, 1, lambda local: [local])
    assert call(scope, obj, "reset").success
    result = call(scope, obj, "read")
    assert result.success
    assert result.ranks[0].allocated_bytes == 100
    assert result.ranks[0].reserved_bytes == 200


@pytest.mark.parametrize("size", [1, 2, 4])
def test_every_tp_rank_resets_and_reads_exact_peaks(api, size):
    for rank in range(size):
        records = evidence(api, size)

        def gathered(local):
            assert local == records[rank]
            return records

        obj, events = scheduler(
            api,
            rank,
            size,
            gathered,
            stats=(100 * (rank + 1), 200 * (rank + 1)),
        )
        assert call(api, obj, "reset").success
        assert events == ["sync", "reset", "gather"]
        records[:] = evidence(api, size, "read")
        events.clear()
        result = call(api, obj, "read")
        assert result.success
        assert events == ["sync", "allocated", "reserved", "gather"]
        assert [r.rank for r in result.ranks] == list(range(size))
        assert (
            sum(r.allocated_bytes for r in result.ranks)
            == {1: 100, 2: 300, 4: 1000}[size]
        )
        assert (
            max(r.reserved_bytes for r in result.ranks)
            == {1: 200, 2: 400, 4: 800}[size]
        )


@pytest.mark.parametrize(
    "operation,failure",
    [
        ("reset", "sync"),
        ("reset", "reset"),
        ("read", "sync"),
        ("read", "allocated"),
        ("read", "reserved"),
    ],
)
def test_local_cuda_error_still_enters_collective_on_all_ranks(api, operation, failure):
    for rank in range(4):
        records = evidence(api, 4, operation)
        records[3] = replace(
            records[3],
            success=False,
            message="injected CUDA error",
            allocated_bytes=None,
            reserved_bytes=None,
        )

        def gathered(local):
            assert local.success is (rank != 3)
            return records

        obj, events = scheduler(
            api, rank, 4, gathered, fail=failure if rank == 3 else None
        )
        obj._cuda_memory_peak_sample_id = "sample-a"
        result = call(api, obj, operation)
        assert not result.success
        assert events[-1] == "gather"
        assert obj._cuda_memory_peak_sample_id is None


def test_failed_reset_cannot_leave_previous_sample_armed(api):
    obj, _ = scheduler(api, 0, 1, lambda local: [local], fail="reset")
    obj._cuda_memory_peak_sample_id = "sample-a"
    assert not call(api, obj, "reset", "sample-b").success
    assert not call(api, obj, "read", "sample-a").success


@pytest.mark.parametrize("operation", ["reset", "read"])
def test_gather_exception_disarms_sample(api, operation):
    obj, events = scheduler(api, 0, 1, lambda local: [local])
    obj._cuda_memory_peak_sample_id = "sample-a"

    def broken_collective(*args, **kwargs):
        events.append("gather-failure")
        raise RuntimeError("CPU collective failed")

    api["torch"].distributed.all_gather_object = broken_collective
    result = call(api, obj, operation)
    assert not result.success
    assert "CPU collective failed" in result.message
    assert obj._cuda_memory_peak_sample_id is None


@pytest.mark.parametrize("sample", ["", "  ", None, 1])
def test_tokenizer_rejects_invalid_sample_before_dispatch(api, sample):
    obj, sent = tokenizer(api, "reset", [])
    with pytest.raises(ValueError):
        asyncio.run(obj.reset_cuda_memory_peak(sample))
    assert not sent


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "none",
        "duplicate",
        "rank",
        "sample",
        "operation",
        "failed",
        "bool-rank",
    ],
)
@pytest.mark.parametrize("operation", ["reset", "read"])
def test_malformed_gather_fails_closed_and_disarms(api, bad, operation):
    records = evidence(api, 2, operation)
    if bad == "missing":
        records.pop()
    elif bad == "none":
        records[1] = None
    else:
        change = {
            "duplicate": {"rank": 0},
            "rank": {"rank": 2},
            "sample": {"sample_id": "other"},
            "operation": {"operation": "other"},
            "failed": {"success": False},
            "bool-rank": {"rank": True},
        }[bad]
        records[1] = replace(records[1], **change)
    obj, _ = scheduler(api, 0, 2, records)
    obj._cuda_memory_peak_sample_id = "sample-a"
    assert not call(api, obj, operation).success
    assert obj._cuda_memory_peak_sample_id is None


def test_complete_permutation_is_returned_in_rank_order(api):
    obj, _ = scheduler(api, 0, 4, evidence(api, 4)[::-1])
    result = call(api, obj, "reset")
    assert result.success
    assert [r.rank for r in result.ranks] == [0, 1, 2, 3]


@pytest.mark.parametrize("armed", [None, "older-sample"])
def test_unarmed_or_stale_read_fails_and_participates(api, armed):
    obj, events = scheduler(api, 0, 1, lambda local: [local])
    obj._cuda_memory_peak_sample_id = armed
    assert not call(api, obj, "read").success
    assert events[-1] == "gather"
    assert obj._cuda_memory_peak_sample_id is None


def test_read_is_consumed_even_after_success(api):
    obj, _ = scheduler(api, 0, 1, lambda local: [local])
    assert call(api, obj, "reset").success
    assert call(api, obj, "read").success
    assert not call(api, obj, "read").success


@pytest.mark.parametrize(
    "stats", [(201, 200), (-1, 2), (True, 200), (100, 2.5), ("100", 200)]
)
def test_invalid_cuda_stats_fail_collectively(api, stats):
    obj, events = scheduler(api, 0, 1, lambda local: [local], stats=stats)
    assert call(api, obj, "reset").success
    assert not call(api, obj, "read").success
    assert events[-1] == "gather"


@pytest.mark.parametrize(
    "stats", [(201, 200), (-1, 2), (True, 200), (100, 2.5), (100, None)]
)
def test_nonzero_rank_invalid_gathered_stats_fail_aggregate(api, stats):
    records = evidence(api, 2, "read")
    records[1] = replace(records[1], allocated_bytes=stats[0], reserved_bytes=stats[1])
    obj, _ = scheduler(api, 0, 2, records)
    obj._cuda_memory_peak_sample_id = "sample-a"
    assert not call(api, obj, "read").success
    assert obj._cuda_memory_peak_sample_id is None


def test_zero_peaks_are_valid_for_generic_control(api):
    obj, _ = scheduler(api, 0, 1, lambda local: [local], stats=(0, 0))
    assert call(api, obj, "reset").success
    result = call(api, obj, "read")
    assert result.success
    assert result.ranks[0].allocated_bytes == result.ranks[0].reserved_bytes == 0


def test_real_group_size_mismatch_enters_collective_as_failure(api):
    obj, events = scheduler(api, 0, 1, lambda local: [local])
    api["torch"].distributed.get_world_size = lambda **kw: 2
    assert not call(api, obj, "reset").success
    assert events == ["gather"]


@pytest.mark.parametrize(
    "change",
    [
        {"rank": 0},
        {"rank": True},
        {"sample_id": "other"},
        {"operation": "other"},
        {"success": 1},
        {"reserved_bytes": 50},
    ],
)
def test_tokenizer_rejects_forged_per_rank_read_evidence(api, change):
    records = evidence(api, 2, "read")
    records[1] = replace(records[1], **change)
    response = api["ReadCudaMemoryPeakReqOutput"](
        sample_id="sample-a", operation="read", success=True, message="", ranks=records
    )
    obj, _ = tokenizer(api, "read", [response])
    with pytest.raises(RuntimeError):
        asyncio.run(obj.read_cuda_memory_peak("sample-a"))


@pytest.mark.parametrize("operation", ["reset", "read"])
@pytest.mark.parametrize("sample", ["", "   ", None, 3])
def test_invalid_sample_fails_collectively(api, operation, sample):
    obj, events = scheduler(api, 0, 1, lambda local: [local])
    assert not call(api, obj, operation, sample).success
    assert events == ["gather"]


def tokenizer(api, operation, responses, size=2, dp=1, pp=1):
    obj = api["TokenizerControlMixin"]()
    obj.server_args = NS(tp_size=size, dp_size=dp, pp_size=pp)
    obj.auto_create_handle_loop = lambda: None
    sent = []

    async def communicate(req):
        sent.append(req)
        return responses

    setattr(obj, f"{operation}_cuda_memory_peak_communicator", communicate)
    api["get_parallel"] = lambda: NS(dp_size=dp, pp_size=pp)
    return obj, sent


def tokenizer_parent_parallel(api, server_args):
    """Use production topology access with no scheduler PP group in the parent."""
    scope = {"_PP": None}
    extract(
        "../distributed/parallel_state.py",
        {"get_pp_group", "get_pipeline_model_parallel_world_size"},
        scope,
    )
    scope["_ps"] = lambda: NS(**scope)
    extract(
        "../runtime_context.py",
        {"ParallelContext"},
        scope,
        methods={"__init__", "__getattr__", "_v", "pp_size"},
    )
    parallel = scope["ParallelContext"]()
    parallel._config = NS(_fields=vars(server_args), **vars(server_args))
    api["get_parallel"] = lambda: parallel


@pytest.mark.parametrize("operation", ["reset", "read"])
def test_tokenizer_parent_dispatches_without_initialized_pp_group(api, operation):
    response = api[f"{operation.title()}CudaMemoryPeakReqOutput"](
        sample_id="sample-a",
        operation=operation,
        success=True,
        message="",
        ranks=evidence(api, 2, operation),
    )
    obj, sent = tokenizer(api, operation, [response])
    tokenizer_parent_parallel(api, obj.server_args)
    result = asyncio.run(getattr(obj, f"{operation}_cuda_memory_peak")("sample-a"))
    assert result is response
    assert len(sent) == 1
    assert isinstance(sent[0], api[f"{operation.title()}CudaMemoryPeakReqInput"])
    assert sent[0].sample_id == "sample-a"


@pytest.mark.parametrize("operation", ["reset", "read"])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_tokenizer_returns_typed_complete_consensus(api, operation, size):
    response = api[f"{operation.title()}CudaMemoryPeakReqOutput"](
        sample_id="sample-a",
        operation=operation,
        success=True,
        message="",
        ranks=evidence(api, size, operation),
    )
    obj, sent = tokenizer(api, operation, [response], size=size)
    result = asyncio.run(getattr(obj, f"{operation}_cuda_memory_peak")("sample-a"))
    assert result is response
    assert sent[0].sample_id == "sample-a"


@pytest.mark.parametrize("operation", ["reset", "read"])
@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "extra",
        "none",
        "type",
        "failed",
        "sample",
        "operation",
        "rank",
        "order",
        "metric",
        "rank-failed",
    ],
)
def test_tokenizer_rejects_incomplete_or_invalid_consensus(api, operation, bad):
    response = api[f"{operation.title()}CudaMemoryPeakReqOutput"](
        sample_id="sample-a",
        operation=operation,
        success=True,
        message="",
        ranks=evidence(api, 2, operation),
    )
    responses = [response]
    if bad == "missing":
        responses = []
    elif bad == "extra":
        responses *= 2
    elif bad == "none":
        responses = [None]
    elif bad == "type":
        responses = [NS(**vars(response))]
    elif bad in ("failed", "sample", "operation"):
        key, value = {
            "failed": ("success", False),
            "sample": ("sample_id", "other"),
            "operation": ("operation", "other"),
        }[bad]
        setattr(response, key, value)
    elif bad == "rank":
        response.ranks.pop()
    elif bad == "order":
        response.ranks.reverse()
    elif bad == "metric":
        response.ranks[1].allocated_bytes = True
    elif bad == "rank-failed":
        response.ranks[1].success = False
    obj, _ = tokenizer(api, operation, responses)
    with pytest.raises(RuntimeError):
        asyncio.run(getattr(obj, f"{operation}_cuda_memory_peak")("sample-a"))


@pytest.mark.parametrize("dp,pp", [(2, 1), (1, 2)])
@pytest.mark.parametrize("operation", ["reset", "read"])
def test_unsupported_topology_rejected_before_dispatch(api, dp, pp, operation):
    obj, sent = tokenizer(api, operation, [], dp=dp, pp=pp)
    tokenizer_parent_parallel(api, obj.server_args)
    with pytest.raises(ValueError):
        asyncio.run(getattr(obj, f"{operation}_cuda_memory_peak")("sample-a"))
    assert not sent


def test_production_request_and_response_tables_route_independently(api):
    """A missing registration or response-type collision loses a control reply."""
    api["OrderedDict"] = OrderedDict
    extract("../../utils.py", {"TypeBasedDispatcher"}, api)
    dispatcher_type = api["TypeBasedDispatcher"]
    obj, _ = scheduler(api, 0, 1, lambda local: [local])
    handlers = MagicMock()
    handlers.reset_cuda_memory_peak = obj.reset_cuda_memory_peak
    handlers.read_cuda_memory_peak = obj.read_cuda_memory_peak
    tree = ast.parse((MANAGERS / "scheduler.py").read_text())
    initializer = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "init_request_dispatcher"
    )
    expression = next(n.value for n in initializer.body if isinstance(n, ast.Assign))
    scope = dict(api, self=handlers)
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and node.id not in scope:
            scope[node.id] = type(node.id, (), {})
    request_dispatcher = eval(
        compile(ast.Expression(expression), "request-table", "eval"), scope
    )

    tree = ast.parse((MANAGERS / "tokenizer_control_mixin.py").read_text())
    expression = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_COMMUNICATOR_SPECS" for t in n.targets
        )
    )
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and node.id not in scope:
            scope[node.id] = type(node.id, (), {})
    specs = eval(compile(ast.Expression(expression), "response-table", "eval"), scope)
    received = {}
    response_dispatcher = dispatcher_type(
        [
            (
                spec[1],
                lambda response, name=spec[0]: received.setdefault(name, response),
            )
            for spec in specs
        ]
    )
    for operation in ("reset", "read"):
        output = request_dispatcher(
            api[f"{operation.title()}CudaMemoryPeakReqInput"](sample_id="sample-a")
        )
        assert output.success
        response_dispatcher(output)
        assert received[f"{operation}_cuda_memory_peak"] is output
    assert len(received) == 2


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_existing_sender_only_emits_consensus_from_rank_zero(api, rank):
    """Nonzero schedulers have no tokenizer socket and must not duplicate replies."""
    obj, _ = scheduler(api, rank, 4, evidence(api, 4))
    output = call(api, obj, "reset")
    output.http_worker_ipc = None
    sent = []
    api["BaseReq"] = (
        api["ResetCudaMemoryPeakReqOutput"],
        api["ReadCudaMemoryPeakReqOutput"],
    )
    api["sock_send"] = lambda socket, value: sent.append((socket, value))
    extract("scheduler_components/output_sender.py", {"SenderWrapper"}, api)
    socket = "tokenizer-socket" if rank == 0 else None
    api["SenderWrapper"](socket).send_output(output, NS(http_worker_ipc="ipc://origin"))
    assert sent == ([("tokenizer-socket", output)] if rank == 0 else [])
    if rank == 0:
        assert output.http_worker_ipc == "ipc://origin"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

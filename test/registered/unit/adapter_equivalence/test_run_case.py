"""Real runner tests at the engine, control and sender process boundaries."""

import asyncio
import importlib
import io
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, wait
from dataclasses import asdict, dataclass, field, make_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence import run_case
from adapter_equivalence.scenarios import ScenarioContractError, lifecycle_steps
from adapter_equivalence.schema import BundleValidationError, Observation
from adapter_equivalence.server import (
    AdapterIdentity,
    ControlResult,
    DistributedPayload,
    ServerSpec,
)


def test_observation_construction_is_not_swallowed(make_runner, monkeypatch):
    """Defect: the old runner silently discarded malformed observation evidence."""
    runner = make_runner()

    def broken(*args, **kwargs):
        raise BundleValidationError("missing scores")

    monkeypatch.setattr(run_case, "capture_generation", broken)
    with pytest.raises(BundleValidationError, match="missing scores"):
        runner.run_selected(("base.initial",))
    assert runner.observations == {}


def test_lora_inputs_bind_cold_start_shape_contract_from_fixture(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "target_modules": ["embed_tokens", "lm_head", "q_proj"],
            }
        )
    )
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(
        json.dumps(
            {
                "policy-a": str(adapter),
                "policy-b": str(adapter),
                "2": str(adapter),
                "3": str(adapter),
            }
        )
    )
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("")
    args = SimpleNamespace(
        mode="native_lora",
        revision_kind="source",
        model_path="/models/qwen3-4b",
        port=31004,
        tp_size=1,
        ep_size=1,
        cuda_graph="off",
        quantization=None,
        moe_runner=None,
        base_gpu_id=1,
        case_id="dense-lora",
        revision_sha="a" * 40,
        architecture="dense",
        precision="bf16",
        checkpoint_manifest=checkpoint,
        prompts_file=prompts,
        fixture_manifest=fixtures,
        bundle_output=tmp_path / "bundle.json",
        completion_output=tmp_path / "complete.json",
        max_new_tokens=32,
        repetition=0,
    )

    spec, _, _, _ = run_case.inputs_from_args(args)

    assert spec.server.max_lora_rank == 8
    assert spec.server.lora_target_modules == (
        "embed_tokens",
        "lm_head",
        "q_proj",
    )


def response(token=10):
    return {
        "output_ids": [token],
        "text": str(token),
        "meta_info": {
            "output_token_logprobs": [(-0.1, token, "token")],
            "output_top_logprobs": [[(-0.1 - i, token + i, "token") for i in range(5)]],
        },
    }


class FakeEngine:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.calls = []
        self.closed = False
        self.fail = None
        self.block = None
        self.lease_release = None

    async def async_generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        if self.block:
            self.block.wait()
        if isinstance(kwargs["input_ids"][0], list):
            return [response(ids[0]) for ids in kwargs["input_ids"]]
        result = response(kwargs["input_ids"][0])
        if not kwargs.get("stream"):
            return result

        async def chunks():
            yield result
            if self.lease_release is not None:
                await self.lease_release.wait()
            yield result

        return chunks()

    def shutdown(self):
        self.closed = True


class FakeSamplingParams:
    __struct_fields__ = (
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "presence_penalty",
    )

    def __init__(self, presence_penalty=0.0, **kwargs):
        self.__dict__.update(kwargs, presence_penalty=presence_penalty)

    def normalize(self, tokenizer):
        self.temperature, self.top_k = 1.0, 1

    def verify(self, vocab_size):
        assert vocab_size > 0


def committed_request_boundary(repo, monkeypatch):
    """Execute real forwarding/normalization; omit annotation-only heavy imports."""
    import ast
    import copy
    import uuid
    from collections import Counter
    from types import ModuleType

    request_path = repo / "python/sglang/srt/managers/io_struct.py"
    engine_path = repo / "python/sglang/srt/entrypoints/engine.py"
    request = next(
        node
        for node in ast.parse(request_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "GenerateReqInput"
    )
    engine = next(
        node
        for node in ast.parse(engine_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    forwarding = next(
        node for node in engine.body if getattr(node, "name", None) == "async_generate"
    )
    module = ModuleType("qualification_request_" + uuid.uuid4().hex)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        dataclass=dataclass,
        field=field,
        uuid=uuid,
        copy=copy,
        Counter=Counter,
        get_return_hidden_states_mode=lambda mode: mode,
    )
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            request,
            forwarding,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(code), str(request_path), "exec"),
        module.__dict__,
    )
    return module.async_generate


class FakeControl:
    def __init__(self, mode):
        self.mode = mode
        self.operations = []
        self.records = {}
        self.focus = None
        self.staged = None
        self.failure = False
        self.transfer_started = threading.Event()
        self.transfer_done = threading.Event()

    def result(self):
        return ControlResult(
            not self.failure, "rejected" if self.failure else "ok", {}, None, None
        )

    def load_path(self, name, path, *, pinned=False):
        self.operations.append(("load_path", name, path))
        self.records[name] = {
            "name": name,
            "id": "live-" + name,
            "version": "0",
            "pinned": pinned,
        }
        self.focus = name
        return self.result()

    def load_tensors(self, name, tensors, config, *, upsert=False):
        self.operations.append(("load_tensors", name, tensors, config, upsert))
        self.records[name] = {
            "name": name,
            "id": "live-" + name,
            "version": "0",
            "pinned": False,
        }
        self.focus = name
        return self.result()

    def load_distributed(self, name, payload, group_name, *, upsert=False):
        self.operations.append(("load_distributed", name, payload, group_name, upsert))
        self.transfer_started.set()
        assert self.transfer_done.wait(
            1
        ), "sender did not run concurrently with request"
        self.transfer_started.clear()
        self.transfer_done.clear()
        self.records[name] = {
            "name": name,
            "id": "live-" + name,
            "version": "0",
            "pinned": False,
        }
        self.focus = name
        return self.result()

    def stage(self, identity, payload, group_name):
        self.operations.append(("stage", identity, payload, group_name))
        self.transfer_started.set()
        assert self.transfer_done.wait(1), "stage sender did not run concurrently"
        self.transfer_started.clear()
        self.transfer_done.clear()
        self.staged = {
            "name": identity.name,
            "id": identity.adapter_id,
            "version": identity.version,
            "pinned": False,
        }
        return self.result()

    def activate(self, identity):
        self.operations.append(("activate", identity))
        assert self.staged["id"] == identity.adapter_id
        self.records[identity.name] = self.staged
        self.focus = identity.name
        self.staged = None
        return self.result()

    def unload(self, name):
        self.operations.append(("unload", name))
        self.records.pop(name)
        self.focus = None
        return self.result()

    def inspect_state(self):
        records = [
            dict(self.records[name], registry_slot=i)
            for i, name in enumerate(sorted(self.records))
        ]
        active = self.records.get(self.focus)
        return {
            "mode": self.mode,
            "registered": records,
            "active": (
                None
                if active is None
                else {k: active[k] for k in ("name", "id", "version")}
            ),
            "staged": self.staged,
            "registry_occupancy": len(records),
            "quarantined": [],
            "tombstoned": [],
            "cache_identity": {
                name: record["id"] for name, record in self.records.items()
            },
        }


PAYLOAD = DistributedPayload(("weight",), ("float32",), ((2, 3),), {"r": 8})


class FakeSender:
    group_name = "group-test"

    def __init__(self, control):
        self.sender = self
        self.control = control
        self.broadcasts = []
        self.payload = PAYLOAD
        self.closed = False
        self.failure = None

    def broadcast_fixture(self, request_id, fixture, *, timeout):
        assert self.control.transfer_started.wait(timeout)
        self.broadcasts.append((request_id, fixture))
        self.control.transfer_done.set()
        return self.payload

    def close(self, timeout):
        self.closed = True
        if self.failure:
            raise self.failure


@pytest.fixture
def make_runner(tmp_path, monkeypatch):
    runners = []
    tensors = {"weight": object()}
    monkeypatch.setattr(
        run_case, "load_fixture", lambda path: (tensors, PAYLOAD.config), raising=False
    )
    monkeypatch.setattr(
        run_case, "distributed_payload_for_fixture", lambda path: PAYLOAD, raising=False
    )

    def make(mode="native_lora", **kwargs):
        engine = kwargs.pop("engine", FakeEngine())
        control = kwargs.pop("control", FakeControl(mode))
        sender = FakeSender(control)
        server = ServerSpec(
            "candidate",
            "/checkpoint",
            mode,
            30000,
            1,
            1,
            False,
            startup_adapters=(
                () if mode == "base" else (("policy-a", str(tmp_path / "a")),)
            ),
            max_lora_rank=8 if mode == "native_lora" else None,
            lora_target_modules=("q_proj",) if mode == "native_lora" else (),
        )
        spec = run_case.RunSpec(
            server=server,
            case_id="test",
            revision_sha="a" * 40,
            architecture="dense",
            precision="bf16",
            checkpoint_manifest=tmp_path / "checkpoint.json",
            prompts_file=tmp_path / "prompts.jsonl",
            fixture_manifest=tmp_path / "fixtures.json",
            bundle_output=tmp_path / "bundle.json",
            completion_output=tmp_path / "complete.json",
        )
        runner = run_case.ShardRunner(
            spec,
            engine=engine,
            control=control,
            sender=sender,
            prompts={"factual": [10], "long-prefix": [11]},
            batches={"batch-8": ["factual", "long-prefix"] * 4},
            fixtures={
                "policy-a": tmp_path / "a",
                "policy-b": tmp_path / "b",
                str(2 + int(mode == "native_oft")): tmp_path / "v1",
                str(3 + int(mode == "native_oft")): tmp_path / "v2",
            },
            diagnostics=io.StringIO(),
            **kwargs,
        )
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        runner.close()


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_real_immediate_mechanisms_and_stable_staging(make_runner, mode):
    """Defect: direct Engine loads, versioned-name stages, and serial collectives."""
    runner = make_runner(mode)
    names = (
        "immediate.path.load",
        "immediate.path.infer",
        "immediate.path.unload",
        "immediate.path.base",
        "immediate.tensor.load",
        "immediate.tensor.unload",
        "immediate.distributed.load",
        "stage.v1",
        "stage.v1.old-active",
        "activate.v1",
        "stage.v2",
        "activate.v2",
    )
    observed = runner.run_selected(names)
    assert tuple(observed) == names
    assert [op[0] for op in runner.control.operations] == [
        "load_path",
        "unload",
        "load_tensors",
        "unload",
        "load_distributed",
        "stage",
        "activate",
        "stage",
        "activate",
    ]
    assert runner.control.operations[2][3] == {"r": 8}
    assert [request for request, _ in runner.sender.broadcasts] == [
        "immediate.distributed.load",
        "stage.v1",
        "stage.v2",
    ]
    for version, stage_index, activation_index in (("2", 5, 6), ("3", 7, 8)):
        identity = AdapterIdentity(
            "policy-a", "live-policy-a", str(int(version) + int(mode == "native_oft"))
        )
        assert runner.control.operations[stage_index][1] == identity
        assert runner.control.operations[activation_index][1] == identity
    assert observed["stage.v1.old-active"].adapter_state["active"]["version"] == "0"
    assert observed["activate.v2"].adapter_state["active"]["version"] == str(
        3 + int(mode == "native_oft")
    )
    for call in runner.engine.calls:
        assert call["return_logprob"] is True
        assert call["top_logprobs_num"] == 5
        assert (
            "adapter_path" if mode == "native_oft" else "lora_path"
        ) in call or call["input_ids"] == [10]


@pytest.mark.parametrize(
    "field,value",
    (
        ("names", ("other",)),
        ("dtypes", ("float16",)),
        ("shapes", ((3, 2),)),
        ("config", {"r": 4}),
    ),
)
def test_sender_metadata_mismatch_fails_without_step_record(make_runner, field, value):
    """Defect: accepting sender/request disagreement in any tensor metadata field."""
    runner = make_runner()
    runner.sender.payload = replace(PAYLOAD, **{field: value})
    with pytest.raises(ScenarioContractError, match="sender.*request"):
        runner.run_selected(("immediate.distributed.load",))
    assert runner.observations == {}


def test_unsuccessful_control_is_not_a_passing_observation(make_runner):
    """Defect: treating a returned product failure as successful execution."""
    runner = make_runner()
    runner.control.failure = True
    with pytest.raises(ScenarioContractError, match="rejected"):
        runner.run_selected(("immediate.path.load",))
    assert runner.observations == {}


def test_selection_uses_only_the_single_lifecycle_table(make_runner, monkeypatch):
    """Defect: duplicated selection tables drift or accept undeclared actions."""
    runner = make_runner()
    steps = lifecycle_steps("native_lora")
    monkeypatch.setattr(run_case, "lifecycle_steps", lambda mode: (steps[0],))
    assert runner.resolve_selection("full") == ("base.initial",)
    with pytest.raises(ScenarioContractError, match="undeclared"):
        runner.run_selected(("immediate.path.load",))
    with pytest.raises(ScenarioContractError, match="duplicate"):
        make_runner().run_selected(("base.initial", "base.initial"))


def test_leased_upsert_does_not_wait_for_collective_before_draining(make_runner):
    """Defect: awaiting the sender blocks the very lease drain needed by upsert."""
    engine = FakeEngine()
    lease_drained = asyncio.Event()
    update_waiting = threading.Event()

    async def generate(**kwargs):
        engine.calls.append(kwargs)

        async def chunks():
            yield response()
            yield response()
            lease_drained.set()

        return chunks()

    engine.async_generate = generate

    class LeasedControl(FakeControl):
        def load_distributed(self, name, payload, group_name, *, upsert=False):
            assert upsert is True

            async def update():
                update_waiting.set()
                await lease_drained.wait()
                return super(LeasedControl, self).load_distributed(
                    name, payload, group_name, upsert=upsert
                )

            return engine.loop.run_until_complete(update())

        def observe_lease_wait(self, name, adapter_id):
            future = Future()
            assert update_waiting.wait(0.5)
            future.set_result(self.inspect_state())
            return future

    control = LeasedControl("native_lora")
    control.load_path("policy-a", "/a")
    runner = make_runner(
        engine=engine,
        control=control,
        timeouts=run_case.Timeouts(collective=0.2, inference=0.2),
    )
    observed = runner.run_selected(
        ("upsert.lease.begin", "upsert.while-leased", "upsert.lease.complete")
    )
    assert observed["upsert.lease.complete"].output_ids == (10,)
    assert runner.sender.broadcasts[0][0] == "upsert.while-leased"
    assert control.operations[-1][-1] is True


def test_rejection_without_required_live_identity_fails_closed(make_runner):
    """Missing prerequisites cannot become a successful rejection surrogate."""
    runner = make_runner()
    with pytest.raises(ScenarioContractError, match="registered"):
        runner.run_selected(("reject.duplicate",))
    assert runner.observations == {}


@pytest.mark.parametrize("mode", ("base", "native_lora", "native_oft"))
def test_startup_and_restart_use_real_launch_specs(make_runner, monkeypatch, mode):
    """Defect: startup.adapter is dynamic load and restart loses the startup manifest."""
    launched = []
    controls = {}

    def launch(spec):
        launched.append(spec)
        engine = FakeEngine()
        control = FakeControl(mode)
        for name, path in spec.startup_adapters:
            control.load_path(name, path)
        controls[engine] = control
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    monkeypatch.setattr(
        run_case, "make_adapter_control", lambda mode, engine: controls[engine]
    )
    runner = make_runner(mode)
    runner.engine = runner.control = runner.sender = None
    runner.spec = replace(
        runner.spec,
        server=replace(
            runner.spec.server,
            max_oft_block_size=128 if mode == "native_oft" else None,
            peft_target_modules=("q_proj",) if mode == "native_oft" else (),
        ),
    )
    names = (
        ("base.initial", "restart.same-manifest", "restart.identity")
        if mode == "base"
        else (
            "base.initial",
            "startup.adapter",
            "restart.same-manifest",
            "restart.identity",
        )
    )
    observed = runner.run_selected(names)
    assert observed["base.initial"].adapter_state["registered"] == ()
    assert launched[0].startup_adapters == ()
    if mode != "base":
        assert len(launched) == 3
        assert launched[1] == launched[2] == runner.spec.server
        assert (
            observed["startup.adapter"].adapter_state
            == observed["restart.identity"].adapter_state
        )
    else:
        assert launched == [runner.spec.server, runner.spec.server]


@pytest.mark.parametrize("mode, offset", (("native_lora", 0), ("native_oft", 1)))
def test_staged_versions_follow_successful_upsert_metadata(mode, offset):
    """Defect: stage version 1 is stale after upsert has advanced active version to 1."""
    steps = {step.name: step for step in lifecycle_steps(mode)}
    assert steps["upsert.while-leased"].input_kind == "distributed"
    assert {
        name: steps[name].version
        for name in (
            "stage.v1",
            "activate.v1",
            "stage.v2",
            "activate.v2",
            "reject.duplicate",
            "reject.stale",
            "reject.wrong-id",
            "failure.update",
            "failure.activation",
        )
    } == {
        "stage.v1": str(2 + offset),
        "activate.v1": str(2 + offset),
        "stage.v2": str(3 + offset),
        "activate.v2": str(3 + offset),
        "reject.duplicate": str(3 + offset),
        "reject.stale": str(2 + offset),
        "reject.wrong-id": str(4 + offset),
        "failure.update": str(4 + offset),
        "failure.activation": str(4 + offset),
    }


def test_base_generates_without_adapter_keyword(make_runner):
    """Defect: base mode rejected or silently routed through native adapter calls."""
    runner = make_runner("base")
    observations = runner.run_selected(("base.initial", "concurrent.non-stream"))
    assert len(observations["concurrent.non-stream"].output_ids) == 8
    assert runner.control.operations == []
    assert all(
        "lora_path" not in call and "adapter_path" not in call
        for call in runner.engine.calls
    )


def test_timeout_preserves_first_error_and_attempts_both_teardowns(make_runner):
    """Defect: a wedged worker or teardown replaces the first failure or hangs."""
    engine = FakeEngine()
    gate = threading.Event()
    engine.block = gate
    runner = make_runner(
        engine=engine,
        timeouts=run_case.Timeouts(
            startup=0.1, inference=0.02, control=0.1, collective=0.1, teardown=0.1
        ),
    )
    runner.sender.failure = RuntimeError("secondary teardown")
    start = time.monotonic()
    try:
        with pytest.raises(run_case.OperationTimeout) as error:
            runner.run_selected(("base.initial",))
        assert time.monotonic() - start < 1
        assert runner.failure is error.value
        assert engine.closed and runner.sender.closed
        assert "secondary teardown" in runner.diagnostics.getvalue()
    finally:
        gate.set()


def test_engine_failure_object_survives_teardown_failure(make_runner):
    """Defect: cleanup hides the original exception object."""
    runner = make_runner()
    original = RuntimeError("first failure")
    runner.engine.fail = original
    runner.sender.failure = RuntimeError("second failure")
    with pytest.raises(RuntimeError) as error:
        runner.run_selected(("base.initial",))
    assert error.value is original
    assert runner.engine.closed and runner.sender.closed


def test_target_shapes_include_global_embedding_and_head():
    """Defect: fixtures omit the global embedding and LM-head binding targets."""
    shapes = run_case.build_target_shapes(
        {
            "hidden_size": 128,
            "vocab_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 384,
        },
        "dense",
        layers=0,
        experts=0,
        suffixes=("q_proj", "down_proj"),
    )
    assert shapes == {
        "model.layers.0.self_attn.q_proj": (128, 128),
        "model.layers.0.mlp.down_proj": (384, 128),
        "model.embed_tokens": (256, 128),
        "lm_head": (128, 256),
    }


def test_cli_requires_qualification_inputs_and_rejects_retired_flags():
    """Defect: executable CLI admits incomplete provenance and retired smoke modes."""
    parser = run_case.build_parser()
    required = {
        "--mode": "base",
        "--bundle-output": "/bundle",
        "--completion-output": "/completion",
        "--case-id": "case",
        "--revision-kind": "source",
        "--revision-sha": "a" * 40,
        "--architecture": "dense",
        "--precision": "bf16",
        "--cuda-graph": "off",
        "--model-path": "/model",
        "--checkpoint-manifest": "/checkpoint",
        "--prompts-file": "/prompts",
        "--fixture-manifest": "/fixtures",
    }
    args = [part for pair in required.items() for part in pair]
    assert parser.parse_args(args).mode == "base"
    for flag in required:
        filtered = [
            part
            for key, value in required.items()
            if key != flag
            for part in (key, value)
        ]
        with pytest.raises(SystemExit):
            parser.parse_args(filtered)
    for retired in ("--transitions", "--no-return-logprob", "--engine-kwarg"):
        with pytest.raises(SystemExit):
            parser.parse_args(args + [retired, "full"])
    assert parser.parse_args(args + ["--selection", "smoke"]).selection == "smoke"


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_switch_mixed_and_streaming_capture_all_requests(make_runner, mode):
    """Defect: replacing batches by one response or losing per-request selectors/scores."""
    runner = make_runner(mode)
    observed = runner.run_selected(
        (
            "switch.a",
            "switch.b",
            "switch.a-again",
            "mixed.base-a-b",
            "concurrent.stream",
            "concurrent.non-stream",
        )
    )
    assert [
        observed[name].adapter_state["active"]["name"]
        for name in ("switch.a", "switch.b", "switch.a-again")
    ] == ["policy-a", "policy-b", "policy-a"]
    assert observed["mixed.base-a-b"].output_ids == (10, 11, 10, 11, 10, 11, 10, 11)
    assert observed["concurrent.stream"] == observed["concurrent.non-stream"]
    assert len(observed["mixed.base-a-b"].selected_logits) == 8
    key = "adapter_path" if mode == "native_oft" else "lora_path"
    batch = next(
        call for call in runner.engine.calls if isinstance(call["input_ids"][0], list)
    )
    assert batch[key] == [
        None,
        "policy-a",
        "policy-b",
        None,
        "policy-a",
        "policy-b",
        None,
        "policy-a",
    ]


def test_oft_switch_setup_supports_wire_upsert(make_runner):
    class OFTControl(FakeControl):
        def load_tensors(self, name, tensors, config, *, upsert=False):
            if upsert and any(
                operation[:2] == ("load_path", name) for operation in self.operations
            ):
                raise ScenarioContractError("disk OFT adapter cannot be wire-upserted")
            return super().load_tensors(name, tensors, config, upsert=upsert)

    runner = make_runner("native_oft", control=OFTControl("native_oft"))
    runner.run_selected(("switch.a", "switch.b", "switch.a-again"))
    result = runner._load("policy-a", "tensors", "upsert", upsert=True)
    assert result.success


def test_capture_error_in_second_batch_response_leaves_no_partial_record(make_runner):
    """Defect: malformed later batch responses are silently discarded."""
    runner = make_runner("base")

    async def malformed_one(**kwargs):
        return {} if kwargs["input_ids"] == [11] else response()

    runner.engine.async_generate = malformed_one
    with pytest.raises(BundleValidationError):
        runner.run_selected(("concurrent.non-stream",))
    assert runner.observations == {}
    assert "step.complete" not in runner.diagnostics.getvalue()


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_fixture_builder_includes_globals_and_reuses_bytes(tmp_path, mode):
    """Defect: different mechanism fixtures are regenerated or omit global bindings."""
    from safetensors.numpy import load_file

    paths = run_case.build_fixture_set(
        tmp_path,
        {
            "hidden_size": 128,
            "vocab_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 384,
        },
        "dense",
        mode,
    )
    offset = int(mode == "native_oft")
    assert paths[str(2 + offset)] == paths["policy-a"]
    assert paths[str(3 + offset)] == paths["policy-b"]
    tensors = load_file(str(paths["policy-a"] / "adapter_model.safetensors"))
    assert any("model.embed_tokens" in name for name in tensors)
    assert any("lm_head" in name for name in tensors)
    if mode == "native_oft":
        assert set(tensors) == {
            "base_model.model.model.layers.0.self_attn.qkv_proj.oft_R",
            "base_model.model.model.layers.0.self_attn.o_proj.oft_R",
            "base_model.model.model.layers.0.mlp.gate_up_proj.oft_R",
            "base_model.model.model.layers.0.mlp.down_proj.oft_R",
            "base_model.model.model.embed_tokens.oft_R",
            "base_model.model.lm_head.oft_R",
        }
    before = (paths["policy-a"] / "adapter_model.safetensors").read_bytes()
    with pytest.raises(ValueError):
        run_case.build_fixture_set(
            tmp_path,
            {
                "hidden_size": 128,
                "vocab_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "intermediate_size": 384,
            },
            "dense",
            mode,
        )
    assert (paths["policy-a"] / "adapter_model.safetensors").read_bytes() == before


def test_help_does_not_import_gpu_runtime():
    """Defect: eager torch/sglang runtime imports break lightweight CLI help."""
    result = subprocess.run(
        [sys.executable, str(Path(run_case.__file__)), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "{base,native_lora,native_oft}" in result.stdout
    assert "--transitions" not in result.stdout


def test_late_engine_startup_is_still_owned_after_timeout(make_runner, monkeypatch):
    """Defect: timed-out startup returns later and leaves an unowned engine alive."""
    engine = FakeEngine()
    gate = threading.Event()
    stopped = threading.Event()

    def launch(spec):
        gate.wait()
        return engine

    def stop(owned):
        assert owned is engine
        stopped.set()

    monkeypatch.setattr(run_case, "launch_engine", launch)
    monkeypatch.setattr(run_case, "stop_engine", stop)
    runner = make_runner(
        "base", timeouts=run_case.Timeouts(startup=0.02, teardown=0.02)
    )
    runner.engine = runner.control = runner.sender = None
    try:
        with pytest.raises(run_case.OperationTimeout):
            runner.run_selected(("base.initial",))
    finally:
        gate.set()
    assert stopped.wait(0.3)


def test_retained_lease_cannot_start_from_an_already_finished_response(make_runner):
    """Defect: calling an already-completed request a leased upsert proves no drain."""
    runner = make_runner()
    runner.control.load_path("policy-a", "/a")

    async def finished(**kwargs):
        async def chunks():
            out = response()
            out["meta_info"]["finish_reason"] = {"type": "length"}
            yield out

        return chunks()

    runner.engine.async_generate = finished
    with pytest.raises(ScenarioContractError, match="finished"):
        runner.run_selected(("upsert.lease.begin",))


def test_cli_timeout_exits_despite_dependency_executor_join():
    """Defect: a timed-out native dependency pool keeps the process alive at exit."""
    script = """
import sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, sys.argv[1])
from adapter_equivalence import run_case
pool = ThreadPoolExecutor(max_workers=1)
pool.submit(threading.Event().wait)
run_case.main = lambda: 2
run_case.cli()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(run_case.__file__).parent.parent)],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 2


@pytest.mark.parametrize("boundary", ("control", "sender"))
def test_mutation_and_sender_deadlines_are_bounded(make_runner, boundary):
    """Defect: mutation or sender calls block outside the inference timeout guard."""
    runner = make_runner(
        timeouts=run_case.Timeouts(control=0.02, collective=0.02, teardown=0.02)
    )
    gate = threading.Event()
    if boundary == "control":

        def blocked(*args, **kwargs):
            gate.wait()
            return runner.control.result()

        runner.control.load_path = blocked
        selection = ("immediate.path.load",)
    else:

        def blocked(*args, **kwargs):
            gate.wait()
            return PAYLOAD

        runner.sender.broadcast_fixture = blocked
        selection = ("immediate.distributed.load",)
    started = time.monotonic()
    try:
        with pytest.raises(run_case.OperationTimeout):
            runner.run_selected(selection)
        assert time.monotonic() - started < 0.5
        assert runner.engine.closed and runner.sender.closed
        assert runner.observations == {}
    finally:
        runner.control.transfer_done.set()
        gate.set()


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_tensor_and_distributed_payloads_read_exact_fixture_files(
    tmp_path, monkeypatch, mode
):
    """Defect: tensor/distributed loads regenerate weights instead of reading path fixtures."""
    from safetensors.numpy import load_file

    paths = run_case.build_fixture_set(
        tmp_path,
        {
            "hidden_size": 128,
            "vocab_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 384,
        },
        "dense",
        mode,
    )
    fixture = paths["policy-a"]
    reads = []

    def tensor_loader(path, *, device):
        assert device == "cpu"
        reads.append(Path(path))
        return load_file(path)

    monkeypatch.setitem(
        sys.modules, "safetensors.torch", SimpleNamespace(load_file=tensor_loader)
    )
    tensors, config = run_case.load_fixture(fixture)
    payload = run_case.distributed_payload_for_fixture(fixture)
    assert reads == [fixture / "adapter_model.safetensors"] * 2
    assert payload.names == tuple(sorted(tensors))
    assert set(payload.dtypes) == {"float32"}
    assert (
        payload.config
        == config
        == json.loads((fixture / "adapter_config.json").read_text())
    )
    assert len(payload.shapes) == len(tensors)


@pytest.mark.parametrize("first_boundary", ("sender", "control"))
@pytest.mark.parametrize("cancelling", (False, True))
def test_concurrent_failures_preserve_the_first_observed_exception(
    make_runner, first_boundary, cancelling
):
    """Defect: unordered completed Futures select a later error over the original."""
    runner = make_runner()
    start = threading.Event()
    first_done = threading.Event()
    original = RuntimeError("original transfer failure")
    secondary = RuntimeError("secondary transfer failure")

    def fail(boundary):
        assert start.wait(0.5)
        if boundary != first_boundary:
            assert first_done.wait(0.5)
            raise secondary
        raise original

    runner.sender.broadcast_fixture = lambda *args, **kwargs: fail("sender")
    transfer = runner._start_distributed(
        "test-transfer", runner.fixtures["policy-a"], lambda *args: fail("control")
    )
    first_future = transfer[1] if first_boundary == "sender" else transfer[0]
    first_future.add_done_callback(lambda future: first_done.set())
    start.set()
    done, pending = wait(transfer[:2], timeout=0.5)
    assert len(done) == 2 and not pending
    with pytest.raises(RuntimeError) as caught:
        if cancelling:
            gate = fault_api().PhaseGate("fan-out")
            runner._finish_cancelled_transfer(transfer, gate)
        else:
            runner._finish_distributed(transfer, "test-transfer")
    assert caught.value is original
    if cancelling:
        assert gate.released.is_set()


def many_token_response(tokens, text="same"):
    result = response()
    result["output_ids"] = tokens
    result["text"] = text
    result["meta_info"]["output_token_logprobs"] = [
        (-0.1, token, "") for token in tokens
    ]
    result["meta_info"]["output_top_logprobs"] = [
        [(-0.1 - i, token + i, "") for i in range(5)] for token in tokens
    ]
    return result


def test_batch_observations_preserve_uneven_request_boundaries(make_runner):
    """Defect: identical flat tokens/text concealed different per-request lengths."""
    observations = []
    for partitions in ([[10], [20, 30]], [[10, 20], [30]]):
        runner = make_runner()
        runner.batches["batch-8"] = ["factual", "long-prefix"]

        async def generate(**kwargs):
            return [many_token_response(tokens) for tokens in partitions]

        runner.engine.async_generate = generate
        observations.append(runner.run_selected(("mixed.base-a-b",))["mixed.base-a-b"])
    assert observations[0].output_ids == observations[1].output_ids
    assert observations[0] != observations[1]
    assert observations[0].request_output_lengths == (1, 2)
    assert observations[1].request_output_lengths == (2, 1)
    assert observations[0].request_texts == ("same", "same")
    for observation in observations:
        assert (
            Observation.from_dict(json.loads(json.dumps(observation.to_dict())))
            == observation
        )


@pytest.mark.parametrize("during_drain", (False, True))
def test_lease_completion_preserves_a_control_failure_after_pending_check(
    make_runner, during_drain
):
    """Defect: a stopped update loop replaced the original failure by drain timeout."""
    engine = FakeEngine()
    fail_now = asyncio.Event()
    waiting = threading.Event()
    original = RuntimeError("update failed after pending observation")
    finalized = threading.Event()

    async def generate(**kwargs):
        async def chunks():
            try:
                yield response()
                if during_drain:
                    fail_now.set()
                    await asyncio.Event().wait()
                yield response()
            finally:
                finalized.set()

        return chunks()

    engine.async_generate = generate

    class Control(FakeControl):
        def load_distributed(self, *args, **kwargs):
            async def update():
                waiting.set()
                await fail_now.wait()
                raise original

            return engine.loop.run_until_complete(update())

        def observe_lease_wait(self, name, adapter_id):
            future = Future()
            assert waiting.wait(0.5)
            future.set_result(self.inspect_state())
            return future

    control = Control("native_lora")
    control.load_path("policy-a", "/a")
    runner = make_runner(
        engine=engine,
        control=control,
        timeouts=run_case.Timeouts(inference=0.03, collective=0.2, teardown=0.03),
    )
    runner.sender.broadcast_fixture = lambda *args, **kwargs: PAYLOAD
    steps = {step.name: step for step in lifecycle_steps("native_lora")}
    runner.execute(steps["upsert.lease.begin"])
    runner.execute(steps["upsert.while-leased"])
    if not during_drain:
        engine.loop.call_soon_threadsafe(fail_now.set)
        wait((runner.upsert_future,), timeout=0.5)
    with pytest.raises(RuntimeError) as caught:
        runner.run_selected(("upsert.lease.complete",))
    assert caught.value is original
    assert finalized.wait(0.3)


def test_upsert_phase_observation_detects_unregister_then_restore(make_runner):
    """Defect: snapshots before update execution missed unregister while leased."""
    engine = FakeEngine()
    proceed = asyncio.Event()
    blocked = asyncio.Event()
    drained = asyncio.Event()

    async def generate(**kwargs):
        async def chunks():
            yield response()
            proceed.set()
            await blocked.wait()
            yield response()
            drained.set()

        return chunks()

    engine.async_generate = generate

    class Control(FakeControl):
        def load_distributed(self, *args, **kwargs):
            async def update():
                await proceed.wait()
                old = self.records.pop("policy-a")
                blocked.set()
                await drained.wait()
                self.records["policy-a"] = old
                return self.result()

            return engine.loop.run_until_complete(update())

        def observe_lease_wait(self, name, adapter_id):
            async def observe():
                proceed.set()
                await blocked.wait()
                return self.inspect_state()

            return asyncio.run_coroutine_threadsafe(observe(), engine.loop)

    control = Control("native_lora")
    control.load_path("policy-a", "/a")
    runner = make_runner(
        engine=engine,
        control=control,
        timeouts=run_case.Timeouts(inference=0.2, collective=0.2, teardown=0.03),
    )
    runner.sender.broadcast_fixture = lambda *args, **kwargs: PAYLOAD
    with pytest.raises(ScenarioContractError, match="registered|lease"):
        runner.run_selected(
            ("upsert.lease.begin", "upsert.while-leased", "upsert.lease.complete")
        )


def fault_api():
    assert (
        importlib.util.find_spec("adapter_equivalence.faults") is not None
    ), "deterministic fault hooks are missing"
    return importlib.import_module("adapter_equivalence.faults")


@pytest.mark.parametrize("rank", (0, 1))
def test_fault_replaces_one_collected_result_and_restores_exact_communicator(rank):
    """Catch mutation of original replies, wrong rank replacement, and leaked hooks."""
    api = fault_api()
    replies = [
        SimpleNamespace(success=True, message="ok", active_adapter_version="3")
        for _ in range(2)
    ]

    async def communicator(request):
        return replies

    manager = SimpleNamespace(update_adapter_from_distributed_communicator=communicator)
    controller = api.FaultController(manager)
    failure = api.RankFailure(rank, "injected stage failure")
    with controller.wrap("stage", failure=failure):
        observed = asyncio.run(
            manager.update_adapter_from_distributed_communicator(object())
        )
        assert [item.success for item in observed] == (
            [False, True] if rank == 0 else [True, False]
        )
        assert observed[rank].message == f"rank {rank}: injected stage failure"
        assert observed[rank].active_adapter_version == "3"
        assert controller.injected == [failure]
        assert all(item.success for item in replies)
    assert manager.update_adapter_from_distributed_communicator is communicator


def test_cancelled_fault_gate_waits_for_real_irreversible_backend_completion():
    """Catch cancellation acknowledged while the real shielding helper still works."""
    api = fault_api()
    from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

    async def scenario():
        events = []

        async def communicator(request):
            events.append("fanout")
            return [SimpleNamespace(success=True, message="ok")]

        manager = SimpleNamespace(activate_adapter_version_communicator=communicator)
        gate = api.PhaseGate("publication")
        controller = api.FaultController(manager)

        async def backend():
            await manager.activate_adapter_version_communicator(object())
            events.append("published")

        with controller.wrap("activate", gate=gate):
            task = asyncio.create_task(finish_irreversible_update(backend))
            await gate.wait_entered(0.5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            gate.release()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert events == ["fanout", "published"]
        assert manager.activate_adapter_version_communicator is communicator

    asyncio.run(scenario())


@pytest.mark.parametrize("cause", ("communicator", "body", "missing_rank"))
def test_fault_failure_restores_exact_original_identity(cause):
    api = fault_api()

    async def communicator(request):
        if cause == "communicator":
            raise RuntimeError("real communicator failed")
        return [SimpleNamespace(success=True, message="ok")]

    manager = SimpleNamespace(discard_adapter_stage_communicator=communicator)
    with pytest.raises((RuntimeError, ScenarioContractError)):
        with api.FaultController(manager).wrap(
            "rollback",
            failure=api.RankFailure(1 if cause == "missing_rank" else 0, "injected"),
        ):
            asyncio.run(manager.discard_adapter_stage_communicator(object()))
            raise RuntimeError("body failed")
    assert manager.discard_adapter_stage_communicator is communicator


@pytest.mark.parametrize(
    "transition,message,code",
    (
        (
            "reject.duplicate",
            "LoRA adapter version 3 must be newer than active version 3.",
            "duplicate_version",
        ),
        (
            "reject.stale",
            "LoRA adapter version 2 must be newer than active version 3.",
            "stale_version",
        ),
        (
            "reject.wrong-id",
            "Requested adapter_id 'wrong-id' does not match expected adapter_id 'live-policy-a'",
            "wrong_id",
        ),
        (
            "reject.wrong-name",
            "Cannot activate name=missing-policy version=4; no native LoRA stage is pending",
            "wrong_name",
        ),
    ),
)
def test_adverse_preflight_classifies_actual_rejection(
    make_runner, transition, message, code
):
    runner = make_runner()
    runner.control.load_path("policy-a", "/a")
    requests = []

    def reject(identity, *args):
        requests.append(identity)
        return ControlResult(False, message, {}, None, None)

    runner.control.stage = runner.control.activate = reject
    observed = runner.run_selected((transition,))[transition]
    assert observed.error["code"] == code
    assert len(requests) == 1
    assert requests[0].name == (
        "missing-policy" if code == "wrong_name" else "policy-a"
    )
    assert (
        requests[0].adapter_id != "live-policy-a"
        if code == "wrong_id"
        else requests[0].adapter_id == "live-policy-a"
    )
    assert runner.sender.broadcasts == []


@pytest.mark.parametrize(
    "result",
    (
        ControlResult(False, "disk failure", {}, None, None),
        ControlResult(
            False,
            "LoRA adapter version 2 must be newer than active version 3.",
            {},
            None,
            None,
        ),
        ControlResult(True, "ok", {}, None, None),
    ),
)
def test_wrong_product_rejection_cannot_satisfy_declared_error(make_runner, result):
    runner = make_runner()
    runner.control.load_path("policy-a", "/a")
    runner.control.stage = lambda *args: result
    with pytest.raises((ScenarioContractError, BundleValidationError)):
        runner.run_selected(("reject.duplicate",))
    assert not runner.observations


def test_returned_fault_marker_cannot_echo_the_expected_action(make_runner):
    runner = make_runner()
    step = next(
        step for step in lifecycle_steps("native_lora") if step.name == "failure.update"
    )
    result = ControlResult(
        False, "rank 0: adapter-harness:activation_failure", {}, None, None
    )
    assert run_case.classify_rejection(result) == "activation_failure"
    with pytest.raises(BundleValidationError, match="does not match"):
        runner._rejection_observation(result, step)


def test_fault_classifier_rejects_extra_unknown_marker():
    result = ControlResult(
        False,
        "adapter-harness:update_failure adapter-harness:unexpected_failure",
        {},
        None,
        None,
    )
    with pytest.raises(ScenarioContractError):
        run_case.classify_rejection(result)


@pytest.mark.parametrize(
    "message",
    (
        "rank 0: adapter-harness:update_failure | rank 1: adapter-harness:activation_failure",
        "rank 0: adapter-harness:update_failure | rank 0: adapter-harness:update_failure",
        "rank 0: adapter-harness:update_failure; restart required",
    ),
)
def test_fault_classifier_rejects_ambiguous_markers_or_failed_cleanup(message):
    with pytest.raises(ScenarioContractError):
        run_case.classify_rejection(ControlResult(False, message, {}, None, None))


@pytest.mark.parametrize("kind,key", (("lora", "r"), ("oft", "oft_block_size")))
@pytest.mark.parametrize("rollback_success", (True, False))
@pytest.mark.parametrize("tp_size", (1, 2, 4))
def test_invalid_config_classifies_real_stage_rollback(
    monkeypatch, make_runner, kind, key, rollback_success, tp_size
):
    """Consensus joins every failed rank before the tokenizer rolls back the stage."""
    helper_path = Path(__file__).parents[1] / "adapter_sync/test_stage_discard.py"
    spec = importlib.util.spec_from_file_location(
        "rollback_runner_helpers", helper_path
    )
    helpers = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, helpers)
    spec.loader.exec_module(helpers)
    runtime = helpers.runtime.__wrapped__(monkeypatch)
    manager, backend, _ = helpers.make_tokenizer(kind, runtime)
    request = helpers.request_for(kind)
    consensus = helpers.load_module(
        helpers.SRT / "managers/scheduler_components/tp_update_consensus.py"
    )

    class Distributed:
        def get_world_size(self, *, group):
            return tp_size

        def all_gather_object(self, results, local_result, *, group):
            results[:] = [local_result] * tp_size

    stage_success, stage_message, _ = consensus.gather_tp_update_result(
        distributed=Distributed(),
        group=object(),
        success=False,
        message=str(KeyError(key)),
        version=None,
    )
    assert stage_success is False

    async def discard(request):
        return [SimpleNamespace(success=rollback_success, message="rollback reply")]

    manager.discard_adapter_stage_communicator = discard

    async def rollback():
        await backend.reserve_stage(request)
        return await manager._rollback_failed_adapter_stage(
            backend, request, stage_message
        )

    success, message = asyncio.run(rollback())
    result = ControlResult(success, message, {}, None, None)
    if rollback_success:
        assert run_case.classify_rejection(result) == "invalid_config"
        assert getattr(manager, f"pending_{kind}_stage") is None
        runner = make_runner(mode="native_" + kind)
        wrong_step = next(
            step
            for step in lifecycle_steps("native_" + kind)
            if step.name == "reject.duplicate"
        )
        with pytest.raises(BundleValidationError, match="does not match"):
            runner._rejection_observation(result, wrong_step)
    else:
        with pytest.raises(ScenarioContractError):
            run_case.classify_rejection(result)


@pytest.mark.parametrize("key", ("r", "oft_block_size"))
@pytest.mark.parametrize("prefix", ("", "TP rank 0: "))
@pytest.mark.parametrize("suffix", ("", "; stage rollback succeeded"))
def test_invalid_config_preserves_bare_and_single_rank_compatibility(
    key, prefix, suffix
):
    result = ControlResult(False, f"{prefix}'{key}'{suffix}", {}, None, None)
    assert run_case.classify_rejection(result) == "invalid_config"


@pytest.mark.parametrize(
    "message",
    (
        "disk failure: 'r'",
        "TP rank 0: 'r'; stage rollback succeeded; restart required",
        "TP rank 0: 'oft_block_size'; stage rollback succeeded; unrelated failure",
        "TP rank 0: 'r'; stage rollback failed: disk failure; restart required",
        "TP rank 0: 'r' | TP rank 1: unrelated failure; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 1: 'oft_block_size'; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 0: 'r'; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 2: 'r'; stage rollback succeeded",
        "TP rank 1: 'r' | TP rank 0: 'r'; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 2: 'r' | TP rank 1: 'r'; stage rollback succeeded",
        "TP rank 1: 'r'; stage rollback succeeded",
        "TP rank 00: 'r'; stage rollback succeeded",
        "TP rank 0: 'r'|TP rank 1: 'r'; stage rollback succeeded",
        "TP rank 0: 'r' || TP rank 1: 'r'; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 1: 'r' | ; stage rollback succeeded",
        "'r' | TP rank 1: 'r'; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 1: 'r'",
        "TP rank 0: 'r' | TP rank 1: 'r'; stage rollback failed: disk failure; restart required",
        "TP rank 0: 'r' | TP rank 1: 'r'; stage rollback succeeded; restart required",
        "TP rank 0: 'r' | TP rank 1: 'r'; stage rollback succeeded; unrelated failure",
        "TP rank 0: 'r' | TP rank 1: 'r'; stage rollback succeeded; stage rollback succeeded",
        "TP rank 0: 'r' | TP rank 1: 'r'; stage rollback succeeded\n",
        "TP rank 0: 'r'; stage rollback succeeded\n",
    ),
)
def test_invalid_config_classifier_rejects_unrelated_or_ambiguous_cleanup(message):
    with pytest.raises(ScenarioContractError):
        run_case.classify_rejection(ControlResult(False, message, {}, None, None))


def test_expected_distributed_rejection_still_requires_sender_completion(make_runner):
    runner = make_runner()

    def reject(payload, group):
        runner.control.transfer_started.set()
        assert runner.control.transfer_done.wait(0.5)
        return ControlResult(False, "rank 0: rejected", {}, None, None)

    result = runner._distributed_call(
        "adverse",
        runner.fixtures["policy-a"],
        reject,
        accept_rejection=True,
    )
    assert result.success is False
    assert runner.sender.broadcasts == [("adverse", runner.fixtures["policy-a"])]


@pytest.mark.parametrize("wrong_id", (False, True))
@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_retry_unload_checks_the_actual_wire_identity(make_runner, wrong_id, mode):
    engine = FakeEngine()
    kind = "lora" if mode == "native_lora" else "oft"
    field = "lora" if kind == "lora" else "adapter"
    tombstones = [
        {"name": "policy-a", "id": "exact-failed-id", "version": "3", "pinned": False}
    ]

    async def communicator(request):
        return [SimpleNamespace(success=True, error_message="", loaded_adapters={})]

    engine.tokenizer_manager = SimpleNamespace(
        **{f"update_{kind}_adapter_communicator": communicator}
    )

    class RetryControl(FakeControl):
        def unload(self, name):
            request = SimpleNamespace(
                **{
                    field + "_name": name,
                    field + "_id": "different-id" if wrong_id else "exact-failed-id",
                }
            )
            engine.loop.run_until_complete(
                getattr(
                    engine.tokenizer_manager, f"update_{kind}_adapter_communicator"
                )(request)
            )
            tombstones.clear()
            return self.result()

        def inspect_state(self):
            return dict(super().inspect_state(), tombstoned=list(tombstones))

    runner = make_runner(mode, engine=engine, control=RetryControl(mode))
    if wrong_id:
        with pytest.raises(ScenarioContractError, match="retry.*identity"):
            runner.run_selected(("failure.unload.retry",))
        assert not runner.observations
        assert len(tombstones) == 1
    else:
        result = runner.run_selected(("failure.unload.retry",))["failure.unload.retry"]
        assert result.adapter_state["tombstoned"] == ()
    assert (
        getattr(engine.tokenizer_manager, f"update_{kind}_adapter_communicator")
        is communicator
    )


@pytest.mark.parametrize(
    "transition,code",
    (
        ("failure.update", "update_failure"),
        ("failure.activation", "activation_failure"),
        ("failure.unload", "unload_failure"),
    ),
)
@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_runner_records_only_real_injected_failure(make_runner, transition, code, mode):
    engine = FakeEngine()
    kind = "lora" if mode == "native_lora" else "oft"
    replies = [SimpleNamespace(success=True, message="ok")]

    async def communicator(request):
        return replies

    engine.tokenizer_manager = SimpleNamespace(
        update_adapter_from_distributed_communicator=communicator,
        activate_adapter_version_communicator=communicator,
    )
    setattr(
        engine.tokenizer_manager, f"update_{kind}_adapter_communicator", communicator
    )

    class Control(FakeControl):
        def __init__(self, mode):
            super().__init__(mode)
            self.tombstones = []

        def stage(self, identity, payload, group):
            self.transfer_started.set()
            assert self.transfer_done.wait(0.5)
            self.transfer_done.clear()
            results = engine.loop.run_until_complete(
                engine.tokenizer_manager.update_adapter_from_distributed_communicator(
                    identity
                )
            )
            if all(item.success for item in results):
                self.staged = {
                    "name": identity.name,
                    "id": identity.adapter_id,
                    "version": identity.version,
                    "pinned": False,
                }
            return ControlResult(
                all(item.success for item in results),
                " | ".join(item.message for item in results),
                {},
                None,
                None,
            )

        def activate(self, identity):
            results = engine.loop.run_until_complete(
                engine.tokenizer_manager.activate_adapter_version_communicator(identity)
            )
            return ControlResult(
                all(item.success for item in results),
                " | ".join(item.message for item in results),
                {},
                None,
                None,
            )

        def unload(self, name):
            results = engine.loop.run_until_complete(
                getattr(
                    engine.tokenizer_manager, f"update_{kind}_adapter_communicator"
                )(SimpleNamespace(name=name))
            )
            self.tombstones = [self.records.pop(name)]
            return ControlResult(
                all(item.success for item in results),
                " | ".join(item.message for item in results),
                {},
                None,
                None,
            )

        def inspect_state(self):
            state = super().inspect_state()
            state["cache_identity"].update(
                {record["name"]: record["id"] for record in self.tombstones}
            )
            return dict(state, tombstoned=list(self.tombstones))

    control = Control(mode)
    control.load_path("policy-a", "/a")
    runner = make_runner(mode, engine=engine, control=control)
    observed = runner.run_selected((transition,))[transition]
    assert observed.error["code"] == code
    assert f"rank 0: adapter-harness:{code}" in observed.error["message"]
    assert all(reply.success for reply in replies)
    assert len(runner.sender.broadcasts) == (0 if transition == "failure.unload" else 1)


@pytest.mark.parametrize(
    "transition,message",
    (
        ("reject.invalid-config", "'r'"),
        (
            "reject.unsupported-target",
            "LoRA adapter rejected-policy with rank 8 is incompatible with the current LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within the configured `--max-lora-rank` and that the target modules are included in `--lora-target-modules`.",
        ),
    ),
)
def test_invalid_adapter_payload_reaches_product_validation(
    make_runner, transition, message
):
    runner = make_runner()
    runner.control.load_path("policy-a", "/a")
    observed_configs = []

    def reject_stage(identity, payload, group):
        observed_configs.append(payload.config)
        runner.control.transfer_started.set()
        assert runner.control.transfer_done.wait(0.5)
        return ControlResult(False, message, {}, None, None)

    def reject_tensors(name, tensors, config):
        assert name == "rejected-policy"
        observed_configs.append(config)
        return ControlResult(False, message, {}, None, None)

    runner.control.stage = reject_stage
    runner.control.load_tensors = reject_tensors
    observation = runner.run_selected((transition,))[transition]
    assert observation.error["code"] == (
        "invalid_config" if transition.endswith("config") else "unsupported_target"
    )
    assert observed_configs == (
        [{}]
        if transition.endswith("config")
        else [{"r": 8, "target_modules": ["adapter_harness_unsupported_target"]}]
    )


def test_oft_unsupported_target_reaches_tensor_name_validation(
    make_runner, monkeypatch
):
    import ast

    path = (
        Path(run_case.__file__).parents[3]
        / "python/sglang/srt/oft/streamed_weight_loader.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_streamed_oft_tensor_groups"
    )
    namespace = {
        "_partition_expert_oft_tensors": lambda tensors, **kwargs: ({}, {}, tensors)
    }
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.utils",
        SimpleNamespace(get_layer_id=lambda name: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.oft.mem_pool",
        SimpleNamespace(normalize_merged_oft_weights=lambda tensors, **kwargs: tensors),
    )
    manager = SimpleNamespace(
        memory_pool=SimpleNamespace(tp_rank=0, R_buffer={}), adapter_modules=[]
    )
    runner = make_runner("native_oft")

    def load(name, tensors, config):
        assert name == "rejected-policy"
        plan, error = namespace["_resolve_streamed_oft_tensor_groups"](
            manager, list(tensors.items()), 32
        )
        return ControlResult(plan is not None, error, {}, None, None)

    runner.control.load_tensors = load
    result = runner.run_selected(("reject.unsupported-target",))[
        "reject.unsupported-target"
    ]
    assert result.error["code"] == "unsupported_target"
    assert (
        result.error["message"]
        == "Unresolved OFT tensor names: adapter_harness_unsupported_target.oft_R"
    )


def test_runner_cancels_caller_and_waits_for_irreversible_publication(make_runner):
    from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

    api = fault_api()
    engine = FakeEngine()
    published = []

    class Manager:
        async def update_adapter_from_distributed(self, obj):
            async def backend():
                await self.update_adapter_from_distributed_communicator(obj)
                published.append("finished")

            return await finish_irreversible_update(backend)

        async def update_adapter_from_distributed_communicator(self, request):
            return [SimpleNamespace(success=True, message="ok")]

    manager = engine.tokenizer_manager = Manager()
    runner = make_runner(engine=engine)
    gate = api.PhaseGate("publication")
    with api.FaultController(manager).wrap("stage", gate=gate):
        future = runner.engine_pool.submit(
            lambda: engine.loop.run_until_complete(
                manager.update_adapter_from_distributed(object())
            )
        )
        cancellation = runner._cancel_at_gate(future, gate)
    assert isinstance(cancellation, asyncio.CancelledError)
    assert published == ["finished"]
    assert gate.exited.is_set()


def test_cancellation_gate_preserves_failure_before_phase(make_runner):
    runner = make_runner()
    original = RuntimeError("original operation failed before phase")

    def fail():
        raise original

    future = runner.engine_pool.submit(fail)
    with pytest.raises(RuntimeError) as caught:
        runner._cancel_at_gate(future, fault_api().PhaseGate("fan-out"))
    assert caught.value is original


@pytest.mark.parametrize("timing", ("before_gate", "waiting_for_gate", "cancel_ack"))
def test_cancelled_transfer_preserves_sender_failure_with_pending_control(
    make_runner, monkeypatch, timing
):
    """Catch a blocked engine/cancel callback masking an earlier sender exception."""
    runner = make_runner(timeouts=run_case.Timeouts(control=0.15, collective=0.5))
    gate = fault_api().PhaseGate("fan-out")
    original = RuntimeError("sender failed before cancellation completed")
    fail_sender = threading.Event()
    timer = None

    def control(*args):
        assert gate.released.wait(1)
        return runner.control.result()

    def sender(*args, **kwargs):
        assert fail_sender.wait(1)
        raise original

    runner.sender.broadcast_fixture = sender
    transfer = runner._start_distributed(
        "cancel-test", runner.fixtures["policy-a"], control
    )
    if timing == "before_gate":
        fail_sender.set()
        assert wait((transfer[1],), timeout=0.5)[0]
    elif timing == "waiting_for_gate":
        timer = threading.Timer(0.02, fail_sender.set)
        timer.start()
    else:
        gate.entered.set()
        schedule = runner.engine.loop.call_soon_threadsafe

        def stalled_callback(callback):
            schedule(callback)
            fail_sender.set()

        monkeypatch.setattr(
            runner.engine.loop, "call_soon_threadsafe", stalled_callback
        )
    try:
        with pytest.raises(RuntimeError) as caught:
            runner._finish_cancelled_transfer(transfer, gate)
        assert caught.value is original
        assert transfer[3][0] is original
        assert not runner.timed_out
        assert gate.released.is_set()
    finally:
        fail_sender.set()
        gate.release()
        if timer is not None:
            timer.join(0.5)
            assert not timer.is_alive()
        assert not wait(transfer[:2], timeout=0.5)[1]
        if timing == "cancel_ack":
            callback_errors = []
            runner.engine.loop.set_exception_handler(
                lambda loop, context: callback_errors.append(context)
            )
            runner.engine.loop.run_until_complete(asyncio.sleep(0))
            assert not callback_errors


@pytest.mark.parametrize("sender_fails", (False, True))
def test_cancelled_transfer_watches_sender_while_native_cancellation_finishes(
    make_runner, sender_fails
):
    """Catch sender failure masked by the cancelled caller's shielded drain timeout."""
    from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

    engine = FakeEngine()
    gate = fault_api().PhaseGate("fan-out")
    released = threading.Event()
    finish = asyncio.Event()
    original = RuntimeError("sender failed during cancellation drain")

    class Manager:
        async def update_adapter_from_distributed(self, obj):
            async def backend():
                await gate.block(obj)
                released.set()
                if sender_fails:
                    await finish.wait()

            return await finish_irreversible_update(backend)

    manager = engine.tokenizer_manager = Manager()
    runner = make_runner(
        engine=engine, timeouts=run_case.Timeouts(control=0.15, collective=0.5)
    )

    def sender(*args, **kwargs):
        assert released.wait(1)
        if sender_fails:
            raise original
        return PAYLOAD

    runner.sender.broadcast_fixture = sender
    transfer = runner._start_distributed(
        "cancel-drain",
        runner.fixtures["policy-a"],
        lambda *args: engine.loop.run_until_complete(
            manager.update_adapter_from_distributed(object())
        ),
    )
    try:
        if sender_fails:
            with pytest.raises(RuntimeError) as caught:
                runner._finish_cancelled_transfer(transfer, gate)
            assert caught.value is original
        else:
            cancellation = runner._finish_cancelled_transfer(transfer, gate)
            assert isinstance(cancellation, asyncio.CancelledError)
            assert transfer[0].exception() is cancellation
            assert transfer[1].result() == PAYLOAD
        assert not runner.timed_out
        assert gate.exited.is_set()
    finally:
        gate.release()
        engine.loop.call_soon_threadsafe(finish.set)
        assert not wait(transfer[:2], timeout=0.5)[1]
        assert not asyncio.all_tasks(engine.loop)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_registry_actions_use_real_limit_trigger_and_reload_before_delete(
    make_runner, mode
):
    engine = FakeEngine()
    args = make_dataclass(
        "FrozenRegistryArgs",
        [
            ("max_loaded_loras", object, field(default=None)),
            ("max_loaded_ofts", object, field(default=None)),
        ],
        frozen=True,
    )()
    engine.tokenizer_manager = SimpleNamespace(server_args=args)
    cache = {}
    events = []

    class LRUControl(FakeControl):
        def load_path(self, name, path, *, pinned=False):
            events.append(("load", name, path))
            result = super().load_path(name, path, pinned=pinned)
            cache[name] = self.records[name]["id"]
            limit = (
                engine.tokenizer_manager.server_args.max_loaded_loras
                if mode == "native_lora"
                else engine.tokenizer_manager.server_args.max_loaded_ofts
            )
            if limit is not None and len(self.records) > limit:
                victim = next(iter(self.records))
                events.append(("lru", victim))
                self.records.pop(victim)
            return result

        def unload(self, name):
            events.append(("delete", name))
            result = super().unload(name)
            cache.pop(name)
            return result

        def inspect_state(self):
            return dict(super().inspect_state(), cache_identity=dict(cache))

    control = LRUControl(mode)
    control.load_path("policy-a", "/a")
    runner = make_runner(mode, engine=engine, control=control)
    observed = runner.run_selected(("registry.fill", "registry.evict", "unload.final"))
    evicted = observed["registry.evict"].adapter_state
    assert [record["name"] for record in evicted["registered"]] == ["policy-b"]
    assert evicted["cache_identity"] == {
        "policy-a": "live-policy-a",
        "policy-b": "live-policy-b",
    }
    assert [event[:2] for event in events][-3:] == [
        ("delete", "policy-b"),
        ("load", "policy-a"),
        ("delete", "policy-a"),
    ]
    assert events[-2][2] == str(runner.fixtures["policy-a"])
    assert ("lru", "policy-a") in events
    assert observed["unload.final"].adapter_state["cache_identity"] == {}
    assert args.max_loaded_loras is None and args.max_loaded_ofts is None
    assert engine.tokenizer_manager.server_args is args
    assert "cleanup.reload-evicted" in runner.diagnostics.getvalue()
    with pytest.raises(RuntimeError, match="eviction failed"):
        with runner._registry_limit(2):
            other = "max_loaded_ofts" if mode == "native_lora" else "max_loaded_loras"
            assert getattr(engine.tokenizer_manager.server_args, other) is None
            raise RuntimeError("eviction failed")
    assert engine.tokenizer_manager.server_args is args


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize("transition", ("cancel.fan-out", "cancel.rollback"))
def test_runner_cancellation_uses_real_stage_and_rollback_code(
    make_runner, monkeypatch, mode, transition
):
    """Exercise actual tokenizer/backends; only worker IPC and ref values are doubled."""
    helper_path = Path(__file__).parents[1] / "adapter_sync/test_stage_discard.py"
    spec = importlib.util.spec_from_file_location(
        "stage_discard_runner_helpers", helper_path
    )
    helpers = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, helpers)
    spec.loader.exec_module(helpers)
    runtime = helpers.runtime.__wrapped__(monkeypatch)
    kind = "lora" if mode == "native_lora" else "oft"
    manager, backend, old = helpers.make_tokenizer(kind, runtime)
    setattr(manager, "failed_" + kind + "_unloads", {})
    sys.modules[
        "sglang.srt.managers.io_struct"
    ].UpdateAdapterFromDistributedReqInput = SimpleNamespace
    events = []

    async def stage(request):
        events.append("real-stage")
        return [SimpleNamespace(success=True, message="staged")]

    async def rollback(request):
        events.append(("real-rollback", request.adapter_id, request.adapter_version))
        return [SimpleNamespace(success=True, message="discarded")]

    manager.update_adapter_from_distributed_communicator = stage
    manager.discard_adapter_stage_communicator = rollback
    engine = FakeEngine()
    engine.tokenizer_manager = manager
    from adapter_equivalence.server import make_adapter_control

    control = make_adapter_control(mode, engine)
    runner = make_runner(mode, engine=engine, control=control)
    runner.sender.broadcast_fixture = lambda *args, **kwargs: PAYLOAD
    before = runner.state()
    observation = runner.run_selected((transition,))[transition]
    assert observation.error is None
    assert runner.state() == before
    assert events == ["real-stage", ("real-rollback", "id-a", "4")]
    assert getattr(manager, "pending_" + kind + "_stage") is None
    assert not backend.lifecycle_lock.locked()


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_lease_drain_cancellation_never_starts_sender(make_runner, mode):
    engine = FakeEngine()
    started = threading.Event()
    cancelled = threading.Event()

    class Manager:
        async def load_lora_adapter_from_distributed(self, obj):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def load_oft_adapter_from_distributed(self, obj):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    engine.tokenizer_manager = Manager()

    class Control(FakeControl):
        def load_distributed(self, name, payload, group, *, upsert=False):
            assert upsert is True
            kind = "lora" if mode == "native_lora" else "oft"
            request = SimpleNamespace(
                **{("lora_name" if kind == "lora" else "adapter_name"): name}
            )
            return engine.loop.run_until_complete(
                getattr(
                    engine.tokenizer_manager, f"load_{kind}_adapter_from_distributed"
                )(request)
            )

        def observe_lease_wait(self, name, uid):
            assert started.wait(0.5)
            future = Future()
            future.set_result(self.inspect_state())
            return future

    runner = make_runner(mode, engine=engine, control=Control(mode))
    result = runner.run_selected(("cancel.lease-drain",))["cancel.lease-drain"]
    assert result.adapter_state["active"]["name"] == "policy-a"
    assert runner.lease is None
    assert cancelled.is_set()
    assert not runner.sender.broadcasts


def test_lease_cancellation_does_not_mask_update_failure_before_barrier(make_runner):
    original = RuntimeError("update failed before lease barrier")
    runner = make_runner(timeouts=run_case.Timeouts(control=0.02, teardown=0.02))

    def fail(*args, **kwargs):
        raise original

    runner.control.load_distributed = fail
    runner.control.observe_lease_wait = lambda *args: Future()
    with pytest.raises(RuntimeError) as caught:
        runner.run_selected(("cancel.lease-drain",))
    assert caught.value is original


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_paused_request_is_rejected_then_resumed_and_drained(
    make_runner, monkeypatch, mode
):
    runner = make_runner(mode)
    runner.control.load_path("policy-a", "/a")
    events = []

    class Manager:
        async def pause_generation(self, request):
            assert request.mode == "in_place"
            events.append("pause")

        async def continue_generation(self, request):
            events.append("continue")

    runner.engine.tokenizer_manager = Manager()
    monkeypatch.setattr(
        run_case,
        "manager_request",
        lambda name, **kw: SimpleNamespace(**kw),
        raising=False,
    )

    def reject(identity):
        assert events == ["pause"]
        events.append("reject")
        return ControlResult(
            False,
            "Cannot activate adapter weights while paused requests are still active; continue generation or abort those requests before retrying.",
            {},
            None,
            None,
        )

    runner.control.activate = reject
    result = runner.run_selected(
        ("pause.retain-kv", "pause.activate-rejected", "pause.resume")
    )
    assert events == ["pause", "reject", "continue"]
    assert result["pause.resume"].output_ids == (11,)
    assert result["pause.activate-rejected"].error["code"] == "paused_requests_active"
    assert runner.lease is None


def test_fault_gate_filters_eviction_from_the_shared_load_communicator():
    api = fault_api()
    requests = []

    async def communicator(request):
        requests.append(request.lora_name)
        return [SimpleNamespace(success=True, error_message="")]

    manager = SimpleNamespace(update_lora_adapter_communicator=communicator)
    gate = api.PhaseGate("eviction")
    gate.release()
    with api.FaultController(manager).wrap(
        "unload", gate=gate, when=lambda request: request.lora_name == "victim"
    ):
        asyncio.run(
            manager.update_lora_adapter_communicator(SimpleNamespace(lora_name="new"))
        )
        assert not gate.entered.is_set()
        asyncio.run(
            manager.update_lora_adapter_communicator(
                SimpleNamespace(lora_name="victim")
            )
        )
        assert gate.request.lora_name == "victim"
    assert requests == ["new", "victim"]


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_cancel_publication_preserves_other_adapter_after_real_activation(
    make_runner, monkeypatch, mode
):
    helper_path = Path(__file__).parents[1] / "adapter_sync/test_stage_discard.py"
    spec = importlib.util.spec_from_file_location(
        "activation_runner_helpers", helper_path
    )
    helpers = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, helpers)
    spec.loader.exec_module(helpers)
    runtime = helpers.runtime.__wrapped__(monkeypatch)
    kind = "lora" if mode == "native_lora" else "oft"
    manager, backend, old = helpers.make_tokenizer(kind, runtime)
    setattr(manager, f"failed_{kind}_unloads", {})
    activation = helpers.load_class(
        "managers/tokenizer_control_mixin.py",
        "TokenizerControlMixin",
        {"activate_adapter_version"},
        FanOutCommunicator=runtime[1].FanOutCommunicator,
    )
    type(manager).activate_adapter_version = activation.activate_adapter_version
    manager.is_pause_cond, manager.is_pause = asyncio.Condition(), False
    manager.model_update_lock = SimpleNamespace(writer_lock=asyncio.Lock())
    io_struct = sys.modules["sglang.srt.managers.io_struct"]
    io_struct.UpdateAdapterFromDistributedReqInput = SimpleNamespace
    io_struct.ActivateAdapterVersionReqInput = SimpleNamespace
    prefix = "lora" if kind == "lora" else "peft"
    registry = getattr(manager, prefix + "_registry")
    cache = getattr(manager, prefix + "_ref_cache")
    names = (
        ("lora_name", "lora_id", "version")
        if kind == "lora"
        else ("adapter_name", "adapter_id", "adapter_version")
    )

    async def refresh(ref):
        registry._registry[getattr(ref, names[0])] = ref

    registry.refresh = refresh

    async def stage(request):
        assert int(request.adapter_version) > int(mode == "native_oft")
        return [SimpleNamespace(success=True, message="staged")]

    async def activate(request):
        return [
            SimpleNamespace(
                success=True,
                message="activated",
                active_adapter_version=request.adapter_version,
            )
        ]

    manager.update_adapter_from_distributed_communicator = stage
    manager.activate_adapter_version_communicator = activate
    engine = FakeEngine()
    engine.tokenizer_manager = manager

    def load(name, path, **kwargs):
        ref = replace(
            old,
            **{names[0]: name, names[1]: "id-b", names[2]: int(mode == "native_oft")},
        )
        registry._registry[name] = cache[name] = ref
        return True, "loaded"

    def unload(name):
        assert getattr(registry._registry[name], names[2]) == 1 + int(
            mode == "native_oft"
        )
        registry._registry.pop(name)
        cache.pop(name)
        return True, "unloaded"

    setattr(engine, f"load_{kind}_adapter", load)
    setattr(engine, f"unload_{kind}_adapter", unload)
    from adapter_equivalence.server import make_adapter_control

    runner = make_runner(
        mode, engine=engine, control=make_adapter_control(mode, engine)
    )
    runner.sender.broadcast_fixture = lambda *args, **kwargs: PAYLOAD
    before = runner.state()
    result = runner.run_selected(("cancel.publication",))["cancel.publication"]
    assert runner.state() == before
    assert result.error is None
    assert getattr(manager, f"pending_{kind}_stage") is None


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_cancel_eviction_targets_victim_and_cleans_reload_catalog(make_runner, mode):
    from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

    engine = FakeEngine()
    cache = {}
    args = SimpleNamespace(max_loaded_loras=None, max_loaded_ofts=None)
    unloads = []
    kind = "lora" if mode == "native_lora" else "oft"
    field = "lora_name" if kind == "lora" else "adapter_name"

    class Manager:
        server_args = args

        async def run_load(self, obj):
            async def work():
                name = getattr(obj, field)
                FakeControl.load_path(control, name, obj.path)
                cache[name] = control.records[name]["id"]
                if (
                    getattr(
                        self.server_args,
                        "max_loaded_loras" if kind == "lora" else "max_loaded_ofts",
                    )
                    == 2
                ):
                    assert set(control.records) == {"policy-a", "policy-b", name}
                    control.records.pop("policy-b")
                    await getattr(self, f"update_{kind}_adapter_communicator")(
                        SimpleNamespace(**{field: "policy-b"})
                    )
                return control.result()

            return await finish_irreversible_update(work)

        async def load_lora_adapter(self, obj):
            return await self.run_load(obj)

        async def load_oft_adapter(self, obj):
            return await self.run_load(obj)

    manager = engine.tokenizer_manager = Manager()

    async def communicator(request):
        unloads.append(getattr(request, field))
        return [SimpleNamespace(success=True, error_message="")]

    setattr(manager, f"update_{kind}_adapter_communicator", communicator)

    class Control(FakeControl):
        def load_path(self, name, path, **kwargs):
            obj = SimpleNamespace(**{field: name, "path": path})
            return engine.loop.run_until_complete(
                getattr(manager, f"load_{kind}_adapter")(obj)
            )

        def unload(self, name):
            result = super().unload(name)
            cache.pop(name)
            return result

        def inspect_state(self):
            return dict(super().inspect_state(), cache_identity=dict(cache))

    control = Control(mode)
    control.load_path("policy-a", "/a")
    runner = make_runner(mode, engine=engine, control=control)
    before = runner.state()
    observed = runner.run_selected(("cancel.eviction",))["cancel.eviction"]
    assert runner.state() == before
    assert observed.error is None
    assert unloads == ["policy-b"]
    assert args.max_loaded_loras is None and args.max_loaded_ofts is None


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_shard_stress_uses_sender_and_real_cancellation_surface(make_runner, mode):
    from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

    engine = FakeEngine()
    kind = "lora" if mode == "native_lora" else "oft"
    name_field = "lora_name" if kind == "lora" else "adapter_name"

    class Manager:
        async def update(self, obj):
            async def work():
                await getattr(self, f"update_{kind}_adapter_communicator")(obj)
                control.records[getattr(obj, name_field)]["version"] = "1"
                return control.result()

            return await finish_irreversible_update(work)

        async def load_lora_adapter_from_distributed(self, obj):
            return await self.update(obj)

        async def load_oft_adapter_from_distributed(self, obj):
            return await self.update(obj)

    manager = engine.tokenizer_manager = Manager()

    async def communicator(request):
        return [SimpleNamespace(success=True, error_message="", loaded_adapters={})]

    setattr(manager, f"update_{kind}_adapter_communicator", communicator)

    class Control(FakeControl):
        def load_distributed(self, name, payload, group, *, upsert=False):
            assert upsert is True and group == "group-test" and payload == PAYLOAD
            self.transfer_started.set()
            assert self.transfer_done.wait(0.5)
            self.transfer_started.clear()
            self.transfer_done.clear()
            return engine.loop.run_until_complete(
                getattr(manager, f"load_{kind}_adapter_from_distributed")(
                    SimpleNamespace(**{name_field: name})
                )
            )

    control = Control(mode)
    runner = make_runner(mode, engine=engine, control=control)
    result = runner.run_stress(job_timeout=10)
    assert result.cycles_completed == 100 and result.requests_completed == 1000
    assert len(runner.sender.broadcasts) == 200
    assert len(engine.calls) == 1000
    key = "lora_path" if kind == "lora" else "adapter_path"
    assert [call.get(key) for call in engine.calls[:10]] == [
        "policy-a",
        "policy-b",
        None,
        "policy-a",
        "policy-b",
        None,
        "policy-a",
        "policy-b",
        None,
        "policy-a",
    ]
    assert sum(call["stream"] for call in engine.calls) == 500
    assert runner.diagnostics.getvalue().count('"event": "stress.cancelled"') == 10


def stress_api():
    assert (
        importlib.util.find_spec("adapter_equivalence.stress_case") is not None
    ), "bounded stress execution is missing"
    return importlib.import_module("adapter_equivalence.stress_case")


def make_stress_spec(api, *, fail=None):
    control = FakeControl("native_lora")
    initial = control.inspect_state()
    batches = []
    upserts = []

    def upsert(name, cycle, cancel):
        assert set(control.records) == {"policy-a", "policy-b"}
        upserts.append((name, cycle, cancel))
        if fail == "upsert":
            return ControlResult(False, "real update failed", {}, None, None)
        if fail != "unchanged":
            control.records[name]["version"] = "1"
        if cancel:
            raise asyncio.CancelledError
        return control.result()

    def generate(count, stream):
        batches.append((count, stream))
        if fail == "generation":
            raise RuntimeError("real inference failed")
        return [
            run_case.capture_generation(
                response(i + 10), control.inspect_state(), top_k=5
            )
            for i in range(count - (fail == "count"))
        ]

    spec = api.StressSpec(
        control=control,
        a_tensors={"a": 1},
        a_config={"r": 8},
        b_tensors={"b": 2},
        b_config={"r": 8},
        generate_mixed=generate,
        upsert_distributed=upsert,
        expected_final_state=initial,
        operation_timeout=0.2,
        job_timeout=10,
    )
    return spec, batches, upserts


def test_stress_bounds_cycle_request_modes_and_real_inplace_upserts():
    api = stress_api()
    spec, batches, upserts = make_stress_spec(api)
    result = api.run_stress(spec)
    assert result.cycles_completed == 100
    assert result.requests_completed == 1000
    assert batches == [(10, bool(cycle % 2)) for cycle in range(100)]
    assert len(upserts) == 200
    assert [cycle for name, cycle, cancel in upserts if cancel] == list(
        range(9, 100, 10)
    )
    assert result.final_state == spec.expected_final_state
    from adapter_equivalence.schema import canonical_sha256

    assert result.completion_hash == canonical_sha256(spec.expected_final_state)
    assert [op[0] for op in spec.control.operations] == [
        "load_tensors",
        "load_tensors",
        "unload",
        "unload",
    ] * 100


@pytest.mark.parametrize("failure", ("upsert", "unchanged", "generation", "count"))
def test_stress_does_not_certify_failed_or_missing_work(failure):
    api = stress_api()
    spec, _, _ = make_stress_spec(api, fail=failure)
    with pytest.raises((ScenarioContractError, RuntimeError)):
        api.run_stress(spec)


@pytest.mark.parametrize("budget", ("operation", "job"))
def test_stress_deadlines_bound_stuck_calls(budget):
    api = stress_api()
    spec, _, _ = make_stress_spec(api)
    release = threading.Event()
    spec = replace(
        spec, generate_mixed=lambda *args: release.wait(), **{budget + "_timeout": 0.02}
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            api.run_stress(spec)
        assert time.monotonic() - started < 0.5
    finally:
        release.set()


@pytest.fixture
def qualifying_runner(tmp_path, monkeypatch):
    from adapter_equivalence.preflight import load_prompt_manifest
    from test_bundle_capture import qualification_inputs

    inputs = qualification_inputs(tmp_path, monkeypatch)
    prompt_manifest = load_prompt_manifest(inputs.spec.prompts_file)
    engines, events, clocks = (
        [],
        [],
        iter((0.0, 9.0, 10.0, 10.5, 20.0, 28.0, 30.0, 30.25, 40.0, 47.0, 50.0, 50.125)),
    )
    failure = {"phase": None, "error": RuntimeError("first failure")}

    def trip(phase):
        if failure["phase"] == phase:
            raise failure["error"]

    def launch(server):
        index = len(engines)
        trip("startup" if index >= 2 else "lifecycle_startup")
        engine = FakeEngine()
        engine.server = server
        engine.normalized_requests = []
        import ast

        tree = ast.parse((inputs.repo / "python/sglang/srt/server_args.py").read_text())
        server_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
        )
        radix = next(
            ast.literal_eval(node.value)
            for node in server_class.body
            if isinstance(node, ast.AnnAssign)
            and node.target.id == "disable_radix_cache"
        )
        values = dict(
            run_case.engine_kwargs(server),
            random_seed=run_case.RUN_SEED,
            disable_radix_cache=radix,
            port=30000 + index,
            tokenizer_path=server.model_path,
            served_model_name=server.model_path,
        )
        engine.server_args = make_dataclass(
            "ResolvedServerArgs",
            [
                (key, object, field(default_factory=lambda value=value: value))
                for key, value in values.items()
            ],
        )()
        engines.append(engine)
        events.append((index, "launch"))

        from types import MethodType

        engine._resolve_routed_dp_rank = lambda rank, old: (
            rank if rank is not None else old
        )
        generate = MethodType(
            committed_request_boundary(inputs.repo, monkeypatch), engine
        )

        def generate_request(obj, request):
            class ResponseIterator:
                done = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.done:
                        raise StopAsyncIteration
                    self.done = True
                    supplied = asdict(obj)
                    obj.normalize_batch_and_arguments()
                    engine.normalized_requests.append(asdict(obj))
                    phase = "warmup" if obj.is_single else "sample"
                    if index >= 2:
                        events.append((index, phase))
                        trip(phase)
                    result = await FakeEngine.async_generate(engine, **asdict(obj))
                    engine.calls[-1] = supplied
                    return result

            return ResponseIterator()

        async def reset(sample_id):
            events.append((index, "reset", sample_id))
            trip("reset")
            return SimpleNamespace(
                success=True,
                operation="reset",
                sample_id=sample_id,
                message="",
                ranks=[
                    SimpleNamespace(
                        rank=rank,
                        operation="reset",
                        sample_id=sample_id,
                        success=True,
                        message="",
                        allocated_bytes=None,
                        reserved_bytes=None,
                    )
                    for rank in range(2)
                ],
            )

        async def read(sample_id):
            events.append((index, "read", sample_id))
            trip("read")
            rows = [
                SimpleNamespace(
                    rank=rank,
                    operation="read",
                    sample_id=sample_id,
                    success=True,
                    message="",
                    allocated_bytes=100 + rank,
                    reserved_bytes=200 + rank,
                )
                for rank in range(2)
            ]
            if failure["phase"] == "zero_memory":
                rows[1].allocated_bytes = 0
            return SimpleNamespace(
                success=True,
                operation="read",
                sample_id=sample_id,
                message="",
                ranks=rows,
            )

        def shutdown():
            events.append((index, "shutdown"))
            engine.closed = True
            trip("teardown" if index >= 2 else "lifecycle_teardown")

        engine.async_generate = generate
        engine.shutdown = shutdown
        engine.tokenizer_manager = SimpleNamespace(
            reset_cuda_memory_peak=reset,
            read_cuda_memory_peak=read,
            resolved_config_dict=lambda base: base,
            sampling_params_class=FakeSamplingParams,
            preferred_sampling_params={},
            tokenizer=None,
            model_config=SimpleNamespace(vocab_size=32000),
            generate_request=generate_request,
        )

        async def internal_state():
            return [
                dict(
                    asdict(engine.server_args),
                    startup_time=float(index),
                    last_gen_throughput=float(index),
                    _resolved_overrides=[
                        ("resolution-audit", {"already_applied": index})
                    ],
                    _ssl_verify_warned=bool(index % 2),
                    effective_max_running_requests_per_dp=128,
                )
            ]

        engine.tokenizer_manager.get_internal_state = internal_state
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    monkeypatch.setattr(run_case.time, "perf_counter", lambda: next(clocks))
    runner = run_case.ShardRunner(
        inputs.spec,
        prompts={
            prompt.id: list(prompt.input_ids) for prompt in prompt_manifest.prompts
        },
        batches={
            batch.id: [request.prompt_id for request in batch.requests]
            for batch in prompt_manifest.batches
        },
        diagnostics=io.StringIO(),
    )
    yield SimpleNamespace(
        runner=runner, inputs=inputs, engines=engines, events=events, failure=failure
    )
    runner.close()


def test_qualifying_shard_publishes_three_fresh_samples_after_teardown(
    qualifying_runner,
):
    """Missing measurements, reused engines, guessed tokens and early publication fail."""
    from adapter_equivalence.schema import RunBundle

    q = qualifying_runner
    bundle = q.runner.run_qualified(argv=["runner", "--repetition", "0"])
    restored = RunBundle.read_json(q.runner.spec.bundle_output)
    assert restored.digest() == bundle.digest()
    assert (
        json.loads(q.runner.spec.completion_output.read_text())["bundle_hash"]
        == bundle.digest()
    )
    assert (
        len(q.engines) == 5
    )  # Two lifecycle launches, then three independent samples.
    assert all(engine.closed for engine in q.engines)
    assert bundle.performance.startup_seconds == (9.0, 8.0, 7.0)
    assert bundle.performance.latency_seconds == (0.5, 0.25, 0.125)
    assert bundle.performance.throughput_tokens_per_second == (64.0, 128.0, 256.0)
    assert bundle.performance.peak_allocated_bytes == (201, 201, 201)
    assert bundle.performance.peak_reserved_bytes == (401, 401, 401)
    assert bundle.manifest["request_order"] == (
        "base.initial",
        "concurrent.non-stream",
        "restart.same-manifest",
        "restart.identity",
    )
    assert bundle.manifest["seed"] == 1729
    assert bundle.manifest["server_args"] == (
        "--model-path",
        q.runner.spec.server.model_path,
        "--base-gpu-id",
        "1",
        "--tp-size",
        "2",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.8",
        "--max-total-tokens",
        "32768",
        "--log-level",
        "error",
        "--random-seed",
        "1729",
    )
    assert bundle.manifest["metadata"]["engine_kwargs"] == {
        "model_path": q.runner.spec.server.model_path,
        "base_gpu_id": 1,
        "tp_size": 2,
        "ep_size": 1,
        "disable_cuda_graph": True,
        "mem_fraction_static": 0.8,
        "max_total_tokens": 32768,
        "log_level": "error",
        "random_seed": 1729,
    }
    assert bundle.provenance["metadata"]["repetition"] == 0
    sample_ids = []
    for index, engine in enumerate(q.engines[2:], 2):
        event_rows = [row for row in q.events if row[0] == index]
        assert [row[1] for row in event_rows] == [
            "launch",
            "warmup",
            "reset",
            "sample",
            "read",
            "shutdown",
        ]
        assert event_rows[2][2] == event_rows[4][2]
        sample_ids.append(event_rows[2][2])
        assert engine.server == q.runner.spec.server
        assert len(engine.calls) == 2
        assert len(engine.calls[1]["input_ids"]) == 32
        assert engine.calls[1]["sampling_params"] == {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 32,
        }
    assert len(set(sample_ids)) == 3


@pytest.mark.parametrize(
    "phase",
    (
        "lifecycle_startup",
        "lifecycle_teardown",
        "startup",
        "warmup",
        "reset",
        "sample",
        "read",
        "zero_memory",
        "teardown",
    ),
)
def test_qualifying_failure_withholds_completion_and_bundle(qualifying_runner, phase):
    q = qualifying_runner
    q.failure["phase"] = phase
    with pytest.raises(
        (RuntimeError, BundleValidationError, ScenarioContractError)
    ) as caught:
        q.runner.run_qualified(argv=["runner"])
    if phase != "zero_memory":
        assert caught.value is q.failure["error"]
    assert not q.runner.spec.bundle_output.exists()
    assert not q.runner.spec.completion_output.exists()


def test_qualifying_late_input_mutation_withholds_completion(qualifying_runner):
    q = qualifying_runner
    original = q.runner.close

    def mutate():
        original()
        (q.inputs.repo / "code.py").write_text("changed after teardown")

    q.runner.close = mutate
    with pytest.raises(BundleValidationError, match="dirty"):
        q.runner.run_qualified(argv=["runner"])
    assert not q.runner.spec.completion_output.exists()


@pytest.mark.parametrize("repetition", (-1, True, 1.5))
def test_repetition_identity_rejects_invalid_values(qualifying_runner, repetition):
    with pytest.raises(ScenarioContractError, match="repetition"):
        replace(qualifying_runner.runner.spec, repetition=repetition)


@pytest.mark.parametrize("field", ("prompts", "batches", "server"))
def test_qualifying_execution_cannot_drift_from_immutable_inputs(
    qualifying_runner, field
):
    q = qualifying_runner
    if field == "prompts":
        q.runner.prompts["factual"] = [999]
    elif field == "batches":
        q.runner.batches["batch-32"] = ["factual"] * 32
    else:
        original = q.runner._restart

        def drift():
            q.runner.spec = replace(
                q.runner.spec,
                server=replace(q.runner.spec.server, mem_fraction_static=0.7),
            )
            original()

        q.runner._restart = drift
    with pytest.raises(ScenarioContractError, match="immutable|manifest|changed"):
        q.runner.run_qualified(argv=["runner"])
    assert not q.runner.spec.completion_output.exists()


@pytest.mark.parametrize(
    "phase", ("lifecycle_validation", "bundle_validation", "publication", "destination")
)
def test_qualifying_validation_and_publication_fail_closed(
    qualifying_runner, monkeypatch, phase
):
    q = qualifying_runner
    error = RuntimeError(phase)

    def fail(*args, **kwargs):
        raise error

    if phase == "lifecycle_validation":
        monkeypatch.setattr(run_case, "validate_lifecycle_observations", fail)
    elif phase == "bundle_validation":
        monkeypatch.setattr(run_case.RunBundle, "create", fail)
    elif phase == "publication":
        monkeypatch.setattr(run_case, "publish_bundle", fail)
    else:
        q.runner.spec.bundle_output.write_text("existing evidence")
    with pytest.raises((RuntimeError, FileExistsError)) as caught:
        q.runner.run_qualified(argv=["runner"])
    if phase == "destination":
        assert not q.engines
        assert q.runner.spec.bundle_output.read_text() == "existing evidence"
    else:
        assert caught.value is error
        assert not q.runner.spec.bundle_output.exists()
    assert not q.runner.spec.completion_output.exists()


def test_qualifying_primary_error_survives_engine_cleanup_failure(
    qualifying_runner, monkeypatch
):
    q = qualifying_runner
    q.failure["phase"] = "sample"
    original = run_case.stop_engine

    def cleanup(engine):
        original(engine)
        if engine is q.engines[-1] and len(q.engines) > 2:
            raise RuntimeError("secondary teardown failure")

    monkeypatch.setattr(run_case, "stop_engine", cleanup)
    with pytest.raises(RuntimeError) as caught:
        q.runner.run_qualified(argv=["runner"])
    assert caught.value is q.failure["error"]
    assert not q.runner.spec.completion_output.exists()


@pytest.mark.parametrize("bad", (0, -1, True, 1.5, None))
@pytest.mark.parametrize("field", ("allocated_bytes", "reserved_bytes"))
def test_scheduler_memory_requires_nonzero_integer_metrics(
    make_runner, monkeypatch, field, bad
):
    runner = make_runner("base")
    row = SimpleNamespace(
        rank=0,
        sample_id="sample",
        operation="read",
        success=True,
        allocated_bytes=100,
        reserved_bytes=200,
    )
    setattr(row, field, bad)

    async def read(sample_id):
        return SimpleNamespace(
            success=True, sample_id=sample_id, operation="read", ranks=[row]
        )

    runner.engine.tokenizer_manager = SimpleNamespace(read_cuda_memory_peak=read)
    with pytest.raises(ScenarioContractError, match="metrics"):
        runner._scheduler_memory("read", "sample")


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_native_performance_uses_identical_preloaded_adapter_state(
    qualifying_runner, monkeypatch, mode
):
    q = qualifying_runner
    q.runner.spec = replace(
        q.runner.spec,
        server=replace(
            q.runner.spec.server,
            mode=mode,
            startup_adapters=(("policy-a", "/immutable/adapter"),),
        ),
    )

    def control(mode, engine):
        result = FakeControl(mode)
        for name, path in engine.server.startup_adapters:
            result.load_path(name, path)
        return result

    monkeypatch.setattr(run_case, "make_adapter_control", control)
    metrics = q.runner._measure_performance("a" * 64)
    assert metrics.throughput_tokens_per_second == (64.0, 128.0, 256.0)
    assert len(q.engines) == 3
    for engine in q.engines:
        assert engine.closed
        assert engine.server == q.runner.spec.server
        assert all(
            call["lora_path" if mode == "native_lora" else "adapter_path"] == "policy-a"
            for call in engine.calls
        )


def test_recorded_seed_reaches_real_engine_constructor(monkeypatch):
    # This test isolates constructor arguments; real origin/spawn checks have
    # dedicated dependency-light module fixtures in test_bundle_capture.
    monkeypatch.setattr(run_case, "bind_runtime", lambda: {})
    monkeypatch.setattr(run_case, "attest_engine_targets", lambda engine: {})
    captured = []
    module = SimpleNamespace(Engine=lambda **kwargs: captured.append(kwargs))
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", module)
    spec = ServerSpec("candidate", "/model", "base", 30000, 1, 1, False)
    run_case.launch_engine(spec)
    assert captured[0]["random_seed"] == 1729
    assert captured[0]["model_path"] == "/model"


@pytest.mark.parametrize("selection", ("full", "smoke"))
def test_cli_qualification_publication_and_smoke_exclusion(
    qualifying_runner, selection
):
    q = qualifying_runner
    spec = q.runner.spec
    args = [
        "--mode",
        "base",
        "--case-id",
        "cell",
        "--revision-kind",
        "source",
        "--revision-sha",
        spec.revision_sha,
        "--architecture",
        "dense",
        "--precision",
        "bf16",
        "--cuda-graph",
        "off",
        "--model-path",
        spec.server.model_path,
        "--tp-size",
        "2",
        "--checkpoint-manifest",
        str(spec.checkpoint_manifest),
        "--prompts-file",
        str(spec.prompts_file),
        "--fixture-manifest",
        str(spec.fixture_manifest),
        "--bundle-output",
        str(spec.bundle_output),
        "--completion-output",
        str(spec.completion_output),
        "--repetition",
        "2",
        "--selection",
        selection,
    ]
    assert run_case.main(args) == 0
    assert spec.completion_output.exists() is (selection == "full")
    if selection == "full":
        from adapter_equivalence.schema import RunBundle

        bundle = RunBundle.read_json(spec.bundle_output)
        assert bundle.provenance["metadata"]["role"] == "source"
        assert bundle.provenance["metadata"]["repetition"] == 2
        assert bundle.manifest["metadata"]["repetition"] == 2


@pytest.mark.parametrize(
    "operation,defect",
    [
        (operation, defect)
        for operation in ("reset", "read")
        for defect in (
            "operation",
            "ranks",
            "row_operation",
            "row_sample",
            "row_rank",
            "row_success",
        )
    ]
    + [("reset", "reset_metrics")],
)
def test_memory_control_rejects_incomplete_boundary_contract(
    make_runner, operation, defect
):
    runner = make_runner("base")
    row = SimpleNamespace(
        rank=0,
        sample_id="sample",
        operation=operation,
        success=True,
        allocated_bytes=100 if operation == "read" else None,
        reserved_bytes=200 if operation == "read" else None,
    )
    result = SimpleNamespace(
        success=True, sample_id="sample", operation=operation, ranks=[row]
    )
    if defect == "operation":
        result.operation = "wrong"
    elif defect == "ranks":
        result.ranks = []
    elif defect == "reset_metrics":
        row.allocated_bytes = 0
    else:
        field = defect.removeprefix("row_")
        field = "sample_id" if field == "sample" else field
        setattr(row, field, False if field == "success" else "wrong")

    async def invoke(sample_id):
        return result

    runner.engine.tokenizer_manager = SimpleNamespace(
        **{f"{operation}_cuda_memory_peak": invoke}
    )
    with pytest.raises(ScenarioContractError, match="CUDA"):
        runner._scheduler_memory(operation, "sample")


def test_qualifying_sender_teardown_failure_withholds_publication(qualifying_runner):
    q = qualifying_runner
    sender = FakeSender(FakeControl("base"))
    sender.failure = RuntimeError("sender teardown failure")
    q.runner.sender = sender
    with pytest.raises(RuntimeError) as caught:
        q.runner.run_qualified(argv=["runner"])
    assert caught.value is sender.failure
    assert sender.closed
    assert all(engine.closed for engine in q.engines)
    assert not q.runner.spec.bundle_output.exists()
    assert not q.runner.spec.completion_output.exists()


def test_performance_procedure_hash_is_comparable_and_binds_sampling(
    qualifying_runner, monkeypatch
):
    q = qualifying_runner
    ticks = iter(range(100))
    monkeypatch.setattr(run_case.time, "perf_counter", lambda: float(next(ticks)))
    bundles = []
    for index, (role, tokens) in enumerate(
        (("source", 32), ("candidate", 32), ("candidate", 16))
    ):
        spec = replace(
            q.runner.spec,
            server=replace(q.runner.spec.server, revision_kind=role),
            repetition=index,
            max_new_tokens=tokens,
            bundle_output=q.runner.spec.bundle_output.with_name(f"bundle-{index}.json"),
            completion_output=q.runner.spec.completion_output.with_name(
                f"complete-{index}.json"
            ),
        )
        runner = run_case.ShardRunner(
            spec,
            prompts=q.runner.prompts,
            batches=q.runner.batches,
            diagnostics=io.StringIO(),
        )
        bundles.append(runner.run_qualified(argv=["runner", str(index)]))
    assert (
        bundles[0].performance.procedure_hash == bundles[1].performance.procedure_hash
    )
    assert (
        bundles[1].performance.procedure_hash != bundles[2].performance.procedure_hash
    )
    from adapter_equivalence.schema import PROVENANCE_HASH_KEYS

    assert all(
        bundles[0].provenance[key] == bundles[1].provenance[key]
        for key in PROVENANCE_HASH_KEYS
    )


def test_comparator_rejects_effective_launch_change(qualifying_runner, monkeypatch):
    from adapter_equivalence.compare import compare_bundles
    from test_compare import _envelope

    q = qualifying_runner
    ticks = iter(range(100))
    monkeypatch.setattr(run_case.time, "perf_counter", lambda: float(next(ticks)))
    first = q.runner.run_qualified(argv=["runner"])
    spec = replace(
        q.runner.spec,
        server=replace(q.runner.spec.server, mem_fraction_static=0.7),
        bundle_output=q.runner.spec.bundle_output.with_name("changed.json"),
        completion_output=q.runner.spec.completion_output.with_name(
            "changed.complete.json"
        ),
    )
    second = run_case.ShardRunner(
        spec,
        prompts=q.runner.prompts,
        batches=q.runner.batches,
        diagnostics=io.StringIO(),
    ).run_qualified(argv=["runner"])
    report = compare_bundles(first, second, _envelope(first))
    assert not report.passed
    assert any(
        mismatch.position == "performance.procedure_hash"
        for mismatch in report.mismatches
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("mem_fraction_static", 0.7),
        ("tp_size", 4),
        ("ep_size", 2),
        ("cuda_graph", True),
        ("quantization", "fp8"),
        ("moe_runner", "triton"),
        ("max_oft_block_size", 64),
        ("peft_target_modules", ("q_proj", "k_proj")),
    ),
)
def test_performance_launch_identity_binds_every_effective_knob(
    qualifying_runner, field, value
):
    q = qualifying_runner
    server = replace(
        q.runner.spec.server,
        mode="native_oft",
        startup_adapters=(("policy-a", "/adapter"),),
        max_oft_block_size=32,
        peft_target_modules=("q_proj",),
    )
    provenance = {
        "checkpoint_hash": "a" * 64,
        "metadata": {"fixture_files": {"policy-a": {"weights": "b" * 64}}},
    }
    before = run_case.performance_launch_identity(server, provenance)
    after = run_case.performance_launch_identity(
        replace(server, **{field: value}), provenance
    )
    assert before != after


def test_performance_launch_identity_accepts_relocated_immutable_inputs(
    qualifying_runner,
):
    server = replace(
        qualifying_runner.runner.spec.server,
        mode="native_lora",
        startup_adapters=(("policy-a", "/source/fixtures/a"),),
        max_lora_rank=8,
        lora_target_modules=("q_proj",),
    )
    provenance = {
        "checkpoint_hash": "a" * 64,
        "metadata": {"fixture_files": {"policy-a": {"weights": "b" * 64}}},
    }
    relocated = replace(
        server,
        model_path="/candidate/checkpoint",
        revision_kind="source",
        startup_adapters=(("policy-a", "/candidate/fixtures/a"),),
    )
    assert run_case.performance_launch_identity(
        server, provenance
    ) == run_case.performance_launch_identity(relocated, provenance)


@pytest.mark.parametrize(
    "change",
    ("relocation", "memory_boundary", "engine_default", "normalization_default"),
)
def test_comparator_uses_normalized_launch_and_memory_boundary(
    qualifying_runner, monkeypatch, change
):
    import shutil
    import subprocess

    from adapter_equivalence.compare import compare_bundles
    from test_compare import _envelope

    q = qualifying_runner
    ticks = iter(range(100))
    monkeypatch.setattr(run_case.time, "perf_counter", lambda: float(next(ticks)))
    first = q.runner.run_qualified(argv=["source"])
    if change == "normalization_default":
        assert q.engines[-1].normalized_requests[-1]["logprob_start_len"] == [-1] * 32
    spec = replace(
        q.runner.spec,
        bundle_output=q.runner.spec.bundle_output.with_name("changed.json"),
        completion_output=q.runner.spec.completion_output.with_name(
            "changed.complete.json"
        ),
    )
    if change == "relocation":
        relocated = spec.checkpoint_manifest.parent / "relocated-model"
        shutil.copytree(spec.server.model_path, relocated)
        manifest = json.loads(spec.checkpoint_manifest.read_text())
        for entry in manifest["checkpoints"]:
            if entry["path"] == spec.server.model_path:
                entry["path"] = str(relocated)
        spec.checkpoint_manifest.write_text(json.dumps(manifest))
        spec = replace(spec, server=replace(spec.server, model_path=str(relocated)))
    else:
        if change == "normalization_default":
            path = q.inputs.repo / "python/sglang/srt/managers/io_struct.py"
            path.write_text(
                path.read_text().replace(
                    'self.logprob_start_len, -1, "logprob_start_len"',
                    'self.logprob_start_len, 0, "logprob_start_len"',
                )
            )
        elif change == "engine_default":
            path = q.inputs.repo / "python/sglang/srt/server_args.py"
            path.write_text(
                path.read_text().replace(
                    'bool, "Disable RadixAttention for prefix caching.", NS("memory")\n    ] = False',
                    'bool, "Disable RadixAttention for prefix caching.", NS("memory")\n    ] = True',
                )
            )
        else:
            path = q.inputs.repo / "python/sglang/srt/managers/scheduler.py"
            path.write_text(
                path.read_text().replace(
                    "torch.cuda.max_memory_allocated(device)",
                    "torch.cuda.memory_allocated(device)",
                )
            )
        subprocess.run(["git", "-C", str(q.inputs.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(q.inputs.repo),
                "-c",
                "user.name=Harness",
                "-c",
                "user.email=harness@example.invalid",
                "commit",
                "-qm",
                "change observation boundary",
            ],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(q.inputs.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        spec = replace(spec, revision_sha=revision)
    second = run_case.ShardRunner(
        spec,
        prompts=q.runner.prompts,
        batches=q.runner.batches,
        diagnostics=io.StringIO(),
    ).run_qualified(argv=["candidate"])
    if change == "normalization_default":
        assert q.engines[-1].normalized_requests[-1]["logprob_start_len"] == [0] * 32
        before_request = first.to_dict()["manifest"]["metadata"][
            "performance_procedure"
        ]["effective_launches"]["preloaded"]["requests"]["batch"]["request"]
        after_request = second.to_dict()["manifest"]["metadata"][
            "performance_procedure"
        ]["effective_launches"]["preloaded"]["requests"]["batch"]["request"]
        assert before_request["logprob_start_len"] == [-1] * 32
        assert after_request["logprob_start_len"] == [0] * 32
        assert "rid" not in before_request and "rid" not in after_request
    report = compare_bundles(first, second, _envelope(first))
    assert report.passed is (change == "relocation")
    if change == "normalization_default":
        assert any(
            mismatch.position == "performance.procedure_hash"
            for mismatch in report.mismatches
        )
    if change == "memory_boundary":
        assert (
            first.provenance["metadata"]["memory_boundary_hash"]
            != second.provenance["metadata"]["memory_boundary_hash"]
        )
        assert any(
            mismatch.position == "performance.procedure_hash"
            for mismatch in report.mismatches
        )


@pytest.mark.parametrize(
    "drift", ("engine", "tokenizer", "scheduler", "sampling", "after_warmup")
)
def test_qualification_rejects_resolved_configuration_drift(
    qualifying_runner, monkeypatch, drift
):
    q = qualifying_runner
    original = run_case.launch_engine

    def launch(spec):
        engine = original(spec)
        if len(q.engines) == 4:
            if drift == "engine":
                engine.server_args.disable_radix_cache = True
            elif drift == "tokenizer":
                engine.tokenizer_manager.resolved_config_dict = lambda base: dict(
                    base, disable_radix_cache=True
                )
            elif drift == "scheduler":

                async def state():
                    return [dict(asdict(engine.server_args), disable_radix_cache=True)]

                engine.tokenizer_manager.get_internal_state = state
            elif drift == "sampling":
                engine.tokenizer_manager.preferred_sampling_params = {
                    "presence_penalty": 1.0
                }
            else:
                generate = engine.async_generate

                async def mutate(**kwargs):
                    result = await generate(**kwargs)
                    engine.server_args.disable_radix_cache = True
                    return result

                engine.async_generate = mutate
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    with pytest.raises(
        ScenarioContractError, match="resolved.*changed|effective.*changed"
    ):
        q.runner.run_qualified(argv=["runner"])
    assert all(engine.closed for engine in q.engines)
    assert not q.runner.spec.completion_output.exists()

    diagnostics = [
        json.loads(line) for line in q.runner.diagnostics.getvalue().splitlines()
    ]
    changes = [
        record for record in diagnostics if record["event"] == "configuration.changed"
    ]
    assert changes
    field_name = "presence_penalty" if drift == "sampling" else "disable_radix_cache"
    assert any(field_name in difference[0] for difference in changes[0]["differences"])


def test_qualification_rejects_capacity_below_fixed_launch_limit(
    qualifying_runner, monkeypatch
):
    q = qualifying_runner
    original = run_case.launch_engine

    def launch(spec):
        engine = original(spec)
        read = engine.tokenizer_manager.get_internal_state

        async def state():
            return [
                dict(
                    row,
                    memory_usage={"token_capacity": 32760, "token_capacity_swa": None},
                )
                for row in await read()
            ]

        engine.tokenizer_manager.get_internal_state = state
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    with pytest.raises(ScenarioContractError, match="token capacity.*launch limit"):
        q.runner.run_qualified(argv=["runner"])
    assert not q.runner.spec.completion_output.exists()


def test_qualified_manifest_records_all_effective_launches_without_dynamic_metrics(
    qualifying_runner,
):
    q = qualifying_runner
    bundle = q.runner.run_qualified(argv=["runner"])
    launches = bundle.manifest["metadata"]["performance_procedure"][
        "effective_launches"
    ]
    assert set(launches) == {"initial", "preloaded"}
    for launch in launches.values():
        assert launch["engine"]["disable_radix_cache"] is False
        assert launch["tokenizer"]["disable_radix_cache"] is False
        assert launch["schedulers"][0]["disable_radix_cache"] is False
        assert "port" not in launch["engine"]
        assert "startup_time" not in launch["schedulers"][0]
        assert "last_gen_throughput" not in launch["schedulers"][0]
        assert launch["requests"]["warmup"]["sampling"]["presence_penalty"] == 0.0
    q.runner.effective_launches.clear()
    assert set(
        bundle.manifest["metadata"]["performance_procedure"]["effective_launches"]
    ) == {"initial", "preloaded"}


def test_resolved_configuration_does_not_publish_credentials(
    qualifying_runner, monkeypatch
):
    q = qualifying_runner
    original = run_case.launch_engine
    credentials = dict(
        api_key="do-not-publish-secret",
        admin_api_key="do-not-publish-admin",
        ssl_keyfile_password="do-not-publish-password",
        ssl_keyfile="/private/secret/key.pem",
    )

    def launch(spec):
        engine = original(spec)
        snapshot = engine.tokenizer_manager.resolved_config_dict
        engine.tokenizer_manager.resolved_config_dict = lambda base: dict(
            snapshot(base), **credentials
        )
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    bundle = q.runner.run_qualified(argv=["runner"])
    serialized = json.dumps(bundle.to_dict())
    assert "do-not-publish" not in serialized
    assert "/private/secret" not in serialized


@pytest.mark.parametrize("valid", (True, False))
def test_qualification_records_and_validates_child_runtime_attestation(
    qualifying_runner, monkeypatch, valid
):
    q = qualifying_runner
    origins = {
        "init_custom_process_group": (
            "python/sglang/srt/utils/common.py" if valid else "/foreign/common.py"
        )
    }
    session = SimpleNamespace(
        sender=SimpleNamespace(runtime_origins=origins), close=lambda timeout: None
    )
    monkeypatch.setattr(
        run_case.DistributedSession, "open", lambda *args, **kwargs: session
    )
    original = q.runner.run_selected

    def lifecycle(selection):
        result = original(selection)
        q.runner._ensure_sender()
        return result

    q.runner.run_selected = lifecycle
    if valid:
        bundle = q.runner.run_qualified(argv=["runner"])
        assert bundle.to_dict()["provenance"]["metadata"]["sender_runtime_origins"] == [
            origins
        ]
    else:
        with pytest.raises(ScenarioContractError, match="sender.*origin"):
            q.runner.run_qualified(argv=["runner"])
        assert not q.runner.spec.completion_output.exists()


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_effective_adapter_refs_normalize_locations_and_ephemeral_ids(
    qualifying_runner, mode
):
    server = replace(
        qualifying_runner.runner.spec.server,
        mode=mode,
        startup_adapters=(("policy-a", "/source/a"),),
    )
    other = replace(
        server, model_path="/other/model", startup_adapters=(("policy-a", "/other/a"),)
    )
    provenance = {
        "checkpoint_hash": "a" * 64,
        "metadata": {"fixture_files": {"policy-a": {"weights": "b" * 64}}},
    }

    def ref(path, identity, pinned=False):
        if mode == "native_lora":
            values = dict(
                lora_id=identity,
                lora_name="policy-a",
                lora_path=path,
                pinned=pinned,
                reloadable=True,
                version=0,
            )
            return SimpleNamespace(__struct_fields__=tuple(values), **values)
        return dict(
            adapter_id=identity,
            adapter_name="policy-a",
            adapter_path=path,
            pinned=pinned,
            adapter_version=1,
            reloadable=True,
        )

    first = run_case.normalized_effective_config(
        ref("/source/a", "a" * 32), server, provenance
    )
    second = run_case.normalized_effective_config(
        ref("/other/a", "b" * 32), other, provenance
    )
    assert first == second
    assert first != run_case.normalized_effective_config(
        ref("/source/a", "a" * 32, True), server, provenance
    )


def test_scheduler_serialized_set_order_is_not_execution_drift(
    qualifying_runner, monkeypatch
):
    import ast
    import os

    source = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/utils/msgspec_utils.py"
    )
    serializer = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "msgspec_to_builtins"
    )
    script = (
        "from __future__ import annotations\nimport json,dataclasses,types\n"
        "msgspec=types.SimpleNamespace(Struct=type('Struct',(),{}))\n"
        + ast.unparse(serializer)
        + "\nprint(json.dumps(msgspec_to_builtins({'_cuda_graph_config_locked': {('decode','backend'),('prefill','backend')}, 'lora_target_modules': {'q_proj','k_proj'}, '_runtime_mutations': [('lora', {'lora_target_modules': {'q_proj','k_proj'}})]})))\n"
    )
    states = [
        json.loads(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=dict(os.environ, PYTHONHASHSEED=str(seed)),
                text=True,
            )
        )
        for seed in range(1, 9)
    ]
    assert len({json.dumps(state) for state in states}) > 1
    q = qualifying_runner
    original = run_case.launch_engine

    def launch(spec):
        engine = original(spec)
        read = engine.tokenizer_manager.get_internal_state
        order = iter([states[0], states[-1]] * 20)

        async def state():
            return [dict(row, **next(order)) for row in await read()]

        engine.tokenizer_manager.get_internal_state = state
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    bundle = q.runner.run_qualified(argv=["runner"])
    assert q.runner.spec.completion_output.exists()

    def normalize(value):
        return run_case.normalized_effective_config(
            value, q.runner.spec.server, bundle.provenance
        )

    assert normalize(states[0]) == normalize(states[-1])
    assert normalize(states[0]) != normalize(
        {"_cuda_graph_config_locked": [["decode", "backend"]]}
    )
    assert normalize({"ordinary": [1, 2]}) != normalize({"ordinary": [2, 1]})
    assert normalize({"lora_target_modules": ["q_proj"]}) != normalize(
        {"lora_target_modules": ["k_proj"]}
    )


@pytest.mark.parametrize(
    "boundary",
    (
        "initial_generation",
        "preloaded_generation",
        "restart",
        "final_lifecycle",
        "first_failure",
    ),
)
def test_lifecycle_post_work_configuration_is_attested_before_teardown(
    qualifying_runner, monkeypatch, boundary
):
    q = qualifying_runner
    original = run_case.launch_engine
    primary = RuntimeError("original generation failure")

    def launch(spec):
        engine = original(spec)
        target = 2 if boundary == "preloaded_generation" else 1
        if len(q.engines) == target and boundary in (
            "initial_generation",
            "preloaded_generation",
            "first_failure",
        ):
            generate = engine.async_generate

            async def mutate(**kwargs):
                before = len(engine.calls)
                result = await generate(**kwargs)
                if len(engine.calls) > before:
                    engine.server_args.disable_radix_cache = True
                    if boundary == "first_failure":
                        raise primary
                return result

            engine.async_generate = mutate
            if boundary == "first_failure":
                shutdown = engine.shutdown

                def fail_teardown():
                    shutdown()
                    raise RuntimeError("secondary teardown failure")

                engine.shutdown = fail_teardown
        return engine

    monkeypatch.setattr(run_case, "launch_engine", launch)
    if boundary == "restart":
        restart = q.runner._restart

        def mutate_restart():
            q.runner.engine.server_args.disable_radix_cache = True
            restart()

        q.runner._restart = mutate_restart
    elif boundary == "final_lifecycle":
        selected = q.runner.run_selected

        def mutate_final(selection):
            result = selected(selection)
            q.runner.engine.server_args.disable_radix_cache = True
            return result

        q.runner.run_selected = mutate_final
    with pytest.raises((ScenarioContractError, RuntimeError)) as caught:
        q.runner.run_qualified(argv=["runner"])
    if boundary == "first_failure":
        assert caught.value is primary
    else:
        assert "configuration changed" in str(caught.value)
    assert all(engine.closed for engine in q.engines)
    assert not q.runner.spec.completion_output.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

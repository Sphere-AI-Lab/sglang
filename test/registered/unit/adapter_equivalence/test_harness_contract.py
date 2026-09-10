import asyncio
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

import adapter_equivalence.scenarios as scenarios
from adapter_equivalence.scenarios import (
    LifecycleStep,
    ScenarioContractError,
    execute_lifecycle,
    lifecycle_steps,
    lifecycle_transition_names,
    validate_lifecycle_observations,
)
from adapter_equivalence.schema import BundleValidationError, CaseKey, Observation
from adapter_equivalence.server import (
    AdapterControl,
    AdapterIdentity,
    BaseControl,
    ControlResult,
    DistributedPayload,
    NativeLoRAControl,
    NativeOFTControl,
    make_adapter_control,
    mode_server_args,
    normalize_control_result,
)

FAKE_CONFIG = {"peft_type": "OFT", "target_modules": ["q_proj"]}
FAKE_TENSORS = {"model.layers.0.self_attn.q_proj.weight": object()}
DISTRIBUTED_PAYLOAD = DistributedPayload(
    names=("model.layers.0.self_attn.q_proj.weight",),
    dtypes=("float32",),
    shapes=((8, 8),),
    config=FAKE_CONFIG,
)

EXPECTED_NATIVE_TRANSITIONS = (
    "base.initial",
    "startup.adapter",
    "immediate.path.load",
    "immediate.path.infer",
    "immediate.path.unload",
    "immediate.path.base",
    "immediate.tensor.load",
    "immediate.tensor.infer",
    "immediate.tensor.unload",
    "immediate.tensor.base",
    "immediate.distributed.load",
    "immediate.distributed.infer",
    "immediate.distributed.unload",
    "immediate.distributed.base",
    "switch.a",
    "switch.b",
    "switch.a-again",
    "mixed.base-a-b",
    "concurrent.stream",
    "concurrent.non-stream",
    "upsert.lease.begin",
    "upsert.while-leased",
    "upsert.lease.complete",
    "upsert.after",
    "stage.v1",
    "stage.v1.old-active",
    "activate.v1",
    "stage.v1.active",
    "stage.v2",
    "stage.v2.v1-active",
    "activate.v2",
    "stage.v2.active",
    "reject.duplicate",
    "reject.stale",
    "reject.wrong-id",
    "reject.wrong-name",
    "reject.invalid-config",
    "reject.unsupported-target",
    "failure.update",
    "failure.update.previous",
    "failure.activation",
    "failure.activation.previous",
    "failure.unload",
    "failure.unload.quarantine",
    "failure.unload.retry",
    "cancel.lease-drain",
    "cancel.fan-out",
    "cancel.publication",
    "cancel.rollback",
    "cancel.eviction",
    "pause.retain-kv",
    "pause.activate-rejected",
    "pause.resume",
    "registry.fill",
    "registry.evict",
    "unload.final",
    "base.restored",
    "restart.same-manifest",
    "restart.identity",
)

EXPECTED_BASE_TRANSITIONS = (
    "base.initial",
    "concurrent.non-stream",
    "restart.same-manifest",
    "restart.identity",
)

EXPECTED_ERROR_CODES = {
    "reject.duplicate": "duplicate_version",
    "reject.stale": "stale_version",
    "reject.wrong-id": "wrong_id",
    "reject.wrong-name": "wrong_name",
    "reject.invalid-config": "invalid_config",
    "reject.unsupported-target": "unsupported_target",
    "failure.update": "update_failure",
    "failure.activation": "activation_failure",
    "failure.unload": "unload_failure",
    "pause.activate-rejected": "paused_requests_active",
}

EXPECTED_NATIVE_OUTPUT_STEPS = (
    "base.initial",
    "startup.adapter",
    "immediate.path.infer",
    "immediate.path.base",
    "immediate.tensor.infer",
    "immediate.tensor.base",
    "immediate.distributed.infer",
    "immediate.distributed.base",
    "switch.a",
    "switch.b",
    "switch.a-again",
    "mixed.base-a-b",
    "concurrent.stream",
    "concurrent.non-stream",
    "upsert.lease.complete",
    "upsert.after",
    "stage.v1.old-active",
    "stage.v1.active",
    "stage.v2.v1-active",
    "stage.v2.active",
    "failure.update.previous",
    "pause.retain-kv",
    "pause.resume",
    "base.restored",
    "restart.identity",
)

EXPECTED_BASE_OUTPUT_STEPS = (
    "base.initial",
    "concurrent.non-stream",
    "restart.identity",
)


class _Request:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class UpdateAdapterFromDistributedReqInput(_Request):
    pass


class ActivateAdapterVersionReqInput(_Request):
    pass


class LoadLoRAAdapterFromTensorsReqInput(_Request):
    pass


class LoadLoRAAdapterFromDistributedReqInput(_Request):
    pass


class _Registry:
    def __init__(self, records=None) -> None:
        self.records = {} if records is None else dict(records)

    def get_all_adapters(self):
        return dict(self.records)


class _FakeTokenizerManager:
    def __init__(self) -> None:
        self.calls = []
        self.lora_registry = _Registry()
        self.peft_registry = _Registry()
        self.pending_lora_stage = None
        self.pending_oft_stage = None
        self.failed_lora_activations = {}
        self.failed_oft_activations = {}
        self.failed_lora_unloads = {}
        self.failed_oft_unloads = {}
        self.lora_ref_cache = {}
        self.peft_ref_cache = {}

    async def update_adapter_from_distributed(self, request, http_request):
        self.calls.append(("update_adapter_from_distributed", request, http_request))
        return True, "staged"

    async def activate_adapter_version(self, request, http_request):
        self.calls.append(("activate_adapter_version", request, http_request))
        return True, "activated"

    async def load_lora_adapter_from_tensors(self, request, http_request):
        self.calls.append(("load_lora_adapter_from_tensors", request, http_request))
        return SimpleNamespace(
            success=True,
            error_message=None,
            loaded_adapters={},
        )

    async def load_lora_adapter_from_distributed(self, request, http_request):
        self.calls.append(("load_lora_adapter_from_distributed", request, http_request))
        return SimpleNamespace(
            success=True,
            error_message=None,
            loaded_adapters={},
        )


class _FakeLoop:
    def run_until_complete(self, coroutine):
        return asyncio.run(coroutine)


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = []
        self.results = {}
        self.serialized = []
        self.loop = _FakeLoop()
        self.tokenizer_manager = _FakeTokenizerManager()

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return self.results.get(
            name,
            SimpleNamespace(
                success=True,
                error_message=None,
                loaded_adapters={},
            ),
        )

    def _serialize_tensors_per_rank(self, tensors, load_format):
        self.serialized.append((tensors, load_format))
        return [b"rank-0", b"rank-1"]

    def load_lora_adapter(self, name, path, *, pinned=False):
        return self._record("load_lora_adapter", name, path, pinned=pinned)

    def load_lora_adapter_from_tensors(self, name, tensors, config):
        return self._record("load_lora_adapter_from_tensors", name, tensors, config)

    def load_lora_adapter_from_distributed(
        self, name, config, names, dtypes, shapes, *, group_name
    ):
        return self._record(
            "load_lora_adapter_from_distributed",
            name,
            config,
            names,
            dtypes,
            shapes,
            group_name=group_name,
        )

    def unload_lora_adapter(self, name):
        return self._record("unload_lora_adapter", name)

    def load_oft_adapter(self, name, path, *, pinned=False):
        return self._record("load_oft_adapter", name, path, pinned=pinned)

    def load_oft_adapter_from_tensors(self, name, tensors, config, *, upsert=False):
        return self._record(
            "load_oft_adapter_from_tensors",
            name,
            tensors,
            config,
            upsert=upsert,
        )

    def load_oft_adapter_from_distributed(
        self, name, config, names, dtypes, shapes, *, group_name, upsert=False
    ):
        return self._record(
            "load_oft_adapter_from_distributed",
            name,
            config,
            names,
            dtypes,
            shapes,
            group_name=group_name,
            upsert=upsert,
        )

    def unload_oft_adapter(self, name):
        return self._record("unload_oft_adapter", name)


@pytest.fixture(autouse=True)
def _stub_control_request_types(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.io_struct",
        SimpleNamespace(
            ActivateAdapterVersionReqInput=ActivateAdapterVersionReqInput,
            LoadLoRAAdapterFromDistributedReqInput=(
                LoadLoRAAdapterFromDistributedReqInput
            ),
            LoadLoRAAdapterFromTensorsReqInput=LoadLoRAAdapterFromTensorsReqInput,
            UpdateAdapterFromDistributedReqInput=(UpdateAdapterFromDistributedReqInput),
        ),
    )


def _case(mode: str) -> CaseKey:
    return CaseKey(
        model="Qwen/Qwen3-4B-Instruct-2507",
        architecture="dense",
        precision="bf16",
        revision="a" * 40,
        mode=mode,
        cuda_graph=False,
        scenario="native-adapter-lifecycle-v2",
    )


def _state(
    mode: str,
    *identities: tuple[str, str, str],
    active: str | None = None,
    staged: tuple[str, str, str] | None = None,
    quarantined: tuple[str, ...] = (),
    tombstoned: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    ordered = sorted(identities)
    registered = [
        {
            "name": name,
            "id": adapter_id,
            "version": version,
            "registry_slot": index,
            "pinned": False,
        }
        for index, (name, adapter_id, version) in enumerate(ordered)
    ]
    active_identity = next(
        (
            {"name": name, "id": adapter_id, "version": version}
            for name, adapter_id, version in ordered
            if name == active
        ),
        None,
    )
    return {
        "mode": mode,
        "registered": registered,
        "active": active_identity,
        "staged": (
            None
            if staged is None
            else {
                "name": staged[0],
                "id": staged[1],
                "version": staged[2],
                "pinned": False,
            }
        ),
        "registry_occupancy": len(registered),
        "quarantined": list(quarantined),
        "tombstoned": list(tombstoned),
        "cache_identity": {name: adapter_id for name, adapter_id, _ in ordered},
    }


def _observation(
    state: dict[str, object],
    *,
    token_id: int | None = None,
    error_code: str | None = None,
) -> Observation:
    output_ids = () if token_id is None else (token_id,)
    row_name = "decode.000.top_logprobs"
    observation = Observation(
        request_output_lengths=() if token_id is None else (1,),
        request_texts=() if token_id is None else (f"token-{token_id}",),
        output_ids=output_ids,
        text="" if token_id is None else f"token-{token_id}",
        token_logprobs=() if token_id is None else (-0.25,),
        selected_logits={} if token_id is None else {row_name: (-0.5, -0.25)},
        selected_token_ids={} if token_id is None else {row_name: (10, 20)},
        adapter_state=state,
        error=(
            None
            if error_code is None
            else {
                "kind": "product_rejection",
                "code": error_code,
                "message": f"expected {error_code}",
            }
        ),
    )
    observation.validate()
    return observation


def complete_base_observations() -> dict[str, Observation]:
    empty = _state("base")
    return {
        "base.initial": _observation(empty, token_id=101),
        "concurrent.non-stream": _observation(empty, token_id=303),
        "restart.same-manifest": _observation(empty),
        "restart.identity": _observation(empty, token_id=101),
    }


def complete_native_observations(
    mode: str = "native_lora",
) -> dict[str, Observation]:
    def version(value):
        return str(value + int(mode == "native_oft"))

    empty = _state(mode)
    startup = _state(
        mode, ("policy-a", "generated-startup", version(0)), active="policy-a"
    )
    path = _state(mode, ("policy-a", "generated-path", version(0)), active="policy-a")
    tensor = _state(
        mode, ("policy-a", "generated-tensor", version(0)), active="policy-a"
    )
    distributed = _state(
        mode, ("policy-a", "generated-distributed", version(0)), active="policy-a"
    )
    switched_a = _state(
        mode,
        ("policy-a", "generated-live-a", version(0)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
    )
    switched_b = _state(
        mode,
        ("policy-a", "generated-live-a", version(0)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-b",
    )
    upserted = _state(
        mode,
        ("policy-a", "generated-live-a", version(1)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
    )
    staged_v1 = _state(
        mode,
        ("policy-a", "generated-live-a", version(1)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
        staged=("policy-a", "generated-live-a", version(2)),
    )
    active_v1 = _state(
        mode,
        ("policy-a", "generated-live-a", version(2)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
    )
    staged_v2 = _state(
        mode,
        ("policy-a", "generated-live-a", version(2)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
        staged=("policy-a", "generated-live-a", version(3)),
    )
    active_v2 = _state(
        mode,
        ("policy-a", "generated-live-a", version(3)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
    )
    failed_activation = _state(
        mode,
        ("policy-a", "generated-live-a", version(3)),
        ("policy-b", "generated-live-b", version(0)),
        active="policy-a",
        staged=("policy-a", "generated-live-a", version(4)),
        quarantined=("policy-a",),
    )
    failed_unload = _state(
        mode,
        ("policy-b", "generated-live-b", version(0)),
        quarantined=("policy-a",),
        tombstoned=(
            {
                "name": "policy-a",
                "id": "generated-live-a",
                "version": version(3),
                "pinned": False,
            },
        ),
    )
    failed_unload["cache_identity"]["policy-a"] = "generated-live-a"
    retry_survivor = _state(
        mode,
        ("policy-b", "generated-live-b", version(0)),
        active="policy-b",
    )
    cancelled = _state(
        mode, ("policy-a", "generated-cancel-a", version(0)), active="policy-a"
    )
    registry_full = _state(
        mode,
        ("policy-a", "generated-registry-a", version(0)),
        ("policy-b", "generated-registry-b", version(0)),
        active="policy-b",
    )
    registry_evicted = _state(
        mode, ("policy-b", "generated-registry-b", version(0)), active="policy-b"
    )
    registry_evicted["cache_identity"]["policy-a"] = "generated-registry-a"

    states = {
        "base.initial": (empty, 101),
        "startup.adapter": (startup, 202),
        "immediate.path.load": (path, None),
        "immediate.path.infer": (path, 211),
        "immediate.path.unload": (empty, None),
        "immediate.path.base": (empty, 101),
        "immediate.tensor.load": (tensor, None),
        "immediate.tensor.infer": (tensor, 212),
        "immediate.tensor.unload": (empty, None),
        "immediate.tensor.base": (empty, 101),
        "immediate.distributed.load": (distributed, None),
        "immediate.distributed.infer": (distributed, 213),
        "immediate.distributed.unload": (empty, None),
        "immediate.distributed.base": (empty, 101),
        "switch.a": (switched_a, 214),
        "switch.b": (switched_b, 215),
        "switch.a-again": (switched_a, 214),
        "mixed.base-a-b": (switched_a, 216),
        "concurrent.stream": (switched_a, 217),
        "concurrent.non-stream": (switched_a, 217),
        "upsert.lease.begin": (switched_a, 214),
        "upsert.while-leased": (switched_a, None),
        "upsert.lease.complete": (switched_a, 214),
        "upsert.after": (upserted, 219),
        "stage.v1": (staged_v1, None),
        "stage.v1.old-active": (staged_v1, 219),
        "activate.v1": (active_v1, None),
        "stage.v1.active": (active_v1, 220),
        "stage.v2": (staged_v2, None),
        "stage.v2.v1-active": (staged_v2, 220),
        "activate.v2": (active_v2, None),
        "stage.v2.active": (active_v2, 221),
        "reject.duplicate": (active_v2, None),
        "reject.stale": (active_v2, None),
        "reject.wrong-id": (active_v2, None),
        "reject.wrong-name": (active_v2, None),
        "reject.invalid-config": (active_v2, None),
        "reject.unsupported-target": (active_v2, None),
        "failure.update": (active_v2, None),
        "failure.update.previous": (active_v2, 221),
        "failure.activation": (failed_activation, None),
        "failure.activation.previous": (failed_activation, None),
        "failure.unload": (failed_unload, None),
        "failure.unload.quarantine": (failed_unload, None),
        "failure.unload.retry": (retry_survivor, None),
        "cancel.lease-drain": (cancelled, None),
        "cancel.fan-out": (cancelled, None),
        "cancel.publication": (cancelled, None),
        "cancel.rollback": (cancelled, None),
        "cancel.eviction": (cancelled, None),
        "pause.retain-kv": (cancelled, 222),
        "pause.activate-rejected": (cancelled, None),
        "pause.resume": (cancelled, 223),
        "registry.fill": (registry_full, None),
        "registry.evict": (registry_evicted, None),
        "unload.final": (empty, None),
        "base.restored": (empty, 101),
        "restart.same-manifest": (startup, None),
        "restart.identity": (startup, 202),
    }
    observations = {
        name: _observation(
            states[name][0],
            token_id=states[name][1],
            error_code=EXPECTED_ERROR_CODES.get(name),
        )
        for name in EXPECTED_NATIVE_TRANSITIONS
    }
    assert tuple(observations) == EXPECTED_NATIVE_TRANSITIONS
    return observations


def _replace_adapter_state(
    observation: Observation, state: dict[str, object]
) -> Observation:
    changed = replace(observation, adapter_state=state)
    changed.validate()
    return changed


def _replace_output_field(observation: Observation, field: str) -> Observation:
    changes = {
        "output_ids": {"output_ids": (999,)},
        "text": {"text": "changed-output"},
        "token_logprobs": {"token_logprobs": (-9.0,)},
        "selected_logits": {
            "selected_logits": {"decode.000.top_logprobs": (-9.0, -8.0)}
        },
        "selected_token_ids": {
            "selected_token_ids": {"decode.000.top_logprobs": (11, 21)}
        },
    }
    changed = replace(observation, **changes[field])
    changed.validate()
    return changed


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_native_lifecycle_is_complete_and_ordered(mode: str) -> None:
    assert lifecycle_transition_names(mode) == EXPECTED_NATIVE_TRANSITIONS
    assert tuple(step.name for step in scenarios.NATIVE_LIFECYCLE_STEPS) == (
        EXPECTED_NATIVE_TRANSITIONS
    )


def test_base_lifecycle_is_exactly_the_four_step_contract() -> None:
    assert lifecycle_transition_names("base") == EXPECTED_BASE_TRANSITIONS
    assert tuple(step.name for step in scenarios.BASE_LIFECYCLE_STEPS) == (
        EXPECTED_BASE_TRANSITIONS
    )


def test_complete_base_lifecycle_validates() -> None:
    validate_lifecycle_observations(complete_base_observations())


def test_lifecycle_steps_freeze_runner_dispatch_metadata() -> None:
    selected = {step.name: step for step in lifecycle_steps("native_oft")}

    assert selected["immediate.distributed.load"] == LifecycleStep(
        "immediate.distributed.load",
        "load",
        adapter="policy-a",
        input_kind="distributed",
    )
    assert selected["stage.v2"] == LifecycleStep(
        "stage.v2",
        "stage",
        adapter="policy-a",
        version="4",
        input_kind="distributed",
    )
    assert selected["reject.wrong-id"] == LifecycleStep(
        "reject.wrong-id",
        "reject_wrong_id",
        adapter="policy-a",
        version="5",
        expected_error_code="wrong_id",
    )
    assert selected["cancel.publication"] == LifecycleStep(
        "cancel.publication", "cancel_publication", adapter="policy-a"
    )
    assert selected["pause.activate-rejected"].expected_error_code == (
        "paused_requests_active"
    )
    assert {
        step.name: step.expected_error_code
        for step in scenarios.NATIVE_LIFECYCLE_STEPS
        if step.expected_error_code is not None
    } == EXPECTED_ERROR_CODES


@pytest.mark.parametrize("name", EXPECTED_NATIVE_OUTPUT_STEPS)
def test_native_inference_steps_require_nonempty_output_evidence(name: str) -> None:
    observations = complete_native_observations()
    observations[name] = _observation(observations[name].to_dict()["adapter_state"])

    with pytest.raises(ScenarioContractError, match="missing inference output"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("name", EXPECTED_BASE_OUTPUT_STEPS)
def test_base_inference_steps_require_nonempty_output_evidence(name: str) -> None:
    observations = complete_base_observations()
    observations[name] = _observation(observations[name].to_dict()["adapter_state"])

    with pytest.raises(ScenarioContractError, match="missing inference output"):
        validate_lifecycle_observations(observations)


def test_immediate_inference_uses_the_identity_recorded_by_its_load() -> None:
    observations = complete_native_observations()
    observations["immediate.path.infer"] = _replace_adapter_state(
        observations["immediate.path.infer"],
        observations["immediate.tensor.load"].to_dict()["adapter_state"],
    )

    with pytest.raises(ScenarioContractError, match="immediate load identity changed"):
        validate_lifecycle_observations(observations)


def test_immediate_load_has_only_the_declared_adapter_identity() -> None:
    observations = complete_native_observations()
    extra_registration = _state(
        "native_lora",
        ("policy-a", "generated-path", "0"),
        ("policy-b", "unexpected-immediate-b", "0"),
        active="policy-a",
    )
    for name in ("immediate.path.load", "immediate.path.infer"):
        observations[name] = _replace_adapter_state(
            observations[name], extra_registration
        )

    with pytest.raises(ScenarioContractError, match="immediate load state changed"):
        validate_lifecycle_observations(observations)


def test_switch_returns_to_exact_a_identity_and_output() -> None:
    observations = complete_native_observations()
    observations["switch.a-again"] = observations["switch.b"]

    with pytest.raises(ScenarioContractError, match="switch did not restore adapter A"):
        validate_lifecycle_observations(observations)


def test_switch_preserves_both_registered_identities() -> None:
    observations = complete_native_observations()
    state = observations["switch.b"].to_dict()["adapter_state"]
    state["registered"][0]["id"] = "different-switch-a"
    state["cache_identity"]["policy-a"] = "different-switch-a"
    observations["switch.b"] = _replace_adapter_state(observations["switch.b"], state)

    with pytest.raises(
        ScenarioContractError, match="switch changed registered identity"
    ):
        validate_lifecycle_observations(observations)


def test_concurrent_stream_modes_have_identical_output() -> None:
    observations = complete_native_observations()
    observations["concurrent.non-stream"] = _replace_output_field(
        observations["concurrent.non-stream"], "output_ids"
    )

    with pytest.raises(ScenarioContractError, match="concurrent output changed"):
        validate_lifecycle_observations(observations)


def test_mixed_and_concurrent_inference_preserve_switched_state() -> None:
    observations = complete_native_observations()
    state = observations["mixed.base-a-b"].to_dict()["adapter_state"]
    state["active"] = None
    observations["mixed.base-a-b"] = _replace_adapter_state(
        observations["mixed.base-a-b"], state
    )

    with pytest.raises(ScenarioContractError, match="mixed inference changed state"):
        validate_lifecycle_observations(observations)


def test_upsert_waits_for_the_leased_identity_to_complete() -> None:
    observations = complete_native_observations()
    observations["upsert.while-leased"] = observations["upsert.after"]

    with pytest.raises(ScenarioContractError, match="upsert changed leased identity"):
        validate_lifecycle_observations(observations)


def test_leased_request_completes_with_pre_upsert_output() -> None:
    observations = complete_native_observations()
    observations["upsert.lease.complete"] = _replace_output_field(
        observations["upsert.lease.complete"], "text"
    )

    with pytest.raises(ScenarioContractError, match="leased request output changed"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    ("name", "message"),
    (
        (
            "stage.v1.old-active",
            "staged inference output changed",
        ),
        (
            "stage.v2.v1-active",
            "staged inference output changed",
        ),
        (
            "failure.update.previous",
            "failed update output changed",
        ),
    ),
)
@pytest.mark.parametrize(
    "field",
    (
        "output_ids",
        "text",
        "token_logprobs",
        "selected_logits",
        "selected_token_ids",
    ),
)
def test_retained_inference_output_is_exact(
    name: str, message: str, field: str
) -> None:
    observations = complete_native_observations()
    observations[name] = _replace_output_field(observations[name], field)

    with pytest.raises(ScenarioContractError, match=message):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    ("name", "expected_message"),
    (
        ("immediate.path.base", "base output changed after unload"),
        ("immediate.tensor.base", "base output changed after unload"),
        ("immediate.distributed.base", "base output changed after unload"),
        ("base.restored", "base output changed after unload"),
    ),
)
def test_each_post_unload_output_exactly_matches_initial_base(
    name: str, expected_message: str
) -> None:
    observations = complete_native_observations()
    observations[name] = replace(observations[name], output_ids=(999,))

    with pytest.raises(ScenarioContractError, match=expected_message):
        validate_lifecycle_observations(observations)


def test_stage_does_not_change_active_identity_or_version() -> None:
    observations = complete_native_observations()
    state = observations["stage.v2"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["active"] = {
        "name": "policy-b",
        "id": "generated-live-b",
        "version": "0",
    }
    observations["stage.v2"] = _replace_adapter_state(observations["stage.v2"], state)

    with pytest.raises(ScenarioContractError, match="stage changed active identity"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(("name", "version"), (("stage.v1", "9"), ("stage.v2", "9")))
def test_stage_records_the_exact_requested_version(name: str, version: str) -> None:
    observations = complete_native_observations()
    state = observations[name].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["staged"]["version"] = version
    observations[name] = _replace_adapter_state(observations[name], state)

    with pytest.raises(ScenarioContractError, match="stage recorded wrong identity"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("field", ("id", "version"))
def test_activation_promotes_the_exact_staged_identity(field: str) -> None:
    observations = complete_native_observations()
    state = observations["activate.v2"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    replacement_identity = {
        "name": "policy-a",
        "id": "generated-different" if field == "id" else "generated-live-a",
        "version": "9" if field == "version" else "3",
    }
    state["registered"][0].update(replacement_identity)
    state["active"] = replacement_identity
    state["cache_identity"]["policy-a"] = replacement_identity["id"]
    observations["activate.v2"] = _replace_adapter_state(
        observations["activate.v2"], state
    )

    with pytest.raises(
        ScenarioContractError, match="activation promoted wrong identity"
    ):
        validate_lifecycle_observations(observations)


def test_activation_clears_the_staged_identity() -> None:
    observations = complete_native_observations()
    state = observations["activate.v2"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["staged"] = {
        "name": "policy-a",
        "id": "generated-live-a",
        "version": "3",
        "pinned": False,
    }
    observations["activate.v2"] = _replace_adapter_state(
        observations["activate.v2"], state
    )

    with pytest.raises(ScenarioContractError, match="activation did not clear staged"):
        validate_lifecycle_observations(observations)


def test_failed_update_retains_the_previous_active_state() -> None:
    observations = complete_native_observations()
    state = observations["failure.update"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["active"] = {
        "name": "policy-b",
        "id": "generated-live-b",
        "version": "0",
    }
    observations["failure.update"] = _replace_adapter_state(
        observations["failure.update"], state
    )

    with pytest.raises(ScenarioContractError, match="failed update changed state"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    ("name", "field", "expected_message"),
    (
        (
            "failure.activation",
            "quarantined",
            "failed activation did not quarantine adapter",
        ),
        (
            "failure.unload",
            "tombstoned",
            "failed unload did not quarantine retry identity",
        ),
    ),
)
@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_failed_mutations_record_the_required_quarantine(
    name: str, field: str, expected_message: str, mode: str
) -> None:
    observations = complete_native_observations(mode)
    state = observations[name].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state[field] = []
    observations[name] = _replace_adapter_state(observations[name], state)

    with pytest.raises(ScenarioContractError, match=expected_message):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_failed_unload_retains_the_activation_quarantine(mode) -> None:
    observations = complete_native_observations(mode)
    state = observations["failure.unload"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["quarantined"] = []
    observations["failure.unload"] = _replace_adapter_state(
        observations["failure.unload"], state
    )

    with pytest.raises(
        ScenarioContractError, match="failed unload did not retain quarantine"
    ):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize(
    "field",
    (
        "id",
        "version",
        "pinned",
        "survivor_id",
        "survivor_version",
        "survivor_pinned",
        "survivor_missing",
    ),
)
def test_failed_unload_preserves_exact_tombstone_and_survivor(mode, field) -> None:
    """A changed old reference or unrelated survivor must invalidate the lifecycle."""
    observations = complete_native_observations(mode)
    for name in ("failure.unload", "failure.unload.quarantine"):
        state = observations[name].to_dict()["adapter_state"]
        if field == "survivor_missing":
            state["registered"] = []
            state["registry_occupancy"] = 0
        else:
            record = (
                state["registered"][0]
                if field.startswith("survivor_")
                else state["tombstoned"][0]
            )
            key = field.removeprefix("survivor_")
            record[key] = True if key == "pinned" else "changed"
            if key == "id":
                state["cache_identity"][record["name"]] = "changed"
        observations[name] = _replace_adapter_state(observations[name], state)

    with pytest.raises(
        ScenarioContractError, match="failed unload changed retry state"
    ):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_unload_retry_requires_the_exact_pre_retry_tombstone(mode) -> None:
    observations = complete_native_observations(mode)
    state = observations["failure.unload.quarantine"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["tombstoned"][0]["id"] = "different-retry-id"
    state["cache_identity"]["policy-a"] = "different-retry-id"
    observations["failure.unload.quarantine"] = _replace_adapter_state(
        observations["failure.unload.quarantine"], state
    )

    with pytest.raises(ScenarioContractError, match="unload retry identity changed"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_unload_retry_removes_only_the_declared_target(mode) -> None:
    observations = complete_native_observations(mode)
    validate_lifecycle_observations(observations)
    failed = observations["failure.unload"].adapter_state
    assert failed["registered"] == (
        {
            "name": "policy-b",
            "id": "generated-live-b",
            "version": str(int(mode == "native_oft")),
            "registry_slot": 0,
            "pinned": False,
        },
    )
    assert failed["registry_occupancy"] == 1
    assert failed["active"] is failed["staged"] is None
    assert failed["tombstoned"] == (
        {
            "name": "policy-a",
            "id": "generated-live-a",
            "version": str(3 + int(mode == "native_oft")),
            "pinned": False,
        },
    )
    assert failed["cache_identity"] == {
        "policy-a": "generated-live-a",
        "policy-b": "generated-live-b",
    }
    assert (
        observations["failure.unload.retry"].adapter_state["registered"]
        == failed["registered"]
    )


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize(
    "survivor_count,expected_active",
    (
        (0, None),
        (1, {"name": "policy-b", "id": "generated-live-b", "version": "0"}),
        (2, None),
    ),
)
def test_unload_retry_oracle_accepts_control_state(
    mode, survivor_count, expected_active
):
    """A cleared focus selects only a sole survivor, and the oracle must agree."""
    offset = int(mode == "native_oft")
    if expected_active is not None:
        expected_active = dict(expected_active, version=str(offset))
    engine = _FakeEngine()
    manager = engine.tokenizer_manager
    control = make_adapter_control(mode, engine)
    assert control.load_path("policy-a", "/adapters/a").success
    survivors = {
        name: _adapter_ref(mode, name, adapter_id, offset)
        for name, adapter_id in (
            ("policy-b", "generated-live-b"),
            ("policy-c", "generated-live-c"),
        )[:survivor_count]
    }
    target = _adapter_ref(mode, "policy-a", "generated-live-a", 3 + offset)
    setattr(manager, control.registry_attribute, _Registry(survivors))
    setattr(manager, control.cache_attribute, dict(survivors, **{"policy-a": target}))
    setattr(manager, control.quarantined_attribute, {"policy-a": "activation failed"})
    setattr(manager, control.tombstoned_attribute, {"policy-a": target})
    before = control.inspect_state()
    assert before["active"] is None

    # Construct the manager's successful retry result; the real control clears focus.
    assert control.unload("policy-a").success
    getattr(manager, control.cache_attribute).pop("policy-a")
    getattr(manager, control.quarantined_attribute).pop("policy-a")
    getattr(manager, control.tombstoned_attribute).pop("policy-a")
    after = control.inspect_state()
    assert after["active"] == expected_active

    observations = complete_native_observations(mode)
    if survivor_count == 1:
        assert observations["failure.unload.retry"].to_dict()["adapter_state"] == after
    for name in ("failure.unload", "failure.unload.quarantine"):
        observations[name] = _replace_adapter_state(observations[name], before)
    observations["failure.unload.retry"] = _replace_adapter_state(
        observations["failure.unload.retry"], after
    )
    scenarios._validate_unload_retry(observations, "policy-a")
    if survivor_count == 1:
        # The frozen full lifecycle specifically keeps B as its sole survivor.
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    "field",
    (
        "pinned",
        "cache",
        "active",
        "staged",
        "target_tombstone",
        "target_cache",
        "target_quarantine",
    ),
)
@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_unload_retry_preserves_complete_unrelated_state(field: str, mode: str) -> None:
    observations = complete_native_observations(mode)
    state = observations["failure.unload.retry"].to_dict()["adapter_state"]
    if field == "pinned":
        state["registered"][0]["pinned"] = True
    elif field == "cache":
        state["registered"][0]["id"] = "different-survivor-id"
        state["cache_identity"]["policy-b"] = "different-survivor-id"
        state["active"]["id"] = "different-survivor-id"
    elif field == "active":
        state["active"] = None
    elif field == "staged":
        state["staged"] = {
            "name": "policy-b",
            "id": "generated-live-b",
            "version": "1",
            "pinned": False,
        }
    elif field == "target_quarantine":
        state["quarantined"] = ["policy-a"]
    else:
        state["cache_identity"]["policy-a"] = "generated-live-a"
        if field == "target_tombstone":
            state["tombstoned"] = [
                {
                    "name": "policy-a",
                    "id": "generated-live-a",
                    "version": "3",
                    "pinned": False,
                }
            ]
    observations["failure.unload.retry"] = _replace_adapter_state(
        observations["failure.unload.retry"], state
    )

    with pytest.raises(ScenarioContractError, match="unload retry result changed"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize("field", ("registered", "active", "staged", "cache"))
def test_failed_unload_rejects_admission_and_cache_disagreement(mode, field):
    """The target cannot remain public or staged after a partial unload."""
    observations = complete_native_observations(mode)
    state = observations["failure.unload"].to_dict()["adapter_state"]
    if field == "registered":
        state["registered"].insert(
            0,
            {
                "name": "policy-a",
                "id": "generated-live-a",
                "version": "3",
                "registry_slot": 0,
                "pinned": False,
            },
        )
        state["registered"][1]["registry_slot"] = 1
        state["registry_occupancy"] = 2
    elif field == "active":
        state["active"] = {"name": "policy-a", "id": "generated-live-a", "version": "3"}
    elif field == "staged":
        state["staged"] = {
            "name": "policy-a",
            "id": "generated-live-a",
            "version": "4",
            "pinned": False,
        }
    else:
        state["cache_identity"]["policy-a"] = "wrong-id"
    observations["failure.unload"] = replace(
        observations["failure.unload"], adapter_state=state
    )
    with pytest.raises(ScenarioContractError):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize("mutation", (None, "tombstone", "quarantine", "cache"))
def test_unload_preserves_unrelated_cleanup_entries(mode, mutation):
    """Retry must clean only its target, even with other failed adapters present."""
    observations = complete_native_observations(mode)
    for name in (
        "failure.activation",
        "failure.activation.previous",
        "failure.unload",
        "failure.unload.quarantine",
        "failure.unload.retry",
    ):
        state = observations[name].to_dict()["adapter_state"]
        state["tombstoned"].append(
            {"name": "policy-z", "id": "id-z", "version": "7", "pinned": True}
        )
        state["quarantined"].append("policy-y")
        state["cache_identity"].update({"policy-z": "id-z", "policy-y": "id-y"})
        if name == "failure.unload.retry":
            if mutation == "tombstone":
                state["tombstoned"] = []
            elif mutation == "quarantine":
                state["quarantined"] = []
            elif mutation == "cache":
                state["cache_identity"].pop("policy-y")
        observations[name] = _replace_adapter_state(observations[name], state)
    if mutation is None:
        validate_lifecycle_observations(observations)
    else:
        with pytest.raises(ScenarioContractError, match="unload retry result changed"):
            validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("active", "cancellation lost retained adapter identity"),
        ("cache", "cancellation lost retained adapter cache"),
        ("quarantine", "cancellation introduced quarantine"),
    ),
)
def test_cancellation_and_pause_are_anchored_to_a_retained_adapter(
    mutation: str, message: str
) -> None:
    observations = complete_native_observations()
    names = (
        "cancel.lease-drain",
        "cancel.fan-out",
        "cancel.publication",
        "cancel.rollback",
        "cancel.eviction",
        "pause.retain-kv",
        "pause.activate-rejected",
        "pause.resume",
    )
    for name in names:
        state = observations[name].to_dict()["adapter_state"]
        if mutation == "active":
            state["active"] = None
        elif mutation == "cache":
            state["cache_identity"] = {}
        else:
            state["quarantined"] = ["policy-a"]
        observations[name] = _replace_adapter_state(observations[name], state)

    with pytest.raises(ScenarioContractError, match=message):
        validate_lifecycle_observations(observations)


def test_paused_request_resume_requires_output_evidence() -> None:
    observations = complete_native_observations()
    observations["pause.resume"] = _observation(
        observations["pause.resume"].to_dict()["adapter_state"]
    )

    with pytest.raises(ScenarioContractError, match="missing inference output"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_registry_eviction_retains_the_exact_reload_cache(mode: str) -> None:
    validate_lifecycle_observations(complete_native_observations(mode))


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
@pytest.mark.parametrize("mutation", ("missing", "wrong-id"))
def test_registry_eviction_requires_the_victims_cached_identity(
    mode: str, mutation: str
) -> None:
    observations = complete_native_observations(mode)
    state = observations["registry.evict"].to_dict()["adapter_state"]
    if mutation == "missing":
        del state["cache_identity"]["policy-a"]
    else:
        state["cache_identity"]["policy-a"] = "wrong-victim-id"
    observations["registry.evict"] = _replace_adapter_state(
        observations["registry.evict"], state
    )

    with pytest.raises(ScenarioContractError, match="registry eviction result changed"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_registry_eviction_removes_the_victim_from_admission(mode: str) -> None:
    observations = complete_native_observations(mode)
    observations["registry.evict"] = _replace_adapter_state(
        observations["registry.evict"],
        observations["registry.fill"].to_dict()["adapter_state"],
    )

    with pytest.raises(ScenarioContractError, match="registry eviction result changed"):
        validate_lifecycle_observations(observations)


def test_registry_eviction_removes_only_the_declared_victim() -> None:
    observations = complete_native_observations()
    wrong_survivor = _state(
        "native_lora",
        ("policy-a", "generated-registry-a", "0"),
        active="policy-a",
    )
    observations["registry.evict"] = _replace_adapter_state(
        observations["registry.evict"], wrong_survivor
    )

    with pytest.raises(ScenarioContractError, match="registry eviction result changed"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    "field",
    (
        "id",
        "version",
        "pinned",
        "cache",
        "quarantined",
        "tombstoned",
        "active",
        "staged",
    ),
)
@pytest.mark.parametrize("mode", ("native_lora", "native_oft"))
def test_registry_eviction_preserves_the_complete_survivor_state(
    mode: str, field: str
) -> None:
    observations = complete_native_observations(mode)
    state = observations["registry.evict"].to_dict()["adapter_state"]
    if field in ("id", "version"):
        value = (
            "wrong-survivor-id" if field == "id" else str(1 + int(mode == "native_oft"))
        )
        state["registered"][0][field] = value
        state["active"][field] = value
        if field == "id":
            state["cache_identity"]["policy-b"] = value
    elif field == "pinned":
        state["registered"][0]["pinned"] = True
    elif field == "cache":
        del state["cache_identity"]["policy-b"]
    elif field == "quarantined":
        state["quarantined"] = ["policy-a"]
    elif field == "tombstoned":
        state["tombstoned"] = [
            {
                "name": "policy-a",
                "id": "generated-registry-a",
                "version": "0",
                "pinned": False,
            }
        ]
    elif field == "active":
        state["active"] = None
    else:
        state["staged"] = {
            "name": "policy-b",
            "id": "generated-registry-b",
            "version": "1",
            "pinned": False,
        }
    observations["registry.evict"] = _replace_adapter_state(
        observations["registry.evict"], state
    )

    with pytest.raises(ScenarioContractError, match="registry eviction result changed"):
        validate_lifecycle_observations(observations)


def test_final_unload_must_leave_a_completely_empty_native_state() -> None:
    observations = complete_native_observations()
    observations["unload.final"] = _replace_adapter_state(
        observations["unload.final"],
        _state(
            "native_lora",
            ("policy-b", "generated-registry-b", "0"),
            active="policy-b",
        ),
    )

    with pytest.raises(ScenarioContractError, match="final adapter state is not empty"):
        validate_lifecycle_observations(observations)


def test_restart_reproduces_the_startup_adapter_identity() -> None:
    observations = complete_native_observations()
    observations["restart.identity"] = _replace_adapter_state(
        observations["restart.identity"],
        _state(
            "native_lora",
            ("policy-a", "different-restart-id", "0"),
            active="policy-a",
        ),
    )

    with pytest.raises(ScenarioContractError, match="restart identity changed"):
        validate_lifecycle_observations(observations)


def test_declared_product_error_code_is_exact() -> None:
    observations = complete_native_observations()
    observations["reject.stale"] = replace(
        observations["reject.stale"],
        error={
            "kind": "product_rejection",
            "code": "wrong_id",
            "message": "expected stale_version",
        },
    )

    with pytest.raises(ScenarioContractError, match="unexpected lifecycle error code"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize("mutation", ("missing", "extra", "out-of-order"))
def test_lifecycle_observation_keys_are_exact_and_ordered(mutation: str) -> None:
    observations = complete_native_observations()
    if mutation == "missing":
        observations.pop("stage.v1")
    elif mutation == "extra":
        observations["undeclared.step"] = observations["base.initial"]
    else:
        items = list(observations.items())
        items[0], items[1] = items[1], items[0]
        observations = dict(items)

    with pytest.raises(ScenarioContractError, match="lifecycle observation order"):
        validate_lifecycle_observations(observations)


def test_lifecycle_validator_validates_every_observation() -> None:
    observations = complete_native_observations()
    observations["switch.b"] = replace(observations["switch.b"], token_logprobs=(0,))

    with pytest.raises(ScenarioContractError, match="invalid lifecycle observation"):
        validate_lifecycle_observations(observations)


def test_lifecycle_mode_is_inferred_from_complete_consistent_state() -> None:
    observations = complete_native_observations()
    state = observations["switch.b"].to_dict()["adapter_state"]
    assert isinstance(state, dict)
    state["mode"] = "native_oft"
    observations["switch.b"] = _replace_adapter_state(observations["switch.b"], state)

    with pytest.raises(ScenarioContractError, match="lifecycle modes differ"):
        validate_lifecycle_observations(observations)


def test_execute_lifecycle_consumes_the_declarative_table_in_order() -> None:
    expected = complete_native_observations()

    class RecordingExecutor:
        def __init__(self) -> None:
            self.seen: list[LifecycleStep] = []

        def execute(self, step: LifecycleStep) -> Observation:
            self.seen.append(step)
            return expected[step.name]

    executor = RecordingExecutor()
    actual = execute_lifecycle("native_lora", executor)

    assert actual == expected
    assert executor.seen == list(scenarios.NATIVE_LIFECYCLE_STEPS)


def test_execute_lifecycle_rejects_observations_from_another_native_mode() -> None:
    wrong_mode = complete_native_observations("native_oft")

    class WrongModeExecutor:
        def execute(self, step: LifecycleStep) -> Observation:
            return wrong_mode[step.name]

    with pytest.raises(ScenarioContractError, match="does not match requested mode"):
        execute_lifecycle("native_lora", WrongModeExecutor())


@pytest.mark.parametrize("mode", ("base", "native_lora", "native_oft"))
def test_native_modes_are_the_only_executable_modes(mode: str) -> None:
    _case(mode).validate()
    if mode != "base":
        assert lifecycle_transition_names(mode)


@pytest.mark.parametrize(
    "retired",
    ("legacy_" + "lora", "legacy_" + "oft", "canonical_" + "oft"),
)
@pytest.mark.parametrize("revision_kind", ("source", "candidate"))
def test_retired_harness_modes_fail_closed(retired: str, revision_kind: str) -> None:
    with pytest.raises(BundleValidationError, match="case_key.mode is invalid"):
        _case(retired).validate()
    with pytest.raises(ValueError, match="unknown .* adapter mode"):
        mode_server_args(revision_kind, retired)


def test_both_revisions_select_native_runtime_surfaces() -> None:
    expected_oft = ("--peft-method", "oft", "--oft-type", "oft")
    assert mode_server_args("source", "native_oft") == expected_oft
    assert mode_server_args("candidate", "native_oft") == expected_oft
    expected_lora = ("--enable-lora", "--enable-lora-staging")
    assert mode_server_args("source", "native_lora") == expected_lora
    assert mode_server_args("candidate", "native_lora") == expected_lora


@pytest.mark.parametrize(
    ("mode", "prefix"),
    (("native_lora", "lora"), ("native_oft", "oft")),
)
def test_control_uses_each_distinct_native_immediate_path(
    mode: str, prefix: str
) -> None:
    engine = _FakeEngine()
    control = make_adapter_control(mode, engine)

    results = (
        control.load_path("policy-a", "/adapters/a", pinned=True),
        control.load_tensors("policy-a", FAKE_TENSORS, FAKE_CONFIG),
        control.load_distributed("policy-a", DISTRIBUTED_PAYLOAD, "group-a"),
        control.unload("policy-a"),
    )

    assert all(result.success for result in results)
    assert [name for name, _, _ in engine.calls] == [
        f"load_{prefix}_adapter",
        f"load_{prefix}_adapter_from_tensors",
        f"load_{prefix}_adapter_from_distributed",
        f"unload_{prefix}_adapter",
    ]
    assert engine.calls[0][1:] == (
        ("policy-a", "/adapters/a"),
        {"pinned": True},
    )
    assert engine.calls[1][1][:3] == (
        "policy-a",
        FAKE_TENSORS,
        FAKE_CONFIG,
    )
    assert engine.calls[2][1] == (
        "policy-a",
        FAKE_CONFIG,
        ["model.layers.0.self_attn.q_proj.weight"],
        ["float32"],
        [[8, 8]],
    )
    assert engine.calls[2][2]["group_name"] == "group-a"


@pytest.mark.parametrize(
    ("mode", "load_format"),
    (("native_lora", "lora_adapter"), ("native_oft", "oft_adapter")),
)
def test_stage_and_activate_use_exact_shared_production_requests(
    mode: str, load_format: str
) -> None:
    engine = _FakeEngine()
    control = make_adapter_control(mode, engine)
    identity = AdapterIdentity("policy-a", "id-a", "2")

    staged = control.stage(identity, DISTRIBUTED_PAYLOAD, "group-a")
    activated = control.activate(identity)

    assert staged == ControlResult(True, "staged", {}, None, None)
    assert activated == ControlResult(True, "activated", {}, None, None)
    assert [name for name, _, _ in engine.tokenizer_manager.calls] == [
        "update_adapter_from_distributed",
        "activate_adapter_version",
    ]
    _, stage_request, stage_http_request = engine.tokenizer_manager.calls[0]
    assert stage_http_request is None
    assert stage_request.names == ["model.layers.0.self_attn.q_proj.weight"]
    assert stage_request.dtypes == ["float32"]
    assert stage_request.shapes == [[8, 8]]
    assert stage_request.group_name == "group-a"
    assert stage_request.adapter_config == FAKE_CONFIG
    assert stage_request.adapter_name == "policy-a"
    assert stage_request.adapter_id == "id-a"
    assert stage_request.adapter_version == "2"
    assert stage_request.load_format == load_format
    assert stage_request.double_buffer is True

    _, activate_request, activate_http_request = engine.tokenizer_manager.calls[1]
    assert activate_http_request is None
    assert activate_request.adapter_name == "policy-a"
    assert activate_request.adapter_id == "id-a"
    assert activate_request.adapter_version == "2"
    assert activate_request.load_format == load_format


def test_lora_upsert_uses_request_field_missing_from_engine_wrapper() -> None:
    engine = _FakeEngine()
    control = make_adapter_control("native_lora", engine)

    tensor_result = control.load_tensors(
        "policy-a", FAKE_TENSORS, FAKE_CONFIG, upsert=True
    )
    distributed_result = control.load_distributed(
        "policy-a", DISTRIBUTED_PAYLOAD, "group-a", upsert=True
    )

    assert tensor_result.success and distributed_result.success
    assert engine.calls == []
    assert engine.serialized == [(FAKE_TENSORS, None)]
    tensor_call, distributed_call = engine.tokenizer_manager.calls
    assert tensor_call[0] == "load_lora_adapter_from_tensors"
    assert tensor_call[1].lora_name == "policy-a"
    assert tensor_call[1].config_dict == FAKE_CONFIG
    assert tensor_call[1].serialized_named_tensors == [b"rank-0", b"rank-1"]
    assert tensor_call[1].upsert is True
    assert distributed_call[0] == "load_lora_adapter_from_distributed"
    assert distributed_call[1].lora_name == "policy-a"
    assert distributed_call[1].config_dict == FAKE_CONFIG
    assert distributed_call[1].names == ["model.layers.0.self_attn.q_proj.weight"]
    assert distributed_call[1].dtypes == ["float32"]
    assert distributed_call[1].shapes == [[8, 8]]
    assert distributed_call[1].group_name == "group-a"
    assert distributed_call[1].upsert is True


def test_control_preserves_explicit_product_failure() -> None:
    engine = _FakeEngine()
    engine.results["load_oft_adapter"] = SimpleNamespace(
        success=False,
        error_message="adapter rejected",
        loaded_adapters=None,
    )
    control = make_adapter_control("native_oft", engine)

    result = control.load_path("policy-a", "/adapters/a")

    assert result == ControlResult(False, "adapter rejected", {}, None, None)
    assert control.inspect_state()["active"] is None


def _adapter_ref(mode: str, name: str, adapter_id: str, version: int, pinned=False):
    if mode == "native_lora":
        return SimpleNamespace(
            lora_name=name,
            lora_id=adapter_id,
            version=version,
            pinned=pinned,
        )
    return SimpleNamespace(
        adapter_name=name,
        adapter_id=adapter_id,
        adapter_version=version,
        pinned=pinned,
    )


@pytest.mark.parametrize(
    "mode,prefix", (("native_lora", "lora"), ("native_oft", "oft"))
)
def test_control_tombstones_preserve_exact_failed_references(mode, prefix):
    """Name-only capture loses the retry ID, old version and pinning."""
    engine = _FakeEngine()
    manager = engine.tokenizer_manager
    setattr(
        manager,
        f"failed_{prefix}_unloads",
        {
            "policy-z": _adapter_ref(mode, "policy-z", "old-z", 9),
            "policy-a": _adapter_ref(mode, "policy-a", "old-a", 3, pinned=True),
        },
    )
    # Deliberately conflicting cache: capture must read the tombstone itself.
    cache_name = "lora_ref_cache" if prefix == "lora" else "peft_ref_cache"
    setattr(
        manager,
        cache_name,
        {
            "policy-a": _adapter_ref(mode, "policy-a", "cache-a", 4),
        },
    )
    assert make_adapter_control(mode, engine).inspect_state()["tombstoned"] == [
        {"name": "policy-a", "id": "old-a", "version": "3", "pinned": True},
        {"name": "policy-z", "id": "old-z", "version": "9", "pinned": False},
    ]


@pytest.mark.parametrize(
    "mode,prefix", (("native_lora", "lora"), ("native_oft", "oft"))
)
def test_control_rejects_mismatched_tombstone_mapping_key(mode, prefix):
    """A mapping key must not silently relabel a failed reference."""
    engine = _FakeEngine()
    setattr(
        engine.tokenizer_manager,
        f"failed_{prefix}_unloads",
        {
            "policy-a": _adapter_ref(mode, "policy-b", "old-b", 3),
        },
    )
    with pytest.raises(ScenarioContractError, match="tombstone.*key.*name"):
        make_adapter_control(mode, engine).inspect_state()


@pytest.mark.parametrize(
    ("mode", "registry_name", "pending_name", "failed_name", "cache_name"),
    (
        (
            "native_lora",
            "lora_registry",
            "pending_lora_stage",
            "failed_lora_activations",
            "lora_ref_cache",
        ),
        (
            "native_oft",
            "peft_registry",
            "pending_oft_stage",
            "failed_oft_activations",
            "peft_ref_cache",
        ),
    ),
)
def test_control_state_is_sorted_and_normalized_across_adapter_types(
    mode: str,
    registry_name: str,
    pending_name: str,
    failed_name: str,
    cache_name: str,
) -> None:
    engine = _FakeEngine()
    manager = engine.tokenizer_manager
    policy_a = _adapter_ref(mode, "policy-a", "id-a", 2, pinned=True)
    policy_b = _adapter_ref(mode, "policy-b", "id-b", 1)
    policy_w = _adapter_ref(mode, "policy-w", "id-w", 4, pinned=True)
    policy_x = _adapter_ref(mode, "policy-x", "id-x", 5)
    pending = _adapter_ref(mode, "policy-a", "id-a", 3, pinned=True)
    setattr(
        manager,
        registry_name,
        _Registry({"policy-b": policy_b, "policy-a": policy_a}),
    )
    setattr(manager, pending_name, pending)
    setattr(manager, failed_name, {"policy-z": "bad", "policy-y": "bad"})
    setattr(
        manager,
        failed_name.replace("activations", "unloads"),
        {"policy-x": policy_x, "policy-w": policy_w},
    )
    setattr(
        manager,
        cache_name,
        {
            "policy-b": policy_b,
            "policy-a": policy_a,
            "policy-w": policy_w,
            "policy-x": policy_x,
        },
    )
    control = make_adapter_control(mode, engine)
    assert control.load_path("policy-a", "/adapters/a").success

    state = control.inspect_state()
    assert state == {
        "mode": mode,
        "registered": [
            {
                "name": "policy-a",
                "id": "id-a",
                "version": "2",
                "registry_slot": 0,
                "pinned": True,
            },
            {
                "name": "policy-b",
                "id": "id-b",
                "version": "1",
                "registry_slot": 1,
                "pinned": False,
            },
        ],
        "active": {"name": "policy-a", "id": "id-a", "version": "2"},
        "staged": {
            "name": "policy-a",
            "id": "id-a",
            "version": "3",
            "pinned": True,
        },
        "registry_occupancy": 2,
        "quarantined": ["policy-y", "policy-z"],
        "tombstoned": [
            {"name": "policy-w", "id": "id-w", "version": "4", "pinned": True},
            {"name": "policy-x", "id": "id-x", "version": "5", "pinned": False},
        ],
        "cache_identity": {
            "policy-a": "id-a",
            "policy-b": "id-b",
            "policy-w": "id-w",
            "policy-x": "id-x",
        },
    }
    _observation(state)


def test_base_control_is_empty_and_rejects_adapter_operations() -> None:
    control = make_adapter_control("base", _FakeEngine())

    assert control.inspect_state() == {
        "mode": "base",
        "registered": [],
        "active": None,
        "staged": None,
        "registry_occupancy": 0,
        "quarantined": [],
        "tombstoned": [],
        "cache_identity": {},
    }
    with pytest.raises(ScenarioContractError, match="base mode"):
        control.load_path("policy-a", "/adapters/a")


def test_result_normalization_rejects_implicit_success_and_normalizes_versions():
    result = normalize_control_result(
        SimpleNamespace(
            success=True,
            message="ok",
            loaded_adapters={"policy-a": "id-a"},
            active_adapter_version=2,
            staged_adapter_version=3,
        )
    )

    assert result == ControlResult(
        True,
        "ok",
        {"policy-a": "id-a"},
        "2",
        "3",
    )
    with pytest.raises(ScenarioContractError, match="explicit success"):
        normalize_control_result(SimpleNamespace(message="ambiguous"))
    with pytest.raises(ScenarioContractError, match="success must be boolean"):
        normalize_control_result(("yes", "ambiguous"))


def test_native_control_surface_is_exported_from_harness_package() -> None:
    import adapter_equivalence as harness

    assert harness.AdapterIdentity is AdapterIdentity
    assert harness.AdapterControl is AdapterControl
    assert harness.BaseControl is BaseControl
    assert harness.ControlResult is ControlResult
    assert harness.DistributedPayload is DistributedPayload
    assert harness.NativeLoRAControl is NativeLoRAControl
    assert harness.NativeOFTControl is NativeOFTControl
    assert harness.make_adapter_control is make_adapter_control
    assert harness.normalize_control_result is normalize_control_result


@pytest.mark.parametrize(
    "mode,phase",
    (
        ("native_lora", "writer"),
        ("native_oft", "counter"),
        ("native_oft", "writer"),
        ("native_oft", "child-writer"),
    ),
)
def test_phase_observer_requires_matching_update_at_actual_lease_wait(mode, phase):
    """Defect: OFT's shielded child wait is missed or an unrelated child qualifies."""
    from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

    lock_type = runpy.run_path(
        str(
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/utils/aio_rwlock.py"
        )
    )["RWLock"]

    class Counter:
        def __init__(self):
            self.released = asyncio.Event()
            self.waiting = asyncio.Event()

        def value(self):
            return 0 if self.released.is_set() else 1

        async def wait_for_zero(self):
            self.waiting.set()
            await self.released.wait()

    class Registry(_Registry):
        async def wait_for_unload(self, uid):
            await self._counters[uid].wait_for_zero()

    class Manager(_FakeTokenizerManager):
        async def load_lora_adapter_from_distributed(self, obj):
            self.entered.set()
            await self.proceed.wait()
            async with self.model_update_lock.writer_lock:
                return True

        async def load_oft_adapter_from_distributed(self, obj):
            self.entered.set()
            await self.proceed.wait()
            if phase == "writer":
                async with self.model_update_lock.writer_lock:
                    return True
            return await self._run_oft_wire_load(obj, hold_dispatch=True)

        async def _run_oft_wire_load(self, obj, hold_dispatch=False):
            async def dispatch_and_finish():
                if hold_dispatch:
                    self.child_entered.set()
                    await self.child_proceed.wait()
                return await self._prepare_oft_wire_load(obj)

            return await finish_irreversible_update(dispatch_and_finish)

        async def _prepare_oft_wire_load(self, obj):
            if phase == "child-writer":
                async with self.model_update_lock.writer_lock:
                    return True
            del self.peft_registry.records[obj.adapter_name]
            self.peft_ref_cache.pop(obj.adapter_name)
            adapter_id = "id-a"
            await self.peft_registry.wait_for_unload(adapter_id)
            return True

    async def exercise():
        manager = Manager()
        manager.entered, manager.proceed = asyncio.Event(), asyncio.Event()
        manager.child_entered, manager.child_proceed = asyncio.Event(), asyncio.Event()
        manager.model_update_lock = lock_type()
        counter = Counter()
        reference = _adapter_ref(mode, "policy-a", "id-a", 0)
        registry = Registry({"policy-a": reference})
        registry._counters = {"id-a": counter, "wrong-id": Counter()}
        manager.lora_registry = manager.peft_registry = registry
        manager.lora_ref_cache = manager.peft_ref_cache = {"policy-a": reference}
        engine = SimpleNamespace(
            loop=asyncio.get_running_loop(), tokenizer_manager=manager
        )
        control = make_adapter_control(mode, engine)
        await manager.model_update_lock.acquire_reader()
        method = (
            manager.load_lora_adapter_from_distributed
            if mode == "native_lora"
            else manager.load_oft_adapter_from_distributed
        )
        obj = SimpleNamespace(lora_name="policy-a", adapter_name="policy-a")
        task = asyncio.create_task(method(obj))
        try:
            await manager.entered.wait()
            observed = control.observe_lease_wait("policy-a", "id-a")
            await asyncio.sleep(0)
            assert not observed.done()
            assert not control._at_lease_wait("policy-a", "id-a")
            if mode == "native_oft" and phase == "counter":
                # Same manager, exact request object, name and ID; not the public
                # request's child. Merely finding this waiter must not qualify.
                unrelated = asyncio.create_task(manager._run_oft_wire_load(obj))
                await counter.waiting.wait()
                manager.proceed.set()
                await manager.child_entered.wait()
                try:
                    # The real parent is now inside the shielding helper, but its
                    # own child has not reached a lease barrier yet.
                    assert not control._at_lease_wait("policy-a", "id-a")
                    assert not observed.done()
                finally:
                    counter.released.set()
                    await unrelated
                    registry.records["policy-a"] = reference
                    manager.peft_ref_cache["policy-a"] = reference
                    counter.released.clear()
                    counter.waiting.clear()
            manager.proceed.set()
            manager.child_proceed.set()
            state = await asyncio.wait_for(asyncio.wrap_future(observed), 0.1)
            assert not control._at_lease_wait("unrelated", "id-a")
            assert not control._at_lease_wait("policy-a", "wrong-id")
            assert [record["name"] for record in state["registered"]] == (
                [] if phase == "counter" else ["policy-a"]
            )
        finally:
            manager.proceed.set()
            manager.child_proceed.set()
            await manager.model_update_lock.release_reader()
            counter.released.set()
            await task

    asyncio.run(exercise())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_oft_restart_allows_new_process_id_but_preserves_live_identity():
    observations = complete_native_observations("native_oft")
    restarted = _state(
        "native_oft", ("policy-a", "fresh-process-id", "1"), active="policy-a"
    )
    for name in ("restart.same-manifest", "restart.identity"):
        observations[name] = _replace_adapter_state(observations[name], restarted)
    validate_lifecycle_observations(observations)
    observations["restart.identity"] = _replace_adapter_state(
        observations["restart.identity"],
        _state("native_oft", ("policy-a", "changed-live-id", "1"), active="policy-a"),
    )
    with pytest.raises(ScenarioContractError, match="restart identity changed"):
        validate_lifecycle_observations(observations)


@pytest.mark.parametrize(
    "field,value", [("version", "2"), ("pinned", True), ("registry_slot", 1)]
)
def test_oft_restart_new_id_cannot_hide_other_state_changes(field, value):
    observations = complete_native_observations("native_oft")
    restarted = _state(
        "native_oft", ("policy-a", "fresh-process-id", "1"), active="policy-a"
    )
    restarted["registered"][0][field] = value
    if field == "version":
        restarted["active"][field] = value
    with pytest.raises((ScenarioContractError, BundleValidationError)):
        for name in ("restart.same-manifest", "restart.identity"):
            observations[name] = _replace_adapter_state(observations[name], restarted)
        validate_lifecycle_observations(observations)

"""Frozen lifecycle contracts for adapter-equivalence runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from .schema import BundleValidationError, Observation


class ScenarioContractError(ValueError):
    """Raised when a lifecycle run violates the frozen oracle contract."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None):
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class LifecycleStep:
    """One observable transition in the adapter lifecycle state machine."""

    name: str
    action: str
    adapter: str | None = None
    version: str | None = None
    prompt_id: str | None = None
    stream: bool | None = None
    max_new_tokens: int | None = None
    input_kind: str | None = None
    expected_error_code: str | None = None


class LifecycleExecutor(Protocol):
    def execute(self, step: LifecycleStep) -> Observation:
        """Execute one transition and return its immutable observation."""


BASE_LIFECYCLE_STEPS = (
    LifecycleStep("base.initial", "generate", prompt_id="factual"),
    LifecycleStep(
        "concurrent.non-stream", "concurrent", prompt_id="batch-8", stream=False
    ),
    LifecycleStep("restart.same-manifest", "restart"),
    LifecycleStep("restart.identity", "generate", prompt_id="factual"),
)


NATIVE_LIFECYCLE_STEPS = (
    LifecycleStep("base.initial", "generate", prompt_id="factual"),
    LifecycleStep(
        "startup.adapter",
        "startup_generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("immediate.path.load", "load", adapter="policy-a", input_kind="path"),
    LifecycleStep(
        "immediate.path.infer",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("immediate.path.unload", "unload", adapter="policy-a"),
    LifecycleStep("immediate.path.base", "generate", prompt_id="factual"),
    LifecycleStep(
        "immediate.tensor.load", "load", adapter="policy-a", input_kind="tensors"
    ),
    LifecycleStep(
        "immediate.tensor.infer",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("immediate.tensor.unload", "unload", adapter="policy-a"),
    LifecycleStep("immediate.tensor.base", "generate", prompt_id="factual"),
    LifecycleStep(
        "immediate.distributed.load",
        "load",
        adapter="policy-a",
        input_kind="distributed",
    ),
    LifecycleStep(
        "immediate.distributed.infer",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("immediate.distributed.unload", "unload", adapter="policy-a"),
    LifecycleStep("immediate.distributed.base", "generate", prompt_id="factual"),
    LifecycleStep("switch.a", "generate", adapter="policy-a", prompt_id="factual"),
    LifecycleStep("switch.b", "generate", adapter="policy-b", prompt_id="factual"),
    LifecycleStep(
        "switch.a-again", "generate", adapter="policy-a", prompt_id="factual"
    ),
    LifecycleStep("mixed.base-a-b", "mixed_batch", prompt_id="batch-8"),
    LifecycleStep("concurrent.stream", "concurrent", prompt_id="batch-8", stream=True),
    LifecycleStep(
        "concurrent.non-stream", "concurrent", prompt_id="batch-8", stream=False
    ),
    LifecycleStep(
        "upsert.lease.begin",
        "begin_lease",
        adapter="policy-a",
        prompt_id="factual",
        stream=True,
    ),
    LifecycleStep(
        "upsert.while-leased",
        "upsert",
        adapter="policy-a",
        input_kind="distributed",
    ),
    LifecycleStep("upsert.lease.complete", "complete_lease", adapter="policy-a"),
    LifecycleStep("upsert.after", "generate", adapter="policy-a", prompt_id="factual"),
    LifecycleStep(
        "stage.v1",
        "stage",
        adapter="policy-a",
        version="2",
        input_kind="distributed",
    ),
    LifecycleStep(
        "stage.v1.old-active",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("activate.v1", "activate", adapter="policy-a", version="2"),
    LifecycleStep(
        "stage.v1.active",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep(
        "stage.v2",
        "stage",
        adapter="policy-a",
        version="3",
        input_kind="distributed",
    ),
    LifecycleStep(
        "stage.v2.v1-active",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep("activate.v2", "activate", adapter="policy-a", version="3"),
    LifecycleStep(
        "stage.v2.active",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep(
        "reject.duplicate",
        "reject_duplicate",
        adapter="policy-a",
        version="3",
        expected_error_code="duplicate_version",
    ),
    LifecycleStep(
        "reject.stale",
        "reject_stale",
        adapter="policy-a",
        version="2",
        expected_error_code="stale_version",
    ),
    LifecycleStep(
        "reject.wrong-id",
        "reject_wrong_id",
        adapter="policy-a",
        version="4",
        expected_error_code="wrong_id",
    ),
    LifecycleStep(
        "reject.wrong-name",
        "reject_wrong_name",
        adapter="missing-policy",
        version="4",
        expected_error_code="wrong_name",
    ),
    LifecycleStep(
        "reject.invalid-config",
        "reject_invalid_config",
        adapter="policy-a",
        version="4",
        input_kind="distributed",
        expected_error_code="invalid_config",
    ),
    LifecycleStep(
        "reject.unsupported-target",
        "reject_unsupported_target",
        adapter="policy-a",
        input_kind="tensors",
        expected_error_code="unsupported_target",
    ),
    LifecycleStep(
        "failure.update",
        "inject_update_failure",
        adapter="policy-a",
        version="4",
        input_kind="distributed",
        expected_error_code="update_failure",
    ),
    LifecycleStep(
        "failure.update.previous",
        "generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
    LifecycleStep(
        "failure.activation",
        "inject_activation_failure",
        adapter="policy-a",
        version="4",
        expected_error_code="activation_failure",
    ),
    LifecycleStep("failure.activation.previous", "inspect_state", adapter="policy-a"),
    LifecycleStep(
        "failure.unload",
        "inject_unload_failure",
        adapter="policy-a",
        expected_error_code="unload_failure",
    ),
    LifecycleStep("failure.unload.quarantine", "inspect_state", adapter="policy-a"),
    LifecycleStep("failure.unload.retry", "retry_unload", adapter="policy-a"),
    LifecycleStep("cancel.lease-drain", "cancel_lease_drain", adapter="policy-a"),
    LifecycleStep("cancel.fan-out", "cancel_fan_out", adapter="policy-a"),
    LifecycleStep("cancel.publication", "cancel_publication", adapter="policy-a"),
    LifecycleStep("cancel.rollback", "cancel_rollback", adapter="policy-a"),
    LifecycleStep("cancel.eviction", "cancel_eviction", adapter="policy-a"),
    LifecycleStep(
        "pause.retain-kv",
        "begin_paused_request",
        adapter="policy-a",
        prompt_id="long-prefix",
        stream=True,
    ),
    LifecycleStep(
        "pause.activate-rejected",
        "reject_paused_activation",
        adapter="policy-a",
        version="1",
        expected_error_code="paused_requests_active",
    ),
    LifecycleStep("pause.resume", "resume_paused_request", adapter="policy-a"),
    LifecycleStep("registry.fill", "fill_registry", adapter="policy-b"),
    LifecycleStep("registry.evict", "evict_registry", adapter="policy-a"),
    # Eviction retains A's reload cache: final cleanup must unload B, reload A
    # from its recorded immutable fixture path, then explicitly unload A.
    LifecycleStep("unload.final", "unload", adapter="policy-b"),
    LifecycleStep("base.restored", "generate", prompt_id="factual"),
    LifecycleStep("restart.same-manifest", "restart"),
    LifecycleStep(
        "restart.identity",
        "startup_generate",
        adapter="policy-a",
        prompt_id="factual",
    ),
)


_LIFECYCLE_STEPS_BY_MODE = {
    "base": BASE_LIFECYCLE_STEPS,
    "native_lora": NATIVE_LIFECYCLE_STEPS,
    # OFT registrations start at one; LoRA starts at zero. The leased upsert
    # advances each once, so OFT's subsequent explicit versions are one higher.
    "native_oft": tuple(
        (
            replace(step, version=str(int(step.version) + 1))
            if step.version is not None
            else step
        )
        for step in NATIVE_LIFECYCLE_STEPS
    ),
}


def lifecycle_steps(mode: str) -> tuple[LifecycleStep, ...]:
    """Return the complete ordered state machine for one executable mode."""

    try:
        return _LIFECYCLE_STEPS_BY_MODE[mode]
    except KeyError as error:
        raise ScenarioContractError(
            f"unknown adapter lifecycle mode: {mode}"
        ) from error


def lifecycle_transition_names(mode: str) -> tuple[str, ...]:
    """Return names derived from the single declarative lifecycle table."""

    return tuple(step.name for step in lifecycle_steps(mode))


def execute_lifecycle(
    mode: str,
    executor: LifecycleExecutor,
) -> dict[str, Observation]:
    """Execute every transition exactly once and validate cross-step invariants."""

    observations: dict[str, Observation] = {}
    for step in lifecycle_steps(mode):
        observation = executor.execute(step)
        if not isinstance(observation, Observation):
            raise ScenarioContractError(
                f"executor returned non-Observation for {step.name}"
            )
        if step.name in observations:
            raise ScenarioContractError(f"duplicate lifecycle observation: {step.name}")
        observations[step.name] = observation
    validate_lifecycle_observations(observations)
    observed_mode = _state(next(iter(observations.values())))["mode"]
    if observed_mode != mode:
        raise ScenarioContractError(
            f"observed lifecycle mode {observed_mode!r} does not match requested mode "
            f"{mode!r}"
        )
    return observations


def _output_payload(observation: Observation) -> tuple[object, ...]:
    return (
        observation.output_ids,
        observation.text,
        observation.token_logprobs,
        observation.selected_logits,
        observation.selected_token_ids,
        observation.request_output_lengths,
        observation.request_texts,
        observation.error,
    )


_OUTPUT_ACTIONS = {
    "generate",
    "startup_generate",
    "mixed_batch",
    "concurrent",
    "complete_lease",
    "begin_paused_request",
    "resume_paused_request",
}


def _model_output_payload(observation: Observation) -> tuple[object, ...]:
    return _output_payload(observation)[:-1]


def _concurrent_failure_details(observations):
    """Retain exact outputs for diagnosis; this does not change the verdict."""
    names = ("concurrent.stream", "concurrent.non-stream")
    outputs = {name: observations[name].to_dict() for name in names}
    fields = (
        "output_ids",
        "text",
        "token_logprobs",
        "selected_logits",
        "selected_token_ids",
        "request_output_lengths",
        "request_texts",
    )
    return {
        "kind": "concurrent-output-mismatch",
        "transitions": list(names),
        "observations": outputs,
        "changed_fields": [
            field
            for field in fields
            if outputs[names[0]][field] != outputs[names[1]][field]
        ],
    }


def validate_lease_prefix(observations: Mapping[str, Observation]) -> None:
    """Check the timing-sized first stream chunk against its own final output."""
    begin = observations.get("upsert.lease.begin")
    if begin is None or begin.adapter_state["mode"] == "base":
        return
    complete = observations.get("upsert.lease.complete")
    if complete is None:
        raise ScenarioContractError("lease prefix has no completed response")
    size = len(begin.output_ids)
    if (
        not 0 < size <= len(complete.output_ids)
        or tuple(begin.request_output_lengths) != (size,)
        or tuple(complete.request_output_lengths) != (len(complete.output_ids),)
        or begin.output_ids != complete.output_ids[:size]
        or begin.token_logprobs != complete.token_logprobs[:size]
        or not complete.text.startswith(begin.text)
        or len(begin.request_texts) != 1
        or len(complete.request_texts) != 1
        or not complete.request_texts[0].startswith(begin.request_texts[0])
    ):
        raise ScenarioContractError("lease prefix differs from completed response")
    names = {f"decode.{position:03d}.top_logprobs" for position in range(size)}
    for field in ("selected_logits", "selected_token_ids"):
        partial_rows, final_rows = getattr(begin, field), getattr(complete, field)
        if set(partial_rows) != names or any(
            name not in final_rows or partial_rows[name] != final_rows[name]
            for name in names
        ):
            raise ScenarioContractError(
                f"lease prefix {field} differs from completed response"
            )


def comparable_lease_observations(
    observations: Mapping[str, Observation],
) -> Mapping[str, Observation]:
    """Comparison-only view: remove first-chunk timing, retaining raw evidence.

    Once every partial output field matches its own completed response,
    represent this step as state-only for comparison. Keep its state/error,
    retain raw evidence, and compare final output independently. Reference
    numeric samples exclude the timing-sized prefix, without duplicating the
    final response or changing measured performance samples/metadata.
    """
    validate_lease_prefix(observations)
    begin = observations.get("upsert.lease.begin")
    if begin is None or begin.adapter_state["mode"] == "base":
        return observations
    view = dict(observations)
    view["upsert.lease.begin"] = replace(
        begin,
        output_ids=(),
        text="",
        token_logprobs=(),
        selected_logits={},
        selected_token_ids={},
        request_output_lengths=(),
        request_texts=(),
    )
    return view


def _state(observation: Observation) -> Mapping[str, object]:
    state = observation.adapter_state
    if not isinstance(state, Mapping):
        raise ScenarioContractError("lifecycle observation state is not an object")
    return state


def _identity(value: object) -> tuple[object, object, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ScenarioContractError("adapter identity is not an object")
    return (value.get("name"), value.get("id"), value.get("version"))


def _active(observation: Observation) -> tuple[object, object, object] | None:
    return _identity(_state(observation)["active"])


def _staged(observation: Observation) -> tuple[object, object, object] | None:
    return _identity(_state(observation)["staged"])


def _registered_by_name(
    observation: Observation,
) -> dict[object, Mapping[str, object]]:
    records = _state(observation)["registered"]
    return {record["name"]: record for record in records}


def _record_without_slot(record: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "registry_slot"}


def _require_active_adapter(
    observation: Observation, adapter: str, message: str
) -> tuple[object, object, object]:
    active = _active(observation)
    if active is None or active[0] != adapter:
        raise ScenarioContractError(message)
    cache = _state(observation)["cache_identity"]
    if not isinstance(cache, Mapping) or cache.get(adapter) != active[1]:
        raise ScenarioContractError(message)
    return active


def _is_empty_state(state: Mapping[str, object]) -> bool:
    return (
        state["registered"] == ()
        and state["active"] is None
        and state["staged"] is None
        and state["registry_occupancy"] == 0
        and state["quarantined"] == ()
        and state["tombstoned"] == ()
        and state["cache_identity"] == {}
    )


def _validate_observations_and_mode(
    observations: Mapping[str, Observation],
) -> str:
    if not observations:
        raise ScenarioContractError("lifecycle observations are empty")

    modes: list[object] = []
    for name, observation in observations.items():
        if not isinstance(observation, Observation):
            raise ScenarioContractError(
                f"lifecycle observation {name!r} is not an Observation"
            )
        try:
            observation.validate(f"lifecycle.{name}")
        except BundleValidationError as error:
            raise ScenarioContractError(
                f"invalid lifecycle observation {name}: {error}"
            ) from error
        modes.append(_state(observation)["mode"])

    mode = modes[0]
    if any(candidate != mode for candidate in modes[1:]):
        raise ScenarioContractError("lifecycle modes differ across observations")
    if not isinstance(mode, str):
        raise ScenarioContractError("lifecycle mode is not a string")
    return mode


def _validate_observation_order(
    observations: Mapping[str, Observation], mode: str
) -> None:
    expected = lifecycle_transition_names(mode)
    actual = tuple(observations)
    if actual != expected:
        raise ScenarioContractError(
            f"lifecycle observation order differs: expected {expected!r}, got {actual!r}"
        )


def _validate_declared_errors(
    observations: Mapping[str, Observation], mode: str
) -> None:
    for step in lifecycle_steps(mode):
        error = observations[step.name].error
        if step.expected_error_code is None:
            if error is not None:
                raise ScenarioContractError(
                    f"undeclared lifecycle error at {step.name}"
                )
            continue
        if error is None or error.get("code") != step.expected_error_code:
            actual = None if error is None else error.get("code")
            raise ScenarioContractError(
                f"unexpected lifecycle error code at {step.name}: "
                f"expected {step.expected_error_code!r}, got {actual!r}"
            )


def _validate_step_outputs(observations: Mapping[str, Observation], mode: str) -> None:
    for step in lifecycle_steps(mode):
        if step.action in _OUTPUT_ACTIONS and not observations[step.name].output_ids:
            raise ScenarioContractError(f"missing inference output at {step.name}")


def _validate_base_lifecycle(observations: Mapping[str, Observation]) -> None:
    initial = observations["base.initial"]
    if _output_payload(observations["restart.identity"]) != _output_payload(initial):
        raise ScenarioContractError("restart identity changed")
    for name, observation in observations.items():
        if not _is_empty_state(_state(observation)):
            raise ScenarioContractError(f"base lifecycle state is not empty at {name}")


def _require_same_state(
    observations: Mapping[str, Observation],
    left: str,
    right: str,
    message: str,
) -> None:
    if _state(observations[left]) != _state(observations[right]):
        raise ScenarioContractError(message)


def _validate_stage(
    observations: Mapping[str, Observation],
    *,
    stage_name: str,
    previous_name: str,
    infer_name: str,
    version: str,
) -> tuple[object, object, object]:
    stage = observations[stage_name]
    if _active(stage) != _active(observations[previous_name]):
        raise ScenarioContractError("stage changed active identity")
    staged = _staged(stage)
    active = _active(stage)
    if (
        staged is None
        or active is None
        or staged[0] != "policy-a"
        or staged[1] != active[1]
        or staged[2] != version
    ):
        raise ScenarioContractError("stage recorded wrong identity")
    _require_same_state(
        observations,
        stage_name,
        infer_name,
        "staged inference changed adapter state",
    )
    if _model_output_payload(observations[infer_name]) != _model_output_payload(
        observations[previous_name]
    ):
        raise ScenarioContractError("staged inference output changed")
    return staged


def _validate_activation(
    observations: Mapping[str, Observation],
    *,
    activation_name: str,
    staged_identity: tuple[object, object, object],
    infer_name: str,
) -> None:
    activated = observations[activation_name]
    if _active(activated) != staged_identity:
        raise ScenarioContractError("activation promoted wrong identity")
    if _staged(activated) is not None:
        raise ScenarioContractError("activation did not clear staged identity")
    _require_same_state(
        observations,
        activation_name,
        infer_name,
        "post-activation inference changed adapter state",
    )


def _validate_immediate_switch_and_upsert(
    observations: Mapping[str, Observation],
) -> None:
    _require_active_adapter(
        observations["startup.adapter"],
        "policy-a",
        "startup did not activate declared adapter",
    )

    for input_kind in ("path", "tensor", "distributed"):
        load_name = f"immediate.{input_kind}.load"
        infer_name = f"immediate.{input_kind}.infer"
        _require_active_adapter(
            observations[load_name],
            "policy-a",
            f"immediate {input_kind} load did not activate policy-a",
        )
        load_state = _state(observations[load_name])
        load_records = _registered_by_name(observations[load_name])
        if (
            set(load_records) != {"policy-a"}
            or load_state["registry_occupancy"] != 1
            or load_state["staged"] is not None
            or load_state["quarantined"]
            or load_state["tombstoned"]
            or load_state["cache_identity"]
            != {"policy-a": load_records["policy-a"]["id"]}
        ):
            raise ScenarioContractError("immediate load state changed")
        _require_same_state(
            observations,
            load_name,
            infer_name,
            "immediate load identity changed before inference",
        )

    switch_a = _require_active_adapter(
        observations["switch.a"],
        "policy-a",
        "switch did not select adapter A",
    )
    _require_active_adapter(
        observations["switch.b"],
        "policy-b",
        "switch did not select adapter B",
    )
    switch_a_state = _state(observations["switch.a"])
    switch_b_state = _state(observations["switch.b"])
    if any(
        switch_a_state[field] != switch_b_state[field]
        for field in (
            "registered",
            "staged",
            "registry_occupancy",
            "quarantined",
            "tombstoned",
            "cache_identity",
        )
    ):
        raise ScenarioContractError("switch changed registered identity")
    switch_a_again = _require_active_adapter(
        observations["switch.a-again"],
        "policy-a",
        "switch did not restore adapter A",
    )
    if (
        switch_a_again != switch_a
        or _state(observations["switch.a-again"]) != _state(observations["switch.a"])
        or _model_output_payload(observations["switch.a-again"])
        != _model_output_payload(observations["switch.a"])
    ):
        raise ScenarioContractError("switch did not restore adapter A")

    for name in (
        "mixed.base-a-b",
        "concurrent.stream",
        "concurrent.non-stream",
    ):
        _require_same_state(
            observations,
            "switch.a-again",
            name,
            "mixed inference changed state",
        )
    if _model_output_payload(
        observations["concurrent.non-stream"]
    ) != _model_output_payload(observations["concurrent.stream"]):
        raise ScenarioContractError(
            "concurrent output changed between stream modes",
            details=_concurrent_failure_details(observations),
        )

    for name in (
        "upsert.lease.begin",
        "upsert.while-leased",
        "upsert.lease.complete",
    ):
        _require_same_state(
            observations,
            "concurrent.non-stream",
            name,
            "upsert changed leased identity before request completion",
        )
    if _model_output_payload(
        observations["upsert.lease.complete"]
    ) != _model_output_payload(observations["switch.a-again"]):
        raise ScenarioContractError("leased request output changed during upsert")

    before = observations["upsert.lease.complete"]
    after = observations["upsert.after"]
    before_active = _require_active_adapter(
        before, "policy-a", "leased request lost policy-a"
    )
    after_active = _require_active_adapter(
        after, "policy-a", "upsert did not activate policy-a"
    )
    before_records = _registered_by_name(before)
    after_records = _registered_by_name(after)
    before_state = _state(before)
    after_state = _state(after)
    if (
        before_active == after_active
        or set(before_records) != {"policy-a", "policy-b"}
        or set(after_records) != {"policy-a", "policy-b"}
        or _record_without_slot(before_records["policy-b"])
        != _record_without_slot(after_records["policy-b"])
        or before_state["cache_identity"].get("policy-b")
        != after_state["cache_identity"].get("policy-b")
        or after_state["staged"] is not None
        or after_state["quarantined"]
        or after_state["tombstoned"]
    ):
        raise ScenarioContractError("upsert result changed unrelated adapter state")


def _validate_unload_retry(
    observations: Mapping[str, Observation], target: str
) -> None:
    before = observations["failure.unload.quarantine"]
    after = observations["failure.unload.retry"]
    before_state = _state(before)
    after_state = _state(after)
    # Successful retry clears the failed target's focus. The native control
    # then reports an active identity only when exactly one adapter survives.
    survivors = before_state["registered"]
    expected_active = (
        {field: survivors[0][field] for field in ("name", "id", "version")}
        if len(survivors) == 1
        else None
    )
    expected_cache = {
        name: adapter_id
        for name, adapter_id in before_state["cache_identity"].items()
        if name != target
    }
    if (
        target not in {record["name"] for record in before_state["tombstoned"]}
        or after_state["registered"] != before_state["registered"]
        or after_state["registry_occupancy"] != before_state["registry_occupancy"]
        or after_state["active"] != expected_active
        or after_state["staged"] != before_state["staged"]
        or after_state["cache_identity"] != expected_cache
        or after_state["quarantined"]
        != tuple(name for name in before_state["quarantined"] if name != target)
        or after_state["tombstoned"]
        != tuple(
            record for record in before_state["tombstoned"] if record["name"] != target
        )
    ):
        raise ScenarioContractError("unload retry result changed")


def _validate_native_lifecycle(
    observations: Mapping[str, Observation],
) -> None:
    initial = observations["base.initial"]
    steps = {step.name: step for step in lifecycle_steps(_state(initial)["mode"])}
    for name in (
        "immediate.path.base",
        "immediate.tensor.base",
        "immediate.distributed.base",
        "base.restored",
    ):
        if _output_payload(observations[name]) != _output_payload(initial):
            raise ScenarioContractError(f"base output changed after unload at {name}")

    for name in (
        "base.initial",
        "immediate.path.unload",
        "immediate.path.base",
        "immediate.tensor.unload",
        "immediate.tensor.base",
        "immediate.distributed.unload",
        "immediate.distributed.base",
        "unload.final",
        "base.restored",
    ):
        if not _is_empty_state(_state(observations[name])):
            if name == "unload.final":
                raise ScenarioContractError("final adapter state is not empty")
            raise ScenarioContractError(f"adapter state is not empty at {name}")

    _validate_immediate_switch_and_upsert(observations)

    staged_v1 = _validate_stage(
        observations,
        stage_name="stage.v1",
        previous_name="upsert.after",
        infer_name="stage.v1.old-active",
        version=steps["stage.v1"].version,
    )
    _validate_activation(
        observations,
        activation_name="activate.v1",
        staged_identity=staged_v1,
        infer_name="stage.v1.active",
    )
    staged_v2 = _validate_stage(
        observations,
        stage_name="stage.v2",
        previous_name="stage.v1.active",
        infer_name="stage.v2.v1-active",
        version=steps["stage.v2"].version,
    )
    _validate_activation(
        observations,
        activation_name="activate.v2",
        staged_identity=staged_v2,
        infer_name="stage.v2.active",
    )

    previous = "stage.v2.active"
    for name in (
        "reject.duplicate",
        "reject.stale",
        "reject.wrong-id",
        "reject.wrong-name",
        "reject.invalid-config",
        "reject.unsupported-target",
    ):
        _require_same_state(
            observations,
            previous,
            name,
            f"rejected mutation changed state at {name}",
        )
        previous = name

    _require_same_state(
        observations,
        "reject.unsupported-target",
        "failure.update",
        "failed update changed state",
    )
    _require_same_state(
        observations,
        "failure.update",
        "failure.update.previous",
        "failed update did not retain previous state",
    )
    if _model_output_payload(
        observations["failure.update.previous"]
    ) != _model_output_payload(observations["stage.v2.active"]):
        raise ScenarioContractError("failed update output changed")

    failed_activation = _state(observations["failure.activation"])
    if "policy-a" not in failed_activation["quarantined"]:
        raise ScenarioContractError("failed activation did not quarantine adapter")
    if _active(observations["failure.activation"]) != _active(
        observations["failure.update.previous"]
    ):
        raise ScenarioContractError(
            "failed activation changed previous active identity"
        )
    pending = _staged(observations["failure.activation"])
    active = _active(observations["failure.activation"])
    if (
        pending is None
        or active is None
        or pending[:2] != active[:2]
        or pending[2] != steps["failure.activation"].version
    ):
        raise ScenarioContractError("failed activation lost staged identity")
    _require_same_state(
        observations,
        "failure.activation",
        "failure.activation.previous",
        "failed activation did not retain previous state",
    )

    failed_unload = _state(observations["failure.unload"])
    if "policy-a" not in failed_unload["quarantined"]:
        raise ScenarioContractError("failed unload did not retain quarantine")
    if "policy-a" not in {record["name"] for record in failed_unload["tombstoned"]}:
        raise ScenarioContractError("failed unload did not quarantine retry identity")
    before_unload = _state(observations["failure.activation.previous"])
    before_records = _registered_by_name(observations["failure.activation.previous"])
    after_records = _registered_by_name(observations["failure.unload"])
    expected_survivors = set(before_records) - {"policy-a"}
    expected_tombstones = tuple(
        sorted(
            (
                *before_unload["tombstoned"],
                _record_without_slot(before_records["policy-a"]),
            ),
            key=lambda record: record["name"],
        )
    )
    if (
        set(after_records) != expected_survivors
        or any(
            _record_without_slot(after_records[name])
            != _record_without_slot(before_records[name])
            for name in expected_survivors & set(after_records)
        )
        or failed_unload["registry_occupancy"] != len(expected_survivors)
        or failed_unload["active"] is not None
        or failed_unload["staged"] is not None
        or any(
            failed_unload[field] != before_unload[field]
            for field in (
                "quarantined",
                "cache_identity",
            )
        )
        or failed_unload["tombstoned"] != expected_tombstones
    ):
        raise ScenarioContractError("failed unload changed retry state")
    _require_same_state(
        observations,
        "failure.unload",
        "failure.unload.quarantine",
        "unload retry identity changed",
    )
    _validate_unload_retry(observations, "policy-a")

    cancellation_names = (
        "cancel.lease-drain",
        "cancel.fan-out",
        "cancel.publication",
        "cancel.rollback",
        "cancel.eviction",
        "pause.retain-kv",
        "pause.activate-rejected",
        "pause.resume",
    )
    retained = observations[cancellation_names[0]]
    retained_state = _state(retained)
    retained_active = _active(retained)
    retained_registered = _registered_by_name(retained)
    if (
        retained_active is None
        or retained_active[0] != "policy-a"
        or set(retained_registered) != {"policy-a"}
        or _identity(retained_registered["policy-a"]) != retained_active
        or retained_state["registry_occupancy"] != 1
        or retained_state["staged"] is not None
    ):
        raise ScenarioContractError("cancellation lost retained adapter identity")
    if retained_state["cache_identity"] != {"policy-a": retained_active[1]}:
        raise ScenarioContractError("cancellation lost retained adapter cache")
    if retained_state["quarantined"] or retained_state["tombstoned"]:
        raise ScenarioContractError("cancellation introduced quarantine")
    for left, right in zip(cancellation_names, cancellation_names[1:]):
        _require_same_state(
            observations,
            left,
            right,
            f"cancellation or pause changed retained state at {right}",
        )

    filled_observation = observations["registry.fill"]
    evicted_observation = observations["registry.evict"]
    filled = _state(filled_observation)
    evicted = _state(evicted_observation)
    filled_registered = _registered_by_name(filled_observation)
    evicted_registered = _registered_by_name(evicted_observation)
    filled_active = _active(filled_observation)
    evicted_active = _active(evicted_observation)
    if (
        set(filled_registered) != {"policy-a", "policy-b"}
        or set(evicted_registered) != {"policy-b"}
        or _record_without_slot(evicted_registered["policy-b"])
        != _record_without_slot(filled_registered["policy-b"])
        or filled["registry_occupancy"] != 2
        or evicted["registry_occupancy"] != 1
        or filled_active != _identity(filled_registered["policy-b"])
        or evicted_active != _identity(evicted_registered["policy-b"])
        or filled["staged"] is not None
        or evicted["staged"] is not None
        or filled["cache_identity"]
        != {name: record["id"] for name, record in filled_registered.items()}
        or evicted["cache_identity"] != filled["cache_identity"]
        or filled["quarantined"]
        or evicted["quarantined"]
        or filled["tombstoned"]
        or evicted["tombstoned"]
    ):
        raise ScenarioContractError("registry eviction result changed")

    validate_restart_observations(observations)


def validate_restart_observations(observations: Mapping[str, Observation]) -> None:
    """Check fresh-engine equivalence while retaining IDs within each engine."""
    startup = observations["startup.adapter"]
    restarted = observations["restart.same-manifest"]
    inferred = observations["restart.identity"]
    for observation in (startup, restarted, inferred):
        observation.validate()
    startup_state = _state(startup)
    restart_state = _state(restarted)
    if _state(inferred) != restart_state:
        raise ScenarioContractError("restart identity changed within restarted engine")
    if startup_state["mode"] == "native_oft":
        # OFT assigns UUID4 IDs on each construction. Align only this fresh
        # engine boundary; all lifecycle and same-engine checks retain raw IDs.
        by_name = {
            record["name"]: record["id"] for record in startup_state["registered"]
        }
        ids = {
            record["id"]: by_name.get(record["name"], record["id"])
            for record in restart_state["registered"]
        }

        def align(record):
            return (
                None
                if record is None
                else dict(record, id=ids.get(record["id"], record["id"]))
            )

        restart_state = dict(
            restart_state,
            registered=tuple(align(record) for record in restart_state["registered"]),
            active=align(restart_state["active"]),
            cache_identity={
                name: ids.get(value, value)
                for name, value in restart_state["cache_identity"].items()
            },
        )
    if restart_state != startup_state:
        raise ScenarioContractError("restart identity changed")
    if _output_payload(inferred) != _output_payload(startup):
        raise ScenarioContractError("restart output identity changed")


def validate_lifecycle_observations(
    observations: Mapping[str, Observation],
) -> None:
    """Validate exact membership and all cross-transition lifecycle invariants."""

    mode = _validate_observations_and_mode(observations)
    _validate_observation_order(observations, mode)
    _validate_declared_errors(observations, mode)
    _validate_step_outputs(observations, mode)
    if mode == "base":
        _validate_base_lifecycle(observations)
    else:
        _validate_native_lifecycle(observations)
        validate_lease_prefix(observations)

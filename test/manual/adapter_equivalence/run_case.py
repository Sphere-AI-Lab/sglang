#!/usr/bin/env python3
"""Execute one fail-closed shard; JSONL is diagnostic, never qualification evidence."""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import math
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    Future,
)
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures import (
    wait,
)
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adapter_equivalence"

from .bundle_capture import (
    PerformanceRecorder,
    attest_engine_targets,
    bind_runtime,
    capture_control,
    capture_generation,
    capture_run_identity,
    publish_bundle,
)
from .compare import _exact_differences
from .distributed_sender import SENDER_RUNTIME_ORIGINS, DistributedSession
from .faults import FaultController, PhaseGate, RankFailure
from .scenarios import (
    LifecycleStep,
    ScenarioContractError,
    lifecycle_steps,
    validate_lifecycle_observations,
)
from .schema import (
    PROVENANCE_HASH_KEYS,
    SCHEMA_VERSION,
    Observation,
    RunBundle,
    canonical_sha256,
)
from .server import (
    AdapterIdentity,
    ControlResult,
    DistributedPayload,
    ServerSpec,
    engine_kwargs,
    make_adapter_control,
    server_other_args,
    stop_engine,
    submit_engine_coroutine,
)

RUN_SEED = 1729


def launch_engine(spec):
    """Bind every lifecycle and performance construction to the recorded seed."""
    bind_runtime()
    from sglang.srt.entrypoints.engine import Engine

    bind_runtime()
    attest_engine_targets(Engine)
    return Engine(**engine_kwargs(spec), random_seed=RUN_SEED)


def performance_launch_identity(server, provenance):
    """Bind actual constructor values while replacing only immutable locations."""
    result = {}
    for name, launch in (
        ("initial", replace(server, startup_adapters=())),
        ("preloaded", server),
    ):
        kwargs = dict(engine_kwargs(launch), random_seed=RUN_SEED)
        kwargs["model_path"] = {"checkpoint_hash": provenance["checkpoint_hash"]}
        for key in ("lora_paths", "peft_paths"):
            if key in kwargs:
                kwargs[key] = [
                    {
                        "name": adapter,
                        "fixture_hash": canonical_sha256(
                            provenance["metadata"]["fixture_files"][adapter]
                        ),
                    }
                    for adapter, _ in launch.startup_adapters
                ]
        result[name] = kwargs
    return result


# Transport addresses and diagnostic observations are not execution settings.
# All other resolved fields (including newly added runtime defaults) are bound.
_NON_EXECUTION_CONFIG = frozenset(
    {
        "host",
        "port",
        "nccl_port",
        "dist_init_addr",
        "grpc_port",
        "forward_pass_metrics_server_port",
        "forward_pass_metrics_server_url",
        "startup_time",
        "last_gen_throughput",
        "avg_spec_accept_length",
        "step_time_dict",
        "dspark_info_record",
        "env_vars",
        "memory_usage",
        "model_config",
        "custom_sigquit_handler",
        # Declaration/audit history is already materialized in resolved fields.
        "_resolved_overrides",
        "_ssl_verify_warned",
    }
)
_CREDENTIAL_CONFIG = frozenset(
    {"api_key", "admin_api_key", "ssl_keyfile_password", "ssl_keyfile"}
)


def normalized_effective_config(value, server, provenance):
    """Preserve resolved values, replacing verified locations with content IDs."""
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "__struct_fields__"):
        value = {key: getattr(value, key) for key in value.__struct_fields__}
    if isinstance(value, Enum):
        return normalized_effective_config(value.value, server, provenance)
    if isinstance(value, dict):
        value = dict(value)
        for field in ("lora_target_modules", "peft_target_modules"):
            targets = value.get(field)
            if targets is not None:
                if not isinstance(targets, (set, frozenset, tuple, list)) or any(
                    type(target) is not str for target in targets
                ):
                    raise ScenarioContractError(f"invalid resolved {field}")
                # Runtime target-module sets cross msgpack as unordered lists.
                value[field] = sorted(targets)
        if "_cuda_graph_config_locked" in value:
            # This set crosses msgpack as a list; ordinary lists keep their order.
            locks = value["_cuda_graph_config_locked"]
            if not isinstance(locks, (set, frozenset, tuple, list)) or any(
                not isinstance(lock, (tuple, list))
                or len(lock) != 2
                or any(type(part) is not str for part in lock)
                for lock in locks
            ):
                raise ScenarioContractError("invalid resolved CUDA graph lock set")
            value["_cuda_graph_config_locked"] = sorted([list(lock) for lock in locks])
        for prefix in ("lora", "adapter"):
            name, path = value.get(f"{prefix}_name"), value.get(f"{prefix}_path")
            if f"{prefix}_id" in value and (name, path) in server.startup_adapters:
                value[f"{prefix}_id"] = {
                    "name": name,
                    "fixture_hash": canonical_sha256(
                        provenance["metadata"]["fixture_files"][name]
                    ),
                }
        return {
            key: (
                {"configured": bool(item)}
                if key in _CREDENTIAL_CONFIG
                else normalized_effective_config(item, server, provenance)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        values = [
            normalized_effective_config(item, server, provenance) for item in value
        ]
        return (
            sorted(values, key=canonical_sha256)
            if isinstance(value, (set, frozenset))
            else values
        )
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        locations = [
            (server.model_path, {"checkpoint_hash": provenance["checkpoint_hash"]})
        ]
        for adapter, path in server.startup_adapters:
            content = {
                "fixture_hash": canonical_sha256(
                    provenance["metadata"]["fixture_files"][adapter]
                )
            }
            locations.extend(
                ((path, content), (f"{adapter}={path}", dict(content, name=adapter)))
            )
        for location, content in locations:
            if value == str(location):
                return content
            if value.startswith(str(location) + "/"):
                return dict(content, relative=value[len(str(location)) + 1 :])
    if value is None or type(value) in (str, bool, int, float):
        canonical_sha256(value)  # Reject non-finite values rather than stringify.
        return value
    raise ScenarioContractError(
        f"unserializable resolved execution value: {type(value).__name__}"
    )


async def effective_request_identity(engine, requests):
    """Exercise real Engine forwarding and request normalization, without dispatch.

    Qualification owns this otherwise-idle Engine. Intercept only the tokenizer
    entrypoint while constructing the contract; never submit these probes to a
    scheduler or count them as warmup/measured model requests.
    """
    manager = engine.tokenizer_manager
    original = manager.generate_request
    had_override = "generate_request" in vars(manager)
    generators = []
    result = {}
    try:
        for name, kwargs in requests.items():
            observed = []

            def intercept(obj, request=None):
                async def normalize():
                    obj.normalize_batch_and_arguments()
                    normalized = copy.deepcopy(vars(obj))
                    # Only transport/request identities are ephemeral; retain
                    # expanded batch fields and all execution-related fallbacks.
                    for field in ("rid", "http_worker_ipc", "received_time"):
                        normalized.pop(field, None)
                    parameters = (
                        [obj.sampling_params] if obj.is_single else obj.sampling_params
                    )
                    samplers = []
                    for parameters_for_request in parameters:
                        sampling = manager.sampling_params_class(
                            **dict(
                                manager.preferred_sampling_params or {},
                                **parameters_for_request,
                            )
                        )
                        sampling.normalize(manager.tokenizer)
                        sampling.verify(manager.model_config.vocab_size)
                        samplers.append(
                            {
                                key: getattr(sampling, key)
                                for key in sampling.__struct_fields__
                            }
                        )
                    observed.append(
                        {
                            "request": normalized,
                            "sampling": samplers[0] if obj.is_single else samplers,
                        }
                    )
                    yield None

                generator = normalize()
                generators.append(generator)
                return generator

            manager.generate_request = intercept
            await engine.async_generate(**copy.deepcopy(kwargs))
            if len(observed) != 1:
                raise ScenarioContractError(
                    "Engine did not forward exactly one normalized performance request"
                )
            result[name] = observed[0]
    finally:
        if had_override:
            manager.generate_request = original
        else:
            del manager.generate_request
        for generator in generators:
            await generator.aclose()
    return result


class OperationTimeout(TimeoutError):
    """An operation exceeded its wall-clock budget."""


class _Worker:
    """Serialized daemon worker: stuck native calls cannot hold CLI exit.

    Deadlines bound callers; teardown runs independently of a wedged worker.
    """

    def __init__(self, name):
        self.jobs = queue.Queue()
        self.closed = False
        threading.Thread(target=self._run, name=name, daemon=True).start()

    def _run(self):
        while (job := self.jobs.get()) is not None:
            future, function = job
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function())
                except BaseException as error:
                    future.set_exception(error)

    def submit(self, function):
        if self.closed:
            raise ScenarioContractError("worker is closed")
        future = Future()
        self.jobs.put((future, function))
        return future

    def close(self):
        self.closed = True
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job[0].cancel()
        self.jobs.put(None)


@dataclass(frozen=True)
class Timeouts:
    startup: float = 1800.0
    inference: float = 300.0
    control: float = 300.0
    collective: float = 300.0
    teardown: float = 30.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ScenarioContractError(
                    f"{name} timeout must be positive and finite"
                )


@dataclass(frozen=True)
class RunSpec:
    server: ServerSpec
    case_id: str
    revision_sha: str
    architecture: str
    precision: str
    checkpoint_manifest: Path
    prompts_file: Path
    fixture_manifest: Path
    bundle_output: Path
    completion_output: Path
    max_new_tokens: int = 32
    repetition: int = 0

    def __post_init__(self):
        if (
            not self.case_id
            or len(self.revision_sha) != 40
            or any(c not in "0123456789abcdef" for c in self.revision_sha)
        ):
            raise ScenarioContractError(
                "case ID and exact lowercase revision SHA are required"
            )
        if self.architecture not in ("dense", "moe") or not self.precision:
            raise ScenarioContractError("architecture and precision are required")
        if self.bundle_output == self.completion_output:
            raise ScenarioContractError(
                "bundle and completion destinations must differ"
            )
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ScenarioContractError("max_new_tokens must be positive")
        if type(self.repetition) is not int or self.repetition < 0:
            raise ScenarioContractError("repetition must be a non-negative integer")


def require_control_success(result, operation):
    if not isinstance(result, ControlResult) or result.success is not True:
        message = (
            result.message
            if isinstance(result, ControlResult)
            else "missing explicit control success"
        )
        raise ScenarioContractError(f"{operation}: {message}")
    return result


def classify_rejection(result):
    """Recognize concrete product diagnostics independently of the desired step."""
    if not isinstance(result, ControlResult) or result.success is not False:
        raise ScenarioContractError("expected an explicit product rejection")
    message = result.message
    codes = set()
    markers = re.findall(r"adapter-harness:([\w-]+)", message)
    if markers:
        if (
            len(markers) != 1
            or markers[0]
            not in {"update_failure", "activation_failure", "unload_failure"}
            or (markers[0] == "update_failure" and "restart required" in message)
        ):
            raise ScenarioContractError("ambiguous fault markers or failed cleanup")
        codes.add(markers[0])
    match = re.fullmatch(
        r"(?:LoRA|OFT) adapter version (\d+) must be newer than active version (\d+)\.",
        message,
    )
    if match:
        requested, active = map(int, match.groups())
        if requested == active:
            codes.add("duplicate_version")
        elif requested < active:
            codes.add("stale_version")
    if re.fullmatch(
        r"Requested adapter_id '.+' does not match expected adapter_id '.+'", message
    ):
        codes.add("wrong_id")
    if re.fullmatch(
        r"Cannot activate name=missing-policy version=\d+; no (?:native LoRA|OFT) stage is pending",
        message,
    ):
        codes.add("wrong_name")
    if (
        message
        == "Cannot activate adapter weights while paused requests are still active; continue generation or abort those requests before retrying."
    ):
        codes.add("paused_requests_active")
    rollback_suffix = "; stage rollback succeeded"
    diagnostics = message.removesuffix(rollback_suffix)
    rank_errors = diagnostics.split(" | ")
    for key in ("r", "oft_block_size"):
        # Keep bare fake and TP1 diagnostics compatible; multi-rank evidence
        # requires successful rollback and the exact homogeneous consensus order.
        if diagnostics in {f"'{key}'", f"TP rank 0: '{key}'"} or (
            message.endswith(rollback_suffix)
            and all(
                error == f"TP rank {rank}: '{key}'"
                for rank, error in enumerate(rank_errors)
            )
        ):
            codes.add("invalid_config")
    if (
        "rejected-policy" in message
        and ("target modules" in message or "target_modules" in message)
        and (
            "incompatible" in message or "Model rejected OFT target modules" in message
        )
    ):
        codes.add("unsupported_target")
    if (
        message
        == "Unresolved OFT tensor names: adapter_harness_unsupported_target.oft_R"
    ):
        codes.add("unsupported_target")
    if len(codes) != 1:
        raise ScenarioContractError(
            f"unclassified or ambiguous product rejection: {message}"
        )
    return codes.pop()


def manager_request(name, **fields):
    from sglang.srt.managers import io_struct

    return getattr(io_struct, name)(**fields)


def load_fixture(path):
    """Read the immutable path-load bytes for tensor and sender requests."""
    from safetensors.torch import load_file

    config = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    if not tensors or not isinstance(config, dict):
        raise ScenarioContractError("fixture tensors/config must be nonempty objects")
    return tensors, config


def distributed_payload_for_fixture(path):
    tensors, config = load_fixture(path)
    names = tuple(sorted(tensors))
    return DistributedPayload(
        names,
        tuple(str(tensors[name].dtype).removeprefix("torch.") for name in names),
        tuple(tuple(int(size) for size in tensors[name].shape) for name in names),
        config,
    )


def build_target_shapes(
    config,
    architecture,
    *,
    layers=0,
    experts=0,
    suffixes=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
):
    """Map transformer and global modules to (input, output) features."""
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or hidden // heads)
    total_layers = int(config["num_hidden_layers"])
    layers = total_layers if layers <= 0 else min(layers, total_layers)
    if architecture == "moe":
        ffn = int(config.get("moe_intermediate_size") or config["intermediate_size"])
        total_experts = int(
            config.get("num_experts")
            or config.get("num_local_experts")
            or config.get("n_routed_experts")
            or 1
        )
        experts = total_experts if experts <= 0 else min(experts, total_experts)
    else:
        ffn, experts = int(config["intermediate_size"]), 0
    attention = {
        "q_proj": (hidden, heads * head_dim),
        "k_proj": (hidden, kv_heads * head_dim),
        "v_proj": (hidden, kv_heads * head_dim),
        "o_proj": (heads * head_dim, hidden),
    }
    mlp = {
        "gate_proj": (hidden, ffn),
        "up_proj": (hidden, ffn),
        "down_proj": (ffn, hidden),
    }
    shapes = {}
    for layer in range(layers):
        for suffix in suffixes:
            if suffix in attention:
                shapes[f"model.layers.{layer}.self_attn.{suffix}"] = attention[suffix]
            elif suffix in mlp:
                prefixes = (
                    (f"mlp.experts.{expert}" for expert in range(experts))
                    if architecture == "moe"
                    else ("mlp",)
                )
                for prefix in prefixes:
                    shapes[f"model.layers.{layer}.{prefix}.{suffix}"] = mlp[suffix]
    vocab = int(config["vocab_size"])
    shapes["model.embed_tokens"] = (vocab, hidden)
    shapes["lm_head"] = (hidden, vocab)
    return shapes


def build_fixture_set(root, config, architecture, mode):
    """Build once; every mechanism consumes these same immutable files."""
    from .fixtures import ADAPTER_SEEDS, build_lora_fixture, build_oft_fixture

    if mode == "base":
        return {}
    lifecycle_steps(mode)
    builder = build_lora_fixture if mode == "native_lora" else build_oft_fixture
    shapes = build_target_shapes(config, architecture)
    paths = {}
    for name, seed in zip(("policy-a", "policy-b"), ADAPTER_SEEDS):
        paths[name] = builder(
            root / name,
            adapter_id=name,
            architecture=architecture,
            seed=seed,
            target_shapes=shapes,
            **({"shared": True} if mode == "native_oft" else {}),
        ).path
    versions = [
        step.version for step in lifecycle_steps(mode) if step.action == "stage"
    ]
    return {**paths, **dict(zip(versions, paths.values()))}


class ShardRunner:
    """Execute core/adverse scenarios; Task 9 publishes complete observations."""

    def __init__(
        self,
        spec: RunSpec,
        *,
        control=None,
        sender=None,
        engine=None,
        prompts=None,
        batches=None,
        fixtures=None,
        diagnostics=None,
        timeouts=None,
    ):
        self.spec = spec
        self.engine, self.control, self.sender = engine, control, sender
        self.prompts, self.batches, self.fixtures = (
            prompts or {},
            batches or {},
            fixtures or {},
        )
        self.diagnostics = diagnostics if diagnostics is not None else sys.stdout
        self.timeouts = timeouts or Timeouts()
        self.engine_pool = _Worker("shard-engine")
        self.sender_pool = _Worker("shard-sender")
        self.observations = {}
        self.selection = ()
        self.failure = None
        self.closed = False
        self.active_adapter = None
        self.staged_identities = {}
        self.lease = None
        self.upsert_future = None
        self.upsert_deadline = None
        self.upsert_transfer = None
        self.preloaded = False
        self.timed_out = False
        self.lease_work = []
        self.qualification_spec = None
        self.qualification_provenance = None
        self.effective_launches = {}
        self.sender_runtime_origins = []

    def emit(self, event, **fields):
        self.diagnostics.write(
            json.dumps({"event": event, **fields}, allow_nan=False, sort_keys=True)
            + "\n"
        )
        self.diagnostics.flush()

    def _wait(self, future, timeout, label, *, check=None):
        deadline = time.monotonic() + timeout
        while True:
            if check is not None:
                check()
            remaining = max(0, deadline - time.monotonic())
            try:
                return future.result(
                    timeout=remaining if check is None else min(0.005, remaining)
                )
            except FutureTimeout as error:
                if check is not None:
                    check()
                if future.done():
                    raise
                if check is not None and time.monotonic() < deadline:
                    continue
                self.timed_out = True
                raise OperationTimeout(f"{label} exceeded {timeout:g}s") from error
            except BaseException:
                if check is not None:
                    check()
                raise

    def _call(self, label, function, timeout):
        return self._wait(self.engine_pool.submit(function), timeout, label)

    def _independent(self, label, function, timeout):
        worker = _Worker(label)
        try:
            return self._wait(worker.submit(function), timeout, label)
        finally:
            worker.close()

    def _async(self, label, factory):
        def drive():
            asyncio.set_event_loop(self.engine.loop)
            return self.engine.loop.run_until_complete(
                asyncio.wait_for(factory(), self.timeouts.inference)
            )

        return self._call(label, drive, self.timeouts.inference)

    def launch(self, *, preload=False):
        if self.qualification_spec is not None:
            self._assert_qualification_inputs()
        if self.engine is None:
            server = (
                self.spec.server
                if preload
                else replace(self.spec.server, startup_adapters=())
            )

            def launch_owned():
                engine = launch_engine(server)
                self.engine = engine
                if self.closed:
                    self._independent(
                        "late.engine.shutdown",
                        lambda: stop_engine(engine),
                        self.timeouts.teardown,
                    )
                return engine

            self.engine = self._call("startup", launch_owned, self.timeouts.startup)
            self.preloaded = preload
            if self.qualification_provenance is not None:
                self._capture_effective_launch(preload)
        if self.control is None:
            self.control = make_adapter_control(self.spec.server.mode, self.engine)

    def _capture_effective_launch(self, preload):
        async def capture():
            base = asdict(self.engine.server_args)
            manager = self.engine.tokenizer_manager
            tokenizer = manager.resolved_config_dict(copy.deepcopy(base))
            schedulers = await manager.get_internal_state()
            if not isinstance(schedulers, list) or len(schedulers) != 1:
                raise ScenarioContractError(
                    "resolved configuration requires one DP1 scheduler response"
                )
            for snapshot in (tokenizer, *schedulers):
                if not isinstance(snapshot, dict) or not (
                    set(base) - _NON_EXECUTION_CONFIG
                ) <= set(snapshot):
                    raise ScenarioContractError("incomplete resolved configuration")

            def execution(snapshot):
                result = {
                    key: value
                    for key, value in snapshot.items()
                    if key not in _NON_EXECUTION_CONFIG
                }
                if "memory_usage" in snapshot:
                    memory = snapshot["memory_usage"]
                    if (
                        type(memory["token_capacity"]) is not int
                        or memory["token_capacity"] != self.spec.server.max_total_tokens
                    ):
                        raise ScenarioContractError(
                            "scheduler token capacity differs from fixed launch limit"
                        )
                    result["token_capacity"] = memory["token_capacity"]
                    result["token_capacity_swa"] = memory["token_capacity_swa"]
                return result

            return {
                "engine": execution(base),
                "tokenizer": execution(tokenizer),
                "schedulers": [execution(snapshot) for snapshot in schedulers],
                "requests": await effective_request_identity(
                    self.engine, self._performance_requests()
                ),
            }

        effective = normalized_effective_config(
            self._async("resolved.configuration", capture),
            self.spec.server,
            self.qualification_provenance,
        )
        key = "preloaded" if preload else "initial"
        if key in self.effective_launches and self.effective_launches[key] != effective:
            self.emit(
                "configuration.changed",
                launch=key,
                differences=_exact_differences(
                    self.effective_launches[key], effective, key
                ),
            )
            raise ScenarioContractError(
                f"resolved {key} configuration changed between constructions"
            )
        self.effective_launches[key] = effective

    def resolve_selection(self, selection):
        steps = lifecycle_steps(self.spec.server.mode)
        if selection == "full":
            return tuple(step.name for step in steps)
        if selection == "smoke":
            return tuple(step.name for step in steps[:6])
        names = tuple(selection)
        if len(names) != len(set(names)):
            raise ScenarioContractError("duplicate selected transition")
        declared = {step.name for step in steps}
        if not names or any(name not in declared for name in names):
            raise ScenarioContractError("undeclared selected transition")
        return names

    def state(self, adapter=None, *, concurrent=False):
        call = self._independent if concurrent else self._call
        state = copy.deepcopy(
            call("inspect_state", self.control.inspect_state, self.timeouts.control)
        )
        return self._select_state(state, adapter)

    def _select_state(self, state, adapter=None):
        selected = self.active_adapter if adapter is None else adapter
        # Control focus tracks mutations. Inference focus is the actual request
        # selector, resolved exclusively against observed registry identities.
        if selected is not None:
            matches = [
                record for record in state["registered"] if record["name"] == selected
            ]
            if len(matches) != 1:
                raise ScenarioContractError(
                    f"requested adapter is not registered: {selected}"
                )
            state["active"] = {
                key: matches[0][key] for key in ("name", "id", "version")
            }
        return state

    def _wait_with_update(self, future, deadline, label):
        """Watch the real transfer alongside loop work; a stopped loop cannot hide failure."""
        pending = {future, *self.upsert_transfer[:2]}
        while True:
            failures = self.upsert_transfer[3]
            if failures:
                future.cancel()
                raise failures[0]
            if future.done():
                return future.result()
            done, pending = wait(
                pending,
                timeout=max(0, deadline - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )
            if self.upsert_transfer[3]:
                future.cancel()
                raise self.upsert_transfer[3][0]
            for completed in done:
                completed.result()
            if not done:
                future.cancel()
                self.timed_out = True
                raise OperationTimeout(f"{label} exceeded its deadline")
            if self.upsert_future.done() and not future.done():
                future.cancel()
                raise ScenarioContractError(f"update completed before {label}")

    def _state_observation(self, state):
        # Start/inspection is a harness event, not a product mutation response.
        # Deferred upsert's real result is required at lease completion.
        observation = Observation(
            output_ids=(),
            request_output_lengths=(),
            request_texts=(),
            text="",
            token_logprobs=(),
            selected_logits={},
            selected_token_ids={},
            adapter_state=state,
            error=None,
        )
        observation.validate()
        return observation

    def _control_observation(self, result, step):
        return capture_control(require_control_success(result, step.name), self.state())

    def _rejection_observation(self, result, step):
        return capture_control(
            result,
            self.state(),
            expected_error_code=step.expected_error_code,
            returned_error_code=classify_rejection(result),
        )

    def _preflight_rejection(self, step):
        active = self.state("policy-a")["active"]
        identity = AdapterIdentity(
            step.adapter,
            "wrong-id" if step.action == "reject_wrong_id" else active["id"],
            step.version,
        )
        if step.action == "reject_wrong_name":

            def operation():
                return self.control.activate(identity)

        else:
            payload = self._independent(
                "fixture.metadata",
                lambda: distributed_payload_for_fixture(self.fixtures["policy-a"]),
                self.timeouts.control,
            )

            def operation():
                return self.control.stage(
                    identity, payload, "preflight-must-not-dispatch"
                )

        result = self._call(step.name, operation, self.timeouts.control)
        return self._rejection_observation(result, step)

    def _faults(self):
        return FaultController(self.engine.tokenizer_manager, self.spec.server.mode)

    def _cancel_native_caller(self, *, request=None, name=None, check=None):
        acknowledged = Future()

        def cancel():
            if not acknowledged.set_running_or_notify_cancel():
                return
            public = {
                "update_adapter_from_distributed",
                "activate_adapter_version",
                "load_lora_adapter_from_distributed",
                "load_oft_adapter_from_distributed",
                "load_lora_adapter",
                "load_oft_adapter",
                "unload_lora_adapter",
                "unload_oft_adapter",
            }
            matches = []
            for task in asyncio.all_tasks(self.engine.loop):
                coroutine = task.get_coro()
                frame = getattr(coroutine, "cr_frame", None)
                if frame is None or frame.f_code.co_name not in public:
                    continue
                obj = frame.f_locals.get("obj")
                if frame.f_locals.get("self") is not self.engine.tokenizer_manager:
                    continue
                same_transaction = (
                    request is not None
                    and all(
                        getattr(request, field, None) is not None
                        for field in ("adapter_name", "adapter_id", "adapter_version")
                    )
                    and all(
                        getattr(obj, field, None) == getattr(request, field)
                        for field in ("adapter_name", "adapter_id", "adapter_version")
                    )
                )
                if (request is not None and (obj is request or same_transaction)) or (
                    request is None
                    and name is not None
                    and (
                        getattr(obj, "lora_name", None)
                        or getattr(obj, "adapter_name", None)
                    )
                    == name
                ):
                    matches.append(task)
            if len(matches) != 1:
                acknowledged.set_exception(
                    ScenarioContractError(
                        "cancellation cannot identify one exact native caller"
                    )
                )
            else:
                matches[0].cancel()
                acknowledged.set_result(True)

        self.engine.loop.call_soon_threadsafe(cancel)
        try:
            self._wait(
                acknowledged, self.timeouts.control, "cancel.request", check=check
            )
        finally:
            acknowledged.cancel()

    def _await_cancelled(self, future, *, check=None):
        try:
            self._wait(future, self.timeouts.control, "cancel.finish", check=check)
        except asyncio.CancelledError as error:
            return error
        raise ScenarioContractError("native caller did not acknowledge cancellation")

    def _finish_cancelled_transfer(self, transfer, gate):
        cancellation = self._cancel_at_gate(transfer[0], gate, transfer=transfer)
        self._wait(
            transfer[1],
            max(0, transfer[2] - time.monotonic()),
            "cancel.sender-finish",
            check=lambda: self._check_transfer_failure(transfer, cancelling=True),
        )
        return cancellation

    def _check_transfer_failure(self, transfer, *, cancelling=False):
        control, sender, _, failures = transfer
        for error in failures:
            if cancelling and isinstance(error, asyncio.CancelledError):
                # Only the native caller's requested cancellation is expected.
                # Failure recording precedes Future completion on each worker.
                if (
                    sender.done()
                    and not sender.cancelled()
                    and sender.exception() is error
                ):
                    raise error
                if not control.done():
                    return
                if not control.cancelled() and control.exception() is error:
                    continue
            raise error

    def _cancel_stage(self, step):
        before = self.state(step.adapter)
        identity = self._next_identity(
            step.adapter, str(int(before["active"]["version"]) + 1)
        )
        phase = "fan-out" if step.action == "cancel_fan_out" else "rollback"
        gate = PhaseGate(phase)
        fault = RankFailure(0, "adapter-harness:update_failure")
        controller = self._faults()
        with (
            controller.wrap(
                "stage", failure=fault, gate=gate if phase == "fan-out" else None
            ),
            controller.wrap("rollback", gate=gate if phase == "rollback" else None),
        ):
            transfer = self._start_distributed(
                step.name,
                self.fixtures["policy-a"],
                lambda payload, group: self.control.stage(identity, payload, group),
                accept_rejection=True,
            )
            self._finish_cancelled_transfer(transfer, gate)
        after = self.state(step.adapter)
        if controller.injected != [fault] or before != after:
            raise ScenarioContractError(
                "cancelled stage did not restore exact prior state"
            )
        self.emit("cancel.complete", phase=phase)
        return self._state_observation(after)

    def _drain_retained(self, label):
        if self.lease is None:
            raise ScenarioContractError("no retained request to drain")

        async def drain():
            last = None
            async for item in self.lease:
                last = item
            return last

        result = self._async(label, drain)
        self.lease = None
        return capture_generation(result, self.state(), top_k=5)

    def _cancel_lease_drain(self, step):
        # Previous failure/retry leaves B. Establish the single retained A once,
        # then preserve that exact identity across all cancellation scenarios.
        present = {record["name"] for record in self.state()["registered"]}
        if "policy-b" in present:
            self._delete("policy-b", "cancel.setup-unload-b")
        self._ensure_loaded(("policy-a",))
        self.active_adapter = "policy-a"
        before = self.state()
        self._begin_lease(replace(step, prompt_id="factual", stream=True))
        self._ensure_sender()
        payload = self._independent(
            "fixture.metadata",
            lambda: distributed_payload_for_fixture(self.fixtures["policy-b"]),
            self.timeouts.control,
        )
        future = self.engine_pool.submit(
            lambda: self.control.load_distributed(
                "policy-a", payload, self.sender.group_name, upsert=True
            )
        )
        observed = self.control.observe_lease_wait("policy-a", before["active"]["id"])
        self.lease_work.append(observed)
        # No broadcast is started: cancellation occurs before any rank can enter
        # the collective, while the real update is blocked behind the lease.
        done, _ = wait(
            (observed, future),
            timeout=self.timeouts.control,
            return_when=FIRST_COMPLETED,
        )
        if future in done:
            result = future.result()
            raise ScenarioContractError(
                f"update completed before lease cancellation barrier: {result!r}"
            )
        if observed not in done:
            self.timed_out = True
            raise OperationTimeout(
                f"cancel.lease-wait exceeded {self.timeouts.control:g}s"
            )
        phase_state = observed.result()
        if self._select_state(phase_state) != before:
            raise ScenarioContractError(
                "lease-drain cancellation changed live identity"
            )
        try:
            self._cancel_native_caller(name="policy-a")
        except ScenarioContractError:
            if future.done():
                future.result()
            raise
        self._await_cancelled(future)
        self._drain_retained("cancel.lease-request-finish")
        after = self.state()
        if after != before:
            raise ScenarioContractError("cancelled lease drain changed retained state")
        return self._state_observation(after)

    def _paused_request(self, step):
        manager = self.engine.tokenizer_manager
        if step.action == "begin_paused_request":
            observation = self._begin_lease(step)
            self._async(
                step.name + ".pause",
                lambda: manager.pause_generation(
                    manager_request("PauseGenerationReqInput", mode="in_place")
                ),
            )
            return observation
        if step.action == "reject_paused_activation":
            identity = self._next_identity(step.adapter, step.version)
            result = self._call(
                step.name,
                lambda: self.control.activate(identity),
                self.timeouts.control,
            )
            return self._rejection_observation(result, step)
        self._async(
            step.name + ".continue",
            lambda: manager.continue_generation(
                manager_request("ContinueGenerationReqInput")
            ),
        )
        return self._drain_retained(step.name)

    def _cancel_publication(self, step):
        before = self.state(step.adapter)
        self._load_named_path(
            "policy-b", self.fixtures["policy-b"], "cancel.publication-load-b"
        )
        active = self.state("policy-b")["active"]
        identity = self._next_identity("policy-b", str(int(active["version"]) + 1))
        require_control_success(
            self._stage_call(identity, step.name + ".stage-b"), "publication.stage"
        )
        gate = PhaseGate("publication")
        with self._faults().wrap("activate", gate=gate):
            future = self.engine_pool.submit(lambda: self.control.activate(identity))
            self._cancel_at_gate(future, gate)
        published = self.state("policy-b")
        if (
            published["active"]
            != {
                "name": identity.name,
                "id": identity.adapter_id,
                "version": identity.version,
            }
            or published["staged"] is not None
        ):
            raise ScenarioContractError(
                "cancelled activation did not finish publication"
            )
        self.emit("cancel.publication-finished", identity=published["active"])
        self._delete("policy-b", "cancel.publication-delete-b")
        self.active_adapter = step.adapter
        after = self.state()
        if before != after:
            raise ScenarioContractError(
                "publication cancellation changed retained adapter"
            )
        return self._state_observation(after)

    def _cancel_eviction(self, step):
        before = self.state(step.adapter)
        self._load_named_path(
            "policy-b", self.fixtures["policy-b"], "cancel.eviction-load-b"
        )
        # Real inference makes retained A most recently used. B, not A, must be
        # selected by the actual registry's eviction policy under load pressure.
        self._generate(
            LifecycleStep(
                "cancel.eviction-touch-a",
                "generate",
                adapter=step.adapter,
                prompt_id="factual",
            )
        )
        temporary = "harness-cancel-trigger"
        field = (
            "lora_name" if self.spec.server.mode == "native_lora" else "adapter_name"
        )
        gate = PhaseGate("eviction")
        with self._registry_limit(2), self._faults().wrap(
            "unload",
            gate=gate,
            when=lambda request: getattr(request, field, None) == "policy-b",
        ):
            future = self.engine_pool.submit(
                lambda: self.control.load_path(
                    temporary, str(self.fixtures["policy-b"])
                )
            )
            self._cancel_at_gate(future, gate, caller_name=temporary)
        if {record["name"] for record in self.state()["registered"]} != {
            step.adapter,
            temporary,
        }:
            raise ScenarioContractError(
                "cancelled LRU did not evict the intended victim"
            )
        self._delete(temporary, "cancel.eviction-delete-trigger")
        self._load_named_path(
            "policy-b", self.fixtures["policy-b"], "cancel.eviction-reload-b"
        )
        self._delete("policy-b", "cancel.eviction-delete-b")
        self.active_adapter = step.adapter
        after = self.state()
        if after != before:
            raise ScenarioContractError(
                "eviction cancellation changed retained identity"
            )
        return self._state_observation(after)

    def _cancel_at_gate(self, future, gate, *, caller_name=None, transfer=None):
        deadline = time.monotonic() + self.timeouts.control
        cancelling = False
        check = (
            None
            if transfer is None
            else lambda: self._check_transfer_failure(transfer, cancelling=cancelling)
        )
        try:
            while True:
                if check is not None:
                    check()
                if future.done():
                    future.result()
                    raise ScenarioContractError(
                        "native operation completed before cancellation phase"
                    )
                if gate.entered.wait(0.005):
                    if check is not None:
                        check()
                    break
                if time.monotonic() >= deadline:
                    if check is not None:
                        check()
                    self.timed_out = True
                    raise OperationTimeout("cancellation phase deadline exceeded")
            cancelling = True
            self._cancel_native_caller(
                request=gate.request if caller_name is None else None,
                name=caller_name,
                check=check,
            )
        finally:
            gate.release()
        return self._await_cancelled(future, check=check)

    def _retry_unload(self, step):
        tombstones = [
            record
            for record in self.state()["tombstoned"]
            if record["name"] == step.adapter
        ]
        if len(tombstones) != 1:
            raise ScenarioContractError("retry requires one exact tombstone identity")
        wanted = tombstones[0]
        observed = []
        kind = "lora" if self.spec.server.mode == "native_lora" else "adapter"

        def inspect(request):
            actual = (
                getattr(request, kind + "_name", None),
                getattr(request, kind + "_id", None),
            )
            if actual != (wanted["name"], wanted["id"]):
                raise ScenarioContractError(
                    "retry wire identity differs from exact tombstone"
                )
            observed.append(actual)

        with self._faults().wrap("unload", observe=inspect):
            result = self._call(
                step.name,
                lambda: self.control.unload(step.adapter),
                self.timeouts.control,
            )
        if observed != [(wanted["name"], wanted["id"])]:
            raise ScenarioContractError(
                "retry did not issue exactly one identity request"
            )
        self.active_adapter = None
        return self._control_observation(result, step)

    def _next_identity(self, name, version):
        active = self.state(name)["active"]
        return AdapterIdentity(name, active["id"], version)

    def _stage_call(
        self, identity, request_id, *, accept_rejection=False, invalid_config=False
    ):
        def stage(payload, group):
            if invalid_config:
                config = dict(payload.config)
                del config[
                    "r" if self.spec.server.mode == "native_lora" else "oft_block_size"
                ]
                payload = replace(payload, config=config)
            return self.control.stage(identity, payload, group)

        return self._distributed_call(
            request_id,
            self.fixtures.get(identity.version, self.fixtures["policy-a"]),
            stage,
            accept_rejection=accept_rejection,
        )

    def _invalid_payload(self, step):
        if step.action == "reject_invalid_config":
            result = self._stage_call(
                self._next_identity(step.adapter, step.version),
                step.name,
                accept_rejection=True,
                invalid_config=True,
            )
        else:
            tensors, config = self._independent(
                "fixture.read",
                lambda: load_fixture(self.fixtures[step.adapter]),
                self.timeouts.control,
            )
            config = dict(config, target_modules=["adapter_harness_unsupported_target"])
            if self.spec.server.mode == "native_oft":
                # The OFT wire loader resolves targets from tensor names,
                # rather than using the config's target_modules to select them.
                tensors = {
                    "adapter_harness_unsupported_target.oft_R": next(
                        iter(tensors.values())
                    )
                }
            # An unused name is necessary: otherwise duplicate-name validation
            # rejects the request before its target modules are examined.
            result = self._call(
                step.name,
                lambda: self.control.load_tensors("rejected-policy", tensors, config),
                self.timeouts.control,
            )
        return self._rejection_observation(result, step)

    def _inject_failure(self, step):
        operation, code = {
            "inject_update_failure": ("stage", "update_failure"),
            "inject_activation_failure": ("activate", "activation_failure"),
            "inject_unload_failure": ("unload", "unload_failure"),
        }[step.action]
        identity = (
            None
            if operation == "unload"
            else self._next_identity(step.adapter, step.version)
        )
        if operation == "activate":
            require_control_success(
                self._stage_call(identity, step.name + ".setup-stage"), "setup-stage"
            )
        failure = RankFailure(0, "adapter-harness:" + code)
        controller = self._faults()
        with controller.wrap(operation, failure=failure):
            if operation == "stage":
                result = self._stage_call(identity, step.name, accept_rejection=True)
            elif operation == "activate":
                result = self._call(
                    step.name,
                    lambda: self.control.activate(identity),
                    self.timeouts.control,
                )
            else:
                result = self._call(
                    step.name,
                    lambda: self.control.unload(step.adapter),
                    self.timeouts.control,
                )
                self.active_adapter = None
        marker = f"rank {failure.rank}: {failure.message}"
        if (
            controller.injected != [failure]
            or not isinstance(result, ControlResult)
            or result.success is not False
            or marker not in result.message
            or (operation == "stage" and "restart required" in result.message)
        ):
            raise ScenarioContractError(
                "injected failure was not returned or rollback failed"
            )
        return self._rejection_observation(result, step)

    @contextmanager
    def _registry_limit(self, limit):
        manager = self.engine.tokenizer_manager
        args = manager.server_args
        field = (
            "max_loaded_loras"
            if self.spec.server.mode == "native_lora"
            else "max_loaded_ofts"
        )

        # Registry enforcement reads this manager's args directly. The startup
        # record is immutable, so scope the test limit to a forwarding view.
        class LimitedArgs:
            def __getattr__(self, name):
                return limit if name == field else getattr(args, name)

        manager.server_args = LimitedArgs()
        try:
            yield
        finally:
            manager.server_args = args

    def _delete(self, name, label):
        result = self._call(
            label, lambda: self.control.unload(name), self.timeouts.control
        )
        require_control_success(result, label)
        self.active_adapter = None
        self.emit(label, adapter=name)
        return result

    def _load_named_path(self, name, fixture, label):
        result = self._call(
            label,
            lambda: self.control.load_path(name, str(fixture)),
            self.timeouts.control,
        )
        require_control_success(result, label)
        self.active_adapter = name
        self.emit(label, adapter=name, fixture=str(fixture))
        return result

    def _evict_registry(self, step):
        before = self.state()
        temporary = "harness-eviction-trigger"
        if {record["name"] for record in before["registered"]} != {
            "policy-a",
            "policy-b",
        }:
            raise ScenarioContractError("LRU trigger requires exactly A and B")
        with self._registry_limit(2):
            self._load_named_path(
                temporary, self.fixtures["policy-b"], "registry.lru-trigger"
            )
        actual = self.state()
        if {record["name"] for record in actual["registered"]} != {
            "policy-b",
            temporary,
        }:
            raise ScenarioContractError("real LRU did not evict policy-a")
        result = self._delete(temporary, "registry.delete-trigger")
        self.active_adapter = "policy-b"
        after = self.state()
        if after["cache_identity"] != before["cache_identity"]:
            raise ScenarioContractError("LRU changed retained reload identities")
        return self._control_observation(result, step)

    def _final_unload(self, step):
        self._delete("policy-b", "cleanup.unload-b")
        if "policy-a" not in self.state()["cache_identity"]:
            raise ScenarioContractError(
                "final cleanup lost the evicted adapter catalog"
            )
        self._load_named_path(
            "policy-a", self.fixtures["policy-a"], "cleanup.reload-evicted"
        )
        result = self._delete("policy-a", "cleanup.unload-reloaded")
        return self._control_observation(result, step)

    def _load(self, name, kind, request_id, *, upsert=False):
        fixture = self.fixtures["policy-b" if upsert else name]
        if kind == "path":
            result = self._call(
                request_id,
                lambda: self.control.load_path(name, str(fixture)),
                self.timeouts.control,
            )
        elif kind == "tensors":
            tensors, config = self._independent(
                "fixture.read", lambda: load_fixture(fixture), self.timeouts.control
            )
            result = self._call(
                request_id,
                lambda: self.control.load_tensors(name, tensors, config, upsert=upsert),
                self.timeouts.control,
            )
        elif kind == "distributed":
            result = self._distributed_call(
                request_id,
                fixture,
                lambda payload, group: self.control.load_distributed(
                    name, payload, group, upsert=upsert
                ),
            )
        else:
            raise ScenarioContractError(f"unimplemented load input: {kind}")
        require_control_success(result, request_id)
        self.active_adapter = name
        return result

    def _ensure_loaded(self, names):
        present = {record["name"] for record in self.state()["registered"]}
        # OFT forbids converting disk-backed registrations through wire upsert.
        # These fixtures feed later update scenarios; path loading is covered
        # separately by startup and the immediate.path lifecycle transitions.
        kind = "tensors" if self.spec.server.mode == "native_oft" else "path"
        for name in names:
            if name not in present:
                capture_control(
                    self._load(name, kind, f"setup.load.{name}"), self.state()
                )
                present.add(name)

    def _ensure_sender(self):
        if self.sender is None:

            def open_owned():
                session = DistributedSession.open(
                    self.engine, self.spec.server.tp_size, self.timeouts.collective
                )
                self.sender = session
                if self.qualification_provenance is not None:
                    origins = getattr(session.sender, "runtime_origins", None)
                    if origins != SENDER_RUNTIME_ORIGINS:
                        raise ScenarioContractError(
                            "sender runtime origin attestation differs from recorded checkout"
                        )
                    self.sender_runtime_origins.append(copy.deepcopy(origins))
                if self.closed:
                    self._independent(
                        "late.sender.close",
                        lambda: session.close(self.timeouts.teardown),
                        self.timeouts.teardown,
                    )
                return session

            self.sender = self._independent(
                "sender.open", open_owned, self.timeouts.collective
            )

    def _start_distributed(
        self, request_id, fixture, operation, *, accept_rejection=False
    ):
        self._ensure_sender()
        payload = self._independent(
            "fixture.metadata",
            lambda: distributed_payload_for_fixture(fixture),
            self.timeouts.collective,
        )
        deadline = time.monotonic() + self.timeouts.collective
        failures = []

        def record_failure(function):
            try:
                return function()
            except BaseException as error:
                failures.append(error)
                raise

        def control_call():
            result = operation(payload, self.sender.group_name)
            if accept_rejection:
                if not isinstance(result, ControlResult):
                    raise ScenarioContractError(
                        "distributed call omitted ControlResult"
                    )
                return result
            return require_control_success(result, request_id)

        future = self.engine_pool.submit(lambda: record_failure(control_call))

        def broadcast():
            result = self.sender.sender.broadcast_fixture(
                request_id, fixture, timeout=self.timeouts.collective
            )
            if result != payload:
                raise ScenarioContractError(
                    "sender and request tensor metadata/config differ"
                )
            return result

        sender_future = self.sender_pool.submit(lambda: record_failure(broadcast))
        return future, sender_future, deadline, failures

    def _finish_distributed(self, transfer, request_id):
        future, sender_future, deadline, failures = transfer
        done, pending = wait(
            (future, sender_future),
            timeout=max(0, deadline - time.monotonic()),
            return_when=FIRST_EXCEPTION,
        )
        if failures:
            raise failures[0]
        for completed in done:
            completed.result()
        if pending:
            self.timed_out = True
            raise OperationTimeout(f"{request_id} collective exceeded its deadline")
        return future.result()

    def _distributed_call(
        self, request_id, fixture, operation, *, accept_rejection=False
    ):
        return self._finish_distributed(
            self._start_distributed(
                request_id, fixture, operation, accept_rejection=accept_rejection
            ),
            request_id,
        )

    def _kwargs(self, prompt, adapter=None, *, stream=False, max_new_tokens=None):
        if prompt not in self.prompts:
            raise ScenarioContractError(f"missing immutable prompt: {prompt}")
        kwargs = {
            "input_ids": list(self.prompts[prompt]),
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": max_new_tokens or self.spec.max_new_tokens,
            },
            "return_logprob": True,
            "top_logprobs_num": 5,
            "stream": stream,
        }
        if adapter is not None:
            kwargs[
                "adapter_path" if self.spec.server.mode == "native_oft" else "lora_path"
            ] = adapter
        return kwargs

    async def _generate_one(self, kwargs):
        result = await self.engine.async_generate(**kwargs)
        if kwargs["stream"]:
            last = None
            async for chunk in result:
                last = chunk
            if last is None:
                raise ScenarioContractError("stream returned no complete output")
            return last
        return result

    def _generate(self, step):
        kwargs = self._kwargs(
            step.prompt_id,
            step.adapter,
            stream=bool(step.stream),
            max_new_tokens=step.max_new_tokens,
        )
        result = self._async(step.name, lambda: self._generate_one(kwargs))
        self.active_adapter = step.adapter or self.active_adapter
        return capture_generation(result, self.state(), top_k=5)

    def _batch(self, step):
        prompts = self.batches[step.prompt_id]
        adapters = (
            [None] * len(prompts)
            if self.spec.server.mode == "base"
            else [(None, "policy-a", "policy-b")[i % 3] for i in range(len(prompts))]
        )
        if self.spec.server.mode != "base":
            self._ensure_loaded(("policy-a", "policy-b"))
        kwargs = [
            self._kwargs(prompt, adapter, stream=bool(step.stream))
            for prompt, adapter in zip(prompts, adapters)
        ]

        async def generate():
            if step.action == "concurrent":
                return await asyncio.gather(
                    *(self._generate_one(item) for item in kwargs)
                )
            combined = {
                key: [item[key] for item in kwargs]
                for key in ("input_ids", "sampling_params")
            }
            combined.update(return_logprob=True, top_logprobs_num=5, stream=False)
            if any(adapters):
                combined[
                    (
                        "adapter_path"
                        if self.spec.server.mode == "native_oft"
                        else "lora_path"
                    )
                ] = adapters
            return await self.engine.async_generate(**combined)

        results = self._async(step.name, generate)
        if not isinstance(results, list) or len(results) != len(prompts):
            raise ScenarioContractError("batch response count differs from requests")
        state = self.state()
        observations = [
            capture_generation(result, state, top_k=5) for result in results
        ]
        observation = Observation(
            request_output_lengths=tuple(len(item.output_ids) for item in observations),
            request_texts=tuple(item.text for item in observations),
            output_ids=tuple(
                token for item in observations for token in item.output_ids
            ),
            text=json.dumps([item.text for item in observations], ensure_ascii=False),
            token_logprobs=tuple(
                score for item in observations for score in item.token_logprobs
            ),
            selected_logits={
                f"decode.{i:03d}.top_logprobs": values
                for i, values in enumerate(
                    values
                    for item in observations
                    for values in item.selected_logits.values()
                )
            },
            selected_token_ids={
                f"decode.{i:03d}.top_logprobs": values
                for i, values in enumerate(
                    values
                    for item in observations
                    for values in item.selected_token_ids.values()
                )
            },
            adapter_state=state,
            error=None,
        )
        observation.validate()
        return observation

    def _begin_lease(self, step):
        if self.lease is not None:
            raise ScenarioContractError("lease already active")

        async def begin():
            stream = await self.engine.async_generate(
                **self._kwargs(step.prompt_id, step.adapter, stream=True)
            )
            first = await anext(stream)
            if first["meta_info"].get("finish_reason") is not None:
                await stream.aclose()
                raise ScenarioContractError("retained lease request already finished")
            self.lease = stream
            return first

        first = self._async(step.name, begin)
        self.active_adapter = step.adapter
        return capture_generation(first, self.state(), top_k=5)

    def _begin_upsert(self, step):
        if self.lease is None or step.input_kind != "distributed":
            raise ScenarioContractError(
                "leased upsert requires a live lease and distributed input"
            )
        before = self.state()
        self.upsert_transfer = self._start_distributed(
            step.name,
            self.fixtures["policy-b"],
            lambda payload, group: self.control.load_distributed(
                step.adapter, payload, group, upsert=True
            ),
        )
        self.upsert_future, _, self.upsert_deadline, _ = self.upsert_transfer
        if self.upsert_future.done():
            require_control_success(self.upsert_future.result(), step.name)
            raise ScenarioContractError(
                "upsert completed before retained lease drained"
            )
        observation_future = self.control.observe_lease_wait(
            step.adapter, before["active"]["id"]
        )
        self.lease_work.append(observation_future)
        after = self._select_state(
            self._wait_with_update(
                observation_future, self.upsert_deadline, "lease wait observation"
            )
        )
        if before != after:
            raise ScenarioContractError("upsert changed state while lease was held")
        return self._state_observation(after)

    def _complete_lease(self, step):
        if self.lease is None or self.upsert_future is None:
            raise ScenarioContractError("no pending leased upsert")
        if self.upsert_transfer[3]:
            raise self.upsert_transfer[3][0]
        before = self.state(concurrent=True)

        async def drain():
            last = None
            async for chunk in self.lease:
                last = chunk
            return last

        # The synchronous control wrapper is currently driving engine.loop.
        future = submit_engine_coroutine(self.engine, drain)
        self.lease_work.append(future)
        result = self._wait_with_update(
            future,
            min(self.upsert_deadline, time.monotonic() + self.timeouts.inference),
            step.name,
        )
        observation = capture_generation(result, before, top_k=5)
        control_result = self._finish_distributed(
            self.upsert_transfer, "upsert.complete"
        )
        require_control_success(control_result, "upsert.complete")
        capture_control(control_result, self.state())
        self.lease, self.upsert_future = None, None
        return observation

    def execute(self, step: LifecycleStep) -> Observation:
        if step not in lifecycle_steps(self.spec.server.mode):
            raise ScenarioContractError(f"undeclared lifecycle step: {step.name}")
        self.launch()
        if step.action in {
            "reject_duplicate",
            "reject_stale",
            "reject_wrong_id",
            "reject_wrong_name",
        }:
            return self._preflight_rejection(step)
        if step.action == "retry_unload":
            return self._retry_unload(step)
        if step.action in {"reject_invalid_config", "reject_unsupported_target"}:
            return self._invalid_payload(step)
        if step.action in {
            "inject_update_failure",
            "inject_activation_failure",
            "inject_unload_failure",
        }:
            return self._inject_failure(step)
        if step.action in {"cancel_fan_out", "cancel_rollback"}:
            return self._cancel_stage(step)
        if step.action == "cancel_lease_drain":
            return self._cancel_lease_drain(step)
        if step.action == "cancel_publication":
            return self._cancel_publication(step)
        if step.action == "cancel_eviction":
            return self._cancel_eviction(step)
        if step.action in {
            "begin_paused_request",
            "reject_paused_activation",
            "resume_paused_request",
        }:
            return self._paused_request(step)
        if step.action == "fill_registry":
            return self._control_observation(
                self._load("policy-b", "path", step.name), step
            )
        if step.action == "evict_registry":
            return self._evict_registry(step)
        if step.name == "unload.final":
            return self._final_unload(step)
        if step.action == "load":
            # Clear the previously observed startup registration before each
            # immediate mechanism. No inference output is substituted.
            if step.adapter in {
                record["name"] for record in self.state()["registered"]
            }:
                result = self._call(
                    "setup.unload",
                    lambda: self.control.unload(step.adapter),
                    self.timeouts.control,
                )
                self.active_adapter = None
                capture_control(
                    require_control_success(result, "setup.unload"), self.state()
                )
            return self._control_observation(
                self._load(step.adapter, step.input_kind, step.name), step
            )
        if step.action in ("generate", "startup_generate"):
            if step.action == "startup_generate" and not self.preloaded:
                self._restart()
            if step.name.startswith("switch."):
                self._ensure_loaded(("policy-a", "policy-b"))
            return self._generate(step)
        if step.action in ("mixed_batch", "concurrent"):
            return self._batch(step)
        if step.action == "begin_lease":
            return self._begin_lease(step)
        if step.action == "upsert":
            return self._begin_upsert(step)
        if step.action == "complete_lease":
            return self._complete_lease(step)
        if step.action == "stage":
            before = self.state(step.adapter)
            identity = AdapterIdentity(
                step.adapter, before["active"]["id"], step.version
            )
            result = self._distributed_call(
                step.name,
                self.fixtures[step.version],
                lambda payload, group: self.control.stage(identity, payload, group),
            )
            observed = self._control_observation(result, step)
            state = observed.adapter_state
            wanted = {
                "name": identity.name,
                "id": identity.adapter_id,
                "version": identity.version,
            }
            if (
                state["active"] != before["active"]
                or state["staged"] is None
                or any(state["staged"][key] != value for key, value in wanted.items())
            ):
                raise ScenarioContractError(
                    "stage changed active or requested staged identity"
                )
            self.staged_identities[(step.adapter, step.version)] = identity
            return observed
        if step.action == "activate":
            identity = self.staged_identities[(step.adapter, step.version)]
            result = self._call(
                step.name,
                lambda: self.control.activate(identity),
                self.timeouts.control,
            )
            observed = self._control_observation(result, step)
            if (
                observed.adapter_state["active"]
                != {
                    "name": identity.name,
                    "id": identity.adapter_id,
                    "version": identity.version,
                }
                or observed.adapter_state["staged"] is not None
            ):
                raise ScenarioContractError(
                    "activation did not promote exact staged identity"
                )
            return observed
        if step.action == "unload":
            result = self._call(
                step.name,
                lambda: self.control.unload(step.adapter),
                self.timeouts.control,
            )
            self.active_adapter = None
            return self._control_observation(result, step)
        if step.action == "inspect_state":
            return self._state_observation(self.state())
        if step.action == "restart":
            self._restart()
            return self._state_observation(self.state())
        raise ScenarioContractError(f"unimplemented lifecycle action: {step.action}")

    def _restart(self):
        self._stop_resources()
        self.engine, self.control, self.sender = None, None, None
        self.active_adapter = None
        self.staged_identities.clear()
        self.launch(preload=True)

    def run_stress(self, *, job_timeout=3600.0):
        from .stress_case import StressSpec, run_stress

        try:
            if self.closed or self.spec.server.mode == "base":
                raise ScenarioContractError("stress requires an open native runner")
            self.launch()
            initial = self.state()
            if (
                initial["registered"]
                or initial["cache_identity"]
                or initial["staged"]
                or initial["quarantined"]
                or initial["tombstoned"]
            ):
                raise ScenarioContractError(
                    "stress requires an initially empty adapter state"
                )
            a_tensors, a_config = self._independent(
                "stress.fixture-a",
                lambda: load_fixture(self.fixtures["policy-a"]),
                self.timeouts.control,
            )
            b_tensors, b_config = self._independent(
                "stress.fixture-b",
                lambda: load_fixture(self.fixtures["policy-b"]),
                self.timeouts.control,
            )

            def upsert(name, cycle, cancel):
                request_id = f"stress.{cycle:03d}.{name}.upsert"

                def operation(payload, group):
                    return self.control.load_distributed(
                        name, payload, group, upsert=True
                    )

                fixture = self.fixtures[
                    "policy-b" if name == "policy-a" else "policy-a"
                ]
                if not cancel:
                    return self._distributed_call(request_id, fixture, operation)
                gate = PhaseGate("fan-out")
                with self._faults().wrap("load", gate=gate):
                    transfer = self._start_distributed(request_id, fixture, operation)
                    cancellation = self._finish_cancelled_transfer(transfer, gate)
                self.emit("stress.cancelled", cycle=cycle, adapter=name)
                raise cancellation

            def generate(count, stream):
                async def mixed():
                    return await asyncio.gather(
                        *(
                            self._generate_one(
                                self._kwargs(
                                    "factual",
                                    ("policy-a", "policy-b", None)[index % 3],
                                    stream=stream,
                                )
                            )
                            for index in range(count)
                        )
                    )

                results = self._async("stress.mixed", mixed)
                state = self.state()
                return [
                    capture_generation(result, state, top_k=5) for result in results
                ]

            return run_stress(
                StressSpec(
                    self.control,
                    a_tensors,
                    a_config,
                    b_tensors,
                    b_config,
                    generate,
                    initial,
                    upsert,
                    max(
                        self.timeouts.control,
                        self.timeouts.collective,
                        self.timeouts.inference,
                    ),
                    job_timeout,
                    emit=self.emit,
                )
            )
        except BaseException as error:
            self.failure = error
            self.close()
            raise

    def run_selected(self, selection="full"):
        try:
            if self.closed or self.observations:
                raise ScenarioContractError("runner is single-use")
            self.selection = self.resolve_selection(selection)
            by_name = {
                step.name: step for step in lifecycle_steps(self.spec.server.mode)
            }
            for name in self.selection:
                observation = self.execute(by_name[name])
                observation.validate()
                self.observations[name] = observation
                self.emit("step.complete", transition=name)
            if self.upsert_future is not None or self.lease is not None:
                raise ScenarioContractError(
                    "selection ended with an incomplete lease/update"
                )
            if selection == "full":
                validate_lifecycle_observations(self.observations)
            return dict(self.observations)
        except BaseException as error:
            self.failure = error
            self._save_failure_details(error)
            try:
                self.emit(
                    "shard.failed", error_type=type(error).__name__, message=str(error)
                )
            except Exception:
                pass  # Diagnostic failure cannot replace the original failure.
            self.close()
            raise

    def _save_failure_details(self, error):
        details = getattr(error, "details", None)
        if details is None:
            return
        path = self.spec.bundle_output.with_name("failure-details.json")
        try:
            artifact = {
                "artifact_kind": "adapter-failure-details-v1",
                "status": "failed",
                "case_id": self.spec.case_id,
                "revision_sha": self.spec.revision_sha,
                "error_type": type(error).__name__,
                "message": str(error),
                "details": details,
            }
            digest = canonical_sha256(artifact)
            encoded = json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True)
            # Preserve earlier evidence and reject invalid JSON before opening.
            with path.open("x") as stream:
                stream.write(encoded + "\n")
            self.emit("failure.details.saved", path=str(path), artifact_hash=digest)
        except Exception as diagnostic_error:
            try:
                self.emit(
                    "failure.details.failed",
                    path=str(path),
                    error_type=type(diagnostic_error).__name__,
                    message=str(diagnostic_error),
                )
            except Exception:
                pass  # Diagnostic failures must not replace the original error.

    def _scheduler_memory(self, operation, sample_id):
        async def control():
            manager = self.engine.tokenizer_manager
            future = submit_engine_coroutine(
                self.engine,
                lambda: getattr(manager, f"{operation}_cuda_memory_peak")(sample_id),
            )
            return await asyncio.wrap_future(future)

        result = self._async(f"performance.{operation}", control)
        if (
            result.success is not True
            or result.sample_id != sample_id
            or result.operation != operation
        ):
            raise ScenarioContractError("CUDA memory sample identity/outcome mismatch")
        if (
            not isinstance(result.ranks, list)
            or len(result.ranks) != self.spec.server.tp_size
        ):
            raise ScenarioContractError("CUDA memory sample has incomplete TP ranks")
        allocated = reserved = 0
        for rank, row in enumerate(result.ranks):
            if (
                type(row.rank) is not int
                or row.rank != rank
                or row.sample_id != sample_id
                or row.success is not True
                or row.operation != operation
            ):
                raise ScenarioContractError(
                    "CUDA memory rank identity/outcome mismatch"
                )
            if operation == "reset":
                if row.allocated_bytes is not None or row.reserved_bytes is not None:
                    raise ScenarioContractError("unexpected CUDA reset metrics")
                continue
            if (
                type(row.allocated_bytes) is not int
                or type(row.reserved_bytes) is not int
                or not 0 < row.allocated_bytes <= row.reserved_bytes
            ):
                raise ScenarioContractError("invalid or zero CUDA memory rank metrics")
            allocated += row.allocated_bytes
            reserved += row.reserved_bytes
        return allocated, reserved

    def _assert_qualification_inputs(self):
        from .preflight import load_prompt_manifest

        if self.spec != self.qualification_spec:
            raise ScenarioContractError(
                "server/run identity changed from qualification manifest"
            )
        prompts = load_prompt_manifest(self.spec.prompts_file)
        expected = {
            "prompts": {
                prompt.id: list(prompt.input_ids) for prompt in prompts.prompts
            },
            "batches": {
                batch.id: [request.prompt_id for request in batch.requests]
                for batch in prompts.batches
            },
        }
        if canonical_sha256(expected) != canonical_sha256(
            {"prompts": self.prompts, "batches": self.batches}
        ):
            raise ScenarioContractError(
                "execution differs from immutable prompt/batch manifest"
            )
        expected_startup = (
            ()
            if self.spec.server.mode == "base"
            else (("policy-a", str(self.fixtures["policy-a"])),)
        )
        if self.spec.server.startup_adapters != expected_startup:
            raise ScenarioContractError(
                "startup adapters differ from immutable fixture manifest"
            )

    def _performance_requests(self):
        adapter = None if self.spec.server.mode == "base" else "policy-a"
        warmup = self._kwargs("factual", adapter)
        batch = self._kwargs("factual", adapter)
        batch["input_ids"] = [
            list(self.prompts[name]) for name in self.batches["batch-32"]
        ]
        if len(batch["input_ids"]) != 32:
            raise ScenarioContractError("performance requires exactly 32 requests")
        return {"warmup": warmup, "batch": batch}

    def _measure_performance(self, procedure_hash):
        recorder = PerformanceRecorder(procedure_hash)
        adapter = None if self.spec.server.mode == "base" else "policy-a"
        requests = self._performance_requests()
        for index in range(3):
            if self.engine is not None:
                raise ScenarioContractError("performance requires a fresh engine")
            start = time.perf_counter()
            self.launch(preload=True)
            recorder.record_startup(float(time.perf_counter() - start))
            state = self.state(adapter)
            result = self._async(
                "performance.warmup",
                lambda: self._generate_one(copy.deepcopy(requests["warmup"])),
            )
            capture_generation(result, state, top_k=5)
            if self.qualification_provenance is not None:
                self._capture_effective_launch(True)
            sample_id = f"{self.spec.case_id}:{self.spec.revision_sha}:{self.spec.server.revision_kind}:{self.spec.repetition}:{index}"
            self._scheduler_memory("reset", sample_id)
            start = time.perf_counter()
            results = self._async(
                "performance.sample",
                lambda: self._generate_one(copy.deepcopy(requests["batch"])),
            )
            latency = float(time.perf_counter() - start)
            allocated, reserved = self._scheduler_memory("read", sample_id)
            if self.qualification_provenance is not None:
                self._capture_effective_launch(True)
            if not isinstance(results, list) or len(results) != 32:
                raise ScenarioContractError(
                    "performance response must contain 32 requests"
                )
            tokens = sum(
                len(capture_generation(result, state, top_k=5).output_ids)
                for result in results
            )
            recorder.record_sample(
                latency_seconds=latency,
                output_tokens=tokens,
                peak_allocated_bytes=allocated,
                peak_reserved_bytes=reserved,
            )
            self._stop_resources()
            self.engine = self.control = self.sender = None
        return recorder.build()

    def run_qualified(self, *, argv):
        """Publish one full shard only after all engines and senders stop cleanly."""
        try:
            if self.engine is not None or self.closed or self.observations:
                raise ScenarioContractError("qualification requires an unused runner")
            for path in (self.spec.bundle_output, self.spec.completion_output):
                if path.exists() or path.is_symlink():
                    raise FileExistsError(path)
            identity = capture_run_identity(self.spec, self.fixtures, argv)
            self.qualification_spec = self.spec
            self._assert_qualification_inputs()
            case, provenance = identity
            self.qualification_provenance = provenance
            procedure = {
                "version": 1,
                "seed": RUN_SEED,
                "samples": 3,
                "effective_launches": self.effective_launches,
                "memory_boundary_hash": provenance["metadata"]["memory_boundary_hash"],
                "launch": performance_launch_identity(self.spec.server, provenance),
                "requests": self._performance_requests(),
                "warmup": "one factual request on a fresh engine, before every sample",
                "workload": "one fixed batch-32, startup policy-a for native modes",
                "sampling": self._kwargs("factual")["sampling_params"],
                "memory": "synchronize/reset each scheduler; request set; synchronize/read each scheduler; sum all TP allocated/reserved peaks",
                "latency": "perf_counter elapsed around async_generate, excluding memory RPCs",
                "throughput": "actual output token IDs across all 32 responses divided by latency",
                "implementation": [
                    inspect.getsource(method)
                    for method in (
                        launch_engine,
                        ShardRunner._measure_performance,
                        ShardRunner._performance_requests,
                        ShardRunner._scheduler_memory,
                        ShardRunner._kwargs,
                        ShardRunner._generate_one,
                        ShardRunner._async,
                    )
                ],
            }
            procedure_hash = canonical_sha256(procedure)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "case_key": case.to_dict(),
                "performance_procedure_hash": procedure_hash,
                "provenance_hashes": {
                    key: provenance[key] for key in PROVENANCE_HASH_KEYS
                },
                "server_args": [
                    "--model-path",
                    self.spec.server.model_path,
                    *server_other_args(self.spec.server),
                    "--random-seed",
                    str(RUN_SEED),
                ],
                "request_order": list(self.resolve_selection("full")),
                "seed": RUN_SEED,
                "metadata": {
                    "case_id": self.spec.case_id,
                    "repetition": self.spec.repetition,
                    "role": self.spec.server.revision_kind,
                    "performance_procedure": procedure,
                    "engine_kwargs": dict(
                        engine_kwargs(self.spec.server), random_seed=RUN_SEED
                    ),
                    "initial_engine_kwargs": dict(
                        engine_kwargs(replace(self.spec.server, startup_adapters=())),
                        random_seed=RUN_SEED,
                    ),
                },
            }
            self.run_selected("full")
            self._stop_resources()
            self.engine = self.control = self.sender = None
            if capture_run_identity(self.spec, self.fixtures, argv) != identity:
                raise ScenarioContractError(
                    "qualification inputs changed during lifecycle"
                )
            performance = self._measure_performance(procedure_hash)
            procedure = copy.deepcopy(procedure)
            manifest["metadata"]["performance_procedure"] = procedure
            procedure_hash = canonical_sha256(procedure)
            performance = replace(performance, procedure_hash=procedure_hash)
            manifest["performance_procedure_hash"] = procedure_hash
            self.close()
            self._assert_qualification_inputs()
            if capture_run_identity(self.spec, self.fixtures, argv) != identity:
                raise ScenarioContractError(
                    "qualification inputs changed before publication"
                )
            bundle = RunBundle.create(
                case_key=case,
                manifest=manifest,
                provenance={
                    **provenance,
                    "metadata": {
                        **provenance["metadata"],
                        "sender_runtime_origins": copy.deepcopy(
                            self.sender_runtime_origins
                        ),
                    },
                },
                observations=self.observations,
                performance=performance,
                completion={"status": "complete", "exit_code": 0, "metadata": {}},
            )
            publish_bundle(bundle, self.spec.bundle_output, self.spec.completion_output)
            return bundle
        except BaseException as error:
            self.failure = self.failure or error
            self.close()
            raise self.failure

    def _stop_resources(self):
        failures = []
        for future in self.lease_work:
            if not future.done():
                future.cancel()
        if self.lease is not None:
            try:
                future = submit_engine_coroutine(self.engine, self.lease.aclose)
                # If a failed control stopped the loop, drive queued cleanup on
                # its serialized owner. A live update instead drives aclose.
                self.engine_pool.submit(
                    lambda: self.engine.loop.run_until_complete(asyncio.sleep(0))
                )
                self._wait(future, self.timeouts.teardown, "lease.close")
            except BaseException as error:
                failures.append(error)
            self.lease = None
        if self.sender is not None:
            try:
                self._independent(
                    "sender.close",
                    lambda: self.sender.close(self.timeouts.teardown),
                    self.timeouts.teardown,
                )
            except BaseException as error:
                failures.append(error)
                # Session.close may stall destroying the engine group before
                # reaching its child. Still terminate that child independently.
                peer = self.sender.sender
                if hasattr(peer, "terminate"):
                    try:
                        self._independent(
                            "sender.terminate", peer.terminate, self.timeouts.teardown
                        )
                    except BaseException as peer_error:
                        failures.append(peer_error)
        if self.engine is not None:
            if (
                self.qualification_provenance is not None
                and self.failure is None
                and not self.timed_out
                and not failures
            ):
                try:
                    bind_runtime()
                    self._capture_effective_launch(self.preloaded)
                except BaseException as error:
                    failures.append(error)
            try:
                self._independent(
                    "engine.shutdown",
                    lambda: stop_engine(self.engine),
                    self.timeouts.teardown,
                )
            except BaseException as error:
                failures.append(error)
        for error in failures:
            try:
                self.emit(
                    "teardown.failed",
                    error_type=type(error).__name__,
                    message=str(error),
                )
            except Exception:
                pass
        if failures:
            raise failures[0]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._stop_resources()
        except BaseException:
            if self.failure is None:
                raise
        finally:
            self.engine_pool.close()
            self.sender_pool.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("base", "native_lora", "native_oft")
    )
    for flag in (
        "bundle-output",
        "completion-output",
        "case-id",
        "revision-sha",
        "precision",
        "model-path",
        "checkpoint-manifest",
        "prompts-file",
        "fixture-manifest",
    ):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument(
        "--revision-kind", required=True, choices=("source", "candidate")
    )
    parser.add_argument("--architecture", required=True, choices=("dense", "moe"))
    parser.add_argument("--cuda-graph", required=True, choices=("on", "off"))
    parser.add_argument(
        "--selection",
        choices=("full", "smoke"),
        default="full",
        help="smoke is non-qualifying; full requires every declared action",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--quantization")
    parser.add_argument("--moe-runner")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repetition", type=int, default=0)
    for field, value in vars(Timeouts()).items():
        parser.add_argument(f"--{field}-timeout", type=float, default=value)
    return parser


def inputs_from_args(args):
    # Task 9 verifies immutable hashes/provenance before publication. Runner
    # execution consumes existing files and never regenerates fixture bytes.
    checkpoint = json.loads(Path(args.checkpoint_manifest).read_text())
    if not isinstance(checkpoint, dict):
        raise ScenarioContractError("checkpoint manifest must be an object")
    manifest = json.loads(Path(args.fixture_manifest).read_text())
    fixtures = {
        key: Path(value).resolve(strict=True) for key, value in manifest.items()
    }
    required_fixtures = {"policy-a", "policy-b"} | {
        step.version for step in lifecycle_steps(args.mode) if step.action == "stage"
    }
    if args.mode != "base" and not required_fixtures <= fixtures.keys():
        raise ScenarioContractError("fixture manifest is missing native fixture paths")
    prompts, batches = {}, {}
    for line in Path(args.prompts_file).read_text().splitlines():
        record = json.loads(line)
        if record["kind"] == "prompt":
            prompts[record["id"]] = record["input_ids"]
        elif record["kind"] == "batch":
            batches[record["id"]] = [
                request["prompt_id"] for request in record["requests"]
            ]
    adapter_config = (
        json.loads((fixtures["policy-a"] / "adapter_config.json").read_text())
        if args.mode != "base"
        else {}
    )
    server = ServerSpec(
        args.revision_kind,
        args.model_path,
        args.mode,
        args.port,
        args.tp_size,
        args.ep_size,
        args.cuda_graph == "on",
        quantization=args.quantization,
        moe_runner=args.moe_runner,
        startup_adapters=(
            () if args.mode == "base" else (("policy-a", str(fixtures["policy-a"])),)
        ),
        base_gpu_id=args.base_gpu_id,
        max_lora_rank=adapter_config.get("r") if args.mode == "native_lora" else None,
        lora_target_modules=(
            tuple(adapter_config.get("target_modules", ()))
            if args.mode == "native_lora"
            else ()
        ),
        max_oft_block_size=(
            adapter_config.get("oft_block_size") if args.mode == "native_oft" else None
        ),
        peft_target_modules=(
            tuple(adapter_config.get("target_modules", ()))
            if args.mode == "native_oft"
            else ()
        ),
    )
    spec = RunSpec(
        server,
        args.case_id,
        args.revision_sha,
        args.architecture,
        args.precision,
        Path(args.checkpoint_manifest),
        Path(args.prompts_file),
        Path(args.fixture_manifest),
        Path(args.bundle_output),
        Path(args.completion_output),
        args.max_new_tokens,
        args.repetition,
    )
    return spec, prompts, batches, fixtures


def main(argv=None):
    args = build_parser().parse_args(argv)
    runner = None
    try:
        spec, prompts, batches, fixtures = inputs_from_args(args)
        runner = ShardRunner(
            spec,
            prompts=prompts,
            batches=batches,
            fixtures=fixtures,
            timeouts=Timeouts(
                **{
                    field: getattr(args, field + "_timeout")
                    for field in vars(Timeouts())
                }
            ),
        )
        if args.selection == "full":
            runner.run_qualified(
                argv=sys.argv if argv is None else [sys.argv[0], *argv]
            )
        else:
            runner.run_selected(args.selection)
            runner.close()
            runner.emit("smoke.complete", qualifying=False)
        return 0
    except BaseException as error:
        if runner is not None:
            runner.failure = runner.failure or error
            runner.close()
        print(
            json.dumps(
                {
                    "event": "run.failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2 if runner is not None and runner.timed_out else 1


def cli():
    status = main()
    if status:
        # Native dependency executors may register unbounded interpreter joins.
        # main already flushed diagnostics and attempted both bounded teardowns.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)


if __name__ == "__main__":
    cli()

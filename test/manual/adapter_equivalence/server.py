"""Server argument contracts for native adapter qualification revisions."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import IO, Literal, Protocol

from .scenarios import ScenarioContractError
from .schema import AdapterMode

_MODE_ARGS = {
    "base": (),
    "native_lora": ("--enable-lora", "--enable-lora-staging"),
    "native_oft": ("--peft-method", "oft", "--oft-type", "oft"),
}

_OFT_MODES = {"native_oft"}


@dataclass(frozen=True)
class ServerSpec:
    """Everything that changes the observable server launch contract."""

    revision_kind: Literal["source", "candidate"]
    model_path: str
    mode: str
    port: int
    tp_size: int
    ep_size: int
    cuda_graph: bool
    quantization: str | None = None
    moe_runner: str | None = None
    startup_adapters: tuple[tuple[str, str], ...] = ()
    max_lora_rank: int | None = None
    lora_target_modules: tuple[str, ...] = ()
    max_oft_block_size: int | None = None
    peft_target_modules: tuple[str, ...] = ()
    base_gpu_id: int = 1
    mem_fraction_static: float = 0.8
    # Keep KV capacity stable across fresh workers and repeated constructions.
    max_total_tokens: int = 32768

    def __post_init__(self) -> None:
        object.__setattr__(self, "startup_adapters", tuple(self.startup_adapters))
        object.__setattr__(self, "lora_target_modules", tuple(self.lora_target_modules))
        object.__setattr__(self, "peft_target_modules", tuple(self.peft_target_modules))
        if not self.model_path:
            raise ScenarioContractError("model_path must be non-empty")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ScenarioContractError("port must be an integer from 1 to 65535")
        for name in ("tp_size", "ep_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ScenarioContractError(f"{name} must be a positive integer")
        if self.ep_size > self.tp_size or self.tp_size % self.ep_size:
            raise ScenarioContractError("ep_size must evenly divide tp_size")
        if type(self.cuda_graph) is not bool:
            raise ScenarioContractError("cuda_graph must be a boolean")
        if type(self.base_gpu_id) is not int or self.base_gpu_id < 0:
            raise ScenarioContractError("base_gpu_id must be a non-negative integer")
        if type(self.max_total_tokens) is not int or self.max_total_tokens <= 0:
            raise ScenarioContractError("max_total_tokens must be a positive integer")
        if (
            type(self.mem_fraction_static) is not float
            or not 0.0 < self.mem_fraction_static < 1.0
        ):
            raise ScenarioContractError(
                "mem_fraction_static must be a float between zero and one"
            )
        names = [name for name, _ in self.startup_adapters]
        if any(not name or not path for name, path in self.startup_adapters):
            raise ScenarioContractError(
                "startup adapter names and paths must be non-empty"
            )
        if len(names) != len(set(names)):
            raise ScenarioContractError("startup adapter names must be unique")
        if self.mode == "base" and self.startup_adapters:
            raise ScenarioContractError("base mode cannot preload adapters")
        mode_server_args(self.revision_kind, self.mode)
        if self.max_lora_rank is not None and (
            type(self.max_lora_rank) is not int or self.max_lora_rank <= 0
        ):
            raise ScenarioContractError(
                "max_lora_rank must be a positive integer when present"
            )
        if any(
            type(module) is not str or not module for module in self.lora_target_modules
        ):
            raise ScenarioContractError(
                "lora_target_modules must contain non-empty strings"
            )
        if len(self.lora_target_modules) != len(set(self.lora_target_modules)):
            raise ScenarioContractError("lora_target_modules must be unique")
        if self.mode != "native_lora" and (
            self.max_lora_rank is not None or self.lora_target_modules
        ):
            raise ScenarioContractError("LoRA shape fields require native_lora mode")
        if (
            self.mode == "native_lora"
            and not self.startup_adapters
            and (self.max_lora_rank is None or not self.lora_target_modules)
        ):
            raise ScenarioContractError(
                "dynamic LoRA requires max_lora_rank and lora_target_modules"
            )
        if self.max_oft_block_size is not None and (
            type(self.max_oft_block_size) is not int or self.max_oft_block_size <= 0
        ):
            raise ScenarioContractError(
                "max_oft_block_size must be a positive integer when present"
            )
        if any(
            type(module) is not str or not module for module in self.peft_target_modules
        ):
            raise ScenarioContractError(
                "peft_target_modules must contain non-empty strings"
            )
        if len(self.peft_target_modules) != len(set(self.peft_target_modules)):
            raise ScenarioContractError("peft_target_modules must be unique")
        if self.mode not in _OFT_MODES and (
            self.max_oft_block_size is not None or self.peft_target_modules
        ):
            raise ScenarioContractError("OFT shape fields require an OFT mode")
        if (
            self.mode in _OFT_MODES
            and not self.startup_adapters
            and (self.max_oft_block_size is None or not self.peft_target_modules)
        ):
            raise ScenarioContractError(
                "dynamic OFT requires max_oft_block_size and peft_target_modules"
            )
        for field in ("lora_target_modules", "peft_target_modules"):
            object.__setattr__(self, field, tuple(sorted(getattr(self, field))))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def mode_server_args(revision_kind: str, mode: str) -> tuple[str, ...]:
    """Return the native adapter-selection arguments for either revision."""

    if revision_kind not in {"source", "candidate"}:
        raise ScenarioContractError(f"unknown revision kind: {revision_kind}")
    try:
        return _MODE_ARGS[mode]
    except KeyError as error:
        raise ScenarioContractError(
            f"unknown {revision_kind} adapter mode: {mode}"
        ) from error


@dataclass(frozen=True)
class AdapterIdentity:
    """Exact logical identity used by staged adapter transactions."""

    name: str
    adapter_id: str
    version: str


@dataclass(frozen=True)
class DistributedPayload:
    """Tensor metadata and config shared by immediate and staged broadcasts."""

    names: tuple[str, ...]
    dtypes: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    config: Mapping[str, object]


@dataclass(frozen=True)
class ControlResult:
    """Adapter-operation result normalized across LoRA and OFT APIs."""

    success: bool
    message: str
    loaded_adapters: dict[str, object]
    active_version: str | None
    staged_version: str | None


def _normalized_version(value: object) -> str | None:
    return None if value is None else str(value)


def normalize_control_result(result: object) -> ControlResult:
    """Require explicit success instead of equating no exception with success."""

    if isinstance(result, tuple) and len(result) == 2:
        success, message = result
        if type(success) is not bool:
            raise ScenarioContractError("control result success must be boolean")
        return ControlResult(success, str(message), {}, None, None)

    success = getattr(result, "success", None)
    if type(success) is not bool:
        raise ScenarioContractError("control result omitted explicit success")
    message = getattr(result, "error_message", None) or getattr(result, "message", "")
    raw_loaded = getattr(result, "loaded_adapters", None)
    if raw_loaded is None:
        loaded_adapters: dict[str, object] = {}
    elif isinstance(raw_loaded, Mapping):
        loaded_adapters = dict(raw_loaded)
    else:
        raise ScenarioContractError("control result loaded_adapters must be an object")
    return ControlResult(
        success=success,
        message=str(message),
        loaded_adapters=loaded_adapters,
        active_version=_normalized_version(
            getattr(result, "active_adapter_version", None)
        ),
        staged_version=_normalized_version(
            getattr(result, "staged_adapter_version", None)
        ),
    )


class AdapterControl(Protocol):
    mode: AdapterMode

    def load_path(
        self, name: str, path: str, *, pinned: bool = False
    ) -> ControlResult: ...

    def load_tensors(
        self,
        name: str,
        tensors: Mapping[str, object],
        config: Mapping[str, object],
        *,
        upsert: bool = False,
    ) -> ControlResult: ...

    def load_distributed(
        self,
        name: str,
        payload: DistributedPayload,
        group_name: str,
        *,
        upsert: bool = False,
    ) -> ControlResult: ...

    def stage(
        self,
        identity: AdapterIdentity,
        payload: DistributedPayload,
        group_name: str,
    ) -> ControlResult: ...

    def activate(self, identity: AdapterIdentity) -> ControlResult: ...

    def unload(self, name: str) -> ControlResult: ...

    def inspect_state(self) -> dict[str, object]: ...

    def observe_lease_wait(self, name: str, adapter_id: str) -> Future: ...


def submit_engine_coroutine(engine, factory) -> Future:
    """Create work only once the loop runs; cancellation cannot leak a coroutine."""
    result = Future()
    tasks = []

    def complete(task):
        try:
            value = task.result()
        except BaseException as error:
            if result.set_running_or_notify_cancel():
                result.set_exception(error)
        else:
            if result.set_running_or_notify_cancel():
                result.set_result(value)

    def start():
        if result.cancelled():
            return

        async def run():
            return await factory()

        task = engine.loop.create_task(run())
        tasks.append(task)
        task.add_done_callback(complete)
        if result.cancelled():
            task.cancel()

    def cancel(future):
        if future.cancelled():
            for task in tasks:
                engine.loop.call_soon_threadsafe(task.cancel)

    result.add_done_callback(cancel)
    engine.loop.call_soon_threadsafe(start)
    return result


class BaseControl:
    mode: AdapterMode = "base"

    def __init__(self, engine: object) -> None:
        self.engine = engine

    @staticmethod
    def _reject() -> ControlResult:
        raise ScenarioContractError("base mode does not support adapter operations")

    def load_path(self, name: str, path: str, *, pinned: bool = False) -> ControlResult:
        return self._reject()

    def load_tensors(
        self,
        name: str,
        tensors: Mapping[str, object],
        config: Mapping[str, object],
        *,
        upsert: bool = False,
    ) -> ControlResult:
        return self._reject()

    def load_distributed(
        self,
        name: str,
        payload: DistributedPayload,
        group_name: str,
        *,
        upsert: bool = False,
    ) -> ControlResult:
        return self._reject()

    def stage(
        self,
        identity: AdapterIdentity,
        payload: DistributedPayload,
        group_name: str,
    ) -> ControlResult:
        return self._reject()

    def activate(self, identity: AdapterIdentity) -> ControlResult:
        return self._reject()

    def unload(self, name: str) -> ControlResult:
        return self._reject()

    def inspect_state(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "registered": [],
            "active": None,
            "staged": None,
            "registry_occupancy": 0,
            "quarantined": [],
            "tombstoned": [],
            "cache_identity": {},
        }


class _NativeAdapterControl:
    mode: AdapterMode
    load_format: str
    path_method: str
    tensors_method: str
    distributed_method: str
    unload_method: str
    registry_attribute: str
    pending_attribute: str
    quarantined_attribute: str
    tombstoned_attribute: str
    cache_attribute: str
    ref_name_attribute: str
    ref_id_attribute: str
    ref_version_attribute: str

    def __init__(self, engine: object) -> None:
        self.engine = engine
        self._focus_name: str | None = None

    @staticmethod
    def _task_frames(task):
        frames = []
        coroutine = task.get_coro()
        while coroutine is not None:
            frame = getattr(coroutine, "cr_frame", None)
            if frame is not None:
                frames.append(frame)
            coroutine = getattr(coroutine, "cr_await", None)
        return frames

    def _oft_child_frames(self, frames, request):
        from sglang.srt.adapter_sync.tokenizer_backend import finish_irreversible_update

        manager = self.engine.tokenizer_manager
        if not any(
            frame.f_code.co_name == "_run_oft_wire_load"
            and frame.f_locals.get("self") is manager
            and frame.f_locals.get("obj") is request
            for frame in frames
        ):
            return []
        for frame in frames:
            if frame.f_code is not finish_irreversible_update.__code__:
                continue
            # Follow only the real shielding helper's exact child, not another
            # task with a similar stack or the same adapter name/counter.
            child = frame.f_locals.get("task")
            operation = frame.f_locals.get("operation")
            if not isinstance(child, asyncio.Task) or child.done():
                continue
            child_frames = self._task_frames(child)
            if child_frames and (
                child_frames[0].f_code is getattr(operation, "__code__", None)
                and child_frames[0].f_code.co_name == "dispatch_and_finish"
                and child_frames[0].f_locals.get("self") is manager
                and child_frames[0].f_locals.get("obj") is request
            ):
                return child_frames
        return []

    def _at_lease_wait(self, name: str, adapter_id: str) -> bool:
        """Observe the exact update at a held-lease barrier on its engine loop.

        LoRA waits at the model writer lock. OFT currently waits in its
        shielded child's old-ID counter; an exact writer wait also qualifies.
        No production locks, methods, registries, or requests are modified.
        """
        manager = self.engine.tokenizer_manager
        method = (
            "load_lora_adapter_from_distributed"
            if self.mode == "native_lora"
            else "load_oft_adapter_from_distributed"
        )
        name_field = "lora_name" if self.mode == "native_lora" else "adapter_name"
        registry = (
            manager.lora_registry
            if self.mode == "native_lora"
            else manager.peft_registry
        )
        for task in asyncio.all_tasks():
            frames = self._task_frames(task)
            public = next(
                (
                    frame
                    for frame in frames
                    if frame.f_code.co_name == method
                    and frame.f_locals.get("self") is manager
                    and getattr(frame.f_locals.get("obj"), name_field, None) == name
                ),
                None,
            )
            if public is None:
                continue
            request = public.f_locals["obj"]
            child_frames = (
                self._oft_child_frames(frames, request)
                if self.mode == "native_oft"
                else []
            )
            counter = registry._counters.get(adapter_id)
            if counter is None or counter.value() <= 0:
                continue
            lock = manager.model_update_lock
            reference = registry.get_all_adapters().get(name)
            if (
                getattr(reference, self.ref_id_attribute, None) == adapter_id
                and lock._readers > 0
                and lock._waiting_writers > 0
                and any(
                    frame.f_code.co_name == "acquire_writer"
                    and frame.f_locals.get("self") is lock
                    for frame in frames + child_frames
                )
            ):
                return True
            if (
                self.mode == "native_oft"
                and any(
                    frame.f_code.co_name == "_prepare_oft_wire_load"
                    and frame.f_locals.get("self") is manager
                    and frame.f_locals.get("obj") is request
                    and frame.f_locals.get("adapter_id") == adapter_id
                    for frame in child_frames
                )
                and any(
                    frame.f_code.co_name == "wait_for_unload"
                    and frame.f_locals.get("self") is registry
                    and frame.f_locals.get("uid") == adapter_id
                    for frame in child_frames
                )
                and any(
                    frame.f_code.co_name == "wait_for_zero"
                    and frame.f_locals.get("self") is counter
                    for frame in child_frames
                )
            ):
                return True
        return False

    def observe_lease_wait(self, name: str, adapter_id: str) -> Future:
        async def observe():
            while not self._at_lease_wait(name, adapter_id):
                await asyncio.sleep(0.005)
            return copy.deepcopy(self.inspect_state())

        return submit_engine_coroutine(self.engine, observe)

    def _record_focus(self, name: str, result: ControlResult) -> ControlResult:
        if result.success:
            self._focus_name = name
        return result

    def load_path(self, name: str, path: str, *, pinned: bool = False) -> ControlResult:
        result = getattr(self.engine, self.path_method)(name, path, pinned=pinned)
        return self._record_focus(name, normalize_control_result(result))

    def load_tensors(
        self,
        name: str,
        tensors: Mapping[str, object],
        config: Mapping[str, object],
        *,
        upsert: bool = False,
    ) -> ControlResult:
        kwargs = {"upsert": True} if upsert else {}
        result = getattr(self.engine, self.tensors_method)(
            name, dict(tensors), dict(config), **kwargs
        )
        return self._record_focus(name, normalize_control_result(result))

    def load_distributed(
        self,
        name: str,
        payload: DistributedPayload,
        group_name: str,
        *,
        upsert: bool = False,
    ) -> ControlResult:
        kwargs: dict[str, object] = {"group_name": group_name}
        if upsert:
            kwargs["upsert"] = True
        result = getattr(self.engine, self.distributed_method)(
            name,
            dict(payload.config),
            list(payload.names),
            list(payload.dtypes),
            [list(shape) for shape in payload.shapes],
            **kwargs,
        )
        return self._record_focus(name, normalize_control_result(result))

    def _manager_call(self, method_name: str, request: object) -> ControlResult:
        method = getattr(self.engine.tokenizer_manager, method_name)
        try:
            raw_result = self.engine.loop.run_until_complete(method(request, None))
        except ValueError as error:
            return ControlResult(False, str(error), {}, None, None)
        return normalize_control_result(raw_result)

    def stage(
        self,
        identity: AdapterIdentity,
        payload: DistributedPayload,
        group_name: str,
    ) -> ControlResult:
        from sglang.srt.managers.io_struct import (
            UpdateAdapterFromDistributedReqInput,
        )

        request = UpdateAdapterFromDistributedReqInput(
            names=list(payload.names),
            dtypes=list(payload.dtypes),
            shapes=[list(shape) for shape in payload.shapes],
            group_name=group_name,
            adapter_version=identity.version,
            load_format=self.load_format,
            adapter_config=dict(payload.config),
            adapter_name=identity.name,
            adapter_id=identity.adapter_id,
            double_buffer=True,
        )
        return self._record_focus(
            identity.name,
            self._manager_call("update_adapter_from_distributed", request),
        )

    def activate(self, identity: AdapterIdentity) -> ControlResult:
        from sglang.srt.managers.io_struct import ActivateAdapterVersionReqInput

        request = ActivateAdapterVersionReqInput(
            adapter_name=identity.name,
            adapter_version=identity.version,
            load_format=self.load_format,
            adapter_id=identity.adapter_id,
        )
        return self._record_focus(
            identity.name,
            self._manager_call("activate_adapter_version", request),
        )

    def unload(self, name: str) -> ControlResult:
        result = normalize_control_result(
            getattr(self.engine, self.unload_method)(name)
        )
        if result.success and self._focus_name == name:
            self._focus_name = None
        return result

    def _reference_record(
        self, reference: object, *, registry_slot: int | None = None
    ) -> dict[str, object]:
        name = getattr(reference, self.ref_name_attribute, None)
        adapter_id = getattr(reference, self.ref_id_attribute, None)
        version = getattr(reference, self.ref_version_attribute, None)
        if type(name) is not str or not name:
            raise ScenarioContractError("adapter reference omitted its name")
        if type(adapter_id) is not str or not adapter_id:
            raise ScenarioContractError("adapter reference omitted its ID")
        if version is None:
            raise ScenarioContractError("adapter reference omitted its version")
        pinned = getattr(reference, "pinned", False)
        if pinned is None:
            pinned = False
        if type(pinned) is not bool:
            raise ScenarioContractError("adapter reference pinned must be boolean")
        record: dict[str, object] = {
            "name": name,
            "id": adapter_id,
            "version": str(version),
        }
        if registry_slot is not None:
            record["registry_slot"] = registry_slot
        record["pinned"] = pinned
        return record

    @staticmethod
    def _state_mapping(value: object, name: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ScenarioContractError(f"{name} must be an object")
        return value

    def inspect_state(self) -> dict[str, object]:
        manager = self.engine.tokenizer_manager
        registry = getattr(manager, self.registry_attribute)
        registry_records = self._state_mapping(
            registry.get_all_adapters(), self.registry_attribute
        )
        ordered_names = sorted(registry_records)
        registered = []
        records_by_name = {}
        for slot, name in enumerate(ordered_names):
            record = self._reference_record(registry_records[name], registry_slot=slot)
            if record["name"] != name:
                raise ScenarioContractError(
                    "adapter registry key does not match reference name"
                )
            registered.append(record)
            records_by_name[name] = record

        active_name = self._focus_name
        if active_name is None and len(ordered_names) == 1:
            active_name = ordered_names[0]
        active_record = records_by_name.get(active_name)
        active = (
            None
            if active_record is None
            else {
                "name": active_record["name"],
                "id": active_record["id"],
                "version": active_record["version"],
            }
        )

        pending = getattr(manager, self.pending_attribute)
        staged = None if pending is None else self._reference_record(pending)
        quarantined = self._state_mapping(
            getattr(manager, self.quarantined_attribute),
            self.quarantined_attribute,
        )
        tombstoned = self._state_mapping(
            getattr(manager, self.tombstoned_attribute),
            self.tombstoned_attribute,
        )
        tombstone_records = []
        for name in sorted(tombstoned):
            record = self._reference_record(tombstoned[name])
            if record["name"] != name:
                raise ScenarioContractError(
                    "adapter tombstone key does not match reference name"
                )
            tombstone_records.append(record)
        cache = self._state_mapping(
            getattr(manager, self.cache_attribute), self.cache_attribute
        )
        cache_identity = {
            name: self._reference_record(cache[name])["id"] for name in sorted(cache)
        }
        return {
            "mode": self.mode,
            "registered": registered,
            "active": active,
            "staged": staged,
            "registry_occupancy": len(registered),
            "quarantined": sorted(quarantined),
            "tombstoned": tombstone_records,
            "cache_identity": cache_identity,
        }


class NativeLoRAControl(_NativeAdapterControl):
    mode: AdapterMode = "native_lora"
    load_format = "lora_adapter"
    path_method = "load_lora_adapter"
    tensors_method = "load_lora_adapter_from_tensors"
    distributed_method = "load_lora_adapter_from_distributed"
    unload_method = "unload_lora_adapter"
    registry_attribute = "lora_registry"
    pending_attribute = "pending_lora_stage"
    quarantined_attribute = "failed_lora_activations"
    tombstoned_attribute = "failed_lora_unloads"
    cache_attribute = "lora_ref_cache"
    ref_name_attribute = "lora_name"
    ref_id_attribute = "lora_id"
    ref_version_attribute = "version"

    def load_tensors(
        self,
        name: str,
        tensors: Mapping[str, object],
        config: Mapping[str, object],
        *,
        upsert: bool = False,
    ) -> ControlResult:
        if not upsert:
            return super().load_tensors(name, tensors, config)

        from sglang.srt.managers.io_struct import (
            LoadLoRAAdapterFromTensorsReqInput,
        )

        serialized = self.engine._serialize_tensors_per_rank(dict(tensors), None)
        request = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=name,
            config_dict=dict(config),
            serialized_named_tensors=serialized,
            upsert=True,
        )
        return self._record_focus(
            name,
            self._manager_call("load_lora_adapter_from_tensors", request),
        )

    def load_distributed(
        self,
        name: str,
        payload: DistributedPayload,
        group_name: str,
        *,
        upsert: bool = False,
    ) -> ControlResult:
        if not upsert:
            return super().load_distributed(name, payload, group_name)

        from sglang.srt.managers.io_struct import (
            LoadLoRAAdapterFromDistributedReqInput,
        )

        request = LoadLoRAAdapterFromDistributedReqInput(
            lora_name=name,
            config_dict=dict(payload.config),
            names=list(payload.names),
            dtypes=list(payload.dtypes),
            shapes=[list(shape) for shape in payload.shapes],
            group_name=group_name,
            upsert=True,
        )
        return self._record_focus(
            name,
            self._manager_call("load_lora_adapter_from_distributed", request),
        )


class NativeOFTControl(_NativeAdapterControl):
    mode: AdapterMode = "native_oft"
    load_format = "oft_adapter"
    path_method = "load_oft_adapter"
    tensors_method = "load_oft_adapter_from_tensors"
    distributed_method = "load_oft_adapter_from_distributed"
    unload_method = "unload_oft_adapter"
    registry_attribute = "peft_registry"
    pending_attribute = "pending_oft_stage"
    quarantined_attribute = "failed_oft_activations"
    tombstoned_attribute = "failed_oft_unloads"
    cache_attribute = "peft_ref_cache"
    ref_name_attribute = "adapter_name"
    ref_id_attribute = "adapter_id"
    ref_version_attribute = "adapter_version"


def make_adapter_control(mode: str, engine: object) -> AdapterControl:
    if mode == "base":
        return BaseControl(engine)
    if mode == "native_lora":
        return NativeLoRAControl(engine)
    if mode == "native_oft":
        return NativeOFTControl(engine)
    raise ScenarioContractError(f"unknown adapter control mode: {mode}")


def _startup_adapter_flag(mode: str) -> str:
    if mode == "native_lora":
        return "--lora-paths"
    if mode == "native_oft":
        return "--peft-paths"
    raise ScenarioContractError(f"mode does not support startup adapters: {mode}")


def server_other_args(spec: ServerSpec) -> tuple[str, ...]:
    """Build the deterministic arguments consumed by SGLang's launch helper."""

    arguments: list[str] = [
        "--base-gpu-id",
        str(spec.base_gpu_id),
        "--tp-size",
        str(spec.tp_size),
    ]
    if spec.ep_size > 1:
        arguments.extend(("--ep-size", str(spec.ep_size)))
    if spec.quantization is not None:
        arguments.extend(("--quantization", spec.quantization))
    if spec.moe_runner is not None:
        arguments.extend(("--moe-runner-backend", spec.moe_runner))
    if not spec.cuda_graph:
        arguments.append("--disable-cuda-graph")
    arguments.extend(mode_server_args(spec.revision_kind, spec.mode))
    if spec.max_lora_rank is not None:
        arguments.extend(("--max-lora-rank", str(spec.max_lora_rank)))
    if spec.lora_target_modules:
        arguments.append("--lora-target-modules")
        arguments.extend(spec.lora_target_modules)
    if spec.max_oft_block_size is not None:
        arguments.extend(("--max-oft-block-size", str(spec.max_oft_block_size)))
    if spec.peft_target_modules:
        arguments.append("--peft-target-modules")
        arguments.extend(spec.peft_target_modules)
    if spec.startup_adapters:
        arguments.append(_startup_adapter_flag(spec.mode))
        arguments.extend(f"{name}={path}" for name, path in spec.startup_adapters)
    arguments.extend(
        (
            "--mem-fraction-static",
            str(spec.mem_fraction_static),
            "--max-total-tokens",
            str(spec.max_total_tokens),
            "--log-level",
            "error",
        )
    )
    return tuple(arguments)


def engine_kwargs(spec: ServerSpec) -> dict[str, object]:
    """Translate a server spec into the offline Engine constructor contract."""

    arguments: dict[str, object] = {
        "model_path": spec.model_path,
        "base_gpu_id": spec.base_gpu_id,
        "tp_size": spec.tp_size,
        "ep_size": spec.ep_size,
        "disable_cuda_graph": not spec.cuda_graph,
        "mem_fraction_static": spec.mem_fraction_static,
        "max_total_tokens": spec.max_total_tokens,
        "log_level": "error",
    }
    if spec.quantization is not None:
        arguments["quantization"] = spec.quantization
    if spec.moe_runner is not None:
        arguments["moe_runner_backend"] = spec.moe_runner
    if spec.max_lora_rank is not None:
        arguments["max_lora_rank"] = spec.max_lora_rank
    if spec.lora_target_modules:
        arguments["lora_target_modules"] = list(spec.lora_target_modules)
    if spec.max_oft_block_size is not None:
        arguments["max_oft_block_size"] = spec.max_oft_block_size
    if spec.peft_target_modules:
        arguments["peft_target_modules"] = list(spec.peft_target_modules)

    if spec.mode == "native_oft":
        arguments.update(peft_method="oft", oft_type="oft")
    elif spec.mode == "native_lora":
        arguments.update(enable_lora=True, enable_lora_staging=True)

    if spec.startup_adapters:
        values = [f"{name}={path}" for name, path in spec.startup_adapters]
        key = "lora_paths" if spec.mode == "native_lora" else "peft_paths"
        arguments[key] = values
    return arguments


def launch_server(
    spec: ServerSpec,
    *,
    timeout: float,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
):
    """Launch one SGLang server using the repository's readiness-checked helper."""

    from sglang.test.test_utils import popen_launch_server

    streams = None if stdout is None and stderr is None else (stdout, stderr)
    return popen_launch_server(
        spec.model_path,
        spec.base_url,
        timeout=timeout,
        other_args=list(server_other_args(spec)),
        return_stdout_stderr=streams,
        device="cuda",
    )


def launch_engine(spec: ServerSpec):
    """Launch the offline Engine for native adapter control."""

    from sglang.srt.entrypoints.engine import Engine

    return Engine(**engine_kwargs(spec))


def stop_engine(engine: object) -> None:
    """Shut down an offline Engine and all scheduler subprocesses it owns."""

    engine.shutdown()


def stop_server(process: object) -> None:
    """Terminate a server and its worker tree with the established cleanup path."""

    from sglang.test.test_utils import terminate_and_kill_process_tree

    terminate_and_kill_process_tree(process)

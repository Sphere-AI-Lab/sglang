"""Server argument contracts for frozen source and candidate revisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Callable, Literal

from .scenarios import ScenarioContractError


_SOURCE_MODE_ARGS = {
    "base": (),
    "legacy_oft": ("--peft-method", "oft", "--oft-impl", "peft"),
    "canonical_oft": ("--peft-method", "oft", "--oft-impl", "sibling"),
    "legacy_lora": ("--peft-method", "lora"),
    "native_lora": ("--enable-lora", "--enable-lora-staging"),
}

_CANDIDATE_MODE_ARGS = {
    "base": (),
    "canonical_oft": ("--peft-method", "oft"),
    "native_lora": ("--enable-lora", "--enable-lora-staging"),
}

_OFT_MODES = {"legacy_oft", "canonical_oft"}
_INTERNAL_OFT_CONTROL_FORMAT = "adapter_equivalence_oft_control"


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
    max_oft_block_size: int | None = None
    peft_target_modules: tuple[str, ...] = ()
    base_gpu_id: int = 1
    mem_fraction_static: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "startup_adapters", tuple(self.startup_adapters))
        object.__setattr__(
            self, "peft_target_modules", tuple(self.peft_target_modules)
        )
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
        if (
            type(self.mem_fraction_static) is not float
            or not 0.0 < self.mem_fraction_static < 1.0
        ):
            raise ScenarioContractError(
                "mem_fraction_static must be a float between zero and one"
            )
        names = [name for name, _ in self.startup_adapters]
        if any(not name or not path for name, path in self.startup_adapters):
            raise ScenarioContractError("startup adapter names and paths must be non-empty")
        if len(names) != len(set(names)):
            raise ScenarioContractError("startup adapter names must be unique")
        if self.mode == "base" and self.startup_adapters:
            raise ScenarioContractError("base mode cannot preload adapters")
        mode_server_args(self.revision_kind, self.mode)
        if self.max_oft_block_size is not None and (
            type(self.max_oft_block_size) is not int
            or self.max_oft_block_size <= 0
        ):
            raise ScenarioContractError(
                "max_oft_block_size must be a positive integer when present"
            )
        if any(
            type(module) is not str or not module
            for module in self.peft_target_modules
        ):
            raise ScenarioContractError(
                "peft_target_modules must contain non-empty strings"
            )
        if len(self.peft_target_modules) != len(set(self.peft_target_modules)):
            raise ScenarioContractError("peft_target_modules must be unique")
        if self.mode not in _OFT_MODES and (
            self.max_oft_block_size is not None or self.peft_target_modules
        ):
            raise ScenarioContractError(
                "OFT shape fields require an OFT mode"
            )
        if (
            self.mode in _OFT_MODES
            and not self.startup_adapters
            and (
                self.max_oft_block_size is None
                or not self.peft_target_modules
            )
        ):
            raise ScenarioContractError(
                "dynamic OFT requires max_oft_block_size and peft_target_modules"
            )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def mode_server_args(revision_kind: str, mode: str) -> tuple[str, ...]:
    """Return the exact adapter-selection arguments for a frozen revision."""

    if revision_kind == "source":
        arguments = _SOURCE_MODE_ARGS
    elif revision_kind == "candidate":
        if mode in {"legacy_oft", "legacy_lora"}:
            raise ScenarioContractError(f"{mode} is a source-only oracle mode")
        arguments = _CANDIDATE_MODE_ARGS
    else:
        raise ScenarioContractError(f"unknown revision kind: {revision_kind}")

    try:
        return arguments[mode]
    except KeyError as error:
        raise ScenarioContractError(
            f"unknown {revision_kind} adapter mode: {mode}"
        ) from error


def _startup_adapter_flag(mode: str) -> str:
    if mode == "native_lora":
        return "--lora-paths"
    if mode in {"legacy_oft", "canonical_oft", "legacy_lora"}:
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
    if spec.max_oft_block_size is not None:
        arguments.extend(
            ("--max-oft-block-size", str(spec.max_oft_block_size))
        )
    if spec.peft_target_modules:
        arguments.append("--peft-target-modules")
        arguments.extend(spec.peft_target_modules)
    if spec.startup_adapters:
        arguments.append(_startup_adapter_flag(spec.mode))
        arguments.extend(
            f"{name}={path}" for name, path in spec.startup_adapters
        )
    arguments.extend(
        (
            "--mem-fraction-static",
            str(spec.mem_fraction_static),
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
        "log_level": "error",
    }
    if spec.quantization is not None:
        arguments["quantization"] = spec.quantization
    if spec.moe_runner is not None:
        arguments["moe_runner_backend"] = spec.moe_runner
    if spec.max_oft_block_size is not None:
        arguments["max_oft_block_size"] = spec.max_oft_block_size
    if spec.peft_target_modules:
        arguments["peft_target_modules"] = spec.peft_target_modules

    if spec.mode == "legacy_oft":
        arguments.update(peft_method="oft", oft_impl="peft")
    elif spec.mode == "canonical_oft":
        arguments["peft_method"] = "oft"
        if spec.revision_kind == "source":
            arguments["oft_impl"] = "sibling"
    elif spec.mode == "legacy_lora":
        arguments["peft_method"] = "lora"
    elif spec.mode == "native_lora":
        arguments.update(enable_lora=True, enable_lora_staging=True)

    if spec.startup_adapters:
        values = tuple(
            f"{name}={path}" for name, path in spec.startup_adapters
        )
        key = "lora_paths" if spec.mode == "native_lora" else "peft_paths"
        arguments[key] = values
    return arguments


@dataclass(frozen=True)
class OFTControlResult:
    """Normalized result from the harness-only OFT control channel."""

    success: bool
    error_message: str = ""


def resolve_oft_update_request_type(revision_kind: str) -> Callable[..., object]:
    """Resolve the existing scheduler request used as the control transport."""

    if revision_kind not in {"source", "candidate"}:
        raise ScenarioContractError(f"unknown revision kind: {revision_kind}")
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

    return UpdateWeightsFromTensorReqInput


def _normalize_oft_control_result(result: object) -> OFTControlResult:
    if isinstance(result, tuple) and len(result) >= 2:
        success, message = result[:2]
        return OFTControlResult(
            success=bool(success),
            error_message="" if success else str(message),
        )
    success = getattr(result, "success", False) is True
    message = getattr(result, "error_message", None)
    return OFTControlResult(
        success=success,
        error_message="" if success else str(message or "unknown failure"),
    )


def _dispatch_internal_oft_control(
    model_runner: object,
    *,
    adapter_config: dict[str, object] | None,
    adapter_id: str | None,
    ref_type: Callable[..., object],
) -> tuple[bool, str]:
    """Execute one control message against the worker's real OFT manager."""

    if getattr(model_runner.server_args, "peft_method", None) != "oft":
        return False, "OFT is not enabled"
    if adapter_config is None or adapter_id is None:
        return False, "internal OFT control requires adapter metadata and adapter_id"

    operation = adapter_config.get("operation")
    if operation not in {"load", "unload"}:
        return False, f"unknown internal OFT operation: {operation}"
    try:
        ref = ref_type(
            adapter_id=adapter_id,
            adapter_name=adapter_config["adapter_name"],
            adapter_path=adapter_config["adapter_path"],
            pinned=adapter_config["pinned"],
        )
        manager = model_runner.oft_manager
        method = (
            manager.load_oft_adapter
            if operation == "load"
            else manager.unload_oft_adapter
        )
        result = method(ref)
    except Exception as exc:
        return False, str(exc)
    normalized = _normalize_oft_control_result(result)
    return normalized.success, normalized.error_message


def _resolve_worker_oft_ref_type(
    model_runner: object,
    revision_kind: str,
) -> Callable[..., object]:
    oft_impl = getattr(model_runner.server_args, "oft_impl", None)
    if revision_kind == "source" and oft_impl not in {"sibling", "staged"}:
        from sglang.srt.peft.oft.oft_registry import OFTRef
    else:
        from sglang.srt.oft.oft_registry import OFTRef
    return OFTRef


def install_internal_oft_control_bridge(revision_kind: str) -> None:
    """Install the harness-only worker dispatch before Engine subprocess spawn."""

    if revision_kind == "source":
        from sglang.srt.peft import integration
    elif revision_kind == "candidate":
        from sglang.srt.oft import integration
    else:
        raise ScenarioContractError(f"unknown revision kind: {revision_kind}")

    marker = "_adapter_equivalence_oft_control_bridge"
    installed_revision = getattr(integration, marker, None)
    if installed_revision == revision_kind:
        return
    if installed_revision is not None:
        raise ScenarioContractError(
            "internal OFT bridge cannot switch revisions in one process"
        )

    original = integration.maybe_load_adapter_format

    def maybe_load_adapter_format(
        model_runner,
        load_format,
        tensors,
        adapter_config,
        adapter_name,
        adapter_id,
        **kwargs,
    ):
        if load_format != _INTERNAL_OFT_CONTROL_FORMAT:
            return original(
                model_runner,
                load_format,
                tensors,
                adapter_config,
                adapter_name,
                adapter_id,
                **kwargs,
            )
        return _dispatch_internal_oft_control(
            model_runner,
            adapter_config=adapter_config,
            adapter_id=adapter_id,
            ref_type=_resolve_worker_oft_ref_type(model_runner, revision_kind),
        )

    integration.maybe_load_adapter_format = maybe_load_adapter_format
    setattr(integration, marker, revision_kind)


class InternalOFTControl:
    """Drive real worker OFT managers through an existing scheduler channel."""

    def __init__(
        self,
        engine: object,
        *,
        revision_kind: str,
        update_request_type: Callable[..., object] | None = None,
    ) -> None:
        if revision_kind not in {"source", "candidate"}:
            raise ScenarioContractError(f"unknown revision kind: {revision_kind}")
        self.engine = engine
        self.update_request_type = (
            update_request_type
            or resolve_oft_update_request_type(revision_kind)
        )

    def _run(self, operation: str, awaitable: object) -> object:
        result = _normalize_oft_control_result(
            self.engine.loop.run_until_complete(awaitable)
        )
        if not result.success:
            raise ScenarioContractError(
                f"internal OFT {operation} failed: {result.error_message}"
            )
        return result

    def _request(
        self,
        *,
        operation: str,
        adapter_name: str,
        adapter_path: str,
        pinned: bool,
        adapter_id: str | None,
    ) -> object:
        return self.update_request_type(
            serialized_named_tensors=self.engine._serialize_tensors_per_rank(
                [], _INTERNAL_OFT_CONTROL_FORMAT
            ),
            load_format=_INTERNAL_OFT_CONTROL_FORMAT,
            flush_cache=True,
            adapter_config={
                "operation": operation,
                "adapter_name": adapter_name,
                "adapter_path": adapter_path,
                "pinned": pinned,
            },
            adapter_name=adapter_name if operation == "load" else None,
            adapter_id=adapter_id,
        )

    async def _load(
        self,
        adapter_name: str,
        adapter_path: str,
        pinned: bool,
    ) -> object:
        manager = self.engine.tokenizer_manager
        async with manager.peft_update_lock:
            request = self._request(
                operation="load",
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                pinned=pinned,
                adapter_id=None,
            )
            try:
                result = await manager.update_weights_from_tensor(request, None)
            except Exception:
                await self._rollback_load_registration(adapter_name)
                raise
            if not _normalize_oft_control_result(result).success:
                await self._rollback_load_registration(adapter_name)
            return result

    async def _rollback_load_registration(self, adapter_name: str) -> None:
        manager = self.engine.tokenizer_manager
        if adapter_name not in manager.peft_ref_cache:
            return
        manager.peft_ref_cache.pop(adapter_name, None)
        await manager.peft_registry.unregister(adapter_name)

    async def _unload(self, adapter_name: str) -> object:
        manager = self.engine.tokenizer_manager
        async with manager.peft_update_lock:
            ref = manager.peft_ref_cache.get(adapter_name)
            if ref is None:
                raise ScenarioContractError(
                    f"internal OFT unload failed: unknown adapter {adapter_name}"
                )
            adapter_id = await manager.peft_registry.unregister(adapter_name)
            await manager.peft_registry.wait_for_unload(adapter_id)
            request = self._request(
                operation="unload",
                adapter_name=adapter_name,
                adapter_path=ref.adapter_path,
                pinned=bool(ref.pinned),
                adapter_id=adapter_id,
            )
            try:
                result = await manager.update_weights_from_tensor(request, None)
            except Exception:
                await manager.peft_registry.register(ref)
                raise
            if _normalize_oft_control_result(result).success:
                manager.peft_ref_cache.pop(adapter_name, None)
            else:
                await manager.peft_registry.register(ref)
            return result

    def load(
        self,
        adapter_name: str,
        adapter_path: str,
        *,
        pinned: bool = False,
    ) -> object:
        return self._run("load", self._load(adapter_name, adapter_path, pinned))

    def unload(self, adapter_name: str) -> object:
        return self._run("unload", self._unload(adapter_name))


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
    """Launch the offline Engine so test-only internal controls stay in-process."""

    from sglang.srt.entrypoints.engine import Engine

    return Engine(**engine_kwargs(spec))


def stop_engine(engine: object) -> None:
    """Shut down an offline Engine and all scheduler subprocesses it owns."""

    engine.shutdown()


def stop_server(process: object) -> None:
    """Terminate a server and its worker tree with the established cleanup path."""

    from sglang.test.test_utils import terminate_and_kill_process_tree

    terminate_and_kill_process_tree(process)

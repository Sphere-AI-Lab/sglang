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
    base_gpu_id: int = 1
    mem_fraction_static: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "startup_adapters", tuple(self.startup_adapters))
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
class OFTRequestTypes:
    """Revision-specific OFT request constructors used by the internal shim."""

    load: Callable[..., object]
    unload: Callable[..., object]


def resolve_oft_request_types(revision_kind: str) -> OFTRequestTypes:
    """Resolve request types without importing removed source modules eagerly."""

    if revision_kind == "source":
        from sglang.srt.peft.io_types import (
            LoadOFTAdapterReqInput,
            UnloadOFTAdapterReqInput,
        )
    elif revision_kind == "candidate":
        from sglang.srt.oft.io_types import (
            LoadOFTAdapterReqInput,
            UnloadOFTAdapterReqInput,
        )
    else:
        raise ScenarioContractError(f"unknown revision kind: {revision_kind}")
    return OFTRequestTypes(LoadOFTAdapterReqInput, UnloadOFTAdapterReqInput)


class InternalOFTControl:
    """Drive the unchanged source's real OFT manager when HTTP routes are absent."""

    def __init__(
        self,
        engine: object,
        *,
        revision_kind: str,
        request_types: OFTRequestTypes | None = None,
    ) -> None:
        if revision_kind not in {"source", "candidate"}:
            raise ScenarioContractError(f"unknown revision kind: {revision_kind}")
        self.engine = engine
        self.request_types = request_types or resolve_oft_request_types(
            revision_kind
        )

    def _run(self, operation: str, awaitable: object) -> object:
        result = self.engine.loop.run_until_complete(awaitable)
        if getattr(result, "success", False) is not True:
            message = getattr(result, "error_message", None) or "unknown failure"
            raise ScenarioContractError(f"internal OFT {operation} failed: {message}")
        return result

    def load(
        self,
        adapter_name: str,
        adapter_path: str,
        *,
        pinned: bool = False,
    ) -> object:
        request = self.request_types.load(
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            pinned=pinned,
        )
        return self._run(
            "load",
            self.engine.tokenizer_manager.load_oft_adapter(request, None),
        )

    def unload(self, adapter_name: str) -> object:
        request = self.request_types.unload(adapter_name=adapter_name)
        return self._run(
            "unload",
            self.engine.tokenizer_manager.unload_oft_adapter(request, None),
        )


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

#!/usr/bin/env python3
"""Parameterized one-shard adapter-lifecycle smoke for the adapter_equivalence harness.

This script drives ONE cell of the adapter-equivalence matrix (one mode, one prompt,
one transition subset) against an in-process SGLang ``Engine`` and emits one flushed
JSON line per step so a ``tail -f`` of the log is machine-parseable.

It deliberately does NOT modify the harness under
``test/manual/adapter_equivalence/``.  Two known harness defects are worked around
here instead:

  Defect 1 -- ``server.engine_kwargs(spec)`` emits startup adapter paths (and
    ``peft_target_modules``) as ``tuple``, but SGLang's server-args validators
    require ``list``/``dict``.  So we call ``engine_kwargs`` ourselves, coerce those
    values to lists, and construct ``Engine(**kwargs)`` directly instead of using
    ``server.launch_engine``.

  Defect 2 -- ``fixtures.build_*_fixture`` refuses to write into a destination that
    already exists.  So every run gets its own unique fixture root
    (``<run_dir>/fixtures/<utc-timestamp>-<pid>-<rand>``) and the script fails loudly
    with a ``setup`` phase line if that root somehow already exists.

Every engine interaction is bounded.  Coroutine work is bounded with
``asyncio.wait_for``; on top of that, *all* engine calls are executed on a single
dedicated worker thread and joined with ``Future.result(timeout=...)``.  If a call
wedges below the asyncio layer (a stuck scheduler subprocess, a hung NCCL collective)
the main thread still regains control, emits the remaining ``summary``/``verdict``
lines, and hard-exits with ``os._exit`` so the wedged thread cannot keep the process
alive.  The script therefore always terminates and always reports.

Exit codes
----------
  0  every selected transition passed
  1  at least one transition failed
  2  an operation timed out (process was hard-exited after reporting)
  3  setup failed before any transition could run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

DEFAULT_HARNESS_DIR = (
    "/data/home/zeju/miles-orbit-dev/impossible/sglang/.worktrees/"
    "retire-legacy-peft/test/manual/adapter_equivalence"
)

# ---------------------------------------------------------------------------
# Transition selection
# ---------------------------------------------------------------------------

# Mirrors scenarios._ADAPTER_LIFECYCLE_TRANSITIONS exactly, in contract order.
# It is re-declared (not imported) only so that --transitions can be resolved and
# printed by --dry-run without importing sglang; run_selection() cross-checks this
# tuple against the harness's own tuple and aborts on any drift.
FULL_TRANSITIONS = (
    "base.initial",
    "startup.adapter",
    "dynamic.load",
    "dynamic.infer",
    "dynamic.unload",
    "dynamic.post-unload-base",
    "switch.a",
    "switch.b",
    "switch.a-again",
    "mixed.base-a-b",
    "concurrent.stream",
    "concurrent.non-stream",
    "prefill.short",
    "prefill.long",
    "decode.short",
    "decode.long",
    "stage.v1",
    "activate.v1",
    "stage.v2",
    "activate.v2",
    "reject.duplicate",
    "reject.stale",
    "reject.invalid-id",
    "reject.invalid-config",
    "rollback.previous",
    "restart.same-manifest",
)

# --transitions short: the minimal round trip named in the task.
SHORT_TRANSITIONS = (
    "dynamic.load",
    "dynamic.infer",
    "dynamic.unload",
    "dynamic.post-unload-base",
)

# --transitions expert: the subset that actually touches MoE *expert* bindings.
#
# Why these and not the others.  On an MoE cell the adapter's gate_proj/up_proj/
# down_proj tensors live under `model.layers.N.mlp.experts.E.*`, so an adapter
# load has to stream a separate weight slice per expert and register a per-expert
# binding; `unload_oft_adapter` documents the inverse (resetting the rotation slot
# to identity and *clearing streamed MoE expert bindings*).  A transition only
# stresses that machinery if it creates, re-points, indexes, or tears down those
# per-expert bindings:
#
#   base.initial              baseline with zero bindings installed; every other
#                             row's "differs from base" answer is meaningless
#                             without it, and it is the reference for the
#                             post-unload teardown check.
#   dynamic.load              installs the per-expert bindings in the first place.
#   dynamic.infer             routes tokens through the freshly installed bindings.
#   switch.a / switch.b /     re-points the same expert slots at a different
#   switch.a-again            adapter's weights, twice, then back -- the case
#                             where a stale binding table survives a switch.
#   mixed.base-a-b            one batch containing base, policy-a and policy-b
#                             requests, so the binding lookup must be per-request
#                             rather than a single global table.
#   concurrent.stream /       several requests in flight over the same binding
#   concurrent.non-stream     tables at once, on both the streaming and the
#                             non-streaming completion path.
#   prefill.long              876 prompt tokens instead of 12: enough tokens to
#                             route across the whole expert set, so bindings for
#                             rarely-selected experts get exercised. A 12-token
#                             prompt may only ever reach a handful of experts.
#   decode.long               128 decode steps, i.e. 128 successive routing
#                             decisions, versus a single prefill pass.
#   dynamic.unload            tears the per-expert bindings back down.
#   dynamic.post-unload-base  proves the teardown was complete: the harness
#                             oracle requires this to be byte-identical to
#                             base.initial, so any expert left bound shows up here.
#
# Excluded on purpose: startup.adapter (launch-time path, no rebinding),
# prefill.short / decode.short (strictly weaker than their long counterparts),
# stage.* / activate.* / rollback.* / reject.* (native-LoRA versioning and
# error-path semantics, not expert binding), restart.same-manifest (process
# lifecycle, and it costs a second full model load).
EXPERT_TRANSITIONS = (
    "base.initial",
    "dynamic.load",
    "dynamic.infer",
    "dynamic.unload",
    "dynamic.post-unload-base",
    "switch.a",
    "switch.b",
    "switch.a-again",
    "mixed.base-a-b",
    "concurrent.stream",
    "concurrent.non-stream",
    "prefill.long",
    "decode.long",
)

TRANSITION_SETS = {
    "full": FULL_TRANSITIONS,
    "expert": EXPERT_TRANSITIONS,
    "short": SHORT_TRANSITIONS,
}

# Transitions that cannot be interpreted without an earlier transition also
# running.  Auto-added to any selection, then re-sorted into contract order.
TRANSITION_REQUIRES = {
    "dynamic.post-unload-base": ("base.initial",),
    "activate.v1": ("stage.v1",),
    "activate.v2": ("stage.v2",),
    "reject.stale": ("stage.v1", "activate.v1", "stage.v2", "activate.v2"),
    "rollback.previous": ("base.initial",),
    "restart.same-manifest": ("base.initial",),
}

# Transitions whose contract expectation is a *failure*: they pass when the
# engine correctly refuses them, and fail when the call succeeds.
EXPECTED_FAILURE_PREFIX = "reject."

# Transitions this script satisfies with a surrogate rather than the contract's
# literal mechanism.  Flagged as "surrogate": true on every emitted step line so a
# log reader is never misled about what was actually exercised.  See --help epilog.
SURROGATE_TRANSITIONS = {
    "startup.adapter": (
        "generates against a launch-time preloaded adapter registered under its "
        "own name (policy-s), because the contract's policy-a is also the "
        "dynamically loaded adapter and cannot be both"
    ),
    "stage.v1": (
        "no Engine-level stage API exists; staging is only reachable through "
        "update_adapter_from_distributed, which needs an external rank on an "
        "NCCL weight-sync group. Surrogate: load the version under a versioned "
        "adapter name (policy-a@v1)"
    ),
    "stage.v2": (
        "same as stage.v1; surrogate loads a second, distinct fixture under "
        "policy-a@v2"
    ),
    "activate.v1": (
        "no Engine-level activate API; surrogate generates against policy-a@v1 "
        "and marks it the active version"
    ),
    "activate.v2": (
        "no Engine-level activate API; surrogate generates against policy-a@v2, "
        "marks it active, and unloads policy-a@v1 so that it becomes genuinely "
        "stale for reject.stale"
    ),
    "reject.stale": (
        "surrogate: generate against policy-a@v1 after activate.v2 superseded "
        "and unloaded it; must be refused"
    ),
}

DEFAULT_MIXED_ADAPTER_PATTERN = (None, "policy-a", "policy-b", None,
                                 "policy-a", "policy-b", None, "policy-a")


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


class Emitter:
    """One JSON object per line, to stdout and to a log file, flushed each time."""

    def __init__(self, log_path: Path | None) -> None:
        self._stream = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = log_path.open("a", encoding="utf-8")
        self._t0 = time.time()

    def emit(self, phase: str, **fields: object) -> dict:
        record: dict[str, object] = {"phase": phase, "t": round(time.time() - self._t0, 3)}
        record.update(fields)
        # sort_keys=True / default=str matches the existing smokes' emit().
        line = json.dumps(record, default=str, sort_keys=True)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        return record

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass


class OperationTimeout(RuntimeError):
    """Raised when a bounded engine operation did not return in time."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def digest_ids(output_ids) -> str:
    payload = ",".join(str(int(token)) for token in (output_ids or ()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_prompts(path: Path) -> tuple[dict, dict]:
    """Read prompts.jsonl -> ({prompt_id: record}, {batch_id: record})."""
    prompts: dict[str, dict] = {}
    batches: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            kind = record.get("kind")
            if kind == "prompt":
                prompts[record["id"]] = record
            elif kind == "batch":
                batches[record["id"]] = record
    if not prompts:
        raise RuntimeError(f"no prompt records found in {path}")
    return prompts, batches


def resolve_selection(name: str) -> tuple[str, ...]:
    chosen = set(TRANSITION_SETS[name])
    changed = True
    while changed:
        changed = False
        for transition in sorted(chosen):
            for required in TRANSITION_REQUIRES.get(transition, ()):
                if required not in chosen:
                    chosen.add(required)
                    changed = True
    return tuple(t for t in FULL_TRANSITIONS if t in chosen)


def parse_engine_kwarg(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--engine-kwarg must be KEY=JSON_VALUE, got {raw!r}"
        )
    key, _, value = raw.partition("=")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("--engine-kwarg key must be non-empty")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


# ---------------------------------------------------------------------------
# Model shape derivation (for fixture target_shapes)
# ---------------------------------------------------------------------------


def read_model_config(model_path: str) -> dict:
    """Return the HF config as a plain dict, local dir or hub id."""
    local = Path(model_path) / "config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    from transformers import AutoConfig  # imported lazily: needs the sglang env

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config.to_dict()


def detect_architecture(config: dict) -> str:
    for key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if config.get(key):
            return "moe"
    if config.get("moe_intermediate_size"):
        return "moe"
    return "dense"


def build_target_shapes(
    config: dict,
    architecture: str,
    *,
    layers: int,
    experts: int,
    suffixes: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    """Map fully qualified module names -> (input_features, output_features)."""
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or (hidden // heads))
    total_layers = int(config["num_hidden_layers"])
    layers = total_layers if layers <= 0 else min(layers, total_layers)

    q_out = heads * head_dim
    kv_out = kv_heads * head_dim

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
        ffn = int(config["intermediate_size"])
        experts = 0

    attn_shapes = {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
    }
    mlp_shapes = {
        "gate_proj": (hidden, ffn),
        "up_proj": (hidden, ffn),
        "down_proj": (ffn, hidden),
    }

    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(layers):
        for suffix in suffixes:
            if suffix in attn_shapes:
                shapes[f"model.layers.{layer}.self_attn.{suffix}"] = attn_shapes[suffix]
            elif suffix in mlp_shapes:
                if architecture == "moe":
                    for expert in range(experts):
                        name = f"model.layers.{layer}.mlp.experts.{expert}.{suffix}"
                        shapes[name] = mlp_shapes[suffix]
                else:
                    shapes[f"model.layers.{layer}.mlp.{suffix}"] = mlp_shapes[suffix]
    return shapes


def write_invalid_fixture(destination: Path) -> Path:
    """Hand-build a structurally broken adapter dir for reject.invalid-config.

    Built by hand rather than via build_*_fixture because the harness builders
    validate their inputs and would refuse to produce a broken artifact.
    """
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "adapter_config.json").write_text(
        json.dumps({"peft_type": "NOT_A_REAL_PEFT_TYPE", "r": -1}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "adapter_model.safetensors").write_bytes(b"not a safetensors file")
    return destination


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ShardRunner:
    def __init__(self, args, emit: Emitter) -> None:
        self.args = args
        # Bind the METHOD, not the Emitter instance: every call site below is
        # self.emit("phase", **fields), and Emitter defines .emit() with no
        # __call__, so storing the object raised
        # TypeError: 'Emitter' object is not callable on the first use.
        self.emitter = emit
        self.emit = emit.emit
        self.engine = None
        self.engine_kwargs: dict[str, object] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
        self.prompts: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}
        self.fixtures: dict[str, str] = {}
        self.loaded: set[str] = set()
        self.active_version: str | None = None
        self.base_output_ids: list[int] | None = None
        self.observations: dict[str, object] = {}
        self.results: list[dict] = []
        self.timed_out = False
        self.is_oft = args.mode == "canonical_oft"
        self.select_kwarg = "adapter_path" if self.is_oft else "lora_path"
        self.startup_adapter_name = "policy-s"

    # -- bounded execution -------------------------------------------------

    def _submit(self, fn, timeout: float, label: str):
        """Run fn on the single engine thread; never block past `timeout`."""
        future = self.pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as error:
            self.timed_out = True
            raise OperationTimeout(
                f"{label} did not return within {timeout}s"
            ) from error

    def call_sync(self, label: str, fn, timeout: float):
        return self._submit(fn, timeout, label)

    def call_async(self, label: str, coro_factory, timeout: float):
        """Drive a coroutine on the engine's own loop, bounded twice.

        Inner bound: asyncio.wait_for(timeout) cancels the coroutine cleanly.
        Outer bound: Future.result(timeout + slack) reclaims the main thread even
        if the loop itself is wedged and wait_for never fires.
        """

        def runner():
            return self.engine.loop.run_until_complete(
                asyncio.wait_for(coro_factory(), timeout=timeout)
            )

        return self._submit(runner, timeout + self.args.timeout_slack, label)

    # -- setup -------------------------------------------------------------

    def prepare_fixtures(self, harness) -> None:
        root = Path(self.args.run_dir) / "fixtures" / self.args.run_id
        if root.exists():
            raise RuntimeError(
                f"fixture root already exists and fixtures are immutable: {root}"
            )
        root.mkdir(parents=True, exist_ok=False)

        config = read_model_config(self.args.model_path)
        architecture = (
            self.args.architecture
            if self.args.architecture != "auto"
            else detect_architecture(config)
        )
        self.architecture = architecture

        if self.is_oft:
            builder = harness.build_oft_fixture
            # fixtures._OFT_REQUIRED_SUFFIXES.  Canonical OFT rotates the INPUT
            # side of a module and SGLang's runtime exposes only the FUSED
            # qkv_proj, whose R buffer is sized 3 * num_blocks;
            # normalize_merged_oft_weights() skips a fused group unless every
            # sibling leaf is present.  A q_proj-only adapter therefore loads
            # through the streamed path but raises on the disk path, and
            # contributes no attention rotation at all -- so the whole q/k/v
            # group is required, not merely allowed.
            suffixes = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")
        else:
            builder = harness.build_lora_fixture
            # fixtures._LORA_REQUIRED_SUFFIXES plus the allowed k_proj.
            suffixes = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")

        shapes = build_target_shapes(
            config,
            architecture,
            layers=self.args.fixture_layers,
            experts=self.args.fixture_experts,
            suffixes=suffixes,
        )
        self.emit(
            "fixtures.plan",
            architecture=architecture,
            targets=len(shapes),
            suffixes=list(suffixes),
            layers=self.args.fixture_layers,
            experts=self.args.fixture_experts,
            root=str(root),
        )

        seeds = {
            "policy-a": harness.ADAPTER_SEEDS[0],
            "policy-b": harness.ADAPTER_SEEDS[1],
            "policy-s": 4099,
            "policy-a@v2": 5077,
        }
        scale_override = self.args.oft_scale if self.is_oft else None
        fixtures_module = sys.modules[builder.__module__]
        original_values = fixtures_module._deterministic_values
        if scale_override is not None:
            def _forced_scale(seed, label, shape, scale, _o=original_values):
                return _o(seed, label, shape, scale_override)
            fixtures_module._deterministic_values = _forced_scale
            self.emit("fixtures.scale_override", oft_scale=scale_override,
                      harness_default=1e-3)
        try:
            for name, seed in seeds.items():
                dest = root / name.replace("@", "-at-")
                fixture = builder(
                    dest,
                    adapter_id=name,
                    architecture=architecture,
                    seed=seed,
                    target_shapes=shapes,
                )
                self.fixtures[name] = str(fixture.path)
                self.emit("fixtures.built", adapter=name,
                          path=str(fixture.path), seed=seed)
        finally:
            fixtures_module._deterministic_values = original_values

        # v1 deliberately reuses policy-a's bytes: staging v1 must be the same
        # weights that the plain policy-a name serves, so rollback.previous is
        # comparable against activate.v1.
        self.fixtures["policy-a@v1"] = self.fixtures["policy-a"]
        self.fixtures["invalid"] = str(write_invalid_fixture(root / "invalid"))
        self.emit("fixtures.built", adapter="invalid", path=self.fixtures["invalid"])

    def build_spec(self, harness):
        startup = ()
        if self.args.startup_adapter:
            startup = ((self.startup_adapter_name,
                        self.fixtures[self.startup_adapter_name]),)
        spec_kwargs = dict(
            revision_kind=self.args.revision_kind,
            model_path=self.args.model_path,
            mode=self.args.mode,
            port=self.args.port,
            tp_size=self.args.tp_size,
            ep_size=self.args.ep_size,
            cuda_graph=self.args.cuda_graph,
            base_gpu_id=self.args.base_gpu_id,
            mem_fraction_static=self.args.mem_fraction_static,
            startup_adapters=startup,
        )
        if self.args.quantization:
            spec_kwargs["quantization"] = self.args.quantization
        if self.args.moe_runner:
            spec_kwargs["moe_runner"] = self.args.moe_runner
        if self.is_oft:
            # ServerSpec rejects these fields outside OFT modes.
            spec_kwargs["max_oft_block_size"] = self.args.max_oft_block_size
            spec_kwargs["peft_target_modules"] = (
                "down_proj", "gate_proj", "o_proj", "q_proj", "up_proj",
            )
        return harness.ServerSpec(**spec_kwargs)

    def launch(self, harness) -> None:
        spec = self.build_spec(harness)
        kwargs = dict(harness.engine_kwargs(spec))

        # DEFECT 1 workaround: engine_kwargs hands back tuples where SGLang's
        # server-arg validators insist on list/dict.  Coerce, do not edit the
        # harness, and construct Engine ourselves instead of launch_engine.
        for key in ("lora_paths", "peft_paths", "peft_target_modules"):
            value = kwargs.get(key)
            if isinstance(value, tuple):
                kwargs[key] = list(value)
        for key, value in self.args.engine_kwarg:
            kwargs[key] = value

        self.engine_kwargs = kwargs
        self.emit("engine.launch", kwargs={k: v for k, v in kwargs.items()})

        from sglang.srt.entrypoints.engine import Engine

        started = time.time()
        self.engine = self.call_sync(
            "engine.launch", lambda: Engine(**kwargs), self.args.startup_timeout
        )
        if self.args.startup_adapter:
            self.loaded.add(self.startup_adapter_name)
        self.emit("engine.ready", seconds=round(time.time() - started, 2))

    def relaunch(self) -> None:
        from sglang.srt.entrypoints.engine import Engine

        engine, self.engine = self.engine, None
        self.call_sync("engine.shutdown", engine.shutdown, self.args.op_timeout)
        self.loaded.clear()
        self.active_version = None
        self.engine = self.call_sync(
            "engine.relaunch",
            lambda: Engine(**self.engine_kwargs),
            self.args.startup_timeout,
        )
        if self.args.startup_adapter:
            self.loaded.add(self.startup_adapter_name)

    def shutdown(self) -> None:
        if self.engine is None:
            return
        engine, self.engine = self.engine, None
        try:
            self.call_sync("engine.shutdown", engine.shutdown, self.args.op_timeout)
        except Exception:
            pass

    # -- adapter operations ------------------------------------------------

    def _load_call(self, name: str, path: str):
        engine = self.engine
        if self.is_oft:
            return lambda: engine.load_oft_adapter(name, path)
        return lambda: engine.load_lora_adapter(name, path)

    def _unload_call(self, name: str):
        engine = self.engine
        if self.is_oft:
            return lambda: engine.unload_oft_adapter(name)
        return lambda: engine.unload_lora_adapter(name)

    @staticmethod
    def _check_result(result, operation: str):
        """Some adapter APIs report failure by return value rather than raising."""
        success = getattr(result, "success", None)
        if success is False:
            message = getattr(result, "message", None) or getattr(
                result, "error_message", "unknown failure"
            )
            raise RuntimeError(f"{operation} reported failure: {message}")
        return result

    def load_adapter(self, name: str, path: str | None = None):
        path = path or self.fixtures[name]
        result = self.call_sync(
            f"load:{name}", self._load_call(name, path), self.args.op_timeout
        )
        self._check_result(result, f"load {name}")
        self.loaded.add(name)
        return result

    def unload_adapter(self, name: str):
        result = self.call_sync(
            f"unload:{name}", self._unload_call(name), self.args.op_timeout
        )
        self._check_result(result, f"unload {name}")
        self.loaded.discard(name)
        return result

    def ensure_loaded(self, name: str) -> bool:
        """Lazily (re)load an adapter a later transition needs.

        The frozen contract orders dynamic.unload BEFORE switch.a/switch.b, which
        both generate against policy-a.  Rather than silently generating against a
        missing adapter, we reload on demand and flag it in the step record as
        `reloaded: true` so the log shows exactly where it happened.
        """
        if name in self.loaded:
            return False
        self.load_adapter(name)
        return True

    # -- generation --------------------------------------------------------

    def prompt_ids(self, prompt_id: str) -> list[int]:
        record = self.prompts.get(prompt_id)
        if record is None:
            raise RuntimeError(f"prompt id not present in prompts.jsonl: {prompt_id}")
        return list(record["input_ids"])

    def resolve_prompt_id(self, transition: str, contract_prompt_id: str | None) -> str:
        """Apply the --prompt shard axis.

        --prompt replaces every step whose contract prompt is "factual", so the
        whole lifecycle can be re-run under an 876-token prefix.  prefill.short is
        exempt: it exists purely to contrast with prefill.long, so forcing it long
        would erase the transition's meaning.
        """
        if contract_prompt_id == "factual" and transition != "prefill.short":
            return self.args.prompt
        return contract_prompt_id or self.args.prompt

    def _sampling(self, max_new_tokens: int | None) -> dict:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": int(max_new_tokens or self.args.max_new_tokens),
        }

    def generate(
        self,
        prompt_id: str,
        *,
        adapter: str | None = None,
        max_new_tokens: int | None = None,
        label: str = "generate",
    ) -> dict:
        engine = self.engine
        kwargs: dict[str, object] = {
            "input_ids": self.prompt_ids(prompt_id),
            "sampling_params": self._sampling(max_new_tokens),
            "return_logprob": self.args.return_logprob,
        }
        if adapter is not None:
            kwargs[self.select_kwarg] = adapter

        async def go():
            return await engine.async_generate(**kwargs)

        return self.call_async(label, go, self.args.op_timeout)

    def generate_batch(
        self,
        prompt_ids: list[str],
        adapters: list[str | None],
        *,
        label: str,
    ) -> list[dict]:
        """One GenerateReqInput carrying N requests with per-request adapters."""
        engine = self.engine
        kwargs: dict[str, object] = {
            "input_ids": [self.prompt_ids(pid) for pid in prompt_ids],
            "sampling_params": [self._sampling(None) for _ in prompt_ids],
            "return_logprob": self.args.return_logprob,
        }
        if any(a is not None for a in adapters):
            kwargs[self.select_kwarg] = list(adapters)

        async def go():
            return await engine.async_generate(**kwargs)

        result = self.call_async(label, go, self.args.batch_timeout)
        return result if isinstance(result, list) else [result]

    def generate_concurrent(
        self,
        prompt_ids: list[str],
        adapters: list[str | None],
        *,
        stream: bool,
        label: str,
    ) -> list[dict]:
        """N independent in-flight requests gathered on the engine's own loop."""
        engine = self.engine
        sampling = self._sampling(None)
        return_logprob = self.args.return_logprob
        select_kwarg = self.select_kwarg
        payloads = [
            (self.prompt_ids(pid), adapter)
            for pid, adapter in zip(prompt_ids, adapters)
        ]

        async def one(input_ids, adapter):
            kwargs: dict[str, object] = {
                "input_ids": input_ids,
                "sampling_params": dict(sampling),
                "return_logprob": return_logprob,
                "stream": stream,
            }
            if adapter is not None:
                kwargs[select_kwarg] = adapter
            result = await engine.async_generate(**kwargs)
            if not stream:
                return result
            last = None
            async for chunk in result:
                last = chunk
            return last

        async def go():
            return await asyncio.gather(
                *(one(ids, adapter) for ids, adapter in payloads)
            )

        return self.call_async(label, go, self.args.batch_timeout)

    # -- observation bookkeeping -------------------------------------------

    @staticmethod
    def _output_ids(result) -> list[int]:
        """Generated token ids, however this revision chooses to report them.

        tokenizer_manager puts "output_ids" in the response dict, but the existing
        smokes in this family read the ids out of
        meta_info["output_token_logprobs"] (entries are (logprob, token_id, text)).
        Prefer the direct field, fall back to the logprob triples, so the script
        works on either shape rather than silently recording an empty output.
        """
        if not isinstance(result, dict):
            return []
        direct = result.get("output_ids")
        if direct:
            return [int(token) for token in direct]
        triples = (result.get("meta_info") or {}).get("output_token_logprobs")
        if triples:
            return [int(entry[1]) for entry in triples]
        return []

    @staticmethod
    def _text(result) -> str:
        if isinstance(result, dict):
            return result.get("text") or ""
        return ""

    def record_observation(self, transition: str, result, harness) -> None:
        """Build a harness Observation so validate_lifecycle_observations can run."""
        try:
            self.observations[transition] = harness.Observation(
                output_ids=tuple(self._output_ids(result)),
                text=self._text(result),
                token_logprobs=(),
                selected_logits={},
                adapter_state={},
                error=None,
            )
        except Exception:  # observation construction must never fail the shard
            pass

    def differs_from_base(self, output_ids: list[int]) -> bool | None:
        if self.base_output_ids is None:
            return None
        return list(output_ids) != list(self.base_output_ids)

    # -- the transition dispatch table -------------------------------------

    def run_transition(self, transition: str, harness) -> dict:
        """Execute one transition; return the fields to attach to its step line."""
        extra: dict[str, object] = {}
        reloaded = False

        def gen(prompt_id, adapter=None, max_new_tokens=None):
            nonlocal reloaded
            if adapter is not None:
                reloaded = self.ensure_loaded(adapter) or reloaded
            resolved = self.resolve_prompt_id(transition, prompt_id)
            extra["prompt_id"] = resolved
            extra["prompt_tokens"] = len(self.prompt_ids(resolved))
            if adapter is not None:
                extra["adapter"] = adapter
            return self.generate(
                resolved,
                adapter=adapter,
                max_new_tokens=max_new_tokens,
                label=transition,
            )

        def finish_single(result):
            output_ids = self._output_ids(result)
            extra["output_ids"] = output_ids
            extra["output_digest"] = digest_ids(output_ids)
            extra["output_len"] = len(output_ids)
            extra["text"] = self._text(result)[: self.args.text_chars]
            extra["differs_from_base"] = self.differs_from_base(output_ids)
            self.record_observation(transition, result, harness)
            if reloaded:
                extra["reloaded"] = True
            return extra

        # --- base / startup ------------------------------------------------
        if transition == "base.initial":
            result = gen("factual")
            self.base_output_ids = self._output_ids(result)
            return finish_single(result)

        if transition == "startup.adapter":
            if not self.args.startup_adapter:
                extra["skipped_reason"] = "--no-startup-adapter"
                extra["status_override"] = "skipped"
                return extra
            return finish_single(gen("factual", adapter=self.startup_adapter_name))

        # --- dynamic load / infer / unload ---------------------------------
        if transition == "dynamic.load":
            self.load_adapter("policy-a")
            extra["adapter"] = "policy-a"
            extra["adapter_path"] = self.fixtures["policy-a"]
            return extra

        if transition == "dynamic.infer":
            return finish_single(gen("factual", adapter="policy-a"))

        if transition == "dynamic.unload":
            self.ensure_loaded("policy-a")
            self.unload_adapter("policy-a")
            extra["adapter"] = "policy-a"
            return extra

        if transition == "dynamic.post-unload-base":
            result = gen("factual")
            finish_single(result)
            if self.base_output_ids is not None:
                extra["matches_base_exactly"] = (
                    self._output_ids(result) == self.base_output_ids
                )
                extra["contract_ok"] = extra["matches_base_exactly"]
            return extra

        # --- switching -----------------------------------------------------
        if transition == "switch.a":
            return finish_single(gen("factual", adapter="policy-a"))
        if transition == "switch.b":
            return finish_single(gen("factual", adapter="policy-b"))
        if transition == "switch.a-again":
            return finish_single(gen("factual", adapter="policy-a"))

        # --- mixed / concurrent --------------------------------------------
        if transition in ("mixed.base-a-b", "concurrent.stream",
                          "concurrent.non-stream"):
            requested = self.batches.get("batch-8", {}).get("requests", 8)
            # prompts.jsonl stores "requests" as the LIST of request objects;
            # int() on it raised TypeError and errored every multi-request
            # transition (mixed.base-a-b, concurrent.stream/non-stream).
            size = len(requested) if isinstance(requested, list) else int(requested)
            single_ids = [
                pid for pid in self.prompts
                if pid != "long-prefix" or self.args.prompt == "long-prefix"
            ] or list(self.prompts)
            prompt_ids = [
                self.resolve_prompt_id(transition, "factual")
                if index % 2 == 0
                else single_ids[index % len(single_ids)]
                for index in range(size)
            ]
            adapters = [
                DEFAULT_MIXED_ADAPTER_PATTERN[index % len(DEFAULT_MIXED_ADAPTER_PATTERN)]
                for index in range(size)
            ]
            for adapter in {a for a in adapters if a}:
                reloaded = self.ensure_loaded(adapter) or reloaded
            extra["requests"] = size
            extra["adapters"] = adapters
            extra["prompt_ids"] = prompt_ids

            if transition == "mixed.base-a-b":
                results = self.generate_batch(prompt_ids, adapters, label=transition)
            else:
                results = self.generate_concurrent(
                    prompt_ids,
                    adapters,
                    stream=(transition == "concurrent.stream"),
                    label=transition,
                )
            per_request = []
            for index, result in enumerate(results):
                output_ids = self._output_ids(result)
                per_request.append({
                    "index": index,
                    "adapter": adapters[index],
                    "output_ids": output_ids,
                    "output_digest": digest_ids(output_ids),
                    "differs_from_base": self.differs_from_base(output_ids),
                })
            extra["results"] = per_request
            extra["returned"] = len(results)
            extra["complete"] = len(results) == size
            if reloaded:
                extra["reloaded"] = True
            if len(results) != size:
                raise RuntimeError(
                    f"expected {size} responses, received {len(results)}"
                )
            return extra

        # --- prefill / decode ----------------------------------------------
        if transition == "prefill.short":
            return finish_single(gen("factual"))
        if transition == "prefill.long":
            return finish_single(gen("long-prefix"))
        if transition == "decode.short":
            return finish_single(gen("factual", max_new_tokens=1))
        if transition == "decode.long":
            return finish_single(gen("factual", max_new_tokens=128))

        # --- staging / activation (surrogate) -------------------------------
        if transition in ("stage.v1", "stage.v2"):
            version = transition.split(".", 1)[1]
            name = f"policy-a@{version}"
            self.load_adapter(name, self.fixtures[name])
            extra["adapter"] = name
            extra["adapter_path"] = self.fixtures[name]
            return extra

        if transition in ("activate.v1", "activate.v2"):
            version = transition.split(".", 1)[1]
            name = f"policy-a@{version}"
            result = gen("factual", adapter=name)
            finish_single(result)
            previous, self.active_version = self.active_version, name
            extra["activated"] = name
            if previous and previous != name and previous in self.loaded:
                # Superseding a version must retire it, otherwise reject.stale has
                # nothing stale to reject.
                self.unload_adapter(previous)
                extra["superseded"] = previous
            return extra

        # --- expected-failure transitions -----------------------------------
        if transition == "reject.duplicate":
            self.ensure_loaded("policy-a")
            self.load_adapter("policy-a")
            extra["unexpected_success"] = "duplicate load was accepted"
            return extra

        if transition == "reject.stale":
            stale = "policy-a@v1"
            extra["adapter"] = stale
            result = self.generate(
                self.resolve_prompt_id(transition, "factual"),
                adapter=stale,
                label=transition,
            )
            extra["unexpected_success"] = "stale version served a request"
            extra["output_digest"] = digest_ids(self._output_ids(result))
            return extra

        if transition == "reject.invalid-id":
            self.unload_adapter("missing")
            extra["unexpected_success"] = "unload of unknown adapter was accepted"
            return extra

        if transition == "reject.invalid-config":
            self.load_adapter("policy-invalid", self.fixtures["invalid"])
            extra["unexpected_success"] = "malformed adapter config was accepted"
            return extra

        # --- rollback / restart ----------------------------------------------
        if transition == "rollback.previous":
            result = gen("factual", adapter="policy-a")
            finish_single(result)
            extra["active_version"] = self.active_version
            return extra

        if transition == "restart.same-manifest":
            self.relaunch()
            result = gen("factual")
            finish_single(result)
            if self.base_output_ids is not None:
                extra["matches_base_exactly"] = (
                    self._output_ids(result) == self.base_output_ids
                )
                extra["contract_ok"] = extra["matches_base_exactly"]
            return extra

        raise RuntimeError(f"transition not implemented by this runner: {transition}")

    # -- driver -------------------------------------------------------------

    def run(self, selection: tuple[str, ...], harness) -> str:
        for index, transition in enumerate(selection):
            expect_failure = transition.startswith(EXPECTED_FAILURE_PREFIX)
            started = time.time()
            record: dict[str, object] = {
                "transition": transition,
                "index": index,
                "total": len(selection),
                "expected": "failure" if expect_failure else "success",
            }
            if transition in SURROGATE_TRANSITIONS:
                record["surrogate"] = True
                record["surrogate_note"] = SURROGATE_TRANSITIONS[transition]

            try:
                extra = self.run_transition(transition, harness)
                status_override = extra.pop("status_override", None)
                record.update(extra)
                if status_override:
                    status = status_override
                elif expect_failure:
                    status = "fail"  # the call was supposed to be refused
                elif extra.get("contract_ok") is False:
                    status = "fail"
                else:
                    status = "pass"
            except OperationTimeout as error:
                record["status"] = "timeout"
                record["error"] = str(error)
                record["seconds"] = round(time.time() - started, 3)
                self.results.append(record)
                self.emit("step", **record)
                return "timeout"
            except BaseException as error:  # noqa: BLE001 - reported, never swallowed
                status = "pass" if expect_failure else "error"
                record["error_type"] = type(error).__name__
                record["error"] = str(error)[: self.args.text_chars]
                if not expect_failure:
                    record["traceback"] = traceback.format_exc()[-self.args.text_chars:]
                else:
                    record["rejected_as_expected"] = True

            record["status"] = status
            record["seconds"] = round(time.time() - started, 3)
            self.results.append(record)
            self.emit("step", **record)

            if status in ("error", "fail") and self.args.fail_fast:
                return "fail"

        return "fail" if any(
            r["status"] in ("fail", "error") for r in self.results
        ) else "pass"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_case.py",
        description="Run one adapter-lifecycle shard against an in-process SGLang Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Transitions implemented as surrogates (flagged \"surrogate\": true in\n"
            "the log): startup.adapter, stage.v1, stage.v2, activate.v1,\n"
            "activate.v2, reject.stale.  The Engine exposes no stage/activate API;\n"
            "real staging goes through update_adapter_from_distributed, which needs\n"
            "an external rank on an NCCL weight-sync group and cannot be driven\n"
            "from a single-process smoke."
        ),
    )
    parser.add_argument("run_dir", help="Directory for logs and per-run fixtures.")
    parser.add_argument("model_path", help="HF model id or local model directory.")
    parser.add_argument(
        "--mode", required=True, choices=("canonical_oft", "native_lora")
    )
    parser.add_argument("--prompt", required=True, choices=("factual", "long-prefix"))
    parser.add_argument(
        "--transitions", required=True, choices=("full", "expert", "short")
    )

    parser.add_argument("--harness-dir", default=DEFAULT_HARNESS_DIR)
    parser.add_argument("--prompts-file", default=None,
                        help="Defaults to <harness-dir>/prompts.jsonl")
    parser.add_argument("--revision-kind", default="candidate",
                        choices=("source", "candidate"))

    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--cuda-graph", action="store_true", default=False)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--moe-runner", default=None)
    parser.add_argument("--max-oft-block-size", type=int, default=128)
    parser.add_argument("--oft-scale", type=float, default=None,
                        help="Magnitude of the OFT fixture's skew-symmetric "
                             "parameters S. The harness default (1e-3) puts "
                             "R = Cayley(S) within ~0.2%% of identity, so the "
                             "adapter cannot flip a token and differs_from_base "
                             "is false whether or not the adapter works. 1e-2 is "
                             "the smallest measured scale that moves tokens while "
                             "still restoring the base exactly on unload, and it "
                             "matches the LoRA fixture scale.")
    parser.add_argument(
        "--architecture", default="auto", choices=("auto", "dense", "moe")
    )

    parser.add_argument("--fixture-layers", type=int, default=0,
                        help="Transformer layers the fixtures cover; 0 = all layers "
                             "(what the existing passing smokes use -- a few-layer "
                             "adapter may be too weak to shift any token, which "
                             "makes every differs_from_base answer uninformative).")
    parser.add_argument("--fixture-experts", type=int, default=8,
                        help="Experts per layer the MoE fixtures cover; 0 = all. "
                             "Kept small by default because all-experts is huge: on "
                             "Qwen3-30B-A3B (48 layers x 128 experts) a full OFT "
                             "fixture is ~1.9 GB, and this script builds four of "
                             "them. Raise it when expert coverage matters more "
                             "than disk.")

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--return-logprob", dest="return_logprob",
                        action="store_true", default=True,
                        help="On by default: it is the fallback source of "
                             "output token ids.")
    parser.add_argument("--no-return-logprob", dest="return_logprob",
                        action="store_false")
    parser.add_argument("--text-chars", type=int, default=2000,
                        help="Cap on text/error strings written to the log.")

    parser.add_argument("--op-timeout", type=float, default=300.0,
                        help="Bound on a single generate / load / unload.")
    parser.add_argument("--batch-timeout", type=float, default=600.0,
                        help="Bound on a batched or concurrent step.")
    parser.add_argument("--startup-timeout", type=float, default=1800.0,
                        help="Bound on engine construction / relaunch.")
    parser.add_argument("--timeout-slack", type=float, default=60.0,
                        help="Extra wall clock the outer thread bound allows.")

    parser.add_argument("--startup-adapter", dest="startup_adapter",
                        action="store_true", default=True)
    parser.add_argument("--no-startup-adapter", dest="startup_adapter",
                        action="store_false")
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Resolve and print the plan; launch nothing.")
    parser.add_argument("--log-name", default=None,
                        help="Log filename inside run_dir (default run-case-<run-id>.jsonl)")
    parser.add_argument(
        "--engine-kwarg", action="append", type=parse_engine_kwarg, default=[],
        metavar="KEY=JSON",
        help="Extra Engine(**kwargs) entry, repeatable "
             "(e.g. --engine-kwarg max_lora_rank=8).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_id = "{}-{}-{:04x}".format(
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        os.getpid(),
        random.getrandbits(16),
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_name = args.log_name or f"run-case-{args.run_id}.jsonl"
    emit_target = Emitter(run_dir / log_name)
    emit = emit_target.emit

    selection = resolve_selection(args.transitions)
    emit(
        "start",
        run_id=args.run_id,
        run_dir=str(run_dir.resolve()),
        log=str((run_dir / log_name).resolve()),
        model_path=args.model_path,
        mode=args.mode,
        prompt=args.prompt,
        transitions=args.transitions,
        selected=list(selection),
        selected_count=len(selection),
        surrogates=[t for t in selection if t in SURROGATE_TRANSITIONS],
        argv=list(argv if argv is not None else sys.argv[1:]),
    )

    if args.dry_run:
        emit("summary", dry_run=True, selected=list(selection),
             counts={"pass": 0, "fail": 0, "error": 0, "skipped": 0, "timeout": 0})
        emit("verdict", verdict="dry-run", exit_code=0)
        emit_target.close()
        return 0

    harness_dir = Path(args.harness_dir)
    prompts_file = Path(args.prompts_file or (harness_dir / "prompts.jsonl"))
    runner = ShardRunner(args, emit_target)

    # ---- setup ----------------------------------------------------------
    try:
        sys.path.insert(0, str(harness_dir.parent))
        import adapter_equivalence as harness_pkg
        from adapter_equivalence import scenarios as harness_scenarios
        from adapter_equivalence import server as harness_server

        if tuple(harness_scenarios._ADAPTER_LIFECYCLE_TRANSITIONS) != FULL_TRANSITIONS:
            raise RuntimeError(
                "harness transition contract has drifted from this script's copy; "
                "refusing to run a stale selection"
            )

        class Harness:
            ServerSpec = harness_server.ServerSpec
            engine_kwargs = staticmethod(harness_server.engine_kwargs)
            build_oft_fixture = staticmethod(harness_pkg.build_oft_fixture)
            build_lora_fixture = staticmethod(harness_pkg.build_lora_fixture)
            Observation = harness_pkg.Observation
            ADAPTER_SEEDS = __import__(
                "adapter_equivalence.fixtures", fromlist=["ADAPTER_SEEDS"]
            ).ADAPTER_SEEDS
            validate_lifecycle_observations = staticmethod(
                harness_scenarios.validate_lifecycle_observations
            )

        harness = Harness
        runner.prompts, runner.batches = load_prompts(prompts_file)
        emit("prompts.loaded", path=str(prompts_file),
             prompt_ids=sorted(runner.prompts),
             batch_ids=sorted(runner.batches),
             lengths={pid: len(rec["input_ids"])
                      for pid, rec in sorted(runner.prompts.items())})
        if args.prompt not in runner.prompts:
            raise RuntimeError(f"--prompt {args.prompt} not present in {prompts_file}")

        runner.prepare_fixtures(harness)
        runner.launch(harness)
    except BaseException as error:  # noqa: BLE001
        emit("setup.error", error_type=type(error).__name__, error=str(error),
             traceback=traceback.format_exc()[-args.text_chars:])
        emit("summary", ok=False, stage="setup", selected=list(selection),
             counts={"pass": 0, "fail": 0, "error": 0, "skipped": 0, "timeout": 0})
        emit("verdict", verdict="setup-error", exit_code=3)
        runner.shutdown()
        emit_target.close()
        return 3

    # ---- run ------------------------------------------------------------
    try:
        outcome = runner.run(selection, harness)
    except BaseException as error:  # noqa: BLE001
        emit("runner.error", error_type=type(error).__name__, error=str(error),
             traceback=traceback.format_exc()[-args.text_chars:])
        outcome = "fail"

    # ---- contract validation (best effort) -------------------------------
    contract = None
    if {"base.initial", "dynamic.post-unload-base"} <= set(runner.observations):
        try:
            harness.validate_lifecycle_observations(runner.observations)
            contract = {"ok": True}
        except BaseException as error:  # noqa: BLE001
            contract = {"ok": False, "error": str(error),
                        "error_type": type(error).__name__}
            outcome = "fail"
        emit("contract.validate", **contract)

    counts = {"pass": 0, "fail": 0, "error": 0, "skipped": 0, "timeout": 0}
    for record in runner.results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    executed = {record["transition"] for record in runner.results}
    not_reached = [t for t in selection if t not in executed]

    emit(
        "summary",
        run_id=args.run_id,
        mode=args.mode,
        prompt=args.prompt,
        transitions=args.transitions,
        model_path=args.model_path,
        selected=list(selection),
        executed=len(runner.results),
        not_reached=not_reached,
        counts=counts,
        base_output_digest=digest_ids(runner.base_output_ids or []),
        base_output_ids=runner.base_output_ids,
        contract=contract,
        surrogate_transitions=[t for t in selection if t in SURROGATE_TRANSITIONS],
        steps=[
            {
                "transition": r["transition"],
                "status": r["status"],
                "seconds": r["seconds"],
                "differs_from_base": r.get("differs_from_base"),
                "output_digest": r.get("output_digest"),
            }
            for r in runner.results
        ],
    )

    if outcome == "timeout" or runner.timed_out:
        exit_code = 2
        verdict = "timeout"
    elif counts["fail"] or counts["error"] or not_reached:
        exit_code = 1
        verdict = "fail"
    else:
        exit_code = 0
        verdict = "pass"

    emit("verdict", verdict=verdict, exit_code=exit_code, counts=counts,
         run_id=args.run_id, log=str((run_dir / log_name).resolve()))

    if verdict == "timeout":
        # A worker thread is still wedged inside the engine; a clean interpreter
        # exit would join it and hang. The report is already flushed to disk.
        emit_target.close()
        os._exit(exit_code)

    runner.shutdown()
    runner.pool.shutdown(wait=False)
    emit_target.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

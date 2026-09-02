from types import SimpleNamespace

import pytest

# Sentinel distinguishing "caller didn't pass oft_target_modules" (auto-fill
# a valid default so unrelated tests don't need to restate it) from "caller
# explicitly passed None" (needed to test the --peft-target-modules
# deprecated-alias copy, which only fires when oft_target_modules is
# genuinely unset).
_UNSET = object()


def _args(
    peft_method,
    *,
    enable_lora=True,
    max_loaded_ofts=None,
    max_ofts_per_batch=2,
    peft_paths=None,
    oft_target_modules=_UNSET,
    peft_target_modules=None,
    max_oft_block_size=None,
    oft_double_buffer=False,
    peft_double_buffer=False,
    oft_impl="sibling",
    cuda_graph_config=None,
    model_has_moe=False,
    enable_dp_attention=False,
):
    ns = SimpleNamespace(
        enable_lora=enable_lora,
        enable_dp_attention=enable_dp_attention,
        peft_method=peft_method,
        peft_paths=peft_paths,
        peft_target_modules=peft_target_modules,
        # --peft-paths is retired, so validate_peft_args's "either paths or
        # (block_size and target_modules)" requirement collapses to always
        # needing block_size+target_modules -- default both to a valid value
        # whenever OFT is enabled and the caller didn't override them, so
        # tests that aren't specifically exercising that assertion don't
        # need to restate it every time.
        oft_target_modules=(
            oft_target_modules
            if oft_target_modules is not _UNSET
            else (["o_proj"] if peft_method is not None else None)
        ),
        max_oft_block_size=(
            max_oft_block_size
            if max_oft_block_size is not None
            else (32 if peft_method is not None else None)
        ),
        max_ofts_per_batch=max_ofts_per_batch,
        max_loaded_ofts=max_loaded_ofts,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        oft_double_buffer=oft_double_buffer,
        peft_double_buffer=peft_double_buffer,
        speculative_algorithm=None,
        cuda_graph_config=cuda_graph_config,
        oft_impl=oft_impl,
    )
    # Stand-in for ServerArgs._late_resolution: validate_peft_args writes its
    # normalized oft_target_modules through this (real ServerArgs is
    # read-only by plain assignment once __post_init__ resolves it).
    ns._late_resolution = lambda source, **fields: ns.__dict__.update(fields)
    # Stand-in for ServerArgs.get_model_config(): the MoE decode-graph guard
    # reads hf_text_config's per-token expert count to tell an actual MoE model
    # from a dense one whose MLP merely uses the same module names.
    ns.get_model_config = lambda: SimpleNamespace(
        hf_text_config=(
            SimpleNamespace(num_experts_per_tok=8)
            if model_has_moe
            else SimpleNamespace()
        )
    )
    return ns


def test_removed_lora_method_is_rejected_even_when_set_programmatically():
    """Regression: argparse's ``choices=["oft"]`` rejects ``--peft-method lora``
    on the CLI, but a caller that builds ``ServerArgs`` directly (e.g. an RL
    launcher passing ``peft_method="lora"`` as a kwarg) bypasses argparse
    entirely and used to sail through ``validate_peft_args`` uncaught after
    srt/peft/lora was deleted -- silently no-op'ing to base-model-only serving
    instead of failing loudly."""
    from sglang.srt.peft.config import validate_peft_args

    with pytest.raises(ValueError, match=r"--peft-method 'lora' is no longer supported"):
        validate_peft_args(_args("lora", enable_lora=False))


def test_native_lora_and_single_active_peft_are_mutually_exclusive():
    """Catch the PEFT method initializing alongside native LoRA."""
    from sglang.srt.peft.config import validate_peft_args

    with pytest.raises(
        ValueError,
        match=r"--enable-lora.*--peft-method.*mutually exclusive",
    ):
        validate_peft_args(_args("oft"))


@pytest.mark.parametrize(
    ("enable_lora", "peft_method"),
    [(True, None), (False, "oft")],
)
def test_native_lora_and_single_active_peft_validate_independently(
    enable_lora, peft_method
):
    """Catch an over-broad guard that rejects either system on its own."""
    from sglang.srt.peft.config import validate_peft_args

    validate_peft_args(_args(peft_method, enable_lora=enable_lora))


def test_max_loaded_ofts_must_be_at_least_max_ofts_per_batch_minus_one():
    """Validate that max_loaded_ofts must be >= max_ofts_per_batch - 1.

    Buffer slot 0 is always reserved for the base/identity placeholder, so
    real per-batch adapter capacity is max_ofts_per_batch - 1, not
    max_ofts_per_batch -- the bound must be checked against that real
    capacity, or the minimum legal configuration silently overcommits by one
    slot (see C1's fix in peft/config.py).
    """
    from sglang.srt.peft.config import validate_peft_args

    # max_loaded_ofts=2 is below max_ofts_per_batch - 1 == 3: must still fail.
    with pytest.raises(AssertionError, match=r"max_loaded_ofts should be greater than or equal"):
        validate_peft_args(_args("oft", enable_lora=False, max_loaded_ofts=2, max_ofts_per_batch=4))


def test_max_loaded_ofts_equal_to_max_ofts_per_batch_minus_one_is_legal():
    """The new minimum legal boundary (max_loaded_ofts == max_ofts_per_batch
    - 1) must be accepted -- regression guard for the fix that moved the
    bound from max_ofts_per_batch to max_ofts_per_batch - 1."""
    from sglang.srt.peft.config import validate_peft_args

    validate_peft_args(
        _args(
            "oft",
            enable_lora=False,
            max_loaded_ofts=3,
            max_ofts_per_batch=4,
        )
    )


def test_peft_paths_is_rejected_as_retired():
    """--peft-paths has been retired: on-disk adapter preload is no longer
    supported. Any truthy value must raise a clear retirement error pointing
    at the native RPC adapter-load mechanism, instead of parsing into
    OFTRef objects (the removed normalization logic) or silently no-op'ing."""
    from sglang.srt.peft.config import validate_peft_args

    with pytest.raises(ValueError, match=r"--peft-paths has been retired"):
        validate_peft_args(
            _args(
                "oft",
                enable_lora=False,
                peft_paths=["/models/adapter"],
            )
        )


def test_oft_target_modules_alone_works_as_canonical_flag():
    """--oft-target-modules, with no deprecated --peft-target-modules alias
    involved at all, must work as the new canonical flag: the value lands on
    oft_target_modules unchanged (as a normalized set)."""
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["o_proj", "down_proj"],
    )
    validate_peft_args(args)
    assert args.oft_target_modules == {"o_proj", "down_proj"}


def test_peft_target_modules_deprecated_alias_still_works_and_warns(caplog):
    """--peft-target-modules is deprecated in favor of --oft-target-modules,
    but must still work as an alias: when only the old flag is set, a
    warning is logged and the value is copied across to oft_target_modules."""
    import logging

    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=None,
        peft_target_modules=["o_proj", "down_proj"],
    )
    with caplog.at_level(logging.WARNING, logger="sglang.srt.peft.config"):
        validate_peft_args(args)

    assert any(
        "--peft-target-modules is deprecated" in message
        for message in caplog.messages
    )
    assert args.oft_target_modules == {"o_proj", "down_proj"}


def test_peft_target_modules_alias_copy_survives_real_server_args_read_only_guard():
    """Regression: the deprecated --peft-target-modules -> --oft-target-modules
    alias copy used to write ``server_args.oft_target_modules = ...`` directly.
    Real ``ServerArgs.__setattr__`` raises ``AttributeError`` for any bare
    (non-underscore) field write once ``__post_init__``'s
    ``materialize_declarations()`` sets ``_declarations_materialized`` --
    which always happens before ``validate_peft_args`` runs (it's called from
    ``check_server_args``, itself called from ``Engine._launch_subprocesses``,
    well after ``__post_init__`` completes). So the ONLY case the alias
    exists for -- old flag set, new flag unset -- crashed with
    ``AttributeError`` at every real server launch.

    The other alias tests in this file use a ``SimpleNamespace`` fixture
    whose ``_late_resolution`` stand-in just does ``ns.__dict__.update(...)``
    -- a plain mutable object happily accepts any attribute write, so that
    fixture cannot catch this bug at all. This test instead drives the REAL,
    read-only ``ServerArgs`` seam: a bare instance (``__new__`` bypasses
    ``__init__``) with ``_declarations_materialized`` set, reproducing the
    exact post-``__post_init__`` state ``validate_peft_args`` always runs
    under for a real launch.
    """
    from sglang.srt.peft.config import validate_peft_args
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs.__new__(ServerArgs)
    fields = dict(
        peft_method="oft",
        enable_lora=False,
        oft_impl="sibling",
        peft_paths=None,
        oft_target_modules=None,
        peft_target_modules=["down_proj"],
        max_oft_block_size=32,
        max_ofts_per_batch=4,
        max_loaded_ofts=None,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        oft_double_buffer=False,
        peft_double_buffer=False,
        speculative_algorithm=None,
        cuda_graph_config=None,
        enable_dp_attention=False,
    )
    for name, value in fields.items():
        object.__setattr__(sa, name, value)
    # Reproduce the read-only guard: real server launches always reach
    # validate_peft_args with this already set (see docstring above).
    object.__setattr__(sa, "_declarations_materialized", True)

    validate_peft_args(sa)  # must not raise AttributeError

    assert sa.oft_target_modules == {"down_proj"}


def test_peft_target_modules_conflicting_with_oft_target_modules_is_rejected():
    """When BOTH flags are set to different values, this must fail loudly
    (mirrors ServerArgs._handle_elastic_ep's --elastic-ep-rejoin conflict
    check) rather than silently picking one and hiding the ambiguity."""
    from sglang.srt.peft.config import validate_peft_args

    with pytest.raises(ValueError, match=r"--peft-target-modules.*conflicts with.*--oft-target-modules"):
        validate_peft_args(
            _args(
                "oft",
                enable_lora=False,
                oft_target_modules=["o_proj"],
                peft_target_modules=["down_proj"],
            )
        )


def test_oft_double_buffer_alone_works_as_canonical_flag():
    """--oft-double-buffer, with no deprecated --peft-double-buffer alias
    involved at all, must work as the new canonical flag: the value lands on
    oft_double_buffer unchanged."""
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        max_ofts_per_batch=3,
        oft_double_buffer=True,
    )
    validate_peft_args(args)
    assert args.oft_double_buffer is True


def test_peft_double_buffer_deprecated_alias_still_works_and_warns(caplog):
    """--peft-double-buffer is deprecated in favor of --oft-double-buffer,
    but must still work as an alias: when the old flag is set, a warning is
    logged and it is OR-merged into oft_double_buffer."""
    import logging

    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        max_ofts_per_batch=3,
        oft_double_buffer=False,
        peft_double_buffer=True,
    )
    with caplog.at_level(logging.WARNING, logger="sglang.srt.peft.config"):
        validate_peft_args(args)

    assert any(
        "--peft-double-buffer is deprecated" in message
        for message in caplog.messages
    )
    assert args.oft_double_buffer is True


def test_peft_double_buffer_alias_copy_survives_real_server_args_read_only_guard():
    """Regression, mirroring
    test_peft_target_modules_alias_copy_survives_real_server_args_read_only_guard:
    the deprecated --peft-double-buffer -> --oft-double-buffer OR-merge must
    not write ``server_args.oft_double_buffer = ...`` directly, since real
    ``ServerArgs.__setattr__`` raises ``AttributeError`` for any bare field
    write once ``_declarations_materialized`` is set -- which is always true
    by the time ``validate_peft_args`` runs at a real server launch. This
    drives the REAL, read-only ``ServerArgs`` seam (not the ``SimpleNamespace``
    fixture the other alias tests use, which can't catch this class of bug).
    """
    from sglang.srt.peft.config import validate_peft_args
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs.__new__(ServerArgs)
    fields = dict(
        peft_method="oft",
        enable_lora=False,
        oft_impl="sibling",
        peft_paths=None,
        oft_target_modules=["down_proj"],
        peft_target_modules=None,
        max_oft_block_size=32,
        max_ofts_per_batch=4,
        max_loaded_ofts=None,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        oft_double_buffer=False,
        peft_double_buffer=True,
        speculative_algorithm=None,
        cuda_graph_config=None,
        enable_dp_attention=False,
    )
    for name, value in fields.items():
        object.__setattr__(sa, name, value)
    # Reproduce the read-only guard: real server launches always reach
    # validate_peft_args with this already set (see docstring above).
    object.__setattr__(sa, "_declarations_materialized", True)

    validate_peft_args(sa)  # must not raise AttributeError

    assert sa.oft_double_buffer is True


def _cuda_graph_config(*, decode_backend):
    """Build a real CudaGraphConfig with the given decode backend (prefill
    is left DISABLED so it never interferes with the decode-specific
    assertions below)."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )

    return CudaGraphConfig(
        decode=PhaseConfig(backend=decode_backend),
        prefill=PhaseConfig(backend=Backend.DISABLED),
    )


def test_moe_target_oft_sibling_with_zero_capacity_disables_decode_cuda_graph():
    """Task 4b (2026-09-01-oft-moe-cuda-graph-dual-capture): the dual-capture
    mechanism (Tasks 1-4 of that plan) makes decode CUDA graphs safe for
    oft_impl=sibling + MoE-expert targeting whenever
    decode_cuda_graph_runner.py's _resolve_record_oft_variant_graph would
    engage it -- but that mechanism only engages when effective per-batch
    adapter capacity (max_ofts_per_batch - 1) is >= 1. At capacity == 0
    (max_ofts_per_batch == 1), dual-capture never engages (only the single
    fast-path graph is captured), so this guard must still disable decode
    CUDA graphs for exactly this configuration."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=1,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.DISABLED


def test_moe_target_oft_sibling_with_real_capacity_keeps_decode_cuda_graph_enabled():
    """Task 4b (2026-09-01-oft-moe-cuda-graph-dual-capture): relaxation of the
    guard above. Once effective per-batch adapter capacity
    (max_ofts_per_batch - 1) is >= 1 -- the same threshold
    decode_cuda_graph_runner.py's _resolve_record_oft_variant_graph uses to
    decide whether to capture the dual (no-real-adapter / any-real-adapter)
    decode graphs -- that mechanism handles the per-token multi-tenant
    routing tensor's pointer stability correctly across CUDA-graph
    capture/replay, so this guard must no longer disable decode CUDA graphs
    for this configuration. Regression guard: before this fix, the guard
    disabled decode graphs unconditionally whenever oft_impl=sibling targeted
    MoE experts on an MoE model, making Tasks 1-4's whole mechanism dead code
    in production."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=2,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_dense_target_oft_leaves_decode_cuda_graph_enabled():
    """Negative case: OFT targeting only dense modules (no MoE experts) must
    not trip the MoE-specific decode-graph guard -- the dense path has its
    own, different, already-correct per-token CUDA-graph mechanism
    (weight_indices via oft.utils.generate_sequence_lengths). max_ofts_per_
    batch=1 (effective capacity 0) so the capacity term alone (which by
    itself would WANT to disable) can't be what's keeping this enabled, and
    model_has_moe=True so _model_has_moe_layers alone can't either -- the
    target-module check ({"o_proj"} has no MoE-expert overlap) must be what's
    actually doing it."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["o_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        max_ofts_per_batch=1,
        model_has_moe=True,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


@pytest.mark.parametrize("oft_target_modules", [["down_proj"], ["all"]])
def test_dense_model_targeting_mlp_module_names_keeps_decode_cuda_graph(
    oft_target_modules,
):
    """Regression guard: the MoE decode-graph guard checked target-module
    NAMES only, never whether the model has any MoE layer -- so it also fired
    for dense models, whose MLP uses those very same names
    (gate_proj/up_proj/down_proj), and for every --oft-target-modules=all
    deployment (``all`` expands to include them). Those deployments have no
    expert-OFT buffers at all and never reach the unstable per-token routing
    path, so they must keep their decode CUDA graphs; silently disabling them
    cost real throughput and stripped decode-graph coverage from
    test/registered/rl/test_oft_load_from_tensor.py (dense Qwen3-0.6B,
    oft_target_modules=["down_proj"], sibling, decode graphs enabled)
    without failing anything. max_ofts_per_batch=1 (effective capacity 0) so
    the capacity term alone (which by itself would WANT to disable) can't be
    what's keeping this enabled -- the model_has_moe=False check must be."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=oft_target_modules,
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=False,
        max_ofts_per_batch=1,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_moe_target_oft_staged_impl_leaves_decode_cuda_graph_enabled():
    """Negative case: oft_impl=staged's double-buffer activate() copies real
    adapter data in place into memory_pool.active_idx, so the single-adapter
    fast path stays genuinely correct under decode-graph replay there -- the
    guard is specific to the plain native-RPC ("sibling") pool and must not
    fire for staged. max_ofts_per_batch=1 (effective capacity 0) so the
    capacity term alone (which by itself would WANT to disable for sibling)
    can't be what's keeping this enabled -- oft_impl=staged must be."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="staged",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        # Real MoE model: oft_impl=staged is the ONLY reason the guard must
        # not fire here.
        model_has_moe=True,
        max_ofts_per_batch=1,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_moe_target_oft_sibling_with_dp_attention_disables_decode_cuda_graph_even_with_capacity():
    """Final whole-branch review C1: --enable-dp-attention must keep decode
    CUDA graphs disabled for MoE-target sibling OFT regardless of capacity.
    decode_cuda_graph_runner.py's _resolve_record_oft_variant_graph excludes
    --enable-dp-attention outright (its cross-rank MoE token gathering is not
    supported by the per-rank persistent slot_ids buffer -- see
    OFTManager._compute_moe_multi_tenant_slot_ids), so dual-capture never
    engages for this combination no matter how much capacity is configured.
    Before this fix, this guard only looked at capacity (Task 4b), so a
    DP-attention server with capacity >= 1 would have kept decode CUDA
    graphs enabled -- "eligible but never dual-captured" -- which is unsafe:
    the single fast-path graph alone does not guarantee a real adapter lands
    at memory_pool.active_idx the way the capacity-only case does."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=4,
        enable_dp_attention=True,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.DISABLED


def test_moe_target_oft_sibling_without_dp_attention_and_with_capacity_keeps_decode_cuda_graph():
    """Negative case for the test above: without --enable-dp-attention, real
    capacity alone must still be sufficient to keep decode CUDA graphs
    enabled (Task 4b's relaxation must not be silently re-broken by adding
    the DP-attention term)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=4,
        enable_dp_attention=False,
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_non_oft_server_leaves_decode_cuda_graph_enabled():
    """Negative case: peft_method=None (OFT disabled entirely) must never
    trip the MoE decode-graph guard."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        None,
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL

from types import SimpleNamespace

import pytest

# Sentinel distinguishing "caller didn't pass oft_target_modules" (auto-fill
# a valid default so unrelated tests don't need to restate it) from "caller
# explicitly passed None".
_UNSET = object()


def _args(
    enable_oft,
    *,
    enable_lora=True,
    max_loaded_ofts=None,
    max_ofts_per_batch=2,
    oft_target_modules=_UNSET,
    max_oft_block_size=None,
    oft_double_buffer=False,
    oft_impl="sibling",
    cuda_graph_config=None,
    model_has_moe=False,
    enable_dp_attention=False,
):
    ns = SimpleNamespace(
        enable_lora=enable_lora,
        enable_dp_attention=enable_dp_attention,
        enable_oft=enable_oft,
        # OFT initialization always requires both max_oft_block_size and
        # oft_target_modules explicitly -- default both to a valid value
        # whenever OFT is enabled and the caller didn't override them, so
        # tests that aren't specifically exercising that assertion don't
        # need to restate it every time.
        oft_target_modules=(
            oft_target_modules
            if oft_target_modules is not _UNSET
            else (["o_proj"] if enable_oft else None)
        ),
        max_oft_block_size=(
            max_oft_block_size
            if max_oft_block_size is not None
            else (32 if enable_oft else None)
        ),
        max_ofts_per_batch=max_ofts_per_batch,
        max_loaded_ofts=max_loaded_ofts,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        oft_double_buffer=oft_double_buffer,
        speculative_algorithm=None,
        cuda_graph_config=cuda_graph_config,
        oft_impl=oft_impl,
    )
    # Stand-in for ServerArgs._late_resolution: validate_oft_args writes its
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


def test_native_lora_and_single_active_peft_are_mutually_exclusive():
    """Catch the PEFT method initializing alongside native LoRA."""
    from sglang.srt.oft.config import validate_oft_args

    with pytest.raises(
        ValueError,
        match=r"--enable-lora.*--enable-oft.*mutually exclusive",
    ):
        validate_oft_args(_args(True))


@pytest.mark.parametrize(
    ("enable_lora", "enable_oft"),
    [(True, False), (False, True)],
)
def test_native_lora_and_single_active_peft_validate_independently(
    enable_lora, enable_oft
):
    """Catch an over-broad guard that rejects either system on its own."""
    from sglang.srt.oft.config import validate_oft_args

    validate_oft_args(_args(enable_oft, enable_lora=enable_lora))


def test_max_loaded_ofts_must_be_at_least_max_ofts_per_batch_minus_one():
    """Validate that max_loaded_ofts must be >= max_ofts_per_batch - 1.

    Buffer slot 0 is always reserved for the base/identity placeholder, so
    real per-batch adapter capacity is max_ofts_per_batch - 1, not
    max_ofts_per_batch -- the bound must be checked against that real
    capacity, or the minimum legal configuration silently overcommits by one
    slot (see C1's fix in oft/config.py).
    """
    from sglang.srt.oft.config import validate_oft_args

    # max_loaded_ofts=2 is below max_ofts_per_batch - 1 == 3: must still fail.
    with pytest.raises(AssertionError, match=r"max_loaded_ofts should be greater than or equal"):
        validate_oft_args(_args(True, enable_lora=False, max_loaded_ofts=2, max_ofts_per_batch=4))


def test_max_loaded_ofts_equal_to_max_ofts_per_batch_minus_one_is_legal():
    """The new minimum legal boundary (max_loaded_ofts == max_ofts_per_batch
    - 1) must be accepted -- regression guard for the fix that moved the
    bound from max_ofts_per_batch to max_ofts_per_batch - 1."""
    from sglang.srt.oft.config import validate_oft_args

    validate_oft_args(
        _args(
            True,
            enable_lora=False,
            max_loaded_ofts=3,
            max_ofts_per_batch=4,
        )
    )


def test_oft_target_modules_alone_works_as_canonical_flag():
    """--oft-target-modules must work as the canonical flag: the value lands
    on oft_target_modules unchanged (as a normalized set)."""
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["o_proj", "down_proj"],
    )
    validate_oft_args(args)
    assert args.oft_target_modules == {"o_proj", "down_proj"}


def test_oft_double_buffer_alone_works_as_canonical_flag():
    """--oft-double-buffer must work as the canonical flag: the value lands
    on oft_double_buffer unchanged."""
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        max_ofts_per_batch=3,
        oft_double_buffer=True,
    )
    validate_oft_args(args)
    assert args.oft_double_buffer is True


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
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=1,
    )
    validate_oft_args(args)
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
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=2,
    )
    validate_oft_args(args)
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
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["o_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        max_ofts_per_batch=1,
        model_has_moe=True,
    )
    validate_oft_args(args)
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
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=oft_target_modules,
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=False,
        max_ofts_per_batch=1,
    )
    validate_oft_args(args)
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
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="staged",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        # Real MoE model: oft_impl=staged is the ONLY reason the guard must
        # not fire here.
        model_has_moe=True,
        max_ofts_per_batch=1,
    )
    validate_oft_args(args)
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
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=4,
        enable_dp_attention=True,
    )
    validate_oft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.DISABLED


def test_moe_target_oft_sibling_without_dp_attention_and_with_capacity_keeps_decode_cuda_graph():
    """Negative case for the test above: without --enable-dp-attention, real
    capacity alone must still be sufficient to keep decode CUDA graphs
    enabled (Task 4b's relaxation must not be silently re-broken by adding
    the DP-attention term)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        True,
        enable_lora=False,
        oft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
        model_has_moe=True,
        max_ofts_per_batch=4,
        enable_dp_attention=False,
    )
    validate_oft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_non_oft_server_leaves_decode_cuda_graph_enabled():
    """Negative case: enable_oft=False (OFT disabled entirely) must never
    trip the MoE decode-graph guard."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.oft.config import validate_oft_args

    args = _args(
        False,
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
    )
    validate_oft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL

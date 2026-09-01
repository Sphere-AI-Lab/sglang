from types import SimpleNamespace

import pytest


def _args(
    peft_method,
    *,
    enable_lora=True,
    max_loaded_ofts=None,
    max_ofts_per_batch=2,
    peft_paths=None,
    peft_target_modules=None,
    oft_impl="sibling",
    cuda_graph_config=None,
):
    ns = SimpleNamespace(
        enable_lora=enable_lora,
        peft_method=peft_method,
        peft_paths=(
            peft_paths
            if peft_paths is not None
            else (["/models/adapter"] if peft_method is not None else None)
        ),
        peft_target_modules=peft_target_modules,
        max_oft_block_size=None,
        max_ofts_per_batch=max_ofts_per_batch,
        max_loaded_ofts=max_loaded_ofts,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        peft_double_buffer=False,
        speculative_algorithm=None,
        cuda_graph_config=cuda_graph_config,
        oft_impl=oft_impl,
    )
    # Stand-in for ServerArgs._late_resolution: validate_peft_args writes its
    # normalized peft_paths/peft_target_modules through this (real ServerArgs
    # is read-only by plain assignment once __post_init__ resolves it).
    ns._late_resolution = lambda source, **fields: ns.__dict__.update(fields)
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
            peft_paths=["/models/adapter"],
        )
    )


def test_peft_paths_count_must_not_exceed_max_loaded_ofts():
    """Validate the second (previously untested) branch of the
    max_loaded_ofts checks: the number of --peft-paths entries must not
    exceed max_loaded_ofts."""
    from sglang.srt.peft.config import validate_peft_args

    with pytest.raises(AssertionError, match=r"should not exceed max_loaded_ofts"):
        validate_peft_args(
            _args(
                "oft",
                enable_lora=False,
                max_loaded_ofts=3,
                max_ofts_per_batch=4,
                peft_paths=["/models/a", "/models/b", "/models/c", "/models/d"],
            )
        )


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


def test_moe_target_oft_sibling_disables_decode_cuda_graph():
    """Final-review C1 fix: in the plain native-RPC ("sibling") pool, buffer
    slot 0 (memory_pool.active_idx) is permanently reserved by the boot-time
    base-request registration, so a real dynamically-loaded adapter can never
    occupy it -- OFTManager._compute_moe_multi_tenant_slot_ids therefore
    always takes its per-token multi-tenant branch for any real MoE-target
    adapter, and that branch's routing tensor has no pointer stability across
    CUDA-graph capture/replay. validate_peft_args must disable decode CUDA
    graphs for exactly this configuration (oft_impl=sibling + MoE expert
    target modules)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        peft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.DISABLED


def test_dense_target_oft_leaves_decode_cuda_graph_enabled():
    """Negative case: OFT targeting only dense modules (no MoE experts) must
    not trip the MoE-specific decode-graph guard -- the dense path has its
    own, different, already-correct per-token CUDA-graph mechanism
    (weight_indices via oft.utils.generate_sequence_lengths)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        peft_target_modules=["o_proj"],
        oft_impl="sibling",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
    )
    validate_peft_args(args)
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_moe_target_oft_staged_impl_leaves_decode_cuda_graph_enabled():
    """Negative case: oft_impl=staged's double-buffer activate() copies real
    adapter data in place into memory_pool.active_idx, so the single-adapter
    fast path stays genuinely correct under decode-graph replay there -- the
    guard is specific to the plain native-RPC ("sibling") pool and must not
    fire for staged."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.peft.config import validate_peft_args

    args = _args(
        "oft",
        enable_lora=False,
        peft_target_modules=["gate_proj", "up_proj", "down_proj"],
        oft_impl="staged",
        cuda_graph_config=_cuda_graph_config(decode_backend=Backend.FULL),
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

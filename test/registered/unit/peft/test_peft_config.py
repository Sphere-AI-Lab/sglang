from types import SimpleNamespace

import pytest


def _args(peft_method, *, enable_lora=True, max_loaded_ofts=None, max_ofts_per_batch=2):
    ns = SimpleNamespace(
        enable_lora=enable_lora,
        peft_method=peft_method,
        peft_paths=["/models/adapter"] if peft_method is not None else None,
        peft_target_modules=None,
        max_oft_block_size=None,
        max_ofts_per_batch=max_ofts_per_batch,
        max_loaded_ofts=max_loaded_ofts,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        peft_double_buffer=False,
        speculative_algorithm=None,
        cuda_graph_config=None,
        oft_impl="sibling",
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


def test_max_loaded_ofts_must_be_at_least_max_ofts_per_batch():
    """Validate that max_loaded_ofts must be >= max_ofts_per_batch."""
    from sglang.srt.peft.config import validate_peft_args

    # Test case where max_loaded_ofts < max_ofts_per_batch should fail
    with pytest.raises(AssertionError, match=r"max_loaded_ofts should be greater than or equal"):
        validate_peft_args(_args("oft", enable_lora=False, max_loaded_ofts=2, max_ofts_per_batch=4))

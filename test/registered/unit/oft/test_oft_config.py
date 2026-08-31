import argparse
from types import SimpleNamespace

import pytest


def _args(peft_method, *, enable_lora=True):
    ns = SimpleNamespace(
        enable_lora=enable_lora,
        peft_method=peft_method,
        peft_paths=["/models/adapter"] if peft_method is not None else None,
        peft_target_modules=None,
        max_oft_block_size=None,
        max_ofts_per_batch=2,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        speculative_algorithm=None,
        cuda_graph_config=None,
    )
    # Stand-in for ServerArgs._late_resolution: validate_oft_args writes its
    # normalized paths/targets through this hook.
    ns._late_resolution = lambda source, **fields: ns.__dict__.update(fields)
    return ns


def test_native_lora_and_oft_are_mutually_exclusive():
    """Catch canonical OFT initializing alongside native LoRA."""
    from sglang.srt.oft.config import validate_oft_args

    with pytest.raises(
        ValueError,
        match=r"--enable-lora.*--peft-method.*mutually exclusive",
    ):
        validate_oft_args(_args("oft"))


@pytest.mark.parametrize(
    ("enable_lora", "peft_method"),
    [(True, None), (False, "oft")],
)
def test_native_lora_and_oft_validate_independently(enable_lora, peft_method):
    """Catch an over-broad guard that rejects either system on its own."""
    from sglang.srt.oft.config import validate_oft_args

    validate_oft_args(_args(peft_method, enable_lora=enable_lora))


def test_lora_is_not_an_oft_method():
    """Catch the retired single-active LoRA branch returning to OFT config."""
    from sglang.srt.oft.config import validate_oft_args

    with pytest.raises(ValueError):
        validate_oft_args(_args("lora", enable_lora=False))


def test_register_oft_args_exposes_only_the_canonical_selector():
    """Catch OFT config reintroducing legacy method or implementation choices."""
    from sglang.srt.oft.config import register_oft_args

    parser = argparse.ArgumentParser()
    register_oft_args(parser)

    assert parser.parse_args(["--peft-method", "oft"]).peft_method == "oft"
    with pytest.raises(SystemExit):
        parser.parse_args(["--peft-method", "lora"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--oft-impl", "peft"])

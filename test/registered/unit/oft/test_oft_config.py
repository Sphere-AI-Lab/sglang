import argparse
from types import SimpleNamespace
from typing import get_type_hints

import pytest


_RETIRED_LORA_METHOD = "lo" + "ra"
_RETIRED_IMPLEMENTATION_FLAG = "--oft-" + "impl"


def _args(
    peft_method,
    *,
    enable_lora=True,
    max_loaded_ofts=None,
    max_ofts_per_batch=2,
    peft_paths=None,
    oft_drain_wait_threshold=0.0,
    enable_oft_overlap_loading=False,
):
    ns = SimpleNamespace(
        enable_lora=enable_lora,
        peft_method=peft_method,
        peft_paths=(
            peft_paths
            if peft_paths is not None
            else (["/models/adapter"] if peft_method is not None else None)
        ),
        peft_target_modules=None,
        max_oft_block_size=None,
        max_ofts_per_batch=max_ofts_per_batch,
        max_loaded_ofts=max_loaded_ofts,
        oft_backend="triton",
        oft_dtype=None,
        oft_type="canonical_oft",
        max_oft_chunk_size=16,
        oft_drain_wait_threshold=oft_drain_wait_threshold,
        enable_oft_overlap_loading=enable_oft_overlap_loading,
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
        validate_oft_args(_args(_RETIRED_LORA_METHOD, enable_lora=False))


def test_register_oft_args_exposes_only_the_canonical_selector():
    """Catch OFT config reintroducing legacy method or implementation choices."""
    from sglang.srt.oft.config import register_oft_args

    parser = argparse.ArgumentParser()
    register_oft_args(parser)

    assert parser.parse_args(["--peft-method", "oft"]).peft_method == "oft"
    with pytest.raises(SystemExit):
        parser.parse_args(["--peft-method", _RETIRED_LORA_METHOD])
    with pytest.raises(SystemExit):
        parser.parse_args([_RETIRED_IMPLEMENTATION_FLAG, "peft"])


def test_register_oft_args_exposes_max_loaded_ofts():
    from sglang.srt.oft.config import register_oft_args

    parser = argparse.ArgumentParser()
    register_oft_args(parser)

    assert parser.parse_args(["--max-loaded-ofts", "3"]).max_loaded_ofts == 3


def test_register_oft_args_exposes_drain_wait_threshold():
    from sglang.srt.oft.config import register_oft_args

    parser = argparse.ArgumentParser()
    register_oft_args(parser)

    assert (
        parser.parse_args(["--oft-drain-wait-threshold", "2.5"])
        .oft_drain_wait_threshold
        == 2.5
    )


def test_register_oft_args_exposes_overlap_loading():
    from sglang.srt.oft.config import register_oft_args

    parser = argparse.ArgumentParser()
    register_oft_args(parser)

    assert parser.parse_args(
        ["--enable-oft-overlap-loading"]
    ).enable_oft_overlap_loading


def test_oft_drain_wait_threshold_must_be_non_negative():
    from sglang.srt.oft.config import validate_oft_args

    with pytest.raises(AssertionError, match="must be non-negative"):
        validate_oft_args(
            _args(
                "oft",
                enable_lora=False,
                oft_drain_wait_threshold=-0.1,
            )
        )


def test_max_loaded_ofts_must_cover_real_per_batch_capacity():
    """Slot zero is base, so real adapter capacity is max_ofts_per_batch - 1."""
    from sglang.srt.oft.config import validate_oft_args

    with pytest.raises(
        AssertionError,
        match=r"max_loaded_ofts should be greater than or equal",
    ):
        validate_oft_args(
            _args(
                "oft",
                enable_lora=False,
                max_loaded_ofts=2,
                max_ofts_per_batch=4,
            )
        )


def test_max_loaded_ofts_accepts_real_capacity_boundary():
    from sglang.srt.oft.config import validate_oft_args

    validate_oft_args(
        _args(
            "oft",
            enable_lora=False,
            max_loaded_ofts=3,
            max_ofts_per_batch=4,
        )
    )


def test_initial_paths_must_fit_max_loaded_ofts():
    from sglang.srt.oft.config import validate_oft_args

    with pytest.raises(AssertionError, match=r"should not exceed max_loaded_ofts"):
        validate_oft_args(
            _args(
                "oft",
                enable_lora=False,
                max_loaded_ofts=3,
                max_ofts_per_batch=4,
                peft_paths=["/models/a", "/models/b", "/models/c", "/models/d"],
            )
        )


def test_oft_args_type_hints_resolve_for_server_args_consumers():
    """Catch forward references that break ServerArgs declaration handling."""
    from sglang.srt.oft.config import OFTArgs

    hints = get_type_hints(OFTArgs, include_extras=True)
    assert "peft_paths" in hints

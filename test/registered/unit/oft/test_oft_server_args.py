import argparse

import pytest


def _parser():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser


def test_server_args_accepts_only_canonical_oft():
    from sglang.srt.server_args import ServerArgs

    args = _parser().parse_args(
        ["--model-path", "dummy", "--peft-method", "oft"]
    )
    server_args = ServerArgs.from_cli_args(args)

    assert server_args.peft_method == "oft"
    assert server_args.oft_backend == "triton"


def test_server_args_rejects_legacy_lora_selector():
    from sglang.srt.server_args import ServerArgs

    with pytest.raises(ValueError, match=r"(?i)only.*oft"):
        ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            peft_method="lora",
        ).check_server_args()


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--peft-method", "lora"],
        ["--oft-impl", "peft"],
    ],
)
def test_server_cli_rejects_legacy_selectors(legacy_args):
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(["--model-path", "dummy", *legacy_args])

    assert exc_info.value.code == 2


def test_oft_ipc_types_are_exported_from_io_struct():
    from sglang.srt.managers.io_struct import (
        LoadOFTAdapterReqInput,
        OFTUpdateOutput,
    )
    from sglang.srt.oft.io_types import (
        LoadOFTAdapterReqInput as CanonicalLoadOFTAdapterReqInput,
    )
    from sglang.srt.oft.io_types import OFTUpdateOutput as CanonicalOFTUpdateOutput

    assert LoadOFTAdapterReqInput is CanonicalLoadOFTAdapterReqInput
    assert OFTUpdateOutput is CanonicalOFTUpdateOutput

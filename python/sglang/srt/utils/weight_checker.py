import logging
from typing import Dict, Iterable, Tuple

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)

logger = logging.getLogger(__name__)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors = None

    def handle(self, action: str):
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_tensors":
            self._reset_tensors()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        assert self._snapshot_tensors is not None

        _check_tensors(
            expect_tensors=_postprocess_tensors(self._snapshot_tensors),
            actual_tensors=_postprocess_tensors(dict(self._model_state())),
        )

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()
        yield from self._model_runner.model.named_buffers()


def _check_tensors(
    expect_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
    actual_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, expect_should_compare, expect), (
        actual_name,
        actual_should_compare,
        actual,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"
        name = expect_name
        should_compare = expect_should_compare

        expect = expect.cuda()
        actual = actual.cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            msg = _mismatch_message(name, expect, actual, get_tensor_info)
            (error_messages if should_compare else info_messages).append(msg)

    logger.info(f"[check_tensors] equal tensors: {good_names}")
    if len(info_messages) > 0:
        logger.info(f"[check_tensors] info: {info_messages}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _mismatch_message(name, expect, actual, get_tensor_info):
    msg = f"name={name} "
    try:
        abs_diff = (actual.float() - expect.float()).abs()
        msg += f"max_abs_err={abs_diff.max()} mean_abs_err={abs_diff.mean()} "
    except RuntimeError as e:
        try:
            expect_bytes = expect.view(torch.uint8)
            actual_bytes = actual.view(torch.uint8)
            byte_mismatch = actual_bytes != expect_bytes
            msg += (
                "float_diff_unavailable="
                f"{type(e).__name__}:{str(e).splitlines()[0]} "
                f"byte_mismatch_count={byte_mismatch.sum()} "
                f"byte_numel={byte_mismatch.numel()} "
            )
        except RuntimeError as byte_e:
            msg += (
                "float_and_byte_diff_unavailable="
                f"{type(e).__name__}:{str(e).splitlines()[0]};"
                f"{type(byte_e).__name__}:{str(byte_e).splitlines()[0]} "
            )
    msg += f"{get_tensor_info(expect)=} {get_tensor_info(actual)=} "
    return msg


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    # DSV4: FP4/FP8 native-quant storage (and related exotic dtypes)
    # cannot be written via copy_ from a float32 tensor — PyTorch lacks
    # the cast. Randomise the underlying bytes directly via uint8 view
    # so the post-update compare still detects a mismatched transfer.
    # We fall back to this path for any dtype that isn't a "standard"
    # float (bf16/fp16/fp32/fp64) / int / bool.
    _STANDARD_FP = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    if dtype in _STANDARD_FP:
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    try:
        info = torch.iinfo(dtype)
        return torch.randint(
            low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
        )
    except TypeError:
        # Exotic FP variant (FP4 e2m1, FP8 e4m3/e8m0, etc.) — randomise
        # raw bytes via uint8 view; it preserves the storage layout and
        # divergence detection without needing a float32 cast path.
        rand_bytes = torch.randint(
            low=0, high=256, size=t.view(torch.uint8).shape,
            device=device, dtype=torch.uint8,
        )
        return rand_bytes.view(dtype)


def _postprocess_tensors(
    raw: Dict[str, torch.Tensor],
) -> Iterable[Tuple[str, bool, torch.Tensor]]:
    from sglang.srt.debug_utils.dumper import get_tensor_info

    skip_compare_names = []

    # dequant fp8
    quant_names = [
        name
        for name in raw
        # Match: `something.weight`, `something.experts.w2_weight`
        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw
    ]
    skip_compare_names += quant_names
    for name in quant_names:
        w_q = raw[name]
        w_s = raw[name.replace("weight", "weight_scale_inv")]

        try:
            # TODO this is only needed for Blackwell
            w_s_inverse_transformed = inverse_transform_scale_ue8m0(
                w_s, mn=w_q.shape[-2]
            )
            w_dequant = block_quant_dequant(
                w_q,
                w_s_inverse_transformed,
                # TODO do not hardcode
                block_size=[128, 128],
                dtype=torch.bfloat16,
            )
            yield name, True, w_dequant
        except Exception as e:
            e.add_note(
                f"when handling {name=} {get_tensor_info(w_q)=} {get_tensor_info(w_s)=}"
            )
            raise

    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]

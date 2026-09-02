"""Keep radix entries separate when one OFT adapter changes weights."""

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


def _req(rid, **kwargs):
    return Req(rid, "hello", [1, 2, 3], SamplingParams(), **kwargs)


def test_same_adapter_different_versions_use_different_radix_keys():
    first = _req("r1", adapter_id="adapter", adapter_version=1)
    second = _req("r2", adapter_id="adapter", adapter_version=2)

    assert first.extra_key == "|oft:adapter:v1"
    assert second.extra_key == "|oft:adapter:v2"
    assert first.extra_key != second.extra_key


def test_base_request_has_no_oft_radix_key():
    request = _req("r1")

    assert request.extra_key is None

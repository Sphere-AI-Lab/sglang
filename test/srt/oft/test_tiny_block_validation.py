import pytest

from sglang.srt.oft.oft_config import OFTConfig
from sglang.srt.oft.utils import validate_oft_block_size


@pytest.mark.parametrize("block_size", [4, 8, 16, 1024])
def test_power_of_two_block_sizes_are_valid(block_size):
    assert validate_oft_block_size(block_size) == block_size


@pytest.mark.parametrize("block_size", [1, 2, 3, 6, 12])
def test_unsupported_block_sizes_fail_before_triton(block_size):
    with pytest.raises(ValueError, match="power of two.*at least 4"):
        validate_oft_block_size(block_size)


@pytest.mark.parametrize("block_size", [True, 4.0, "4"])
def test_block_size_must_be_an_integer(block_size):
    with pytest.raises(TypeError, match="must be an integer"):
        validate_oft_block_size(block_size)


def test_zero_is_only_allowed_as_an_explicit_runtime_sentinel():
    with pytest.raises(ValueError, match="power of two.*at least 4"):
        validate_oft_block_size(0)
    assert validate_oft_block_size(0, allow_zero=True) == 0


def test_adapter_config_accepts_bs4_and_rejects_bs2():
    base = {"target_modules": ["q_proj"], "oft_block_size": 4}
    assert OFTConfig.from_dict(base).block_size == 4
    with pytest.raises(ValueError, match="power of two.*at least 4"):
        OFTConfig.from_dict({**base, "oft_block_size": 2})

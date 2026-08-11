import importlib.util
from pathlib import Path


_HELPER_PATH = Path(__file__).with_name("tiny_block_benchmark_report.py")
_SPEC = importlib.util.spec_from_file_location("tiny_block_benchmark_report", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
try:
    _SPEC.loader.exec_module(_HELPER)
except FileNotFoundError:
    _HELPER = None


def test_relative_to_bs16_is_partitioned_by_shape_mode_and_batch():
    assert _HELPER is not None, "tiny-block benchmark report helper is missing"
    rows = [
        {
            "shape": "llama31-8b-tp2-qkv",
            "mode": "rotate",
            "M": 8,
            "BS": 4,
            "ms": 1.5,
        },
        {
            "shape": "llama31-8b-tp2-qkv",
            "mode": "rotate",
            "M": 8,
            "BS": 8,
            "ms": 1.2,
        },
        {
            "shape": "llama31-8b-tp2-qkv",
            "mode": "rotate",
            "M": 8,
            "BS": 16,
            "ms": 1.0,
        },
        {
            "shape": "llama31-8b-tp2-qkv",
            "mode": "identity",
            "M": 8,
            "BS": 4,
            "ms": 0.8,
        },
        {
            "shape": "llama31-8b-tp2-qkv",
            "mode": "identity",
            "M": 8,
            "BS": 16,
            "ms": 1.0,
        },
    ]

    enriched = _HELPER.relative_to_bs16(rows)

    by_key = {(row["mode"], row["BS"]): row for row in enriched}
    assert by_key[("rotate", 4)]["vs_bs16"] == 1.5
    assert by_key[("rotate", 8)]["vs_bs16"] == 1.2
    assert by_key[("rotate", 16)]["vs_bs16"] == 1.0
    assert by_key[("identity", 4)]["vs_bs16"] == 0.8

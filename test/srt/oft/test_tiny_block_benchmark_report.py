import importlib.util
from pathlib import Path

import pytest


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


def _row(bs, ms, *, m=8, shape="llama", legacy_ms=1.0):
    return {
        "shape": shape,
        "mode": "rotate",
        "M": m,
        "BS": bs,
        "ms": ms,
        "legacy_ms": legacy_ms,
    }


def test_median_rows_uses_the_median_for_each_benchmark_key():
    runs = [
        [_row(4, 1.0), _row(8, 2.0)],
        [_row(4, 3.0), _row(8, 4.0)],
        [_row(4, 2.0), _row(8, 3.0)],
    ]
    rows = _HELPER.median_rows(runs)
    by_bs = {row["BS"]: row for row in rows}
    assert by_bs[4]["ms"] == 2.0
    assert by_bs[8]["ms"] == 3.0


def test_tiny_acceptance_requires_30_percent_geomean_improvement():
    baseline = [[_row(4, 1.0), _row(8, 1.0), _row(16, 1.0)]] * 3
    current = [[_row(4, 0.7), _row(8, 0.7), _row(16, 1.0)]] * 3
    report = _HELPER.tiny_acceptance_report(baseline, current)
    assert report["passed"]
    assert report["tiny_geomean_ratio"] == pytest.approx(0.7)
    assert report["tiny_improvement"] == pytest.approx(0.3)


def test_tiny_acceptance_rejects_one_slow_tiny_row():
    baseline = [[_row(4, 1.0), _row(8, 1.0), _row(16, 1.0)]] * 3
    current = [[_row(4, 0.3), _row(8, 1.11), _row(16, 1.0)]] * 3
    report = _HELPER.tiny_acceptance_report(baseline, current)
    assert not report["passed"]
    assert any("BS8" in failure and "10%" in failure for failure in report["failures"])


def test_tiny_acceptance_rejects_bs16_regression():
    baseline = [[_row(4, 1.0), _row(8, 1.0), _row(16, 1.0)]] * 3
    current = [[_row(4, 0.5), _row(8, 0.5), _row(16, 1.051)]] * 3
    report = _HELPER.tiny_acceptance_report(baseline, current)
    assert not report["passed"]
    assert any("BS16" in failure and "5%" in failure for failure in report["failures"])


def test_block_size_regressions_checks_every_row_at_or_above_floor():
    baseline = [[_row(16, 1.0), _row(32, 2.0)]] * 3
    current = [[_row(16, 1.04), _row(32, 2.12)]] * 3
    failures = _HELPER.block_size_regressions(
        baseline, current, min_bs=16, tolerance=0.05
    )
    assert len(failures) == 1
    assert "BS32" in failures[0]

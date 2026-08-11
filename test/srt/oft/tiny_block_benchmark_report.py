"""Pure reporting helpers shared by OFT GPU benchmarks."""

from __future__ import annotations

import math
import statistics


def _key(row: dict) -> tuple[str, str, int, int]:
    return row["shape"], row["mode"], row["M"], row["BS"]


def median_rows(runs: list[list[dict]]) -> list[dict]:
    """Collapse repeated benchmark runs to one median row per workload."""
    grouped: dict[tuple[str, str, int, int], list[dict]] = {}
    for run in runs:
        for row in run:
            grouped.setdefault(_key(row), []).append(row)

    result = []
    for key in sorted(grouped):
        copies = grouped[key]
        row = dict(copies[0])
        for field in ("ms", "legacy_ms"):
            values = [copy[field] for copy in copies if copy.get(field) is not None]
            row[field] = statistics.median(values) if values else None
        result.append(row)
    return result


def tiny_acceptance_report(
    baseline_runs: list[list[dict]],
    current_runs: list[list[dict]],
) -> dict:
    """Evaluate the BS4/8 speedup and the focused BS16 regression guard."""
    baseline = {_key(row): row for row in median_rows(baseline_runs)}
    current = {_key(row): row for row in median_rows(current_runs)}
    tiny_ratios = []
    failures = []
    beats_unfused = True

    for key, base in baseline.items():
        shape, mode, batch, bs = key
        if mode != "rotate" or batch not in {1, 8, 32, 64} or bs not in {4, 8, 16}:
            continue
        now = current.get(key)
        if now is None or now.get("ms") is None or base.get("ms") is None:
            failures.append(f"missing row shape={shape} M={batch} BS{bs}")
            continue
        ratio = now["ms"] / base["ms"]
        if bs in {4, 8}:
            tiny_ratios.append(ratio)
            if ratio > 1.10:
                failures.append(
                    f"BS{bs} shape={shape} M={batch} exceeds 10% slowdown"
                )
            legacy_ms = now.get("legacy_ms")
            beats_unfused &= legacy_ms is not None and now["ms"] < legacy_ms
        elif ratio > 1.05:
            failures.append(f"BS16 shape={shape} M={batch} exceeds 5% slowdown")

    geomean = (
        math.exp(sum(math.log(ratio) for ratio in tiny_ratios) / len(tiny_ratios))
        if tiny_ratios
        else math.inf
    )
    if geomean - 0.70 > 1e-12:
        failures.append(f"tiny geometric-mean ratio {geomean:.3f} exceeds 0.700")
    return {
        "passed": not failures,
        "tiny_geomean_ratio": geomean,
        "tiny_improvement": 1.0 - geomean,
        "beats_unfused": beats_unfused,
        "failures": failures,
    }


def block_size_regressions(
    baseline_runs: list[list[dict]],
    current_runs: list[list[dict]],
    *,
    min_bs: int,
    tolerance: float,
) -> list[str]:
    """Report rows at or above ``min_bs`` that exceed ``tolerance``."""
    baseline = {_key(row): row for row in median_rows(baseline_runs)}
    current = {_key(row): row for row in median_rows(current_runs)}
    failures = []
    for key, base in baseline.items():
        shape, mode, batch, bs = key
        if bs < min_bs or base.get("ms") is None:
            continue
        now = current.get(key)
        if now is None or now.get("ms") is None:
            failures.append(f"missing row shape={shape} mode={mode} M={batch} BS{bs}")
            continue
        ratio = now["ms"] / base["ms"]
        if ratio > 1.0 + tolerance:
            failures.append(
                f"BS{bs} shape={shape} mode={mode} M={batch} "
                f"is {(ratio - 1.0) * 100:.1f}% slower"
            )
    return failures


def relative_to_bs16(rows: list[dict]) -> list[dict]:
    """Add latency ratios relative to matching BS16 benchmark rows."""
    baselines = {
        (row["shape"], row["mode"], row["M"]): row["ms"]
        for row in rows
        if row["BS"] == 16 and row.get("ms") is not None
    }
    enriched = []
    for row in rows:
        copy = dict(row)
        base = baselines.get((row["shape"], row["mode"], row["M"]))
        copy["vs_bs16"] = (
            None
            if base is None or row.get("ms") is None
            else row["ms"] / base
        )
        enriched.append(copy)
    return enriched

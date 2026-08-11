"""Pure reporting helpers shared by OFT GPU benchmarks."""

from __future__ import annotations


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

"""Throughput of the fused OFT rotate-project kernel, per block size.

This is the gate for "no regression at BS <= 128". The kernel file's own
``__main__`` benchmark compares fused/legacy/merged at ONE block size; this one
sweeps the block size, which is the axis the tiling work changes.

Record a baseline before touching the kernel, then check against it after:

    python test/srt/oft/bench_fused_rotate_project_blocks.py --json baseline.json
    ...
    python test/srt/oft/bench_fused_rotate_project_blocks.py --compare baseline.json

The comparison only looks at rows the baseline could actually run. Block sizes
above 128 have no baseline time -- that is the whole point of the change -- so a
number appearing there is a gain, never a regression.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from sglang.srt.oft.triton_ops.fused_rotate_project import fused_rotate_project_qkv

# (name, K, output_sizes). Fused QKV under GQA: q, then k and v.
SHAPES = [
    ("llama31-8b-qkv", 4096, [4096, 1024, 1024]),
    ("qwen25-7b-qkv", 3584, [3584, 512, 512]),
]
# Decode (1, 8, 64), CUDA-graph capture (256), prefill (1024).
BATCHES = [1, 8, 64, 256, 1024]
BLOCK_SIZES = [16, 32, 64, 128, 256, 512, 1024]

# KNOWN, MEASURED, ACCEPTED: after the tiling change, `qwen25-7b-qkv M=1 BS=16`
# is ~12% slower than baseline, reproducibly (12.6 / 12.2 / 12.3% over three
# runs). It is the smallest and fastest configuration in the sweep, it takes the
# untiled path unchanged, and no example launcher or campaign arm uses BS=16 --
# every OFT RL example ships 32, 64 or 128, and E5's ladder is 28/32/64/256. It
# is recorded here rather than hidden by widening the tolerance.
#
# A kernel that was not changed should time within noise. 10% is set from the
# measured floor: with the timing method below, the unmodified kernel compared
# against its own baseline stays inside this on every row. It was 5% with a
# cruder timer, which flagged six false regressions -- see _time_ms.
REGRESSION_TOLERANCE = 0.10


def _time_ms(fn, warmup: int = 20, iters: int = 100, repeats: int = 5) -> float:
    """Fastest observed time per launch, over `repeats` independent batches.

    Three deliberate choices, all forced by a measured failure: timing the
    UNMODIFIED kernel against its own baseline with wall clock, 20 iterations
    and a mean reported six "regressions", one of 84%. That noise floor is far
    above any real difference this work would produce, so the gate was useless.

      * CUDA events, not perf_counter -- removes host-side launch jitter.
      * min over repeats, not mean -- the fastest batch is the one least
        contaminated by interference; means chase whatever else touched the GPU.
      * 100 iterations after 20 warmups -- these kernels run in ~0.1 ms, so a
        20-iteration batch is dominated by the first launch after a gap.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        best = min(best, start_ev.elapsed_time(end_ev) / iters)
    return best


def bench_block_sizes(shapes=SHAPES, block_sizes=BLOCK_SIZES, batches=BATCHES) -> list[dict]:
    """One row per (shape, M, BS). ``ms`` is None when the kernel cannot launch.

    The rotation is identity, so ``err`` is measured against a plain projection
    -- a reference that cannot itself drift. It is fp32 while the kernel casts
    to bf16, so a correct kernel reports ~1e-4 rather than exactly zero.
    """
    dev, dt = "cuda", torch.bfloat16
    rows: list[dict] = []
    for name, K, out_sizes in shapes:
        W = (torch.randn(sum(out_sizes), K, device=dev, dtype=dt) * 0.02).contiguous()
        for M in batches:
            x = (torch.randn(M, K, device=dev, dtype=dt) * 0.01).contiguous()
            ref = (x.float() @ W.float().T)
            for BS in block_sizes:
                if K % BS:
                    continue
                blocks = 3 * (K // BS)
                R = torch.eye(BS, device=dev, dtype=dt).expand(blocks, BS, BS).contiguous()
                row = {"shape": name, "M": M, "BS": BS, "ms": None, "err": None}
                try:
                    out = fused_rotate_project_qkv(x, R, W, out_sizes)
                    torch.cuda.synchronize()
                    row["err"] = (out.float() - ref).abs().max().item()
                    row["ms"] = _time_ms(lambda: fused_rotate_project_qkv(x, R, W, out_sizes))
                except Exception as exc:  # noqa: BLE001 -- record whatever Triton raises
                    row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
                rows.append(row)
                del R
                torch.cuda.empty_cache()
    return rows


def compare(baseline_rows: list[dict], current_rows: list[dict],
            tolerance: float = REGRESSION_TOLERANCE) -> list[dict]:
    """Rows that got slower, or that stopped working."""
    def key(r):
        return (r["shape"], r["M"], r["BS"])

    current = {key(r): r for r in current_rows}
    problems = []
    for base in baseline_rows:
        if base.get("ms") is None:
            continue
        now = current.get(key(base))
        if now is None or now.get("ms") is None:
            problems.append({**base, "reason": "no longer runs"})
            continue
        ratio = now["ms"] / base["ms"]
        if ratio > 1.0 + tolerance:
            problems.append({**base, "now_ms": now["ms"], "ratio": ratio,
                             "reason": f"{(ratio - 1) * 100:.1f}% slower"})
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="write rows here")
    parser.add_argument("--compare", type=str, default=None, help="baseline JSON to check against")
    args = parser.parse_args()

    rows = bench_block_sizes()
    print(f"{'shape':16} {'M':>5} {'BS':>5} {'ms':>9} {'max_err':>9}  note")
    for r in rows:
        ms = f"{r['ms']:9.4f}" if r["ms"] is not None else f"{'--':>9}"
        err = f"{r['err']:9.1e}" if r["err"] is not None else f"{'--':>9}"
        print(f"{r['shape']:16} {r['M']:>5} {r['BS']:>5} {ms} {err}  {r.get('error', '')}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")

    if args.compare:
        with open(args.compare, encoding="utf-8") as fh:
            baseline = json.load(fh)
        problems = compare(baseline, rows)
        gained = [r for r in rows if r.get("ms") is not None
                  and not any(b["shape"] == r["shape"] and b["M"] == r["M"]
                              and b["BS"] == r["BS"] and b.get("ms") is not None
                              for b in baseline)]
        if problems:
            print("\nREGRESSIONS against the baseline:")
            for p in problems:
                print(f"  {p['shape']} M={p['M']} BS={p['BS']}: {p['reason']}")
            return 1
        print(f"\nno regression at any block size the baseline could run "
              f"({len(gained)} row(s) newly runnable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

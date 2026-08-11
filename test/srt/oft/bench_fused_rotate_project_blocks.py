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
import torch.nn.functional as F

from sglang.srt.oft.triton_ops.gemm_oft_r import gemm_oft_r_fwd
from sglang.srt.oft.triton_ops.fused_rotate_project import fused_rotate_project_qkv

from tiny_block_benchmark_report import relative_to_bs16

# (name, K, output_sizes). Fused QKV under GQA: q, then k and v.
SHAPES = [
    ("llama31-8b-tp2-qkv", 4096, [2048, 512, 512]),
    ("qwen25-7b-qkv", 3584, [3584, 512, 512]),
]
# Decode, CUDA-graph capture, and prefill sizes.
BATCHES = [1, 8, 32, 64, 256, 1024]
BLOCK_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
MODES = ["rotate", "identity"]

# KNOWN, MEASURED, ACCEPTED: after the tiling change, `qwen25-7b-qkv` at BS=16
# times 12-19% above baseline, reproducibly, at M=1 or M=8 depending on the run.
# It is the smallest and fastest configuration in the sweep -- 0.09 ms, 224
# rotation blocks per slice, so it is loop-overhead bound -- and it takes the
# untiled path unchanged. No example launcher or campaign arm uses BS=16: every
# OFT RL example ships 32, 64 or 128, and E5's ladder is 28/32/64/256. Recorded
# here rather than hidden by widening the tolerance.
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


def _reference_qkv(x, R, W, output_sizes, mode):
    if mode == "identity":
        rotated = [x.float()] * len(output_sizes)
    else:
        blocks_per_slice = x.shape[1] // R.shape[-1]
        x_blocks = x.reshape(x.shape[0], blocks_per_slice, R.shape[-1]).float()
        rotated = [
            torch.einsum(
                "mbk,bkc->mbc",
                x_blocks,
                R[0, slice_idx * blocks_per_slice : (slice_idx + 1) * blocks_per_slice].float(),
            ).reshape_as(x.float())
            for slice_idx in range(len(output_sizes))
        ]
    refs = []
    output_offset = 0
    for x_slice, output_size in zip(rotated, output_sizes, strict=True):
        refs.append(x_slice @ W[output_offset : output_offset + output_size].float().T)
        output_offset += output_size
    return torch.cat(refs, dim=-1)


def _legacy_qkv(x, R, W, output_sizes, slot_idx_t, bsv_t):
    """Production unfused path: rotate, project each slice, then concatenate."""
    K = x.shape[-1]
    rotated = gemm_oft_r_fwd(
        x, R, slot_idx_t, bsv_t, num_slices=len(output_sizes)
    )
    input_slices = torch.split(rotated, K, dim=-1)
    weight_slices = torch.split(W, output_sizes, dim=0)
    return torch.cat(
        [
            F.linear(input_slice, weight_slice)
            for input_slice, weight_slice in zip(
                input_slices, weight_slices, strict=True
            )
        ],
        dim=-1,
    )


def bench_block_sizes(
    shapes=SHAPES,
    block_sizes=BLOCK_SIZES,
    batches=BATCHES,
    modes=MODES,
) -> list[dict]:
    """One row per (shape, mode, M, BS), including compile and steady-state time."""
    dev, dt = "cuda", torch.bfloat16
    rows: list[dict] = []
    for name, K, out_sizes in shapes:
        W = (torch.randn(sum(out_sizes), K, device=dev, dtype=dt) * 0.02).contiguous()
        for M in batches:
            x = (torch.randn(M, K, device=dev, dtype=dt) * 0.01).contiguous()
            for BS in block_sizes:
                if K % BS:
                    continue
                blocks = 3 * (K // BS)
                eye = torch.eye(BS, device=dev, dtype=dt)
                R = eye.expand(1, blocks, BS, BS).clone()
                R.add_(torch.randn_like(R) * 0.005)
                slot_idx_t = torch.zeros((), dtype=torch.int32, device=dev)
                bsv_t = torch.empty((), dtype=torch.int32, device=dev)
                for mode in modes:
                    bsv_t.fill_(BS if mode == "rotate" else 0)
                    ref = _reference_qkv(x, R, W, out_sizes, mode)
                    row = {
                        "shape": name,
                        "mode": mode,
                        "M": M,
                        "BS": BS,
                        "compile_ms": None,
                        "ms": None,
                        "err": None,
                        "legacy_ms": None,
                        "fused_vs_legacy": None,
                    }
                    try:
                        start = time.perf_counter()
                        out = fused_rotate_project_qkv(
                            x,
                            R,
                            W,
                            out_sizes,
                            slot_idx_t=slot_idx_t,
                            bsv_t=bsv_t,
                        )
                        torch.cuda.synchronize()
                        row["compile_ms"] = (time.perf_counter() - start) * 1000.0
                        row["err"] = (out.float() - ref).abs().max().item()
                        row["ms"] = _time_ms(
                            lambda: fused_rotate_project_qkv(
                                x,
                                R,
                                W,
                                out_sizes,
                                slot_idx_t=slot_idx_t,
                                bsv_t=bsv_t,
                            )
                        )
                        if BS <= 16:
                            row["legacy_ms"] = _time_ms(
                                lambda: _legacy_qkv(
                                    x,
                                    R,
                                    W,
                                    out_sizes,
                                    slot_idx_t,
                                    bsv_t,
                                )
                            )
                            row["fused_vs_legacy"] = (
                                row["ms"] / row["legacy_ms"]
                            )
                    except Exception as exc:  # noqa: BLE001 -- retain unsupported rows
                        row["error"] = (
                            f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                        )
                    rows.append(row)
                del R
                torch.cuda.empty_cache()
    return relative_to_bs16(rows)


def compare(baseline_rows: list[dict], current_rows: list[dict],
            tolerance: float = REGRESSION_TOLERANCE) -> list[dict]:
    """Rows that got slower, or that stopped working."""
    def key(r):
        return (r["shape"], r["mode"], r["M"], r["BS"])

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
    print(
        f"{'shape':20} {'mode':>8} {'M':>5} {'BS':>5} "
        f"{'compile_ms':>11} {'ms':>9} {'vs16':>7} {'max_err':>9}  note"
    )
    for r in rows:
        compile_ms = (
            f"{r['compile_ms']:11.2f}" if r["compile_ms"] is not None else f"{'--':>11}"
        )
        ms = f"{r['ms']:9.4f}" if r["ms"] is not None else f"{'--':>9}"
        ratio = f"{r['vs_bs16']:7.3f}" if r["vs_bs16"] is not None else f"{'--':>7}"
        err = f"{r['err']:9.1e}" if r["err"] is not None else f"{'--':>9}"
        print(
            f"{r['shape']:20} {r['mode']:>8} {r['M']:>5} {r['BS']:>5} "
            f"{compile_ms} {ms} {ratio} {err}  {r.get('error', '')}"
        )

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
                              and b.get("mode", "rotate") == r["mode"]
                              and b["BS"] == r["BS"] and b.get("ms") is not None
                              for b in baseline)]
        if problems:
            print("\nREGRESSIONS against the baseline:")
            for p in problems:
                print(
                    f"  {p['shape']} mode={p['mode']} M={p['M']} "
                    f"BS={p['BS']}: {p['reason']}"
                )
            return 1
        print(f"\nno regression at any block size the baseline could run "
              f"({len(gained)} row(s) newly runnable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

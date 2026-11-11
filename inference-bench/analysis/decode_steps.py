"""Split an nsys trace (exported to SQLite) into engine steps and break decode steps down.

Method (one rank's trace_rank<r>.sqlite, table CUPTI_ACTIVITY_KIND_KERNEL):
  1. Step boundaries: `_fused_metadata_kernel`, which the MoE dispatcher launches once per step.
     A step is [start of one metadata kernel, start of the next). The boundary may sit a few
     kernels into the forward pass; that shifts every step equally, so periods are exact.
  2. A step must contain exactly 2 x MOE_LAYERS `_fused_moe_kernel` launches (one forward).
  3. Classification by the routed-expert GEMM duration: prefill if any > PREFILL_US, "idle" (no or
     almost no tokens on this rank, e.g. after its requests finished) if the median < IDLE_US,
     decode otherwise.
  4. Per decode step: period (next boundary - this one), GPU busy (union of kernel intervals over
     all streams, so overlapped streams are not double-counted), idle = period - busy, and the
     summed kernel time per category (real per-step sums, not medians).
  5. Stalls: decode steps whose period exceeds STALL_X x the median period; the step's first
     collective usually absorbs the wait for a late rank.

Export first (in a container with nsys):
    nsys export --type sqlite -o <run>/trace_rank4.sqlite <run>/trace_rank4.nsys-rep
or `nsys stats --force-export=true ...`, which writes the same SQLite next to the report.

    python3 inference-bench/analysis/decode_steps.py \
        tp4=inference-bench/experiments/<run_a> tp8=inference-bench/experiments/<run_b> [--ranks 4 44]

All times are printed in ms (per step) or us (per kernel).
"""

import argparse
import os
import sqlite3
import statistics as st
from collections import defaultdict

from kernel_breakdown import CATEGORIES, categorize

MOE_LAYERS = 48  # Ultra v3
PREFILL_US = 100.0
IDLE_US = 10.0
STALL_X = 1.5


def load_kernels(path):
    """[(name, start_ns, end_ns)] sorted by start."""
    con = sqlite3.connect(path)
    rows = con.execute(
        "select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on k.shortName = s.id order by k.start"
    ).fetchall()
    con.close()
    return rows


def busy_ns(intervals):
    """Length of the union of [start, end) intervals."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def split_steps(kernels):
    """List of dicts, one per complete step, in time order."""
    bounds = [i for i, (n, _, _) in enumerate(kernels) if n == "_fused_metadata_kernel"]
    t0 = kernels[0][1]
    steps = []
    for a, b in zip(bounds, bounds[1:]):
        ks = kernels[a:b]
        moe = [(e - s) / 1000 for n, s, e in ks if n == "_fused_moe_kernel"]
        if len(moe) != 2 * MOE_LAYERS:
            kind = "partial"
        elif max(moe) > PREFILL_US:
            kind = "prefill"
        elif st.median(moe) < IDLE_US:
            kind = "idle"
        else:
            kind = "decode"
        period_ms = (kernels[b][1] - kernels[a][1]) / 1e6
        cats = defaultdict(float)
        for n, s, e in ks:
            cats[categorize(n)] += (e - s) / 1000
        steps.append({
            "index": len(steps),
            "at_ms": (kernels[a][1] - t0) / 1e6,
            "kind": kind,
            "period_ms": period_ms,
            "busy_ms": busy_ns([(s, e) for _, s, e in ks]) / 1e6,
            "first_collective_us": (ks[0][2] - ks[0][1]) / 1000,
            "moe_us_max": max(moe) if moe else 0.0,
            "cats_us": cats,
        })
    return steps


def summarize(label, run, rank):
    path = os.path.join(run, f"trace_rank{rank}.sqlite")
    steps = split_steps(load_kernels(path))
    kinds = defaultdict(list)
    for s in steps:
        kinds[s["kind"]].append(s)
    dec = kinds["decode"]
    med_period = st.median(s["period_ms"] for s in dec)
    stalls = [s for s in dec if s["period_ms"] > STALL_X * med_period]
    clean = [s for s in dec if s not in stalls]
    print(f"\n=== {label} rank {rank}: {len(steps)} steps "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(kinds.items())))
    q = lambda xs: st.quantiles(xs, n=10) if len(xs) > 1 else [xs[0]] * 9
    per = [s["period_ms"] for s in clean]
    bus = [s["busy_ms"] for s in clean]
    print(f"  decode period (excl. stalls): p10 {q(per)[0]:.2f}  p50 {st.median(per):.2f}  "
          f"p90 {q(per)[8]:.2f} ms | GPU busy p50 {st.median(bus):.2f} ms | "
          f"idle p50 {st.median([p - b for p, b in zip(per, bus)]):.2f} ms")
    for s in kinds["prefill"]:
        print(f"  prefill step {s['index']} @ {s['at_ms']:.0f} ms: period {s['period_ms']:.1f} ms")
    stall_ms = sum(s["period_ms"] - med_period for s in stalls)
    print(f"  stalls: {len(stalls)} decode steps > {STALL_X}x median, {stall_ms:.0f} ms extra in total")
    for s in stalls:
        print(f"    step {s['index']} @ {s['at_ms']:.0f} ms: period {s['period_ms']:.1f} ms, "
              f"first collective {s['first_collective_us']:.0f} us")
    cat_med = {c: st.median(s["cats_us"].get(c, 0.0) for s in clean) for c in
               [c for c, _ in CATEGORIES] + ["Elementwise / other"]}
    return {"period": st.median(per), "busy": st.median(bus), "cats": cat_med,
            "stall_ms": stall_ms, "n_stalls": len(stalls)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="label=run_dir")
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 44])
    p.add_argument("--moe-layers", type=int, default=MOE_LAYERS, help="48 Ultra v3, 23 nano-3.5")
    args = p.parse_args()
    globals()["MOE_LAYERS"] = args.moe_layers

    results = {}
    for spec in args.runs:
        label, run = spec.split("=", 1)
        per_rank = [summarize(label, run, r) for r in args.ranks]
        results[label] = {
            "period": st.mean(r["period"] for r in per_rank),
            "busy": st.mean(r["busy"] for r in per_rank),
            "cats": {c: st.mean(r["cats"][c] for r in per_rank) for c in per_rank[0]["cats"]},
        }

    labels = list(results)
    two = len(labels) == 2
    print(f"\nTypical decode step (median over clean decode steps, mean of ranks "
          f"{', '.join(map(str, args.ranks))}); kernel time summed per step, in us")
    print("| | " + " | ".join(labels) + (" | Delta |" if two else " |"))
    print("|---|" + "---|" * len(labels) + ("---|" if two else ""))
    for c in results[labels[0]]["cats"]:
        v = [results[l]["cats"][c] for l in labels]
        print(f"| {c} | " + " | ".join(f"{x:.0f}" for x in v) + (f" | {v[1] - v[0]:+.0f} |" if two else " |"))
    for key, name in (("busy", "GPU busy (ms)"), ("period", "Step period (ms)")):
        v = [results[l][key] for l in labels]
        print(f"| **{name}** | " + " | ".join(f"{x:.2f}" for x in v)
              + (f" | {v[1] - v[0]:+.2f} |" if two else " |"))
    print("\nCategory sums can exceed GPU busy: kernels on the shared-expert stream overlap the main stream.")


if __name__ == "__main__":
    main()

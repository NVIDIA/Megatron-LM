"""Per-concurrency decode step time and kernel breakdown from a client --profile-all sweep trace.

The client (server/client.py --profile-all --pause N) profiles every batch size's timed batch in
one capture, idling N seconds between batch sizes. This splits one rank's trace (SQLite export)
at GPU gaps longer than --gap seconds, one segment per batch size in sweep order, then:
  steps    boundaries at `_fused_metadata_kernel` (launched once per step by the NVLS dispatcher)
  decode   steps whose period is <= STALL_X x the segment's median period (drops the eager
           prefill steps and stalls; a batch is ~1 prefill + OSL decode steps)
  report   median decode period, MoE layers per step (`_sigmoid_topk_kernel` launches), and the
           mean per-step kernel time per category (kernel_breakdown.CATEGORIES) over decode steps

Export first (mcore container): nsys export --type sqlite -o <run>/trace_rank0.sqlite <run>/trace_rank0.nsys-rep

    python3 inference-bench/analysis/sweep_steps.py vllm=<run_a> trtllm=<run_b> [--rank 0] [--top 8]
"""

import argparse
import bisect
import os
import re
import sqlite3
import statistics as st
from collections import defaultdict

from kernel_breakdown import CATEGORIES, categorize, short

STALL_X = 1.5


def load(run, rank):
    con = sqlite3.connect(f"file:{os.path.join(run, f'trace_rank{rank}.sqlite')}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store=MEMORY")
    rows = con.execute(
        "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName order by k.start"
    ).fetchall()
    return rows


def segments(rows, gap_ns):
    segs, cur, last_end = [], [], None
    for r in rows:
        if last_end is not None and r[0] - last_end > gap_ns:
            segs.append(cur)
            cur = []
        cur.append(r)
        last_end = r[1] if last_end is None else max(last_end, r[1])
    if cur:
        segs.append(cur)
    return segs


def analyze(seg):
    bounds = [r[0] for r in seg if r[2] == "_fused_metadata_kernel"]
    if len(bounds) < 3:
        return None
    periods = [b - a for a, b in zip(bounds, bounds[1:])]
    med = st.median(periods)
    decode = [i for i, p in enumerate(periods) if p <= STALL_X * med]
    starts = [r[0] for r in seg]
    cat_us, kern_us, layers = defaultdict(float), defaultdict(float), []
    for i in decode:
        lo, hi = bisect.bisect_left(starts, bounds[i]), bisect.bisect_left(starts, bounds[i + 1])
        n_router = 0
        for s, e, name in seg[lo:hi]:
            cat_us[categorize(name)] += (e - s) / 1e3
            kern_us[short(name)] += (e - s) / 1e3
            n_router += name == "_sigmoid_topk_kernel"
        layers.append(n_router)
    n = len(decode)
    return {
        "steps": len(periods),
        "decode": n,
        "period_ms": st.median(periods[i] for i in decode) / 1e6,
        "layers": st.median(layers),
        "cat": {k: v / n for k, v in cat_us.items()},
        "kern": {k: v / n for k, v in kern_us.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="label=run_dir")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--gap", type=float, default=3.0, help="seconds of GPU idle that split segments")
    p.add_argument("--batch-sizes", type=int, nargs="*", default=None,
                   help="sweep order (default: BENCH from the run's config.env)")
    p.add_argument("--top", type=int, default=8, help="largest kernels to list per segment")
    args = p.parse_args()

    results = {}
    for label, run in (s.split("=", 1) for s in args.runs):
        bs = args.batch_sizes
        if bs is None:
            env = open(os.path.join(run, "config.env")).read()
            m = re.findall(r"^BENCH=([\d ]+?) BENCH_DATASET", env, re.M)
            bs = [int(x) for x in m[-1].split()] if m else None
        segs = [a for a in (analyze(s) for s in segments(load(run, args.rank), args.gap * 1e9)) if a]
        if bs and len(segs) != len(bs):
            print(f"{label}: {len(segs)} segments but {len(bs)} batch sizes; labelling by index")
            bs = None
        results[label] = dict(zip(bs or range(len(segs)), segs))

    labels = list(results)
    keys = list(results[labels[0]])
    cats = [c for c, _ in CATEGORIES] + ["Elementwise / other"]
    print(f"rank {args.rank}; decode step period (median) and mean kernel time per decode step\n")
    print(f"{'conc':>5} | " + " | ".join(f"{l + ' ms':>12}" for l in labels)
          + (" | delta" if len(labels) == 2 else "") + " | decode steps, MoE layers")
    for k in keys:
        cells = [results[l][k]["period_ms"] if k in results[l] else float("nan") for l in labels]
        delta = f" | {cells[1] - cells[0]:+.3f} ({100 * (cells[1] / cells[0] - 1):+.1f}%)" if len(labels) == 2 else ""
        info = ", ".join(f"{results[l][k]['decode']}/{results[l][k]['steps']} L{results[l][k]['layers']:.0f}"
                         for l in labels if k in results[l])
        print(f"{k:>5} | " + " | ".join(f"{c:12.3f}" for c in cells) + delta + f" | {info}")

    for k in keys:
        print(f"\nconcurrency {k}: us per decode step")
        print(f"  {'category':28s} " + " ".join(f"{l:>12}" for l in labels))
        for c in cats:
            vals = [results[l][k]["cat"].get(c, 0.0) if k in results[l] else 0.0 for l in labels]
            if any(v > 0.5 for v in vals):
                print(f"  {c:28s} " + " ".join(f"{v:12.1f}" for v in vals))
        for l in labels:
            if k in results[l]:
                top = sorted(results[l][k]["kern"].items(), key=lambda x: -x[1])[: args.top]
                print(f"  {l} top: " + ", ".join(f"{n[:40]} {v:.0f}" for n, v in top))


if __name__ == "__main__":
    main()

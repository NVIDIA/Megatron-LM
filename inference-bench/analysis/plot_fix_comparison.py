"""Before / after a fix, one long steady level each: columns = workloads, rows = throughput, clean
decode step, step with a prefill on any EP rank (interfered + prefill), longest active request, mean
and median active-request context.

Optionally (--fa4-results) the clean-step panel also shows the microbenchmark's projection for the
"after" run: before-run clean step - attn_layers x (FA4 time at num_splits=0 - at --splits), with the
FA4 times from microbench/fa4_decode_splits_bench.py on the before-run's KV snapshots (slower DP rank),
interpolated to each window's midpoint.

    python3 inference-bench/analysis/plot_fix_comparison.py \
        --pair math <before dir> <after dir> --pair wildchat <before dir> <after dir> \
        [--fa4-results <results.jsonl> ...] --out <png>
"""

import argparse
import bisect
import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

BEFORE, AFTER, PROJ = "#D55E00", "#0072B2", "#56B4E9"
EXTRA = ["#009E73", "#CC79A7", "#E69F00"]  # further runs (--extra), in this order


def load(d):
    read = lambda f: [json.loads(line) for line in open(os.path.join(d, f))]  # noqa: E731
    requests = read("steady.requests.jsonl")
    t0 = min(r["start"] for r in requests)
    windows = read("steady.jsonl")
    rows = []
    for w in windows:
        h = w["step_time_ms"].get("all", {})
        mixed = [h[c] for c in ("decode_interfered", "prefill") if c in h]
        n = sum(c["count"] for c in mixed)
        rows.append({
            "t": (w["window_start_s"] + w["window_s"] / 2) / 60,
            "tput": w["output_tok_s"] / (64 if "wide" in d else 8),
            "clean": h.get("decode_clean", {}).get("mean_ms", float("nan")),
            "mixed": sum(c["count"] * c["mean_ms"] for c in mixed) / n if n else float("nan"),
        })
    samples = [s for s in read("steady.engines.jsonl") if s["engines"]]
    longest = [((s["t"] - t0) / 60, max(e["max_context_tokens"] for e in s["engines"].values()) / 1e3)
               for s in samples if all("max_context_tokens" in e for e in s["engines"].values())]
    # Mean over all active requests (total KV / running); median from per-request contexts if recorded.
    mean, median = [], []
    for s in samples:
        es = s["engines"].values()
        t = (s["t"] - t0) / 60
        running = sum(e.get("requests_running", 0) for e in es)
        if running and all("kv_tokens" in e for e in es):
            mean.append((t, sum(e["kv_tokens"] for e in es) / running / 1e3))
        ctx = sorted(v for e in es for v in e.get("requests", {}).values())
        if ctx:
            median.append((t, ctx[(len(ctx) - 1) // 2] / 1e3))
    return rows, {"longest": longest, "mean": mean, "median": median}


def fa4_saving_ms(paths, dataset, splits, layers):
    """[(minutes, ms saved per step by `splits` vs the heuristic)] for one dataset."""
    t = collections.defaultdict(float)
    for line in (line for p in paths for line in open(p)):
        r = json.loads(line)
        d, m, _ = r["snapshot"].split("_", 2)
        if d == dataset:
            key = (float(m[:-3]), r["num_splits"])
            t[key] = max(t[key], r["us"])
    mins = sorted({m for m, _ in t})
    return [(m, layers * (t[(m, 0)] - t[(m, splits)]) / 1e3) for m in mins if (m, splits) in t]


def interp(points, x):
    xs = [p[0] for p in points]
    i = bisect.bisect_left(xs, x)
    if i == 0:
        return points[0][1]
    if i == len(xs):
        return points[-1][1]
    (x0, y0), (x1, y1) = points[i - 1], points[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("NAME", "BEFORE", "AFTER"), help='BEFORE may be "none"')
    ap.add_argument("--before-label", default="before: FA4 heuristic (1 split)")
    ap.add_argument("--after-label", default="after: max(heuristic, 8) splits")
    ap.add_argument("--extra", nargs=3, action="append", default=[], metavar=("NAME", "DIR", "LABEL"),
                    help="another run drawn in column NAME (repeatable); not used for the %% labels")
    ap.add_argument("--fa4-results", nargs="*", default=[])
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--attn-layers", type=int, default=8)
    ap.add_argument("--title", default="Super narrow EP, 32 requests per GPU, one level for an hour: "
                    "FA4 decode split-KV fix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(5, len(args.pair), figsize=(max(5.5 * len(args.pair), 9), 13.5), sharex=True,
                             squeeze=False)
    for col, (name, before_dir, after_dir) in enumerate(args.pair):
        # "none" = no before run (yet): draw the after run alone.
        before, before_longest = load(before_dir) if before_dir != "none" else ([], {"longest": [], "mean": [], "median": []})
        after, after_longest = load(after_dir)
        ax_t, ax_c, ax_m, ax_l, ax_s = axes[:, col]
        runs = [(before, before_longest, BEFORE, args.before_label),
                (after, after_longest, AFTER, args.after_label)]
        for i, (_, extra_dir, extra_label) in enumerate(e for e in args.extra if e[0] == name):
            runs.append((*load(extra_dir), EXTRA[i % len(EXTRA)], extra_label))
        for rows, ctx, color, label in runs:
            xs = [r["t"] for r in rows]
            # The "after" run on top of the others.
            style = dict(color=color, marker="o", ms=4, lw=2, zorder=5 if color == AFTER else 2)
            ax_t.plot(xs, [r["tput"] for r in rows], label=label, **style)
            ax_c.plot(xs, [r["clean"] for r in rows], **style)
            ax_m.plot(xs, [r["mixed"] for r in rows], **style)
            ax_l.plot([p[0] for p in ctx["longest"]], [p[1] for p in ctx["longest"]], color=color, lw=1.5,
                      zorder=style["zorder"])
            ax_s.plot([p[0] for p in ctx["mean"]], [p[1] for p in ctx["mean"]], color=color, lw=1.5)
            ax_s.plot([p[0] for p in ctx["median"]], [p[1] for p in ctx["median"]], color=color, lw=1.2,
                      ls="--")
        if args.fa4_results and before:
            saving = fa4_saving_ms(args.fa4_results, name, args.splits, args.attn_layers)
            if saving:
                ax_c.plot([r["t"] for r in before], [r["clean"] - interp(saving, r["t"]) for r in before],
                          color=PROJ, ls="--", lw=1.8, label=f"projected from microbenchmark")
                ax_c.legend(frameon=False, fontsize=8, loc="upper left")
        # Throughput change per window where both runs have it.
        for b, a in zip(before, after):
            ax_t.annotate(f"{a['tput'] / b['tput'] - 1:+.0%}", (a["t"], a["tput"]), textcoords="offset points",
                          xytext=(0, 6), ha="center", fontsize=7, color=AFTER)
        ax_t.set_title(f"{name}" + ("" if before else " (no before run yet)") + "\nThroughput (tokens/s/GPU)",
                       fontsize=9, loc="left")
        ax_c.set_title("Clean decode step (ms): no prefill on any EP rank", fontsize=9, loc="left")
        ax_m.set_title("Step with a prefill on any EP rank (ms)", fontsize=9, loc="left")
        ax_l.set_title("Longest active request (tokens, over DP ranks)", fontsize=9, loc="left")
        ax_s.set_title("Mean (solid) and median (dashed) context over all active requests (tokens)",
                       fontsize=9, loc="left")
        for ax in (ax_l, ax_s):
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}k"))
        ax_s.set_xlabel("Minutes since the level started", fontsize=9)
        for ax in (ax_t, ax_c, ax_m, ax_l, ax_s):
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            ax.yaxis.grid(ls="dotted")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
        ax_t.set_ylim(top=max(r["tput"] for run in runs for r in run[0]) * 1.15)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(args.title, fontsize=10, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()

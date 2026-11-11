"""End-to-end steady-level results against KV skew instead of time: one marker style per run, one point
per measurement window, from steady_client runs with per-request samples (--sample-interval).

x: the window's KV skew = mean over its engine samples of max over DP ranks of
   (longest active context / mean active context) on that rank (the EP group waits for the slower
   rank). Rows: throughput per GPU, clean decode step, step with a prefill on any EP rank.

    python3 inference-bench/analysis/plot_e2e_vs_skew.py --run LABEL DIR [--run LABEL DIR ...] --out <png>
"""

import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#D55E00", "#009E73", "#CC79A7", "#0072B2", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load(d):
    read = lambda f: [json.loads(line) for line in open(os.path.join(d, f))]  # noqa: E731
    gpus = 64 if "wide" in d else 8
    t0 = min(r["start"] for r in read("steady.requests.jsonl"))
    samples = []
    for s in read("steady.engines.jsonl"):
        skews = []
        for e in s["engines"].values():
            ctx = list(e.get("requests", {}).values())
            if len(ctx) >= 2:
                skews.append(max(ctx) * len(ctx) / sum(ctx))
        if skews:
            samples.append((s["t"] - t0, max(skews)))
    points = []
    for w in read("steady.jsonl"):
        a, b = w["window_start_s"], w["window_start_s"] + w["window_s"]
        sk = [k for t, k in samples if a <= t < b]
        if not sk:
            continue
        h = w["step_time_ms"]["all"]
        mixed = [h[c] for c in ("decode_interfered", "prefill") if c in h]
        n = sum(c["count"] for c in mixed)
        points.append({
            "skew": statistics.mean(sk),
            "tput": w["output_tok_s"] / gpus,
            "clean": h.get("decode_clean", {}).get("mean_ms", float("nan")),
            "mixed": sum(c["count"] * c["mean_ms"] for c in mixed) / n if n else float("nan"),
        })
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "DIR"))
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    rows = (("tput", "Throughput (tokens/s/GPU)"), ("clean", "Clean decode step (ms): no prefill on any EP rank"),
            ("mixed", "Step with a prefill on any EP rank (ms)"))
    for i, (label, d) in enumerate(args.run):
        pts = load(d)
        for ax, (key, _) in zip(axes, rows):
            ax.scatter([p["skew"] for p in pts], [p[key] for p in pts], color=COLORS[i % len(COLORS)],
                       marker=MARKERS[i % len(MARKERS)], s=30, alpha=0.8, label=label)
    for ax, (_, title) in zip(axes, rows):
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(ls="dotted")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    ticks = [2, 3, 5, 10, 20, 30]
    axes[-1].set_xticks(ticks, [str(t) for t in ticks])
    axes[-1].minorticks_off()
    axes[-1].set_xlabel("KV skew in the window: longest / mean active context (slower DP rank), "
                        "one point per 5-minute window", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()

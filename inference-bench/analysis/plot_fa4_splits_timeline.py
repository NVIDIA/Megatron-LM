"""FA4 decode attention time over a run, one line per num_splits, from
microbench/fa4_decode_splits_bench.py on KV snapshots named "<dataset>_<minutes>min_<engine>".

Per snapshot time: per-layer kernel time x --attn-layers, taking the slower DP rank (the EP group
runs every MoE layer in lockstep, so the step waits for it). One panel per dataset.

    python3 inference-bench/analysis/plot_fa4_splits_timeline.py <results.jsonl> [...] --out <png>
"""

import argparse
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Heuristic in the accent color; explicit split counts as a light-to-dark sequential ramp.
HEURISTIC_COLOR = "#D55E00"
RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, nargs="*", default=None,
                    help="num_splits values to draw (0 = FA4 heuristic); default all")
    ap.add_argument("--attn-layers", type=int, default=8, help="attention layers per forward (Super: 8)")
    args = ap.parse_args()

    # (dataset, minutes, splits) -> slowest rank's per-layer us
    t = collections.defaultdict(float)
    for line in (line for path in args.results for line in open(path)):
        r = json.loads(line)
        dataset, minutes, _ = r["snapshot"].split("_", 2)
        key = (dataset, float(minutes[:-3]), r["num_splits"])
        t[key] = max(t[key], r["us"])
    datasets = sorted({k[0] for k in t}, reverse=True)
    splits = sorted({k[2] for k in t if args.show is None or k[2] in args.show})

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.2),
                             squeeze=False)
    for ax, d in zip(axes[0], datasets):
        explicit = [s for s in splits if s > 0]
        for s in splits:
            pts = sorted((m, us) for (dd, m, ss), us in t.items() if dd == d and ss == s)
            if s == 0:
                style = dict(color=HEURISTIC_COLOR, lw=2.5, label="1 split (FA4 default: num_splits=0 picks 1)")
            else:
                color = RAMP[len(RAMP) - len(explicit) + explicit.index(s)]
                style = dict(color=color, lw=1.8, label=f"{s} splits")
            ax.plot([m for m, _ in pts], [us * args.attn_layers / 1e3 for _, us in pts],
                    marker="o", ms=3, **style)
        ax.set_title(f"{d}", fontsize=10, loc="left")
        ax.set_xlabel("Minutes since the level started", fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(ls="dotted")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel(f"Decode attention per step, {args.attn_layers} layers (ms)", fontsize=9)
    axes[0][0].legend(title="num_splits", frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle("FA4 decode attention on the recorded KV lengths over the run (Super narrow EP, "
                 "slower of the 2 DP ranks; microbenchmark, L2 flushed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()

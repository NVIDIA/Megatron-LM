"""KV skew (longest / mean active-request context, per DP rank) over a long steady level, from
steady_client's per-request engine samples (--sample-interval). One line per run and DP rank.

    python3 inference-bench/analysis/plot_skew_timeline.py --run LABEL DIR [--run LABEL DIR ...] --out <png>
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#E69F00", "#0072B2", "#009E73", "#CC79A7"]
RANK_STYLES = ["-", "--", ":", "-."]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "DIR"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (label, d) in enumerate(args.run):
        read = lambda f: [json.loads(line) for line in open(os.path.join(d, f))]  # noqa: E731
        t0 = min(r["start"] for r in read("steady.requests.jsonl"))
        series = {}  # engine -> [(minutes, skew)]
        for s in read("steady.engines.jsonl"):
            for engine, e in s["engines"].items():
                ctx = list(e.get("requests", {}).values())
                if len(ctx) >= 2:
                    series.setdefault(engine, []).append(((s["t"] - t0) / 60, max(ctx) * len(ctx) / sum(ctx)))
        for j, (engine, pts) in enumerate(sorted(series.items())):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[i % len(COLORS)],
                    ls=RANK_STYLES[j % len(RANK_STYLES)], lw=1.6,
                    label=f"{label}, DP rank {j}")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(left=0)
    ax.set_xlabel("Minutes since the level started", fontsize=9)
    ax.set_ylabel("KV skew: longest context / mean context", fontsize=9)
    ax.set_title("KV skew per DP rank over an hour (Super narrow EP, 128 requests per rank, "
                 "FA4 heuristic run)", fontsize=9, loc="left")
    ax.yaxis.grid(ls="dotted", color="0.85")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(args.out)


if __name__ == "__main__":
    main()

"""FA4 decode time per attention layer vs num_splits, from microbench/fa4_decode_splits_bench.py: one
panel per KV snapshot (rows: dataset, columns: time into the run, DP rank 1), real batch vs the same
batch at its mean length, with the KV-read floor.

    python3 inference-bench/analysis/plot_fa4_splits.py <results.jsonl> --out <png>
"""

import argparse
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", default="mp-coord-1")
    args = ap.parse_args()
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(args.results):
        r = json.loads(line)
        if r["snapshot"].endswith(args.rank):
            runs[r["snapshot"]][r["variant"]].append(r)
    names = sorted(runs, key=lambda n: (n.split("_")[0], float(n.split("_")[1][:-3])))
    datasets = sorted({n.split("_")[0] for n in names}, reverse=True)
    cols = max(sum(n.startswith(d) for n in names) for d in datasets)
    fig, axes = plt.subplots(len(datasets), cols, figsize=(3.6 * cols, 3.2 * len(datasets)), squeeze=False)
    for row, d in enumerate(datasets):
        for col, n in enumerate([n for n in names if n.startswith(d)]):
            ax = axes[row][col]
            for variant, style in (("real", dict(color="#D55E00", marker="o", lw=2)),
                                   ("balanced", dict(color="#0072B2", marker="s", lw=1.5, ls="--"))):
                rs = sorted(runs[n][variant], key=lambda r: r["num_splits"])
                xs = [max(r["num_splits"], 1) for r in rs]  # 0 = FA4 heuristic, which picks 1 here
                ax.plot(xs, [r["us"] for r in rs], label=f"{variant} batch", ms=4, **style)
            r0 = runs[n]["real"][0]
            ax.axhline(r0["kv_read_floor_us"], color="0.5", lw=1, ls=":", label="KV-read floor (8 TB/s)")
            best = min(runs[n]["real"], key=lambda r: r["us"])
            heur = next(r for r in runs[n]["real"] if r["num_splits"] == 0)
            ax.set_title(f"{d}, {n.split('_')[1].replace('min', ' min')}: longest {r0['max_len'] / 1e3:.0f}k, "
                         f"mean {r0['mean_len'] / 1e3:.1f}k\nheuristic {heur['us']:.0f} us, best "
                         f"{best['us']:.0f} us at {best['num_splits']} splits ({heur['us'] / best['us']:.1f}x)",
                         fontsize=8, loc="left")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["0\n(heur.=1)", "2", "4", "8", "16", "32", "64", "128"])
            ax.set_ylim(bottom=0)
            ax.yaxis.grid(ls="dotted")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if row == len(datasets) - 1:
                ax.set_xlabel("num_splits (0 = FA4 heuristic, which picks 1 here)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Decode attention, one layer (us)", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("FA4 paged decode on recorded KV lengths (Super, TP4 rank: 128 requests, 8 q heads, "
                 "1 KV head, 256-token pages; L2 flushed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()

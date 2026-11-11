"""Decode attention effective bandwidth vs KV skew, one point per recorded per-rank batch and kernel,
from microbench/flashinfer_decode_bench.py results.

x: skew = longest context / mean context in the batch (1 = every request the same length).
   Lines: least-squares fit of y against log10(skew) per kernel and dataset.
y: effective bandwidth = KV bytes read (2 x tokens x kv_heads x head_dim x 2 B) / kernel time,
   with a line at the GPU's peak HBM bandwidth (--peak-tb-s).

    python3 inference-bench/analysis/plot_attention_vs_skew.py <results.jsonl> [...] --out <png>
"""

import argparse
import collections
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KERNELS = (
    ("fa4_1split", "FA4 b20, 1 split (its heuristic)", "#D55E00", "o"),
    ("fa4_8split", "FA4 b20, 8 splits (our floor)", "#0072B2", "s"),
    ("fi_fa2_tc", "FlashInfer decode, tensor cores", "#009E73", "^"),
    ("fa4b38_auto", "FA4 beta38, auto (SwapAB decode)", "black", "D"),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--peak-tb-s", type=float, default=8.0, help="peak HBM bandwidth (GB200: 8 TB/s)")
    args = ap.parse_args()

    runs = collections.defaultdict(dict)  # snapshot -> kernel -> record
    for line in (line for p in args.results for line in open(p)):
        r = json.loads(line)
        if r.get("us") is not None and r["variant"] == "real":
            runs[r["snapshot"]][r["kernel"]] = r

    datasets = ("math", "wildchat")
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4.6), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for kernel, label, color, marker in KERNELS:
            xs, ys = [], []
            for name, k in runs.items():
                if name.startswith(dataset) and kernel in k:
                    r = k[kernel]
                    kv_bytes = 2 * r["mean_len"] * args.batch * args.kv_heads * args.head_dim * 2
                    xs.append(r["max_len"] / r["mean_len"])
                    ys.append(kv_bytes / (r["us"] * 1e-6) / 1e12)
            ax.scatter(xs, ys, s=18, marker=marker, color=color, alpha=0.55, label=label)
            # Least-squares line in log(skew), i.e. straight on this log-x axis.
            if len(xs) > 1:
                lx = np.log10(xs)
                slope, icpt = np.polyfit(lx, ys, 1)
                fx = np.linspace(lx.min(), lx.max(), 50)
                ax.plot(10**fx, slope * fx + icpt, color=color, lw=2, ls=":")
        ax.axhline(args.peak_tb_s, color="black", lw=1.2, ls="--")
        ax.text(0.01, args.peak_tb_s * 1.02, f"peak HBM bandwidth ({args.peak_tb_s:g} TB/s)", fontsize=8,
                va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20) if min(xs) / 1.2 <= t <= max(xs) * 1.2]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, args.peak_tb_s * 1.15)
        ax.set_xlabel("KV skew: longest context / mean context (1 = all requests equal)", fontsize=9)
        ax.set_title(dataset, fontsize=10, loc="left")
        ax.yaxis.grid(ls="dotted", color="0.85")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Effective bandwidth: KV bytes / kernel time (TB/s)", fontsize=9)
    # Legend top to bottom in the order the points sit (fastest kernel first).
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Decode attention vs KV skew (Super TP4 rank: 128 requests, 8 q heads, 1 KV head); "
                 "each point: one recorded batch, every 2 min over an hour, both DP ranks", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(args.out)


if __name__ == "__main__":
    main()

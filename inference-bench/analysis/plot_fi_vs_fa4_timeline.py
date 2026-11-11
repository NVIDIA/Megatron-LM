"""Decode attention per step over a run: FA4 (1 split, 8 splits) vs FlashInfer (tensor-core decode),
from microbench/flashinfer_decode_bench.py on KV snapshots named "<dataset>_<minutes>min_<engine>".

Per snapshot: per-layer kernel time x --attn-layers, slower DP rank. One panel per dataset. The
FlashInfer plan() host time (once per step, outside the kernel) is given in each panel's title.

    python3 inference-bench/analysis/plot_fi_vs_fa4_timeline.py <results.jsonl> [...] --out <png>
"""

import argparse
import collections
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KERNELS = (
    ("fa4_1split", "FA4 b20, 1 split (its heuristic)", dict(color="#D55E00", lw=2.5)),
    ("fa4_8split", "FA4 b20, 8 splits (our floor)", dict(color="#0072B2", lw=2)),
    ("fi_fa2_tc", "FlashInfer decode, tensor cores (plan() load balancing)", dict(color="#009E73", lw=2)),
    ("fa4b38_auto", "FA4 beta38, auto (SwapAB decode, per-request splits)", dict(color="black", lw=2)),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-layers", type=int, default=8)
    args = ap.parse_args()

    t = collections.defaultdict(float)  # (dataset, minutes, kernel) -> slower rank's us
    plan = collections.defaultdict(list)  # dataset -> plan() host us
    for line in (line for p in args.results for line in open(p)):
        r = json.loads(line)
        if r.get("us") is None or r["variant"] != "real":
            continue
        d, m, _ = r["snapshot"].split("_", 2)
        key = (d, float(m[:-3]), r["kernel"])
        t[key] = max(t[key], r["us"])
        if r.get("plan_us"):
            plan[d].append(r["plan_us"])
    datasets = sorted({k[0] for k in t}, reverse=True)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.8 * len(datasets), 4.4), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for kernel, label, style in KERNELS:
            pts = sorted((m, us) for (dd, m, k), us in t.items() if dd == d and k == kernel)
            ax.plot([m for m, _ in pts], [us * args.attn_layers / 1e3 for _, us in pts], marker="o", ms=3,
                    label=label, **style)
        p = f"; FlashInfer plan(): ~{statistics.median(plan[d]):.0f} us host per step" if plan[d] else ""
        ax.set_title(f"{d}{p}", fontsize=9, loc="left")
        ax.set_xlabel("Minutes since the level started", fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(ls="dotted")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel(f"Decode attention per step, {args.attn_layers} layers (ms)", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Decode attention on the recorded KV lengths over the run (Super narrow EP, slower of "
                 "the 2 DP ranks; microbenchmark, L2 flushed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()

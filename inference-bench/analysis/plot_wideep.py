"""Does wide EP help decode? TPOT and per-GPU throughput vs concurrency per GPU.

Each run is a server/client.py sweep (bench.json) with a fixed concurrency per GPU: total
concurrency = c x GPUs, so every DP group (one TP group) sees the same batch and only the expert
sharding differs. TPOT = batch wall / OSL (long OSL, so prefill is amortized away).

Left: TPOT vs concurrency. Right: InferenceMAX-style throughput vs interactivity, i.e. output
tokens/s on --cluster-gpus GPUs vs tokens/s per user (1000 / TPOT); up and right is better.

    python3 inference-bench/analysis/plot_wideep.py "TP8 x EP64=<run>:64" "TP8 x EP8=<run>:8" \
        --out inference-bench/analysis/ultrav3_wideep_tpot.png
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Okabe-Ito, colorblind-safe; fixed per series in argument order, widest EP first: blue solid, then
# (three runs) bluish green dashed, and the narrowest EP always vermillion dotted.
STYLES_2 = [("#0072B2", "o", "-"), ("#D55E00", "^", ":")]
STYLES_3 = [("#0072B2", "o", "-"), ("#009E73", "s", "--"), ("#D55E00", "^", ":")]


def load(run, gpus):
    rows = json.load(open(os.path.join(run, "bench.json")))
    out = []
    for r in sorted(rows, key=lambda r: r["batch_size"]):
        out.append((r["batch_size"] // gpus, 1000 * r["wall_s"] / r["osl"], r["throughput_tok_s"] / gpus))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir:gpus")
    ap.add_argument("--title", default="Does wideEP help decodes? "
                    "NVIDIA Nemotron 3 Ultra 550B-A55B, TP8, OSL 8192")
    ap.add_argument("--out", default="inference-bench/analysis/ultrav3_wideep_tpot.png")
    ap.add_argument("--cluster-gpus", type=int, default=64,
                    help="express load as total concurrency on this many GPUs (c per GPU x N)")
    args = ap.parse_args()

    plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "DejaVu Sans"]
    fig, (tpot_ax, pareto_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    series = []
    # Point labels: first series above-right, last below-left, middle one unlabelled (the outer two
    # bracket it), so labels never collide.
    styles = STYLES_2 if len(args.runs) <= 2 else STYLES_3
    offsets = [((8, 6), "left")] + [None] * (len(args.runs) - 2) + [((-7, -12), "right")]
    for spec, (color, marker, ls), label_offset in zip(args.runs, styles, offsets):
        label, rest = spec.split("=", 1)
        run, gpus = rest.rsplit(":", 1)
        pts = load(run, int(gpus))
        series.append((label, pts))
        c, tpot, tput = zip(*pts)
        kw = dict(color=color, marker=marker, linestyle=ls, linewidth=2, markersize=8, label=label)
        tpot_ax.plot([ci * args.cluster_gpus for ci in c], tpot, **kw)
        pareto_ax.plot([1000 / t for t in tpot], [y * args.cluster_gpus for y in tput], **kw)
        for ci, t, y in pts if label_offset else []:
            pareto_ax.annotate(f"{ci * args.cluster_gpus}", (1000 / t, y * args.cluster_gpus),
                               textcoords="offset points", xytext=label_offset[0],
                               ha=label_offset[1], fontsize=8, color="#555555")

    # Direct-label the TPOT gap of the first series (widest EP) vs the last (narrowest) per point.
    if len(series) >= 2:
        a, b = dict((c, t) for c, t, _ in series[0][1]), dict((c, t) for c, t, _ in series[-1][1])
        for c in sorted(set(a) & set(b)):
            tpot_ax.annotate(f"{100 * (a[c] / b[c] - 1):+.0f}%", (c * args.cluster_gpus, a[c]),
                             textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8,
                             color="#333333")

    tpot_ax.set_xscale("log", base=2)
    tpot_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    tpot_ax.set_xlabel(f"Concurrent requests on {args.cluster_gpus} GPUs")
    tpot_ax.set_ylabel("TPOT (ms)")
    tpot_ax.set_ylim(bottom=0)
    tpot_ax.set_title("TPOT (time per output token), lower is better", fontsize=11)
    pareto_ax.set_xlabel("Interactivity (output tokens/s per user)")
    pareto_ax.set_ylabel(f"Output tokens/s ({args.cluster_gpus} GPUs)")
    pareto_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    pareto_ax.set_ylim(bottom=0)
    pareto_ax.set_title("Throughput vs interactivity, up and right is better", fontsize=11)
    pareto_ax.set_xlim(left=0)
    pareto_ax.text(0.02, 0.03, f"labels: concurrent requests on {args.cluster_gpus} GPUs",
                   transform=pareto_ax.transAxes, ha="left", va="bottom", fontsize=8, color="#555555")
    for ax in (tpot_ax, pareto_ax):
        ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = tpot_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(labels),
               frameon=False)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

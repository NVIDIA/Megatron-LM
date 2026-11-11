"""Throughput vs interactivity Pareto from steady_client.py runs (steady.jsonl), in the PSSG style.

One point per concurrency level: x = median tokens/s per user (1000 / median TPOT, TTFT
included), y = output tokens/s per GPU (from the server's /metrics over the window). A thin
whisker reaches left to the p90 TPOT (the slow tail); --label-c labels points with c. Up and to the right is better. GPUs per
run come from NODES in the run's config.env (4 GPUs per node) unless given as label=run:gpus.

Needs pssgplot (https://github.com/hpcgroup/pssg-plots); install a clone in editable mode so it
finds its bundled Gill Sans font: `pip install -e pssg-plots/`.

    python3 inference-bench/analysis/plot_pareto.py "NVLS/wide=<run>" "DeepEP/wide=<run>" \
        "NVLS/narrow=<run>" "DeepEP/narrow=<run>" --title "Super v3" --out pareto.pdf

Labels "<dispatcher>/<wide|narrow>" get fixed encodings: NVLS blue, DeepEP orange, NVL-A2A green,
vLLM AG/RS (its default all-gather / reduce-scatter dispatch) reddish purple, vLLM A2A (its FlashInfer
NVLink one-sided all-to-all) vermilion;
wide EP solid, narrow EP dashed. Other labels get the PSSG palette in order.
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pssgplot import PALETTE, PlotEnvironment  # noqa: E402


# Fixed encodings for "dispatcher/ep=run" labels: color and marker by dispatcher, dashes by EP
# width. Other labels get the PSSG palette in order and solid lines.
DISPATCHER_STYLE = {"NVLS": ("#0072B2", "o"), "DeepEP": ("#E69F00", "^"), "NVL-A2A": ("#009E73", "s"),
                    "vLLM": ("#CC79A7", "D"), "vLLM-A2A": ("#D55E00", "v"), "vLLM-TP": ("#CC79A7", "D"),
                    "vLLM-A2A-alt": ("#56B4E9", "s"), "vLLM-TP-alt": ("#56B4E9", "s"),
                    "vLLM-TP-v": ("#D55E00", "v"), "MInf-A2A": ("#009E73", "s"),
                    "MInf": ("#0072B2", "o")}
# Legend names where the label key is terse: vLLM's default dispatch is all-gather/reduce-scatter;
# vLLM-A2A is its FlashInfer NVLink one-sided all-to-all (--all2all-backend flashinfer_nvlink_one_sided).
# MInf is Megatron Core inference with NVLS, for plots against other frameworks. vLLM-TP is vLLM with
# one DP rank (EP = TP): no EP dispatch; tokens are replicated across TP and the all-reduce combines.
# vLLM-A2A-alt: the same dispatcher in a second color, for plotting two vLLM-A2A runs together.
DISPATCHER_NAME = {"vLLM": "vLLM (AG/RS)", "vLLM-A2A": "vLLM (FlashInfer A2A)",
                   "vLLM-A2A-alt": "vLLM (FlashInfer A2A)", "vLLM-TP-alt": "vLLM (no dispatch, EP = TP)",
                   "vLLM-TP-v": "vLLM (no dispatch, EP = TP)", "MInf-A2A": "MInf (NVL-A2A)",
                   "vLLM-TP": "vLLM (no dispatch, EP = TP)", "MInf": "MInf (NVLS)"}
EP_DASHES = {"wide": "-", "narrow": "--"}


def gpus_of(run):
    m = re.search(r"\bNODES=(\d+)", open(os.path.join(run, "config.env")).read())
    return 4 * int(m.group(1)) if m else 4


def load(spec):
    label, rest = spec.split("=", 1)
    run, _, gpus = rest.partition(":")
    gpus = int(gpus) if gpus else gpus_of(run)
    rows = []
    for line in open(os.path.join(run, "steady.jsonl")):
        r = json.loads(line)
        if r["tpot_ms_p50"] != r["tpot_ms_p50"]:  # NaN: no request finished in the window
            continue
        rows.append({"config": label, "c": r["concurrency"], "user_tok_s": r["user_tok_s_p50"],
                     "user_tok_s_p90": r["user_tok_s_p90_slow"],
                     "tok_s_gpu": r["output_tok_s"] / gpus, "c_per_gpu": r["concurrency"] / gpus,
                     "tpot_p50": r["tpot_ms_p50"], "tpot_p90": r["tpot_ms_p90"]})
    return rows


def style_of(label, i, show_ep=True):
    """(color, marker, linestyle, legend text) for a series label. show_ep=False leaves the EP
    width out of the legend text (when every series shares it, the title says it)."""
    dispatcher, _, ep = label.partition("/")
    if dispatcher in DISPATCHER_STYLE and ep in EP_DASHES:
        color, marker = DISPATCHER_STYLE[dispatcher]
        name = DISPATCHER_NAME.get(dispatcher, dispatcher)
        return color, marker, EP_DASHES[ep], f"{name}, {ep} EP" if show_ep else name
    return PALETTE[i % len(PALETTE)], "o", "-", label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir[:gpus]; label 'NVLS/wide', 'DeepEP/narrow', ...")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figsize", type=float, nargs=2, default=(5, 3.5))
    ap.add_argument("--label-c", action="store_true", help="label each point with its concurrency")
    ap.add_argument("--legend-loc", default="best",
                    help="Pareto legend location: a matplotlib loc, or 'above' for a two-column "
                         "legend between the title and the axes")
    ap.add_argument("--drop", nargs="*", default=[], metavar="LABEL:C",
                    help="drop a series' point at total concurrency C (e.g. a warmup outlier)")
    ap.add_argument("--tpot-out", default=None,
                    help="also plot median TPOT (whisker to p90) vs concurrency per GPU to this file")
    ap.add_argument("--interactivity-out", default=None,
                    help="also plot interactivity (whisker to the p90 TPOT) vs concurrency per GPU")
    args = ap.parse_args()

    df = pd.DataFrame([row for spec in args.runs for row in load(spec)])
    for item in args.drop:
        label, _, c = item.rpartition(":")
        df = df[~((df["config"] == label) & (df["c"] == int(c)))]
    labels = list(dict.fromkeys(df["config"]))
    # The legend names the EP width only when the series differ in it.
    args.show_ep = len({label.partition("/")[2] for label in labels}) > 1
    with PlotEnvironment():
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        for i, label in enumerate(labels):
            sub = df[df["config"] == label].sort_values("c")
            color, marker, ls, text = style_of(label, i, args.show_ep)
            ax.plot(sub["user_tok_s"], sub["tok_s_gpu"], color=color, marker=marker, linestyle=ls,
                    markeredgewidth=0, label=text, zorder=3)
            for _, r in sub.iterrows():
                # Whisker to the p90 TPOT (the slow tail), and optionally the concurrency.
                ax.plot([r["user_tok_s_p90"], r["user_tok_s"]], [r["tok_s_gpu"]] * 2,
                        color=color, linewidth=1, alpha=0.6)
                if args.label_c:
                    ax.annotate(f"{r['c']}", (r["user_tok_s"], r["tok_s_gpu"]),
                                textcoords="offset points", xytext=(5, 4), fontsize=7, color=color)
        # PSSG axis style (as pssgplot's LinePlot), with headroom so edge points are not clipped.
        ax.set_xlim(0, 1.1 * df["user_tok_s"].max())
        ax.set_ylim(0, 1.12 * df["tok_s_gpu"].max())
        ax.set_xlabel("Interactivity (tokens/s/user)")
        ax.set_ylabel("Throughput (tokens/s/GPU)")
        if args.title:
            ax.set_title(args.title, pad=40 if args.legend_loc == "above" else 6)
        style_axes(ax)
        if args.legend_loc == "above":
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                      handlelength=2.5, columnspacing=1.0, fontsize="small")
        else:
            ax.legend(loc=args.legend_loc, frameon=False, handlelength=3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
        print(f"-> {args.out}")
        if args.tpot_out:
            plot_vs_concurrency(df, labels, args, "tpot_p50", "tpot_p90", "TPOT (ms), median",
                                "upper left", args.tpot_out)
            print(f"-> {args.tpot_out}")
        if args.interactivity_out:
            plot_vs_concurrency(df, labels, args, "user_tok_s", "user_tok_s_p90",
                                "Interactivity (tokens/s/user)", "upper right",
                                args.interactivity_out)
            print(f"-> {args.interactivity_out}")


def style_axes(ax):
    """PSSG axis style, as pssgplot's LinePlot applies it."""
    ax.yaxis.grid(linestyle="dotted", zorder=0)
    ax.spines["left"].set_color("#606060")
    ax.spines["bottom"].set_color("#606060")
    ax.tick_params(axis="both", direction="in", color="#606060")


def plot_vs_concurrency(df, labels, args, y, y_tail, ylabel, legend_loc, out):
    """Median of a per-request metric (TPOT or interactivity, TTFT included) vs concurrency per
    GPU on a log2 axis, with a whisker to its p90-TPOT value (the slow tail)."""
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for i, label in enumerate(labels):
        sub = df[df["config"] == label].sort_values("c")
        color, marker, ls, text = style_of(label, i, args.show_ep)
        ax.plot(sub["c_per_gpu"], sub[y], color=color, marker=marker, linestyle=ls,
                markeredgewidth=0, label=text, zorder=3)
        for _, r in sub.iterrows():
            ax.plot([r["c_per_gpu"]] * 2, [r[y], r[y_tail]], color=color, linewidth=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ticks = sorted(set(df["c_per_gpu"]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(0, 1.12 * max(df[y].max(), df[y_tail].max()))
    ax.set_xlabel("Concurrency per GPU")
    ax.set_ylabel(ylabel)
    if args.title:
        ax.set_title(args.title)
    style_axes(ax)
    ax.legend(loc=legend_loc, frameon=False, handlelength=3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()

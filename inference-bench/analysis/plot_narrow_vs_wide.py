"""Narrow vs wide EP, one plot per model and framework (MInf with NVLS, vLLM with the one-sided
all-to-all): throughput vs p90 interactivity (Pareto) and p90 TPOT vs requests per GPU. Each point
is the mean of the paper's three trials.

    python3 inference-bench/analysis/plot_narrow_vs_wide.py
Writes inference-bench/analysis/pareto/narrow_vs_wide/{superv3,ultrav3}_{minf,vllm}_{pareto,tpot}.png,
and MInf vs vLLM for Ultra with wide EP (ultrav3_wide_minf_vs_vllm_{pareto,tpot}.png).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_figures as pf  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto", "narrow_vs_wide")
MODELS = (("Nemotron-3-Super-120B-A12B", "superv3"), ("Nemotron-3-Ultra-550B-A55B", "ultrav3"))
FRAMEWORK = {"MInf": "MInf (NVLS)", "vLLM": "vLLM (one-sided A2A)"}
# One color, marker and line style per EP width.
WIDTH = {"wide": ("#0072B2", "o", "-", "Wide EP (64 GPUs)"),
         "narrow": ("#E69F00", "s", "--", "Narrow EP (8 GPUs)")}


def series(trials):
    """[(requests per GPU, mean tok/s/GPU, mean p90 tok/s/user, mean p90 TPOT ms)] over levels
    every trial measured, without the 1-request-per-GPU level."""
    gpus = pf.gpus_of(trials[0])
    per_trial = []
    for run in trials:
        recs = {}
        for line in open(os.path.join(pf.EXPERIMENTS, run, "steady.jsonl")):
            r = json.loads(line)
            if r["concurrency"] > gpus:
                recs[r["concurrency"]] = r
        per_trial.append(recs)
    levels = sorted(set.intersection(*(set(t) for t in per_trial)))
    mean = lambda f, c: sum(f(t[c]) for t in per_trial) / len(per_trial)  # noqa: E731
    return [(c // gpus,
             mean(lambda r: r["output_tok_s"] / gpus, c),
             mean(lambda r: r["user_tok_s_p90_slow"], c),
             mean(lambda r: r["tpot_ms_p90"], c)) for c in levels]


def runs_for(model):
    """{(framework, width): trials} from the paper figures' run lists."""
    out = {}
    for width in ("wide", "narrow"):
        for title, minf, vllm, _ in pf.FIGURES[f"e2e_{width}"]:
            if title[0] == model:
                out[("MInf", width)], out[("vLLM", width)] = minf, vllm
    return out


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5)
    ax.yaxis.grid(ls="dotted")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    for model, short in MODELS:
        runs = runs_for(model)
        for fw, fw_label in FRAMEWORK.items():
          data = {width: series(runs[(fw, width)]) for width in ("narrow", "wide")}
          for kind in ("pareto", "tpot"):
            fig, ax = plt.subplots(figsize=(5.4, 3.9))
            for width, pts in data.items():
                color, marker, ls, w_label = WIDTH[width]
                if kind == "pareto":
                    xs, ys = [p[2] for p in pts], [p[1] for p in pts]
                else:
                    xs, ys = [p[0] for p in pts], [p[3] for p in pts]
                ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=2, ms=6,
                        label=w_label)
                if kind == "pareto":
                    for (c, _, x, _), y in zip(pts, ys):
                        ax.annotate(f"{c}", (x, y), textcoords="offset points", xytext=(5, 4),
                                    fontsize=6.5, color="0.35")
            if kind == "pareto":
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                style(ax, "P90 interactivity (tokens/s/user)", "Throughput (tokens/s/GPU)",
                      f"{model}, {fw_label}: narrow vs wide EP\nmean of 3 trials; point labels = requests per GPU")
            else:
                ax.set_xscale("log", base=2)
                ticks = sorted({p[0] for pts in data.values() for p in pts})
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])
                ax.set_ylim(bottom=0)
                style(ax, "Requests per GPU", "P90 TPOT (ms/token, TTFT included)",
                      f"{model}, {fw_label}: narrow vs wide EP\nmean of 3 trials; lower is better")
            fig.tight_layout()
            path = os.path.join(OUT, f"{short}_{fw.lower()}_{kind}.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            print(path)


def plot_frameworks(model, short, width):
    """MInf vs vLLM at one EP width: {short}_{width}_minf_vs_vllm_{pareto,tpot}.png."""
    runs = runs_for(model)
    colors = {"MInf": ("#0072B2", "o", "-"), "vLLM": ("#CC79A7", "D", "--")}
    data = {fw: series(runs[(fw, width)]) for fw in FRAMEWORK}
    for kind in ("pareto", "tpot"):
        fig, ax = plt.subplots(figsize=(5.4, 3.9))
        for fw, pts in data.items():
            color, marker, ls = colors[fw]
            if kind == "pareto":
                xs, ys = [p[2] for p in pts], [p[1] for p in pts]
            else:
                xs, ys = [p[0] for p in pts], [p[3] for p in pts]
            ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=2, ms=6, label=FRAMEWORK[fw])
            if kind == "pareto":
                for p_, x, y in zip(pts, xs, ys):
                    ax.annotate(f"{p_[0]}", (x, y), textcoords="offset points", xytext=(5, 4),
                                fontsize=6.5, color="0.35")
        title = f"{model}, {width} EP ({64 if width == "wide" else 8} GPUs): MInf vs vLLM\nmean of 3 trials"
        if kind == "pareto":
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            style(ax, "P90 interactivity (tokens/s/user)", "Throughput (tokens/s/GPU)",
                  title + "; point labels = requests per GPU")
        else:
            ax.set_xscale("log", base=2)
            ticks = sorted({p_[0] for pts in data.values() for p_ in pts})
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_ylim(bottom=0)
            style(ax, "Requests per GPU", "P90 TPOT (ms/token, TTFT included)",
                  title + "; lower is better")
        fig.tight_layout()
        path = os.path.join(OUT, f"{short}_{width}_minf_vs_vllm_{kind}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
    plot_frameworks("Nemotron-3-Ultra-550B-A55B", "ultrav3", "wide")

"""Prefill interference from the step-time histograms (steady_client's per-level `step_time_ms`):
percentage of total engine time spent in interfered decode steps (a decode-only rank waiting on
another EP rank's prefill) vs requests per GPU.

    python3 inference-bench/analysis/plot_interference.py
Writes inference-bench/analysis/pareto/interference/interfered_decode_time.png and prints the table.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(HERE, "..", "experiments")
OUT = os.path.join(HERE, "pareto", "interference")
# (label, run, GPUs, color, marker, line style)
RUNS = [
    ("Super, wide EP (EP64, 16 engines)", "20261102_032910_math_superv3_wide_ep_nvls_interference",
     64, "#0072B2", "o", "-"),
    ("Super, narrow EP (EP8, 2 engines)", "20261102_031636_math_superv3_narrow_ep_nvls_interference",
     8, "#E69F00", "s", "--"),
    ("Ultra, wide EP (EP64, 8 engines)", "20261102_041957_math_ultrav3_wide_ep_nvls_interference",
     64, "#009E73", "D", "-"),
]


def interference(run, gpus):
    """[(requests per GPU, % of engine time in interfered decodes, % time lost to interference)]."""
    out = []
    for line in open(os.path.join(EXPERIMENTS, run, "steady.jsonl")):
        r = json.loads(line)
        h = r["step_time_ms"].get("all", {})
        get = lambda k: (h.get(k, {}).get("count", 0), h.get(k, {}).get("mean_ms", 0.0))  # noqa: E731
        (nc, mc), (ni, mi), (npf, mp) = get("decode_clean"), get("decode_interfered"), get("prefill")
        total = nc * mc + ni * mi + npf * mp
        out.append((r["concurrency"] // gpus, 100 * ni * mi / total, 100 * ni * (mi - mc) / total))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for label, run, gpus, color, marker, ls in RUNS:
        pts = interference(run, gpus)
        print(label)
        for c, share, lost in pts:
            print(f"  {c:3d} req/GPU: {share:5.2f}% of time in interfered decodes ({lost:5.2f}% lost)")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, ls=ls, lw=2,
                ms=6, label=label)
    ax.set_xscale("log", base=2)
    ticks = sorted({p[0] for _, run, gpus, *_ in RUNS for p in interference(run, gpus)})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Requests per GPU")
    ax.set_ylabel("Engine time in interfered decode steps (%)")
    ax.set_title("Prefill interference, MInf (NVLS)\n"
                 "decode steps of ranks waiting on another EP rank's prefill", fontsize=8.5)
    ax.yaxis.grid(ls="dotted")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    path = os.path.join(OUT, "interfered_decode_time.png")
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()

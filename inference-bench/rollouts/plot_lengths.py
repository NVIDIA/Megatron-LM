"""CDF of natural generation lengths (completion tokens) from collect_rollouts.py JSONL files.

Errored rollouts are dropped; rollouts that hit the max_tokens cap are kept at the cap (marked
in the legend). Groups: one curve per --group value (dataset file, or a field such as level /
source).

    python3 inference-bench/rollouts/plot_lengths.py "MATH-500=<exp>/rollouts.jsonl" \
        "AIME / HMMT / BeyondAIME=<exp>/rollouts_hard_math.jsonl" --out inference-bench/rollouts/lengths.png
"""

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Okabe-Ito, colorblind-safe, fixed order.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), "-"]


def load(path):
    return [r for r in map(json.loads, open(path)) if "error" not in r]


def cdf(ax, values, label, i):
    v = sorted(values)
    n = len(v)
    ax.plot(v, [(k + 1) / n for k in range(n)], color=COLORS[i % 8], linestyle=LINESTYLES[i % 8],
            linewidth=2, label=label)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="label=rollouts.jsonl")
    ap.add_argument("--split", default="", help="also plot one panel split by this field (e.g. level)")
    ap.add_argument("--title", default="Natural generation lengths, NVIDIA Nemotron 3 Super 120B-A12B")
    ap.add_argument("--out", default="inference-bench/rollouts/lengths.png")
    args = ap.parse_args()

    plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "DejaVu Sans"]
    panels = [("By dataset", [(lbl, load(p)) for lbl, p in (s.split("=", 1) for s in args.files)])]
    if args.split:
        groups = {}
        for _, rows in panels[0][1]:
            for r in rows:
                groups.setdefault(str(r[args.split]), []).append(r)
        panels.append((f"By {args.split}", sorted(groups.items(), key=lambda kv: kv[0])))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.2), squeeze=False)
    for ax, (title, groups) in zip(axes[0], panels):
        for i, (label, rows) in enumerate(groups):
            toks = [r["completion_tokens"] for r in rows]
            capped = sum(r.get("finish_reason") == "length" for r in rows)
            v = sorted(toks)
            med, p90 = v[len(v) // 2], v[int(0.9 * (len(v) - 1))]
            extra = f", {capped} hit cap" if capped else ""
            cdf(ax, toks, f"{label} (n={len(v)}, p50 {med / 1000:.1f}k, p90 {p90 / 1000:.1f}k{extra})", i)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"))
        ax.set_xlabel("Completion tokens (reasoning + answer)")
        ax.set_ylabel("Fraction of rollouts")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

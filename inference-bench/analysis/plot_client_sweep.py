"""Client-side decode sweep (bench.json) for one or two runs: batch wall time and, with two runs,
the second run's speedup over the first.

    python3 inference-bench/analysis/plot_client_sweep.py \
        nvl_a2a=inference-bench/experiments/<run_a> nvls=inference-bench/experiments/<run_b> \
        --title "..." --out inference-bench/analysis/client_sweep.png
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

# Okabe-Ito, colorblind-safe; fixed per series in argument order.
STYLES = [("#0072B2", "o", "-"), ("#D55E00", "^", "--")]


def load(exp_dir):
    rows = json.load(open(os.path.join(exp_dir, "bench.json")))
    return {r["batch_size"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="label=experiment_dir (one or two)")
    parser.add_argument("--title", default="Decode sweep")
    parser.add_argument("--out", default="inference-bench/analysis/client_sweep.png")
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "DejaVu Sans"]
    runs = [(spec.split("=", 1)[0], load(spec.split("=", 1)[1])) for spec in args.runs]
    if len(runs) == 2:
        fig, (wall_ax, sp_ax) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]}
        )
        axes = [wall_ax, sp_ax]
    else:
        fig, wall_ax = plt.subplots(figsize=(8, 5))
        axes = [wall_ax]

    for (label, rows), (color, marker, ls) in zip(runs, STYLES):
        bs = sorted(rows)
        wall_ax.plot(bs, [rows[b]["wall_s"] for b in bs], color=color, marker=marker,
                     linestyle=ls, linewidth=2, markersize=6, label=label)

    wall_ax.set_yscale("log", base=2)
    wall_ax.yaxis.set_major_locator(ticker.FixedLocator([8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 56, 64, 80]))
    wall_ax.yaxis.set_minor_locator(ticker.NullLocator())
    wall_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    osl = runs[0][1][min(runs[0][1])]["osl"]
    wall_ax.set_ylabel(f"Batch wall time (s, {osl:,} output tokens)")
    wall_ax.legend(frameon=False, loc="upper left")
    wall_ax.set_title(args.title)

    if len(runs) == 2:
        (la, ra), (lb, rb) = runs
        common = sorted(set(ra) & set(rb))
        speedup = [100 * (ra[b]["wall_s"] / rb[b]["wall_s"] - 1) for b in common]
        sp_ax.axhline(0, color="#999999", linewidth=1)
        sp_ax.bar(common, speedup, width=[b * 0.35 for b in common], color="#555555")
        for b, s in zip(common, speedup):
            sp_ax.annotate(f"{s:+.1f}%", (b, s), textcoords="offset points",
                           xytext=(0, 4 if s >= 0 else -11), ha="center", fontsize=8,
                           color="#333333")
        lim = max(abs(s) for s in speedup) * 1.3
        sp_ax.set_ylim(-lim, lim)
        sp_ax.set_ylabel(f"{lb} speedup\nover {la} (%)")

    axes[-1].set_xscale("log", base=2)
    axes[-1].xaxis.set_major_locator(ticker.FixedLocator(sorted(runs[0][1])))
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[-1].set_xlabel("Concurrent requests (1,024 output tokens each)")
    for ax in axes:
        ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()

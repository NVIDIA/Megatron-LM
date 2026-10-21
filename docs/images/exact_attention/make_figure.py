# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Draw docs/images/exact_attention/fold.svg: the super-block fold and the G trade-off.

Run: python docs/images/exact_attention/make_figure.py
The figure is a schematic drawn from formulas only.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

G = 4
SUPER_BLOCKS = 3
TILES = G * SUPER_BLOCKS
COLORS = ["#76b900", "#4a90d9", "#e0a030"]


def fold_panel(ax):
    ax.set_title("A. One fold for every phase", loc="left", fontsize=11)
    for t in range(TILES):
        ax.add_patch(Rectangle((t, 5.2), 0.95, 0.6, color=COLORS[t // G], alpha=0.35))
    for s in range(SUPER_BLOCKS):
        ax.text(s * G + G / 2, 6.05, f"super-block {s}", ha="center", fontsize=8)
    ax.text(-0.3, 5.5, "key tiles", ha="right", va="center", fontsize=8)
    rows = [
        ("training forward / prefill", [(0, TILES)]),
        ("chunked prefill", [(0, 6), (6, TILES)]),
        ("split decode", [(0, G), (G, 2 * G), (2 * G, TILES)]),
        ("verify", [(0, G), (G, 2 * G), (2 * G, TILES)]),
    ]
    for i, (name, calls) in enumerate(rows):
        y = 4.2 - i * 0.9
        ax.text(-0.3, y + 0.2, name, ha="right", va="center", fontsize=8)
        for a, b in calls:
            ax.add_patch(Rectangle((a, y), b - a - 0.05, 0.4, fill=False, lw=1.2))
    for s in range(SUPER_BLOCKS):
        x = s * G + G / 2
        ax.add_patch(Rectangle((x - 0.9, 0.1), 1.8, 0.5, color=COLORS[s], alpha=0.6))
        ax.text(x, 0.35, f"(m, l, O)$_{s}$", ha="center", va="center", fontsize=8)
        if s:
            ax.add_patch(
                FancyArrowPatch(((s - 1) * G + G / 2 + 0.9, 0.35), (x - 0.9, 0.35), arrowstyle="->")
            )
    ax.text(TILES + 0.2, 0.35, "fold left to right, one BF16 rounding", va="center", fontsize=8)
    ax.text(-0.3, 0.35, "same partials", ha="right", va="center", fontsize=8)
    ax.set_xlim(-6, TILES + 6)
    ax.set_ylim(-0.2, 6.5)
    ax.axis("off")


def total_time_panel(ax):
    ax.set_title("B. Goal: lower total time over all phases", loc="left", fontsize=11)
    phases = ["forward", "backward", "prefill", "decode", "verify"]
    base = [1.0, 2.0, 1.0, 1.6, 0.8]
    extra = [0.12, 0.0, 0.12, 0.1, 0.08]
    for i, (p, b, e) in enumerate(zip(phases, base, extra)):
        ax.barh(i, b, color="#bbbbbb")
        ax.barh(i, e, left=b, color="#76b900")
    ax.set_yticks(range(len(phases)), phases, fontsize=8)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel("time (schematic, no scale)", fontsize=8)
    ax.text(0.02, -0.9, "grey: the phase itself    green: fixed cost of exactness", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def g_panel(ax):
    ax.set_title("C. The G dial", loc="left", fontsize=11)
    gs = [1, 2, 4, 8, 16, 32]
    items = [1.0 / g for g in gs]
    ax.loglog(gs, items, marker="o", color="#4a90d9")
    ax.set_xticks(gs, [str(g) for g in gs], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("G (tiles per super-block)", fontsize=8)
    ax.set_ylabel("ceil(L / (128 G))", fontsize=8)
    ax.text(1.2, 0.05, "larger G: fewer fold steps,\ncheaper prefill and forward", fontsize=8)
    ax.text(5, 0.6, "smaller G: more decode\nwork items per sequence", fontsize=8)


def main():
    fig = plt.figure(figsize=(11, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    fold_panel(fig.add_subplot(grid[0, :]))
    total_time_panel(fig.add_subplot(grid[1, 0]))
    g_panel(fig.add_subplot(grid[1, 1]))
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fold.svg")
    fig.savefig(out, metadata={"Date": None})
    print(out)


if __name__ == "__main__":
    main()

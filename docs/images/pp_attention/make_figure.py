# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Draw docs/images/pp_attention/fold.png: every phase builds the same super-block partials.

Run: python docs/images/pp_attention/make_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

G = 4
TILES = 14
SUPER = [(s * G, min(TILES, (s + 1) * G)) for s in range((TILES + G - 1) // G)]
X0 = 4.2
SB_COLORS = ["#cfe3f5", "#d9efc4", "#f8e2c0", "#e9d6f0"]
PATH_COLOR = "#3b6fb6"
FOLD_COLOR = "#1b9e77"


def box(ax, x, y, w, h, text, fc, ec="0.3", fs=9, weight="normal"):
    """Rounded box with centered text."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.8
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, a, b, color="0.35"):
    """Arrow from a to b."""
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9))


def main():
    """Draw and save the figure."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, X0 + TILES + 1.0)
    ax.set_ylim(-0.4, 9.4)
    ax.axis("off")

    # Rows of the three call patterns, each split at the same super-block boundaries.
    rows = [
        ("training forward, prefill", "one kernel walks all tiles", [(0, TILES)]),
        ("chunked prefill", "hatched: keys already in the KV cache", [(0, 6), (6, TILES)]),
        ("decode, verify", "one work item per super-block", SUPER),
    ]
    for r, (label, note, spans) in enumerate(rows):
        y = 8.3 - r * 1.0
        ax.text(X0 - 0.3, y + 0.36, label, ha="right", va="center", fontsize=9)
        ax.text(X0 - 0.3, y + 0.0, note, ha="right", va="center", fontsize=7.5, color="0.4")
        for k, (a, b) in enumerate(spans):
            hatch = "////" if r == 1 and k == 0 else None
            ax.add_patch(
                Rectangle(
                    (X0 + a + 0.05, y),
                    b - a - 0.1,
                    0.5,
                    fc="white" if hatch else "#dbe6f5",
                    ec=PATH_COLOR,
                    hatch=hatch,
                    lw=0.9,
                )
            )

    # Key tiles, grouped into super-blocks at absolute positions.
    ty = 4.9
    ax.text(X0 - 0.3, ty + 0.3, "key tiles (128 keys)", ha="right", va="center", fontsize=9)
    for t in range(TILES):
        s = t // G
        ax.add_patch(Rectangle((X0 + t + 0.05, ty), 0.9, 0.6, fc=SB_COLORS[s], ec="0.4", lw=0.6))
        ax.text(X0 + t + 0.5, ty + 0.3, f"{t}", ha="center", va="center", fontsize=8)
    for s, (a, b) in enumerate(SUPER):
        ax.plot([X0 + a + 0.1, X0 + b - 0.1], [ty - 0.2, ty - 0.2], color="0.3", lw=0.8)
        ax.text(
            X0 + (a + b) / 2, ty - 0.5, f"super-block {s}", ha="center", va="center", fontsize=8.5
        )
        for gx in (a, b):
            if 0 < gx < TILES:
                ax.plot([X0 + gx, X0 + gx], [ty - 0.1, 8.95], color="0.55", lw=0.7, ls=(0, (3, 3)))
    ax.text(X0 + TILES + 0.1, ty + 0.3, "query", ha="left", va="center", fontsize=8, rotation=90)

    # One FP32 partial per super-block, then one left-to-right fold.
    py, fy = 2.6, 0.5
    ax.text(X0 - 0.3, py + 0.3, "FP32 partials", ha="right", va="center", fontsize=9)
    ax.text(X0 - 0.3, fy + 0.3, "fixed left fold", ha="right", va="center", fontsize=9)
    prev = None
    for s, (a, b) in enumerate(SUPER):
        cx = X0 + (a + b) / 2
        box(ax, cx - 1.1, py, 2.2, 0.6, f"$(m_{s},\\ \\ell_{s},\\ O_{s})$", SB_COLORS[s])
        arrow(ax, (cx, ty - 0.75), (cx, py + 0.65))
        if s == 0:
            prev = (cx, py)
            continue
        box(ax, cx - 0.35, fy, 0.7, 0.6, "fold", "#e3f4ee", ec=FOLD_COLOR, fs=8)
        arrow(ax, (cx, py - 0.05), (cx, fy + 0.65), FOLD_COLOR)
        arrow(
            ax,
            (prev[0] + (0.35 if s > 1 else 0), fy + 0.3 if s > 1 else prev[1]),
            (cx - 0.38, fy + 0.3),
            FOLD_COLOR,
        )
        prev = (cx, fy)
    box(ax, X0 + TILES - 0.2, fy, 0.8, 0.6, "out", "white", ec=FOLD_COLOR, fs=8, weight="bold")
    arrow(ax, (prev[0] + 0.35, fy + 0.3), (X0 + TILES - 0.22, fy + 0.3), FOLD_COLOR)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fold.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()

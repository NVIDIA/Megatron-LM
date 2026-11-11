"""Schematic of FA4 split-KV scheduling: 2 SMs, 2 decode requests, 1 KV head, time in KV tiles.

Rows: equal KV lengths vs skewed KV lengths (same total). Columns: FA4's heuristic
(num_SMs // (batch * kv_heads) = 2 // 2 = 1 split: one CTA per request) vs a floor of 2 splits
(each request's KV cut into 2 equal chunks, 4 CTAs greedily placed on the 2 SMs, then a combine).

    python3 inference-bench/analysis/plot_fa4_split_schematic.py --out <png>
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COLORS = {"R0": "#E69F00", "R1": "#0072B2"}
COMBINE = 0.4  # combine step, drawn in tile units (small next to the KV walk)


def schedule(lengths, splits):
    """Greedy list scheduling of CTAs (request, chunk tiles) onto 2 SMs, in grid order.
    Returns per-SM [(start, tiles, request, label)] and the kernel end time."""
    ctas = []
    for name, n in lengths.items():
        per = -(-n // splits)  # FA4: ceil(own tiles / splits) per chunk, last chunk shorter
        for i in range(splits):
            tiles = max(0, min(per, n - i * per))
            ctas.append((name, tiles, f"{name}" if splits == 1 else f"{name}.{i}"))
    sms = [[], []]
    free = [0.0, 0.0]
    for name, tiles, label in ctas:
        sm = free.index(min(free))
        sms[sm].append((free[sm], tiles, name, label))
        free[sm] += tiles
    end = max(free)
    return sms, end


def draw(ax, lengths, splits, title):
    sms, end = schedule(lengths, splits)
    combine = COMBINE if splits > 1 else 0.0
    for row, segs in enumerate(sms):
        y = 1 - row
        busy_until = 0.0
        for start, tiles, name, label in segs:
            if tiles == 0:
                continue
            ax.barh(y, tiles, left=start, height=0.6, color=COLORS[name], edgecolor="white", lw=2)
            ax.text(start + tiles / 2, y, f"{label}: {tiles} tiles" if tiles > 1 else label,
                    ha="center", va="center", fontsize=8, color="white" if name == "R1" else "black")
            busy_until = start + tiles
        if busy_until < end:  # idle SM: waits for the kernel to finish
            ax.barh(y, end - busy_until, left=busy_until, height=0.6, color="white",
                    edgecolor="0.6", hatch="///", lw=0.8)
    if combine:
        ax.barh(0.5, combine, left=end, height=1.6, color="0.35", edgecolor="white", lw=2)
        ax.text(end - 0.05, 1.45, "combine", ha="right", va="bottom", fontsize=7, color="0.25")
    total = end + combine
    ax.axvline(total, color="black", lw=1.2, ls="--")
    ax.text(total + 0.1, -0.55, f"kernel done: {total:g}", fontsize=8, va="center")
    ax.set_yticks([1, 0], ["SM 0", "SM 1"])
    ax.set_xlim(0, 9.5)
    ax.set_ylim(-0.8, 1.8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelsize=8)
    ax.xaxis.grid(ls="dotted", color="0.8")
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = (("Equal KV: R0 = 4 tiles, R1 = 4 tiles", {"R0": 4, "R1": 4}),
             ("Skewed KV: R0 = 7 tiles, R1 = 1 tile (same total)", {"R0": 7, "R1": 1}))
    cols = ((1, "FA4 heuristic: 2 SMs // 2 requests = 1 split\n(one CTA walks each request's whole KV)"),
            (2, "Floor of 2 splits (ours: max(heuristic, floor))\n(each request's KV cut in 2 chunks)"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), sharex=True)
    for r, (case, lengths) in enumerate(cases):
        for c, (splits, col_title) in enumerate(cols):
            draw(axes[r][c], lengths, splits, (col_title + "\n" if r == 0 else "") + case)
    for ax in axes[1]:
        ax.set_xlabel("Time (KV tiles walked by one SM)", fontsize=8)
    fig.legend(handles=[Patch(color=COLORS["R0"], label="request R0"),
                        Patch(color=COLORS["R1"], label="request R1"),
                        Patch(facecolor="white", edgecolor="0.6", hatch="///", label="SM idle"),
                        Patch(color="0.35", label="split-KV combine")],
               loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Decode attention, 2 SMs, 2 requests, 1 KV head", fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()

"""One step, one MoE layer of a routing-count dump (MCORE_ROUTING_COUNTS): per-expert routed rows,
and the per-rank loads the same routing gives with 64 experts per rank (EP8) and 8 (EP64).

    python3 inference-bench/analysis/plot_routing_load.py <exp dir> [--layer N] [--step-index I]
Writes <exp dir>/routing_load_layer<N>.png.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--layer", type=int, default=20, help="MoE layer index")
    ap.add_argument("--step-index", type=int, default=-1,
                    help="which full-load decode step (default: the last)")
    args = ap.parse_args()
    recs = [json.loads(line) for line in open(os.path.join(args.exp_dir, "routing_counts.jsonl"))]
    full = max(r["routed_rows"] for r in recs if r["routed_rows"] % 22 == 0 and r["routed_rows"] <= 256 * 22)
    rec = [r for r in recs if r["routed_rows"] == full][args.step_index]
    counts = rec["counts"][args.layer]
    num_experts = len(counts)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.bar(range(num_experts), counts, width=1.0, color="0.45")
    mean_e = sum(counts) / num_experts
    ax.axhline(mean_e, color="#D55E00", lw=1, ls="--", label=f"mean per expert ({mean_e:.0f})")
    for b in range(0, num_experts + 1, num_experts // 8):
        ax.axvline(b - 0.5, color="#0072B2", lw=1.2)
    ax.set_xlim(-0.5, num_experts - 0.5)
    ax.set_xlabel("Expert id (blue lines: EP8 rank boundaries, 64 experts each; EP64 ranks are 8 experts wide)")
    ax.set_ylabel("Routed rows")
    ax.set_title(f"Super narrow EP, step {rec['step']} ({full // 22} tokens x top-22 = {full} routed rows), "
                 f"MoE layer {args.layer}", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    for ax, ep, color in ((axes[1], 8, "#0072B2"), (axes[2], 64, "#009E73")):
        per = num_experts // ep
        loads = [sum(counts[r * per:(r + 1) * per]) for r in range(ep)]
        mean = sum(loads) / ep
        rel = [x / mean for x in loads]
        ax.bar(range(ep), rel, width=0.8, color=color)
        ax.axhline(1.0, color="#D55E00", lw=1, ls="--")
        busiest = max(range(ep), key=lambda r: rel[r])
        ax.annotate(f"busiest {rel[busiest]:.2f}x", (busiest, rel[busiest]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xlim(-0.6, ep - 0.4)
        ax.set_ylim(0, max(rel) * 1.15)
        ax.set_ylabel("Rank load / mean")
        ax.set_xlabel(f"EP{ep} rank ({per} experts per rank)")
        ax.set_title(f"EP{ep}: each MoE layer waits for the busiest rank -> {rel[busiest]:.2f}x the "
                     f"balanced expert time", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(ls="dotted")
    fig.tight_layout()
    path = os.path.join(args.exp_dir, f"routing_load_layer{args.layer}.png")
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()

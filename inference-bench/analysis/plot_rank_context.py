"""Per-DP-rank mean context length per request (KV tokens / running requests) over time, from
steady_client's engine samples: a rank x time heatmap, and each rank's line with the max / mean /
min across ranks.

    python3 inference-bench/analysis/plot_rank_context.py <exp dir>
Writes --out, or <exp dir>/rank_context.png.
"""

import argparse
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--out", default=None, help="output PNG (default: <exp dir>/rank_context.png)")
    args = ap.parse_args()
    d = args.exp_dir
    samples = [json.loads(line) for line in open(os.path.join(d, "steady.engines.jsonl"))]
    t0 = min(json.loads(line)["start"] for line in open(os.path.join(d, "steady.requests.jsonl")))
    ranks = sorted({r for s in samples for r in s["engines"]}, key=lambda r: int(r.rsplit("-", 1)[-1]))
    ts = [(s["t"] - t0) / 60 for s in samples]
    ctx = [[s["engines"][r]["kv_tokens"] / 1e3 / max(s["engines"][r]["requests_running"], 1)
            if r in s["engines"] else float("nan") for s in samples] for r in ranks]

    fig, (ax_h, ax_l) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1.1, 1]})
    im = ax_h.imshow(ctx, aspect="auto", cmap="Blues", origin="lower", interpolation="nearest",
                     extent=[ts[0] - 0.5, ts[-1] + 0.5, -0.5, len(ranks) - 0.5])
    ax_h.set_yticks(range(len(ranks)))
    ax_h.set_yticklabels([r.rsplit("-", 1)[-1] for r in ranks], fontsize=7)
    ax_h.set_ylabel("DP rank (engine)")
    fig.colorbar(im, ax=ax_h, label="Mean context per request (K tokens)", pad=0.01)
    ax_h.set_title(f"{os.path.basename(os.path.normpath(d))}: mean context length per request "
                   f"on each DP rank", fontsize=9)

    for row in ctx:
        ax_l.plot(ts, row, color="#0072B2", lw=0.8, alpha=0.35)
    cols = list(zip(*ctx))
    ax_l.plot(ts, [max(c) for c in cols], color="#D55E00", lw=2, label="max over ranks")
    ax_l.plot(ts, [statistics.mean(c) for c in cols], color="black", lw=1.5, ls="--",
              label="mean over ranks")
    ax_l.plot(ts, [min(c) for c in cols], color="#009E73", lw=2, label="min over ranks")
    ax_l.plot([], [], color="#0072B2", lw=0.8, alpha=0.6, label="each rank")
    ax_l.set_ylabel("Mean context per request (K tokens)")
    ax_l.set_xlabel("Minutes since the level started")
    ax_l.set_ylim(bottom=0)
    ax_l.yaxis.grid(ls="dotted")
    ax_l.spines[["top", "right"]].set_visible(False)
    ax_l.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = args.out or os.path.join(d, "rank_context.png")
    fig.savefig(path, dpi=150)
    print(path)
    last = cols[-1]
    print(f"latest sample (t={ts[-1]:.0f} min): min {min(last):.1f}K mean {statistics.mean(last):.1f}K "
          f"max {max(last):.1f}K, max/mean {max(last) / statistics.mean(last):.2f}")
    print("per rank:", " ".join(f"{r.rsplit('-', 1)[-1]}:{v:.0f}K" for r, v in zip(ranks, last)))


if __name__ == "__main__":
    main()

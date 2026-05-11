"""Plot lm-loss curves from two Megatron training logs (overlay).

Usage:
    python plot_loss.py <log_a> <label_a> <log_b> <label_b> <out.png>

Example (cross-arch B300 vs H100):
    python plot_loss.py \\
        b300_run.log "B300 TP=1 (tp-invariant)" \\
        h100_run.log "H100 TP=1 (tp-invariant)" \\
        b300_vs_h100_tp1_invariant.png
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ITER_LOSS_RE = re.compile(r"iteration\s+(\d+)/\s+\d+.*?lm loss:\s+([\d.E+-]+)")


def parse_log(path):
    iters, losses = [], []
    for line in Path(path).read_text().splitlines():
        m = ITER_LOSS_RE.search(line)
        if m:
            iters.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return iters, losses


def main():
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    log_a, label_a, log_b, label_b, out_png = sys.argv[1:]

    a_it, a_loss = parse_log(log_a)
    b_it, b_loss = parse_log(log_b)
    assert a_it == b_it, f"iter index mismatch: {len(a_it)} vs {len(b_it)}"
    print(f"{len(a_it)} iters; bitwise identical: {a_loss == b_loss}")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    # B drawn first thicker (background), A on top thinner — both visible when overlapping
    ax.plot(b_it, b_loss, color="#43a047", linewidth=2.8, label=label_b, alpha=0.9)
    ax.plot(a_it, a_loss, color="#26c6da", linewidth=1.4, label=label_a)

    ax.set_title("lm loss", fontsize=14, pad=14)
    ax.set_xlim(0, max(a_it))
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
    ax.text(max(a_it) - 0.5, ax.get_ylim()[0] + 0.05, "Step", color="#888", fontsize=10, ha="right")

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11,
                    handlelength=1.5, columnspacing=2.2)
    for text, color in zip(leg.get_texts(), ["#43a047", "#26c6da"]):
        text.set_color(color)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()

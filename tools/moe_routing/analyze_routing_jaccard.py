#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compute and visualize per-token Jaccard similarity of expert routing
across consecutive MoE layers.

Given JSONL traces produced by ``--moe-routing-trace-path``, this script
computes Jaccard similarity J(L_prev, L_curr) = |A∩B| / |A∪B| where A is
the set of experts a token selected at the previous MoE layer and B is
the set at the next MoE layer. "Previous" and "next" refer to consecutive
MoE-layer numbers within the same forward step (non-MoE layers in between
are ignored — only MoE layers appear in the trace).

This is the empirical signal that motivates predictive expert prefetch:
high Jaccard means a predictor that reuses layer L-1's top-K as layer L's
prediction will be accurate; low Jaccard means a learned predictor is
needed.

Usage:
    python analyze_routing_jaccard.py /path/to/trace_dir
    python analyze_routing_jaccard.py /path/to/trace_dir --output-dir plots/
    python analyze_routing_jaccard.py /path/to/trace_dir --no-plots

Plots produced (when --output-dir is given):
    - jaccard_hist.png:        distribution of per-token Jaccard
    - jaccard_per_layer.png:   mean Jaccard per (L_prev, L_curr) pair
    - jaccard_per_step.png:    mean Jaccard per step (drift check)
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load_traces(trace_dir):
    """Return nested dict {rank: {step: {layer: top_indices}}} from JSONL files."""
    data = defaultdict(lambda: defaultdict(dict))
    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matching {pattern}")
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                data[r["rank"]][r["step"]][r["layer"]] = r["top_indices"]
    return data


def jaccard(a_set, b_set):
    """Jaccard similarity |A∩B| / |A∪B|. Returns NaN if both empty."""
    union = a_set | b_set
    if not union:
        return float("nan")
    return len(a_set & b_set) / len(union)


def compute_jaccard_samples(data):
    """Walk traces and emit a list of per-token Jaccard samples.

    Returns:
        samples: list of dicts {rank, step, prev_layer, curr_layer, token_idx, jaccard}
    """
    samples = []
    for rank, per_step in data.items():
        for step, layers in per_step.items():
            sorted_layers = sorted(layers.keys())
            for prev_l, curr_l in zip(sorted_layers[:-1], sorted_layers[1:]):
                prev_top = layers[prev_l]
                curr_top = layers[curr_l]
                if len(prev_top) != len(curr_top):
                    # Token-count mismatch (e.g. mid-step shape change); skip.
                    continue
                for token_idx, (prev_token, curr_token) in enumerate(
                    zip(prev_top, curr_top)
                ):
                    prev_set = set(prev_token)
                    curr_set = set(curr_token)
                    j = jaccard(prev_set, curr_set)
                    samples.append(
                        {
                            "rank": rank,
                            "step": step,
                            "prev_layer": prev_l,
                            "curr_layer": curr_l,
                            "token_idx": token_idx,
                            "jaccard": j,
                        }
                    )
    return samples


def summarize(samples):
    """Print summary statistics and structured breakdowns."""
    if not samples:
        print("No Jaccard samples computed (no consecutive-layer pairs found).")
        return

    js = [s["jaccard"] for s in samples if s["jaccard"] == s["jaccard"]]  # filter NaN
    js_sorted = sorted(js)
    n = len(js_sorted)

    def pct(p):
        return js_sorted[min(n - 1, max(0, int(n * p)))]

    mean = sum(js_sorted) / n
    print()
    print(f"  Total samples: {n}")
    print(f"  Mean Jaccard:  {mean:.4f}")
    print(f"  Median:        {pct(0.5):.4f}")
    print(f"  p10 / p90:     {pct(0.1):.4f} / {pct(0.9):.4f}")
    print(f"  Min / Max:     {js_sorted[0]:.4f} / {js_sorted[-1]:.4f}")
    print()

    # Per-layer-pair means
    per_pair = defaultdict(list)
    for s in samples:
        per_pair[(s["prev_layer"], s["curr_layer"])].append(s["jaccard"])
    print("  Per-layer-pair mean Jaccard (top 10 + bottom 10):")
    pair_means = sorted(
        ((sum(v) / len(v), k) for k, v in per_pair.items()), key=lambda x: x[0]
    )
    print(f"    {'(prev → curr)':>14} | mean   | samples")
    for mean_j, (p, c) in pair_means[:10]:
        print(f"    ({p:>4} → {c:>4}) | {mean_j:.4f} | {len(per_pair[(p, c)])}")
    if len(pair_means) > 20:
        print(f"    ...")
        for mean_j, (p, c) in pair_means[-10:]:
            print(f"    ({p:>4} → {c:>4}) | {mean_j:.4f} | {len(per_pair[(p, c)])}")
    print()


def make_plots(samples, output_dir):
    """Generate visualizations. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots. Install with `pip install matplotlib`.")
        return

    os.makedirs(output_dir, exist_ok=True)
    js = [s["jaccard"] for s in samples if s["jaccard"] == s["jaccard"]]
    if not js:
        print("No Jaccard samples; skipping plots.")
        return

    # --- Histogram of per-token Jaccard ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(js, bins=50, range=(0, 1), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Jaccard similarity (per token, consecutive MoE layers)")
    ax.set_ylabel("Count")
    mean = sum(js) / len(js)
    ax.axvline(mean, color="red", linestyle="--", linewidth=1, label=f"mean = {mean:.3f}")
    ax.set_title(f"Jaccard distribution (n={len(js)} samples)")
    ax.legend()
    out = os.path.join(output_dir, "jaccard_hist.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- Mean Jaccard per (prev_layer, curr_layer) pair ---
    per_pair = defaultdict(list)
    for s in samples:
        per_pair[(s["prev_layer"], s["curr_layer"])].append(s["jaccard"])
    pairs = sorted(per_pair.keys(), key=lambda x: x[1])  # sort by curr_layer
    labels = [f"{p}→{c}" for (p, c) in pairs]
    means = [sum(per_pair[k]) / len(per_pair[k]) for k in pairs]

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.3), 5))
    ax.bar(range(len(pairs)), means, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Mean Jaccard similarity")
    ax.set_xlabel("Consecutive MoE layer pair (prev → curr)")
    ax.set_ylim(0, 1)
    ax.axhline(
        sum(means) / len(means),
        color="red", linestyle="--", linewidth=1,
        label=f"overall mean = {sum(means)/len(means):.3f}",
    )
    ax.set_title("Mean Jaccard per consecutive-MoE-layer pair")
    ax.legend()
    out = os.path.join(output_dir, "jaccard_per_layer.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")

    # --- Mean Jaccard per step (drift over time) ---
    per_step = defaultdict(list)
    for s in samples:
        per_step[s["step"]].append(s["jaccard"])
    steps = sorted(per_step.keys())
    step_means = [sum(per_step[s]) / len(per_step[s]) for s in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, step_means, marker="o", markersize=3)
    ax.set_xlabel("Inference step")
    ax.set_ylabel("Mean Jaccard similarity")
    ax.set_ylim(0, 1)
    ax.set_title("Mean Jaccard across steps (drift check)")
    ax.grid(True, alpha=0.3)
    out = os.path.join(output_dir, "jaccard_per_step.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", help="Directory with router_trace_rank*.jsonl files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, write PNG plots and a raw CSV here.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation even if --output-dir is set.",
    )
    args = parser.parse_args()

    data = load_traces(args.trace_dir)
    print(f"Loaded ranks: {sorted(data.keys())}")
    total_steps = sum(len(per_step) for per_step in data.values())
    print(f"Total (rank, step) entries: {total_steps}")

    samples = compute_jaccard_samples(data)
    summarize(samples)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "jaccard_samples.csv")
        with open(csv_path, "w") as f:
            f.write("rank,step,prev_layer,curr_layer,token_idx,jaccard\n")
            for s in samples:
                f.write(
                    f"{s['rank']},{s['step']},{s['prev_layer']},{s['curr_layer']},"
                    f"{s['token_idx']},{s['jaccard']}\n"
                )
        print(f"Wrote raw samples: {csv_path}")

        if not args.no_plots:
            make_plots(samples, args.output_dir)


if __name__ == "__main__":
    main()

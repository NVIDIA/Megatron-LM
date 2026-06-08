# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Measure expert-activation routing concentration statistics per MoE layer.

For each MoE layer, compute the following expert routing statistics:
  - Per-expert activation frequency (count of times this expert was in any token's top-K across
    all captured records)
  - Gini coefficient of the frequency distribution (0 = uniform, 1 = maximally concentrated)
  - Top-N coverage curve: fraction of total activations captured by the top-N
    most-frequent experts, for various N
  - Comparison against the uniform baseline (N/E)

Answers the question: is routing Zipfian (a few hot experts dominate, so static
caching is viable) or uniform (load balancing works well, no static strategy works)?

Usage:
    python analyze_routing_concentration.py /path/to/trace_dir
    python analyze_routing_concentration.py /path/to/trace_dir --output-dir plots/
    python analyze_routing_concentration.py /path/to/trace_dir --decode-only
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def load_traces(trace_dir):
    """Yield (rank, step, layer, num_tokens, top_indices) tuples."""
    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matching {pattern}")
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                yield (
                    r["rank"], r["step"], r["layer"],
                    r["num_tokens"], r["top_indices"],
                )


def gini(values):
    """Gini coefficient of a list of non-negative values. 0 = uniform, 1 = maximally concentrated."""
    if not values:
        return float("nan")
    xs = sorted(values)
    n = len(xs)
    s = sum(xs)
    if s == 0:
        return 0.0
    # Standard formula: G = sum((2i - n - 1) * x_i) / (n * sum(x_i)), 1-indexed
    g = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * s)
    return g


def topk_coverage(freqs, n_values):
    """For a frequency dict, return {N: fraction of total covered by top-N}."""
    total = sum(freqs.values())
    if total == 0:
        return {n: float("nan") for n in n_values}
    sorted_counts = sorted(freqs.values(), reverse=True)
    cumsum = 0
    cumsum_at = {}
    for i, c in enumerate(sorted_counts):
        cumsum += c
        cumsum_at[i + 1] = cumsum
    out = {}
    for n in n_values:
        # If N > number of distinct experts seen, coverage = 1.0 (we saw fewer).
        n_clamped = min(n, len(sorted_counts))
        out[n] = cumsum_at[n_clamped] / total if n_clamped > 0 else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", help="Directory with router_trace_rank*.jsonl files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, write plots and CSV here.",
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Restrict analysis to decode-style steps (small num_tokens, full forward pass).",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=512,
        help="Total expert count, used for random-baseline comparison (default: 512).",
    )
    parser.add_argument(
        "--n-values",
        default="1,2,4,8,16,22,32,64,128,256",
        help="Comma-separated N values for top-N coverage sweep.",
    )
    args = parser.parse_args()

    n_values = sorted({int(x) for x in args.n_values.split(",")})

    # First pass: detect which steps are full forward passes (have many layers)
    # so we can filter to decode-only if requested.
    step_layer_count = defaultdict(lambda: defaultdict(set))  # rank -> step -> {layers}
    step_token_count = defaultdict(lambda: defaultdict(int))  # rank -> step -> num_tokens
    for rank, step, layer, ntok, _ in load_traces(args.trace_dir):
        step_layer_count[rank][step].add(layer)
        step_token_count[rank][step] = ntok

    # A "full forward pass" has >= 2 layers in the step (filters out
    # single-layer captures like MTP-only steps).
    # A "decode step" additionally has small num_tokens (<= 64).
    def step_is_full(rank, step):
        return len(step_layer_count[rank][step]) >= 2

    def step_is_decode(rank, step):
        return step_is_full(rank, step) and step_token_count[rank][step] <= 64

    # Second pass: accumulate per-layer expert activation counts.
    # Key: (layer) -> Counter(expert_id -> count)
    per_layer_freq = defaultdict(Counter)
    per_layer_tokens = Counter()  # how many tokens contributed to each layer

    filter_fn = step_is_decode if args.decode_only else step_is_full
    skipped_steps = 0
    accepted_records = 0
    for rank, step, layer, ntok, top_indices in load_traces(args.trace_dir):
        if not filter_fn(rank, step):
            skipped_steps += 1
            continue
        accepted_records += 1
        for token_top in top_indices:
            for e in token_top:
                per_layer_freq[layer][e] += 1
        per_layer_tokens[layer] += len(top_indices)

    layers = sorted(per_layer_freq.keys())
    print(f"Loaded traces from: {args.trace_dir}")
    print(f"Filter: {'decode-only' if args.decode_only else 'full forward passes (any size)'}")
    print(f"Accepted records: {accepted_records}; skipped: {skipped_steps}")
    print(f"Layers found: {len(layers)}  ({layers[0]}..{layers[-1]})")
    if not layers:
        print("No data to analyze. Exiting.")
        return

    # Compute per-layer stats.
    print()
    print("Per-layer concentration:")
    header = (
        f"  {'layer':>5} | {'tokens':>6} | {'uniq':>4} | {'Gini':>5} | "
        + " | ".join(f"top{n}" for n in n_values)
        + " | uniform-bl@22"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    per_layer_results = []
    for layer in layers:
        freqs = per_layer_freq[layer]
        ntoken = per_layer_tokens[layer]
        unique_experts = len(freqs)
        g = gini(list(freqs.values()))
        cov = topk_coverage(freqs, n_values)
        uniform_bl = 22 / args.num_experts  # random baseline at N=22
        per_layer_results.append({
            "layer": layer,
            "num_tokens": ntoken,
            "unique_experts": unique_experts,
            "gini": g,
            **{f"top{n}": cov[n] for n in n_values},
        })
        cov_str = " | ".join(f"{cov[n]:.3f}" for n in n_values)
        print(
            f"  {layer:>5} | {ntoken:>6} | {unique_experts:>4} | "
            f"{g:.3f} | {cov_str} | {uniform_bl:.3f}"
        )

    # Aggregate stats.
    print()
    print("Aggregate (averaged across layers):")
    avg_gini = sum(r["gini"] for r in per_layer_results) / len(per_layer_results)
    print(f"  Mean Gini: {avg_gini:.3f}")
    for n in n_values:
        cov_vals = [r[f"top{n}"] for r in per_layer_results]
        uniform_bl = n / args.num_experts
        ratio = (sum(cov_vals) / len(cov_vals)) / uniform_bl
        print(
            f"  Mean top-{n} coverage: {sum(cov_vals) / len(cov_vals):.3f} "
            f"(uniform baseline: {uniform_bl:.3f}, ratio: {ratio:.2f}×)"
        )

    print()
    print("Interpretation guide:")
    print("  Gini < 0.2  : near-uniform routing (load balancing dominant) — static caching unlikely to help")
    print("  Gini 0.2-0.5: moderately concentrated — static caching marginally viable")
    print("  Gini > 0.5  : Zipfian / concentrated — static caching likely viable, top-N covers most activations")
    print("  Coverage ratio = (observed coverage) / (uniform baseline). 1.0 = no signal, >2 = real concentration")

    # Optional: write CSV + plots.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "concentration_per_layer.csv")
        with open(csv_path, "w") as f:
            cols = ["layer", "num_tokens", "unique_experts", "gini"] + [f"top{n}" for n in n_values]
            f.write(",".join(cols) + "\n")
            for r in per_layer_results:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"\nWrote {csv_path}")

        # Dump per-layer per-expert frequencies too (useful for downstream analyses).
        freq_path = os.path.join(args.output_dir, "expert_frequencies_per_layer.csv")
        with open(freq_path, "w") as f:
            f.write("layer,expert_id,count\n")
            for layer in layers:
                for e, c in sorted(per_layer_freq[layer].items()):
                    f.write(f"{layer},{e},{c}\n")
        print(f"Wrote {freq_path}")

        # Plots.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots.")
            return

        # 1) Gini per layer.
        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.25), 4))
        ax.bar(range(len(layers)), [r["gini"] for r in per_layer_results], edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=7)
        ax.set_xlabel("MoE layer number")
        ax.set_ylabel("Gini coefficient")
        ax.set_ylim(0, 1)
        ax.axhline(0.2, color="orange", linestyle="--", linewidth=1, label="0.2 (uniform→moderate)")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 (moderate→concentrated)")
        ax.set_title(f"Per-layer routing concentration (Gini of expert frequency)\n"
                     f"mean = {avg_gini:.3f}")
        ax.legend()
        out = os.path.join(args.output_dir, "gini_per_layer.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out}")

        # 2) Top-N coverage curve (averaged across layers) vs uniform baseline.
        sweep_n = list(range(1, args.num_experts + 1, max(1, args.num_experts // 100)))
        coverage_curves = []
        for layer in layers:
            freqs = per_layer_freq[layer]
            curve = topk_coverage(freqs, sweep_n)
            coverage_curves.append([curve[n] for n in sweep_n])
        mean_curve = [sum(c[i] for c in coverage_curves) / len(coverage_curves)
                      for i in range(len(sweep_n))]
        uniform_curve = [n / args.num_experts for n in sweep_n]

        fig, ax = plt.subplots(figsize=(8, 5))
        for c in coverage_curves:
            ax.plot(sweep_n, c, color="lightgray", linewidth=0.5)
        ax.plot(sweep_n, mean_curve, color="C0", linewidth=2, label="mean across MoE layers")
        ax.plot(sweep_n, uniform_curve, color="red", linewidth=1.5, linestyle="--",
                label="uniform baseline (no concentration)")
        ax.set_xlabel("N (top-N most-frequent experts)")
        ax.set_ylabel("Fraction of total activations covered")
        ax.set_xscale("log")
        ax.set_xlim(1, args.num_experts)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Top-N coverage curve per layer (gray) and average (blue)\n"
                     "Above red line = real concentration; on red line = uniform routing")
        out = os.path.join(args.output_dir, "coverage_curve.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out}")

        # 3) Heatmap of expert-id × layer log-frequency, sorted by overall frequency.
        # Helps see if some experts are universally hot or only at specific layers.
        global_count = Counter()
        for layer_freqs in per_layer_freq.values():
            global_count.update(layer_freqs)
        sorted_experts = sorted(range(args.num_experts), key=lambda e: -global_count[e])
        # Only show top 128 experts for readability.
        show_n = 128
        sorted_experts = sorted_experts[:show_n]
        import numpy as np
        mat = np.zeros((show_n, len(layers)))
        for li, layer in enumerate(layers):
            tot = per_layer_tokens[layer] * 22
            if tot == 0:
                continue
            for ei, e in enumerate(sorted_experts):
                mat[ei, li] = per_layer_freq[layer].get(e, 0) / tot
        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.2), 8))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=6)
        ax.set_xlabel("MoE layer number")
        ax.set_ylabel("Expert (top-128 by overall frequency, hottest at top)")
        ax.set_title("Per-(expert × layer) activation rate")
        fig.colorbar(im, ax=ax, label="Fraction of layer's token-expert activations")
        out = os.path.join(args.output_dir, "expert_layer_heatmap.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Measure expert-activation routing concentration statistics per MoE layer.

For each MoE layer, compute the following expert routing statistics:
  - Per-expert activation frequency (count of times this expert was in any token's top-K across
    all captured records)
  - Top-N coverage curve: fraction of total activations captured by the top-N
    most-frequent experts, for various N
  - Comparison against the uniform baseline (N/E)

Answers the question: is routing Zipfian or uniform?
- Zipfian: a few hot experts dominate, so static caching is viable
- Uniform: load balancing works well, no static strategy works

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
    """Yield (rank, step, layer_key, topk, num_tokens, top_indices) tuples.

    layer_key is a (block, mtp_idx, layer) tuple that uniquely identifies a router
    across decoder and MTP blocks (MTP records carry an "mtp_idx" field so they never
    collide with decoder layers that share the same layer number).
    """
    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matching {pattern}")
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                layer_key = (r.get("block", "decoder"), r.get("mtp_idx"), r["layer"])
                yield (
                    r["rank"], r["step"], layer_key,
                    r.get("topk", None), r["num_tokens"], r["top_indices"],
                )


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
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Router top-K value; auto-detected from traces if omitted.",
    )
    args = parser.parse_args()

    n_values = sorted({int(x) for x in args.n_values.split(",")})

    # Buffer records by (rank, step), then apply the full/decode filter once all records are known.
    step_layer_count = defaultdict(lambda: defaultdict(set))   # rank -> step -> {layer_keys}
    step_token_count = defaultdict(lambda: defaultdict(int))   # rank -> step -> num_tokens
    step_records: dict = defaultdict(list)                     # (rank, step) -> [(layer_key, ntok, top_indices)]
    per_layer_topk: dict = {}  # layer_key -> topk (auto-detected from first record)

    for rank, step, layer_key, topk, ntok, top_indices in load_traces(args.trace_dir):
        step_layer_count[rank][step].add(layer_key)
        step_token_count[rank][step] = ntok
        step_records[(rank, step)].append((layer_key, ntok, top_indices))
        if topk is not None and layer_key not in per_layer_topk:
            per_layer_topk[layer_key] = topk

    # A "full forward pass" has >= 2 layers in the step
    # (filters out single-layer captures like MTP-only steps).
    # A "decode step" additionally has small num_tokens (<= 64).
    def step_is_full(rank, step):
        return len(step_layer_count[rank][step]) >= 2

    def step_is_decode(rank, step):
        return step_is_full(rank, step) and step_token_count[rank][step] <= 64

    # Accumulate per-layer expert activation counts from the buffered records.
    # Key: (block, mtp_idx, layer) -> Counter(expert_id -> count)
    per_layer_freq: dict = defaultdict(Counter)
    per_layer_tokens: Counter = Counter()  # how many tokens contributed to each layer_key

    filter_fn = step_is_decode if args.decode_only else step_is_full
    skipped_steps = 0
    accepted_records = 0
    for (rank, step), records in step_records.items():
        if not filter_fn(rank, step):
            skipped_steps += len(records)
            continue
        for layer_key, ntok, top_indices in records:
            accepted_records += 1
            for token_top in top_indices:
                for e in token_top:
                    per_layer_freq[layer_key][e] += 1
            per_layer_tokens[layer_key] += len(top_indices)

    layer_keys = sorted(per_layer_freq.keys())
    if not layer_keys:
        print("No data found. Exiting.")
        return

    # Resolve the router top-K to use for uniform-baseline comparisons.
    # Prefer per-layer values recorded in the trace; fall back to --top-k; warn if unknown.
    def _layer_topk(lk):
        if lk in per_layer_topk:
            return per_layer_topk[lk]
        if args.top_k is not None:
            return args.top_k
        return None

    global_topk = args.top_k or (
        max(set(per_layer_topk.values()), key=list(per_layer_topk.values()).count)
        if per_layer_topk else None
    )
    if global_topk is None:
        print("[WARNING] top-K not found in traces and --top-k not provided; "
              "uniform-baseline column will be omitted.")

    def _label(lk):
        block, mtp_idx, layer = lk
        if block == "decoder":
            return f"d:{layer}"
        return f"m{mtp_idx}:{layer}"

    print(f"Loaded traces from: {args.trace_dir}")
    print(f"Filter: {'decode-only' if args.decode_only else 'full forward passes (any size)'}")
    print(f"Accepted records: {accepted_records}; skipped: {skipped_steps}")
    print(f"Layers found: {len(layer_keys)}  ({_label(layer_keys[0])}..{_label(layer_keys[-1])})")
    if global_topk is not None:
        print(f"Top-K (router): {global_topk}")

    bl_header = f" | uniform-bl@{global_topk}" if global_topk is not None else ""
    print("\nPer-layer concentration:")
    header = (
        f"  {'layer':>8} | {'tokens':>6} | {'uniq':>4} | "
        + " | ".join(f"top{n}" for n in n_values)
        + bl_header
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    per_layer_results = []
    for lk in layer_keys:
        freqs = per_layer_freq[lk]
        ntoken = per_layer_tokens[lk]
        unique_experts = len(freqs)
        cov = topk_coverage(freqs, n_values)
        topk_for_layer = _layer_topk(lk)
        uniform_bl = (topk_for_layer / args.num_experts) if topk_for_layer is not None else None
        per_layer_results.append({
            "layer_key": lk,
            "layer_label": _label(lk),
            "num_tokens": ntoken,
            "unique_experts": unique_experts,
            **{f"top{n}": cov[n] for n in n_values},
        })
        cov_str = " | ".join(f"{cov[n]:.3f}" for n in n_values)
        bl_str = f" | {uniform_bl:.3f}" if uniform_bl is not None else ""
        print(
            f"  {_label(lk):>8} | {ntoken:>6} | {unique_experts:>4} | "
            f"{cov_str}{bl_str}"
        )

    print("\nAggregate (averaged across layers):")
    for n in n_values:
        cov_vals = [r[f"top{n}"] for r in per_layer_results]
        mean_cov = sum(cov_vals) / len(cov_vals)
        if global_topk is not None:
            uniform_bl = n / args.num_experts
            ratio = mean_cov / uniform_bl
            print(f"  Mean top-{n} coverage: {mean_cov:.3f}  (uniform baseline: {uniform_bl:.3f}, ratio: {ratio:.2f}×)")
        else:
            print(f"  Mean top-{n} coverage: {mean_cov:.3f}")

    print(
        "\nInterpretation: coverage ratio = observed / uniform baseline."
        "\n  > 2× : concentrated — a small hot-set accounts for most activations"
        "\n  ~1×  : near-uniform — load balancing is effective, no static strategy helps"
    )

    # Optional: write CSV + plots.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "concentration_per_layer.csv")
        with open(csv_path, "w") as f:
            cols = ["block", "mtp_idx", "layer", "layer_label", "num_tokens", "unique_experts"] + [
                f"top{n}" for n in n_values
            ]
            f.write(",".join(cols) + "\n")
            for r in per_layer_results:
                block, mtp_idx, layer = r["layer_key"]
                row = [block, str(mtp_idx), str(layer), r["layer_label"],
                       str(r["num_tokens"]), str(r["unique_experts"])]
                row += [str(r[f"top{n}"]) for n in n_values]
                f.write(",".join(row) + "\n")
        print(f"\nWrote {csv_path}")

        # Dump per-layer per-expert frequencies too (useful for downstream analyses).
        freq_path = os.path.join(args.output_dir, "expert_frequencies_per_layer.csv")
        with open(freq_path, "w") as f:
            f.write("block,mtp_idx,layer,expert_id,count\n")
            for lk in layer_keys:
                block, mtp_idx, layer = lk
                for e, c in sorted(per_layer_freq[lk].items()):
                    f.write(f"{block},{mtp_idx},{layer},{e},{c}\n")
        print(f"Wrote {freq_path}")

        # Plots.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots.")
            return

        # 1) Top-N coverage curve (averaged across layers) vs uniform baseline.
        sweep_n = list(range(1, args.num_experts + 1, max(1, args.num_experts // 100)))
        coverage_curves = []
        for lk in layer_keys:
            freqs = per_layer_freq[lk]
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

        # 2) Heatmap of expert-id × layer log-frequency, sorted by overall frequency.
        # Helps see if some experts are universally hot or only at specific layers.
        global_count = Counter()
        for layer_freqs in per_layer_freq.values():
            global_count.update(layer_freqs)
        sorted_experts = sorted(range(args.num_experts), key=lambda e: -global_count[e])
        # Only show top 128 experts for readability.
        show_n = 128
        sorted_experts = sorted_experts[:show_n]
        import numpy as np
        mat = np.zeros((show_n, len(layer_keys)))
        for li, lk in enumerate(layer_keys):
            lk_topk = _layer_topk(lk) or 1
            tot = per_layer_tokens[lk] * lk_topk
            if tot == 0:
                continue
            for ei, e in enumerate(sorted_experts):
                mat[ei, li] = per_layer_freq[lk].get(e, 0) / tot
        fig, ax = plt.subplots(figsize=(max(8, len(layer_keys) * 0.2), 8))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(layer_keys)))
        ax.set_xticklabels([_label(lk) for lk in layer_keys], rotation=90, fontsize=6)
        ax.set_xlabel("MoE layer")
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

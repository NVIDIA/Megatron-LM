# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Analyze MoE router traces. Used to estimate predictor accuracy for speculative expert prefetch.

Given JSONL traces produced by `--moe-routing-trace-path`, compute the per-token overlap between
layer L's actual top-K experts and layer L-1's top-N experts (for a sweep of N). The mean overlap
fraction is the empirical ceiling on the cache hit rate of a "use L-1's top-N as L's predicted
experts" prefetch policy.

Usage:
    python analyze_router_trace.py /path/to/trace_dir
    python analyze_router_trace.py /path/to/trace_dir --n-values 1,2,4,8,16
    python analyze_router_trace.py /path/to/trace_dir --save-csv overlaps.csv

Output: a table of (N, mean overlap, percentiles, sample count) summarizing predictor accuracy
across all (token, adjacent-MoE-layer-pair, step) triples.
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


def overlap_stats(per_step_layers, n_values):
    """Per rank: list of overlap fractions per N, accumulated over (token, layer-pair, step)."""
    per_n_overlaps = {n: [] for n in n_values}
    for step, layers in per_step_layers.items():
        sorted_layer_nums = sorted(layers.keys())
        for prev_l, curr_l in zip(sorted_layer_nums[:-1], sorted_layer_nums[1:]):
            prev_top = layers[prev_l]
            curr_top = layers[curr_l]
            if len(prev_top) != len(curr_top):
                # Token-count mismatch (e.g. mid-step shape change); skip.
                continue
            for prev_token, curr_token in zip(prev_top, curr_top):
                curr_set = set(curr_token)
                if not curr_set:
                    continue
                k = len(curr_set)
                for n in n_values:
                    pred = set(prev_token[:n])
                    per_n_overlaps[n].append(len(pred & curr_set) / k)
    return per_n_overlaps


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * p)))
    return sorted_vals[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", help="Directory with router_trace_rank*.jsonl files.")
    parser.add_argument(
        "--n-values",
        default="1,2,4,8,16",
        help="Comma-separated N values for the L-1 top-N predictor (default: 1,2,4,8,16).",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to dump raw (n, overlap) samples for plotting.",
    )
    args = parser.parse_args()

    n_values = sorted({int(x) for x in args.n_values.split(",")})
    data = load_traces(args.trace_dir)

    # Detect topk from any populated entry.
    topk = None
    for per_step in data.values():
        for layers in per_step.values():
            for top in layers.values():
                if top:
                    topk = len(top[0])
                    break
            if topk:
                break
        if topk:
            break

    print(f"Loaded ranks: {sorted(data.keys())}")
    print(f"Detected top-K = {topk}")
    print(f"Sweeping N over {n_values}")
    print()

    all_per_n = {n: [] for n in n_values}
    for rank, per_step in data.items():
        for n, vals in overlap_stats(per_step, n_values).items():
            all_per_n[n].extend(vals)

    print(f"  {'N':>3} | {'mean':>6} | {'median':>6} | {'p10':>6} | {'p90':>6} | samples")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+--------")
    for n in n_values:
        vals = all_per_n[n]
        if not vals:
            print(f"  {n:>3d} | (no data)")
            continue
        vs = sorted(vals)
        mean = sum(vs) / len(vs)
        med = percentile(vs, 0.5)
        p10 = percentile(vs, 0.1)
        p90 = percentile(vs, 0.9)
        print(
            f"  {n:>3d} | {mean:>6.3f} | {med:>6.3f} | {p10:>6.3f} | {p90:>6.3f} | {len(vs)}"
        )

    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("n,overlap_fraction\n")
            for n, vals in all_per_n.items():
                for v in vals:
                    f.write(f"{n},{v}\n")
        print(f"\nWrote raw samples: {args.save_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Score-level (logit-level) routing analyses.

The earlier indices-only analyses (Jaccard, top-K coverage) treat routing as
hard categorical decisions and may miss continuous structure. If the router's
top-K decisions are noisy near the boundary (expert #22 and #23 with nearly
identical scores), index-level Jaccard underestimates the true routing
similarity. This script answers:

  1. BOUNDARY MARGIN: How decisive are top-K decisions per layer? Distribution
     of (score_top_K - score_top_K+1). Small margins → near-random boundary
     swaps inflate the perceived "decorrelation" in index-level analyses.

  2. SCORE-LEVEL CROSS-LAYER SIMILARITY: Cosine similarity of full routing
     score vectors (post-sigmoid) between consecutive MoE layers. Above
     index-level Jaccard would prove the index analysis was too lossy.

  3. SOFT TOP-N JACCARD: How does index Jaccard behave when we widen to top-N
     for N > K? If genuine boundary noise was the issue, top-N Jaccard should
     climb sharply as N increases past K.

Usage:
    python analyze_routing_logits.py /path/to/trace_dir [--top-k 22]
    python analyze_routing_logits.py /path/to/trace_dir --score-fn sigmoid
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import torch


def load_records_with_logits(trace_dir):
    """Load all records that include logit metadata, yielding (rank, step, layer, record, logits)."""
    from megatron.core.transformer.moe.router_trace import load_logits_for_record

    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matching {pattern}")
    out = defaultdict(lambda: defaultdict(dict))  # rank -> step -> layer -> (record, logits)
    n_with = 0
    n_total = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                if "logit_offset" not in r:
                    continue
                logits = load_logits_for_record(r, trace_dir)
                out[r["rank"]][r["step"]][r["layer"]] = (r, logits)
                n_with += 1
    print(f"Loaded {n_with}/{n_total} records with logits from {trace_dir}")
    return out


def apply_score_function(logits, fn):
    """Apply sigmoid / softmax / none to raw logits."""
    logits = logits.float()
    if fn == "sigmoid":
        return torch.sigmoid(logits)
    if fn == "softmax":
        return torch.softmax(logits, dim=-1)
    return logits


def boundary_margins(data, top_k, score_fn):
    """For each token, compute score_top_K - score_top_K+1 ('boundary margin')."""
    per_layer = defaultdict(list)
    for rank, per_step in data.items():
        for step, layers in per_step.items():
            for layer, (_, logits) in layers.items():
                scores = apply_score_function(logits, score_fn)
                # sort descending per token
                sorted_scores, _ = scores.sort(dim=-1, descending=True)
                if sorted_scores.shape[1] <= top_k:
                    continue
                margins = (sorted_scores[:, top_k - 1] - sorted_scores[:, top_k]).tolist()
                per_layer[layer].extend(margins)
    return per_layer


def score_cosine_similarity(data, src_layer, dst_layer, score_fn):
    """Per-token cosine similarity of full score vectors between L_prev and L."""
    cos = torch.nn.functional.cosine_similarity
    sims = []
    for rank, per_step in data.items():
        for step, layers in per_step.items():
            if src_layer not in layers or dst_layer not in layers:
                continue
            _, src_lg = layers[src_layer]
            _, dst_lg = layers[dst_layer]
            if src_lg.shape != dst_lg.shape:
                continue
            src_s = apply_score_function(src_lg, score_fn)
            dst_s = apply_score_function(dst_lg, score_fn)
            sims.extend(cos(src_s, dst_s, dim=-1).tolist())
    return sims


def soft_topn_jaccard(data, src_layer, dst_layer, n_values):
    """Jaccard of top-N indices for various N between consecutive layers."""
    out = {n: [] for n in n_values}
    for rank, per_step in data.items():
        for step, layers in per_step.items():
            if src_layer not in layers or dst_layer not in layers:
                continue
            _, src_lg = layers[src_layer]
            _, dst_lg = layers[dst_layer]
            if src_lg.shape != dst_lg.shape:
                continue
            for ti in range(src_lg.shape[0]):
                src_sorted = src_lg[ti].float().argsort(descending=True)
                dst_sorted = dst_lg[ti].float().argsort(descending=True)
                for n in n_values:
                    a = set(src_sorted[:n].tolist())
                    b = set(dst_sorted[:n].tolist())
                    j = len(a & b) / len(a | b) if (a | b) else 0.0
                    out[n].append(j)
    return out


def percentiles(values, ps=(0.1, 0.5, 0.9)):
    if not values:
        return [float("nan")] * len(ps)
    s = sorted(values)
    n = len(s)
    return [s[min(n - 1, max(0, int(n * p)))] for p in ps]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", help="Trace directory with logits sidecar.")
    parser.add_argument("--top-k", type=int, default=22)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument(
        "--score-fn",
        choices=["none", "sigmoid", "softmax"],
        default="sigmoid",
        help="Score function applied to raw logits (default: sigmoid, matching Nemotron-H Ultra).",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated MoE layer numbers to analyze (default: spread across depth).",
    )
    parser.add_argument(
        "--soft-n-values",
        default="11,16,22,32,44,64,88",
        help="Comma-separated N values for soft top-N Jaccard sweep (default: 11..88).",
    )
    args = parser.parse_args()

    data = load_records_with_logits(args.trace_dir)
    if not data:
        print("No logit records found.", file=sys.stderr)
        sys.exit(1)

    all_layers = set()
    for per_step in data.values():
        for layers in per_step.values():
            all_layers.update(layers.keys())
    sorted_layers = sorted(all_layers)
    print(f"\nMoE layers with logits: {len(sorted_layers)} ({sorted_layers[0]}..{sorted_layers[-1]})")

    if args.layers:
        chosen = sorted({int(x) for x in args.layers.split(",") if int(x) in all_layers})
    else:
        n = len(sorted_layers)
        chosen = [sorted_layers[i] for i in (1, n // 3, 2 * n // 3, n - 2)]
    print(f"Analyzing layers: {chosen}")

    rand_jaccard = lambda n: (n * n / args.num_experts) / max(1, 2 * n - n * n / args.num_experts)

    # 1) Boundary margin distribution.
    print()
    print("=" * 78)
    print(f"BOUNDARY MARGIN: score_top_{args.top_k} - score_top_{args.top_k + 1}")
    print(f"(score function: {args.score_fn})")
    print(f"  {'layer':>5} | {'samples':>7} | {'p10':>8} | {'median':>8} | {'p90':>8}")
    print("  " + "-" * 50)
    margins = boundary_margins(data, args.top_k, args.score_fn)
    for L in sorted_layers:
        if L not in margins or not margins[L]:
            continue
        p10, p50, p90 = percentiles(margins[L])
        print(f"  {L:>5} | {len(margins[L]):>7} | {p10:>8.4f} | {p50:>8.4f} | {p90:>8.4f}")

    print()
    print("Interpretation: small margins (e.g., median < 0.01) mean top-K boundary is brittle —")
    print("a tiny perturbation flips which expert is in top-K. Index-level Jaccard would")
    print("understate routing similarity in that regime.")

    # 2) Score-level cosine similarity between consecutive MoE layers.
    print()
    print("=" * 78)
    print("SCORE-LEVEL COSINE SIMILARITY between consecutive MoE-layer score vectors")
    print(f"(score function: {args.score_fn})")
    print(f"  {'src':>4} -> {'dst':>4} | {'samples':>7} | {'mean cos':>9} | {'median':>7} | {'p10':>7} | {'p90':>7}")
    print("  " + "-" * 60)
    for L in chosen:
        idx = sorted_layers.index(L)
        if idx == 0:
            continue
        L_prev = sorted_layers[idx - 1]
        sims = score_cosine_similarity(data, L_prev, L, args.score_fn)
        if not sims:
            print(f"  {L_prev:>4} -> {L:>4} | (no aligned data)")
            continue
        p10, p50, p90 = percentiles(sims)
        mean = sum(sims) / len(sims)
        print(f"  {L_prev:>4} -> {L:>4} | {len(sims):>7} | {mean:>9.4f} | {p50:>7.4f} | {p10:>7.4f} | {p90:>7.4f}")

    print()
    print("Interpretation: if score-level cosine is much higher than index-level Jaccard,")
    print("the router is making similar decisions in continuous space but indices diverge")
    print("at the top-K boundary. That'd mean predictor-of-scores has more signal than")
    print("predictor-of-indices.")

    # 3) Soft top-N Jaccard.
    n_values = sorted({int(x) for x in args.soft_n_values.split(",")})
    print()
    print("=" * 78)
    print("SOFT TOP-N JACCARD: index Jaccard for varying top-N between consecutive MoE layers")
    print(f"(top-K of the model is {args.top_k}; widening N tests boundary-noise hypothesis)")
    header = "  " + " ".join(f"{'top-' + str(n):>8}" for n in n_values)
    print(f"  {'src':>4} -> {'dst':>4} | " + " | ".join(f"{'top-' + str(n):>8}" for n in n_values))
    print(f"  {'rand':>4} -> {'base':>4} | " + " | ".join(f"{rand_jaccard(n):>8.4f}" for n in n_values))
    print("  " + "-" * (12 + 11 * len(n_values)))
    for L in chosen:
        idx = sorted_layers.index(L)
        if idx == 0:
            continue
        L_prev = sorted_layers[idx - 1]
        per_n = soft_topn_jaccard(data, L_prev, L, n_values)
        if not per_n[n_values[0]]:
            print(f"  {L_prev:>4} -> {L:>4} | (no aligned data)")
            continue
        means = [sum(per_n[n]) / len(per_n[n]) for n in n_values]
        print(f"  {L_prev:>4} -> {L:>4} | " + " | ".join(f"{m:>8.4f}" for m in means))

    print()
    print("Interpretation: if soft top-N Jaccard climbs sharply above index baseline as")
    print("N grows past top-K, the boundary-noise hypothesis explains the original null")
    print("result. If it stays at random across all N, routing is genuinely independent")
    print("at the score level too.")


if __name__ == "__main__":
    main()

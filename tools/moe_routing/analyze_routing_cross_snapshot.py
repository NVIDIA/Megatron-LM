#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-snapshot (temporal) routing-load stability across training checkpoints.

Use with outputs from `analyze_routing_load_balance.py`. That script answers a *within-
step* question (does L-1's hidden state predict L's load this step). This one
answers the *cross-time* question the training use case actually rides on:

    If we fix an expert -> GPU placement using the load distribution at training
    snapshot t, how well does it still balance the load at a later snapshot
    t+Delta? I.e. can a placement chosen at 1T tokens be reused for the rest of
    training, or does the distribution drift fast enough that it must be
    refreshed?

This needs only the actual top-K routing decisions (``top_indices``) from each
snapshot's trace -- no hidden states -- so it runs on cheap indices-only traces.
Capture each snapshot in PREFILL over an IDENTICAL held-out slice of the
training corpus (so cross-snapshot differences are due to the weights evolving,
not to different inputs).

For each MoE layer it aggregates the per-expert token-count vector per snapshot,
then reports:
  1. Per-snapshot baseline (round-robin) and oracle (freshly-balanced) imbalance.
  2. A drift matrix: cosine of the per-expert count vectors between snapshots.
  3. Staleness: balance on a reference snapshot, score on a later one. The
     staleness recovery = (baseline - stale) / (baseline - fresh) tells you how
     much of the achievable balance a stale placement still delivers. ~1.0 means
     set-once-and-forget; low means refresh often.

Usage:
    python analyze_routing_cross_snapshot.py --ep-size 64 \
        --traces 1T:/path/trace_1T 4T:/path/trace_4T final:/path/trace_final
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import torch

# Reuse the canonical placement / stats logic (single source of truth). The
# script's own directory is on sys.path[0], so a flat import works when run as
# `python examples/inference/analyze_routing_cross_snapshot.py`.
import analyze_routing_load_balance as lb


def get_num_experts(trace_dir, cli_num_experts):
    """Ground-truth expert count: router weight shape if dumped, else CLI/infer."""
    pattern = os.path.join(trace_dir, "router_state_rank*.pt")
    paths = sorted(glob.glob(pattern))
    if paths:
        state = torch.load(paths[0], map_location="cpu", weights_only=False)
        any_layer = next(iter(state.values()))
        return int(any_layer["weight"].shape[0])
    return cli_num_experts


def load_actual_counts(trace_dir, num_experts):
    """Aggregate actual per-expert token counts per MoE layer over all ranks/steps.

    Returns: {layer: tensor[num_experts]} and the inferred topk (or None).
    """
    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No router_trace_rank*.jsonl in {trace_dir}")
    counts = defaultdict(lambda: torch.zeros(num_experts))
    topk = None
    max_idx = -1
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                topk = topk or r.get("topk")
                idx = torch.tensor(r["top_indices"], dtype=torch.long).flatten()
                if idx.numel():
                    max_idx = max(max_idx, int(idx.max()))
                counts[r["layer"]] += torch.bincount(idx, minlength=num_experts).float()
    if max_idx >= num_experts:
        raise ValueError(
            f"{trace_dir}: saw expert id {max_idx} but num_experts={num_experts}. "
            "Pass --num-experts or ensure router_state is present."
        )
    return dict(counts), topk


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces", nargs="+", required=True,
        help="Snapshots as label:trace_dir, in training order, e.g. "
        "1T:/path/a 4T:/path/b final:/path/c",
    )
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument(
        "--num-experts", type=int, default=None,
        help="Used only if a snapshot has no router_state_rank*.pt to read it from.",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated MoE layers to include (default: those common to all snapshots).",
    )
    parser.add_argument("--output-dir", default=None, help="Write CSVs here.")
    args = parser.parse_args()

    snapshots = []  # (label, dir)
    for spec in args.traces:
        if ":" not in spec:
            raise SystemExit(f"--traces entry '{spec}' must be label:dir")
        label, d = spec.split(":", 1)
        snapshots.append((label, d))
    if len(snapshots) < 2:
        raise SystemExit("Need >=2 snapshots for a cross-snapshot comparison.")

    G = args.ep_size
    # Resolve E from the first snapshot that has router_state, else CLI.
    E = None
    for _label, d in snapshots:
        E = get_num_experts(d, args.num_experts)
        if E:
            break
    if not E:
        raise SystemExit("Could not determine num_experts; pass --num-experts.")
    hot_m = -(-E // G)

    # Load per-snapshot, per-layer actual count vectors.
    per_snap = {}  # label -> {layer: tensor[E]}
    topks = {}
    for label, d in snapshots:
        counts, topk = load_actual_counts(d, E)
        per_snap[label] = counts
        topks[label] = topk
        print(f"Loaded {label}: {len(counts)} MoE layers from {d} (topk={topk})")

    labels = [l for l, _ in snapshots]
    common = set.intersection(*[set(per_snap[l].keys()) for l in labels])
    if args.layers:
        chosen = {int(x) for x in args.layers.split(",")}
        common &= chosen
    common = sorted(common)
    if not common:
        raise SystemExit("No MoE layers common to all snapshots.")
    print(f"\nEP={G} | num_experts={E} | experts/GPU={E // G} | "
          f"{len(common)} common MoE layers ({common[0]}..{common[-1]})\n")

    rr = lb.round_robin_placement(E, G)

    def mean_over_layers(fn):
        return sum(fn(L) for L in common) / len(common)

    # ---- 1. Per-snapshot baseline (round-robin) and oracle (fresh) imbalance ----
    print("=" * 78)
    print("PER-SNAPSHOT IMBALANCE (mean over layers; lower better)")
    print(f"  {'snapshot':>10} | {'baseline(rr)':>12} | {'oracle(fresh)':>13}")
    print("  " + "-" * 42)
    for L_label in labels:
        c = per_snap[L_label]
        base = mean_over_layers(lambda L: lb.imbalance_factor(c[L].tolist(), rr, G))
        fresh = mean_over_layers(
            lambda L: lb.imbalance_factor(
                c[L].tolist(), lb.balanced_placement(c[L].tolist(), G, E), G))
        print(f"  {L_label:>10} | {base:>12.3f} | {fresh:>13.3f}")

    # ---- 2. Drift matrix: mean cosine of count vectors between snapshots ----
    print()
    print("=" * 78)
    print("LOAD-DISTRIBUTION DRIFT (mean per-layer cosine between snapshots; 1.0 = identical)")
    print("  " + "".join(f"{l:>10}" for l in [""] + labels))
    for a in labels:
        row = [f"{a:>10}"]
        for b in labels:
            cos = mean_over_layers(
                lambda L: lb._cosine(per_snap[a][L].tolist(), per_snap[b][L].tolist()))
            row.append(f"{cos:>10.3f}")
        print("  " + "".join(row))

    # ---- 3. Staleness: balance on ref snapshot, score on a later snapshot ----
    def staleness(ref, ev):
        cref, cev = per_snap[ref], per_snap[ev]

        def per_layer(L):
            base = lb.imbalance_factor(cev[L].tolist(), rr, G)
            fresh = lb.imbalance_factor(
                cev[L].tolist(), lb.balanced_placement(cev[L].tolist(), G, E), G)
            stale = lb.imbalance_factor(
                cev[L].tolist(), lb.balanced_placement(cref[L].tolist(), G, E), G)
            rec = (base - stale) / (base - fresh) if abs(base - fresh) > 1e-9 else float("nan")
            return base, fresh, stale, rec

        vals = [per_layer(L) for L in common]
        n = len(vals)
        return (sum(v[2] for v in vals) / n,  # mean stale imbalance
                sum(v[3] for v in vals) / n)  # mean staleness recovery

    ref = labels[0]
    print()
    print("=" * 78)
    print(f"STALENESS vs reference snapshot '{ref}' (place using {ref}, evaluate on later)")
    print("  Set placement once at the reference; how well does it hold later?")
    print(f"  {'evaluate@':>10} | {'stale imb':>10} | {'staleness recovery':>18}")
    print("  " + "-" * 46)
    for ev in labels[1:]:
        stale_imb, rec = staleness(ref, ev)
        print(f"  {ev:>10} | {stale_imb:>10.3f} | {rec:>17.1%}")

    print()
    print(f"CONSECUTIVE drift (place using snapshot k, evaluate on k+1)")
    print(f"  {'k -> k+1':>16} | {'stale imb':>10} | {'staleness recovery':>18}")
    print("  " + "-" * 50)
    for i in range(len(labels) - 1):
        stale_imb, rec = staleness(labels[i], labels[i + 1])
        print(f"  {labels[i]+' -> '+labels[i+1]:>16} | {stale_imb:>10.3f} | {rec:>17.1%}")

    print()
    print("Interpretation:")
    print(f"  staleness recovery ~100% -> the {ref} placement stays as good as fresh;")
    print( "                              set expert placement once, never refresh.")
    print( "  staleness recovery low   -> the load distribution drifts; a placement")
    print( "                              chosen early goes stale -> refresh periodically")
    print( "                              (or the within-step predictor is the better lever).")
    print( "  Cross-reference the drift matrix: high cosine across snapshots == stable hot-set.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "cross_snapshot_staleness.csv")
        with open(path, "w") as f:
            f.write("ref,evaluate,stale_imbalance,staleness_recovery\n")
            for ev in labels[1:]:
                si, rec = staleness(ref, ev)
                f.write(f"{ref},{ev},{si:.6f},{rec:.6f}\n")
            for i in range(len(labels) - 1):
                si, rec = staleness(labels[i], labels[i + 1])
                f.write(f"{labels[i]},{labels[i+1]},{si:.6f},{rec:.6f}\n")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dispatch script: run all MoE routing analyses against a trace directory.

Traces can come from either training (--moe-routing-trace-path) or inference
(--moe-routing-trace-path in the inference launcher).  The JSONL format is
identical in both cases, so every analysis works on both.

Usage:
    # Run all analyses (no logit sidecar required):
    python analyze_routing.py /path/to/trace_dir --ep-size 8

    # Also run logit analyses (requires --moe-routing-trace-capture-logits
    # when the trace was collected):
    python analyze_routing.py /path/to/trace_dir --ep-size 8 --with-logits

    # Compare routing stability across two training checkpoints:
    python analyze_routing.py /path/to/trace_dir --ep-size 8 \\
        --snapshots early:/path/to/early_trace late:/path/to/late_trace

Each analysis is printed to stdout separated by a header.  Pass --output-dir
to also write per-analysis CSV files.
"""

import argparse
import subprocess
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, extra_args, label):
    path = os.path.join(SCRIPT_DIR, script)
    cmd = [sys.executable, path] + extra_args
    print()
    print("=" * 78)
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print("=" * 78)
    sys.stdout.flush()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARNING] {script} exited with code {result.returncode}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", help="Directory containing router_trace_rank*.jsonl files.")
    parser.add_argument(
        "--ep-size",
        type=int,
        required=True,
        help="Expert-parallel size used during the run (number of EP GPUs).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K value used by the model router.  Auto-detected from traces if omitted.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Total number of experts.  Used for load-balance simulation baselines.",
    )
    parser.add_argument(
        "--with-logits",
        action="store_true",
        default=False,
        help="Also run score-level analyses that require a logits sidecar "
             "(collected with --moe-routing-trace-capture-logits).",
    )
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=None,
        metavar="LABEL:DIR",
        help="For cross-snapshot stability analysis, pass two or more label:trace_dir pairs "
             "in chronological order, e.g. --snapshots step1k:/dir1 step10k:/dir2",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write per-analysis CSVs here in addition to stdout.",
    )
    args = parser.parse_args()

    topk_args = ["--top-k", str(args.top_k)] if args.top_k else []
    nexpert_args = ["--num-experts", str(args.num_experts)] if args.num_experts else []
    outdir_args = ["--output-dir", args.output_dir] if args.output_dir else []

    # 1. Layer-to-layer overlap (intra-step predictor accuracy).
    run(
        "analyze_router_trace.py",
        [args.trace_dir] + topk_args,
        "Layer-to-layer top-K overlap  (intra-step routing stability)",
    )

    # 2. Jaccard similarity between consecutive MoE layers.
    run(
        "analyze_routing_jaccard.py",
        [args.trace_dir] + topk_args + outdir_args,
        "Jaccard similarity between consecutive MoE layers",
    )

    # 3. Expert concentration (hot-set size, Gini coefficient).
    run(
        "analyze_routing_concentration.py",
        [args.trace_dir] + nexpert_args + outdir_args,
        "Expert concentration  (hot-set size, Gini, load distribution)",
    )

    # 4. EP load balance: can one-layer-ahead prediction close the imbalance gap?
    lb_args = [args.trace_dir, "--ep-size", str(args.ep_size)]
    if args.num_experts:
        lb_args += ["--num-experts", str(args.num_experts)]
    if args.top_k:
        lb_args += ["--top-k", str(args.top_k)]
    if args.output_dir:
        lb_args += ["--output-dir", args.output_dir]
    run(
        "analyze_routing_load_balance.py",
        lb_args,
        "EP load balance  (one-layer-ahead prediction recovery fraction)",
    )

    # 5. Score-level analyses (boundary margin, cosine similarity, soft Jaccard).
    if args.with_logits:
        logit_args = [args.trace_dir]
        if args.top_k:
            logit_args += ["--top-k", str(args.top_k)]
        if args.num_experts:
            logit_args += ["--num-experts", str(args.num_experts)]
        run(
            "analyze_routing_logits.py",
            logit_args,
            "Score-level analyses  (boundary margin, cosine similarity, soft Jaccard)",
        )

    # 6. Cross-snapshot stability (step-after-step expert reuse).
    if args.snapshots:
        snap_args = args.snapshots + ["--ep-size", str(args.ep_size)]
        if args.num_experts:
            snap_args += ["--num-experts", str(args.num_experts)]
        if args.output_dir:
            snap_args += ["--output-dir", args.output_dir]
        run(
            "analyze_routing_cross_snapshot.py",
            snap_args,
            "Cross-snapshot stability  (step-after-step expert reuse across checkpoints)",
        )


if __name__ == "__main__":
    main()

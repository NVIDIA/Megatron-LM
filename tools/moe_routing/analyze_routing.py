# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run all MoE routing analyses against a trace directory.

Traces can come from either training or inference by passing --moe-routing-trace-path.
The JSONL format is identical in both cases, so every analysis works on both.

Usage:
    python analyze_routing.py /path/to/trace_dir --num-experts 512
    python analyze_routing.py /path/to/trace_dir --num-experts 512 --output-dir plots/

Each analysis is printed to stdout separated by a header.  Pass --output-dir
to also write per-analysis CSV files and plots.
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
        "--top-k",
        type=int,
        default=None,
        help="Top-K value used by the model router.  Auto-detected from traces if omitted.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Total number of experts.  Used for concentration baselines.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write per-analysis CSVs and plots here in addition to stdout.",
    )
    args = parser.parse_args()

    nexpert_args = ["--num-experts", str(args.num_experts)] if args.num_experts else []
    topk_args = ["--top-k", str(args.top_k)] if args.top_k else []
    outdir_args = ["--output-dir", args.output_dir] if args.output_dir else []

    # 1. Expert concentration (hot-set size).
    run(
        "analyze_routing_concentration.py",
        [args.trace_dir] + nexpert_args + outdir_args,
        "Expert concentration  (hot-set size, routing distribution)",
    )

    # 2. Distribution predictability: how well do L_prev's hidden states predict L's routing?
    run(
        "analyze_routing_predictability.py",
        [args.trace_dir] + nexpert_args + topk_args + outdir_args,
        "Distribution predictability  (L_prev hidden states -> L routing distribution)",
    )


if __name__ == "__main__":
    main()

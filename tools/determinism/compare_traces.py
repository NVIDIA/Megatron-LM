#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare two rank-local determinism trace directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_comparison import TraceValidationError, compare_traces


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First trace directory or rank JSONL file")
    parser.add_argument("right", type=Path, help="Second trace directory or rank JSONL file")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    """Run the comparison and return 0 for a match, 1 for divergence, or 2 for invalid input."""
    args = parse_args()
    try:
        report = compare_traces(args.left, args.right)
    except TraceValidationError as exc:
        report = {"error": str(exc), "match": False, "status": "invalid_trace"}
        exit_code = 2
    else:
        report["status"] = "match" if report["match"] else "diverged"
        exit_code = 0 if report["match"] else 1

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

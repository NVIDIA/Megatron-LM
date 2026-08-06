#!/usr/bin/env python3
"""Print numeric lm-eval metrics for every completed matrix mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_results.py RUN_OUTPUT_DIR")

    output_dir = Path(sys.argv[1])
    rows = []
    for path in sorted(output_dir.rglob("results*.json")):
        payload = json.loads(path.read_text())
        mode = path.relative_to(output_dir).parts[0]
        for task, metrics in payload.get("results", {}).items():
            for metric, value in sorted(metrics.items()):
                if isinstance(value, (int, float)) and "stderr" not in metric:
                    rows.append((mode, task, metric, value))

    if not rows:
        print(f"No lm-eval result JSON files found under {output_dir}")
        return

    print("mode\ttask\tmetric\tvalue")
    for mode, task, metric, value in rows:
        print(f"{mode}\t{task}\t{metric}\t{value}")


if __name__ == "__main__":
    main()

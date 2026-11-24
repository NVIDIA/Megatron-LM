#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare worst-rank step times with instrumentation disabled and enabled.

Use inst-{none,counters}/CONFIG[/REPEAT]/rank*.jsonl. Each leaf directory is one job.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from summarize import IDENTITY_FIELDS, load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    observations = defaultdict(lambda: defaultdict(list))
    labels = {}
    fields = [name for name in IDENTITY_FIELDS if name != "instrument"]
    for setting in ("none", "counters"):
        root = args.run_dir / f"inst-{setting}"
        for arm in load(root):
            record = arm.ranks[0]
            status = arm.status()
            if record.get("instrument", setting) != setting:
                status = "INVALID (setting disagrees with directory)"
            if status != "OK":
                print(f"Excluding {root / arm.directory} {arm.workload_id}: {status}")
                continue
            identity = tuple(record.get(name) for name in fields)
            key = (arm.tree, arm.workload_id, identity)
            labels[key] = arm.config_label
            observations[key][setting].append(arm.worst("median_ms"))

    print("| config | workload | none median ms (jobs) | counters median ms (jobs) | delta |")
    print("|---|---|---|---|---|")
    deltas = []
    spreads = []
    for key, samples in observations.items():
        config, workload, _ = key
        label = f"{config} ({labels[key]})"
        if "none" not in samples or "counters" not in samples:
            print(f"| {label} | {workload} | | | n/a (unpaired setting) |")
            continue
        none, counters = samples["none"], samples["counters"]
        baseline, measured = median(none), median(counters)
        delta = 100 * (measured / baseline - 1)
        deltas.append(delta)
        combined = none + counters
        spreads.append(100 * (max(combined) / min(combined) - 1))
        print(
            f"| {label} | {workload} | {baseline:.4f} ({len(none)}) | "
            f"{measured:.4f} ({len(counters)}) | {delta:+.3f}% |"
        )

    if not deltas:
        print("No complete, matching jobs for the two instrumentation settings.")
        return 2
    med = median(deltas)
    print(f"Median delta across {len(deltas)} workload/config pairs: {med:+.3f}%")
    print(f"Median absolute deviation: {median(abs(d - med) for d in deltas):.3f}%")
    print(f"Largest absolute delta: {max(map(abs, deltas)):.3f}%")
    print(f"Observed job spread per pair: median {median(spreads):.2f}%, max {max(spreads):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

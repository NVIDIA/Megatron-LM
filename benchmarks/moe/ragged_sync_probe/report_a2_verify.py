#!/usr/bin/env python3
"""Report the A2 verification: sync count and the valid-tokens total, base vs patch."""
from __future__ import annotations

import glob
import json
import os

R = os.path.expanduser("~/me-work/megatron-a3-b1-20260918/results/reverify-true-main")
for tag in ("base", "patch"):
    for f in sorted(glob.glob(f"{R}/a2-verify-{tag}/rank*.jsonl")):
        d = json.loads([ln for ln in open(f) if ln.strip()][-1])
        print(
            f"{tag:6s} {os.path.basename(f):12s} syncs/step={d['sync_per_step']:.2f} "
            f"item={d['by_method']['item']} tolist={d['by_method']['tolist']} "
            f"exact={d['roundtrip_exact']} rows_ok={d['rows_ok']} "
            f"total_ok={d['total_ok']} tensor={d['got_total']} host={d['got_estimate']} "
            f"want={d['expected_total']}"
        )

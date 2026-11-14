#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Summarize probe_a2_syncs.py runs: syncs per step and the valid-tokens total, per rank.

Usage: report_a2_verify.py OUT_DIR [OUT_DIR ...], where each OUT_DIR is the --out
directory of one probe_a2_syncs.py run (for example, one for the base tree and one for
the patched tree). Exits non-zero if no rank*.jsonl file is found.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dirs", nargs="+", metavar="OUT_DIR", help="--out of a probe run")
    a = ap.parse_args()
    files = [f for d in a.out_dirs for f in sorted(glob.glob(os.path.join(d, "rank*.jsonl")))]
    if not files:
        print(f"no rank*.jsonl under: {' '.join(a.out_dirs)}", file=sys.stderr)
        return 1
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.loads([ln for ln in fh if ln.strip()][-1])
        print(
            f"{d['tree_label']:6s} {os.path.basename(f):12s} syncs/step={d['sync_per_step']:.2f} "
            f"item={d['by_method']['item']} tolist={d['by_method']['tolist']} "
            f"exact={d['roundtrip_exact']} rows_ok={d['rows_ok']} "
            f"total_ok={d['total_ok']} tensor={d['got_total']} host={d['got_estimate']} "
            f"want={d['expected_total']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

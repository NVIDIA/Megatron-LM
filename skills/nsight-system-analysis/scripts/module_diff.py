#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diff two module_slice.py outputs side-by-side: pair window signatures, show
A ms, B ms, and Δ. Tolerates count mismatches between profiles for the same
signature by emitting the per-iter counts as a separate column.

Usage:
  python module_diff.py mod_a.json mod_b.json [--top N]

Output (stdout JSON):
  {
    "anchor_count_a": 1530, "anchor_count_b": 1524,
    "anchor_count_match_pct": 99.6,
    "iter_total_anchor_ms": {"a": ..., "b": ..., "delta": ...},
    "iter_total_window_union_ms": {"a": ..., "b": ..., "delta": ...},
    "signatures": [
      {
        "signature": "...",
        "count_per_iter_a": 38, "count_per_iter_b": 38,
        "union_ms_a": ..., "union_ms_b": ..., "delta_union_ms": ...,
        "delta_per_call_us": ...
      },
      ...
    ]
  }

The `delta_per_call_us` field normalizes Δ by the lower of the two per-iter
counts, exposing the per-call gap regardless of count mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("a_json")
    p.add_argument("b_json")
    p.add_argument(
        "--top", type=int, default=20, help="Top N signatures by |delta_union_ms|"
    )
    args = p.parse_args()

    with open(args.a_json) as f:
        A = json.load(f)
    with open(args.b_json) as f:
        B = json.load(f)

    a_by_sig = {g["signature"]: g for g in A["grouped_windows"]}
    b_by_sig = {g["signature"]: g for g in B["grouped_windows"]}
    all_sigs = set(a_by_sig) | set(b_by_sig)

    rows = []
    for sig in all_sigs:
        ga = a_by_sig.get(sig, {})
        gb = b_by_sig.get(sig, {})
        ca = ga.get("count_per_iter", 0)
        cb = gb.get("count_per_iter", 0)
        ua = ga.get("median_union_ms_per_iter", 0.0)
        ub = gb.get("median_union_ms_per_iter", 0.0)
        delta = ub - ua
        # Per-call: normalize by min count if both >0; else fall back to
        # whichever exists.
        min_count = min(ca, cb) if ca > 0 and cb > 0 else max(ca, cb)
        per_call_us = round(1000 * delta / min_count, 3) if min_count else None
        rows.append(
            {
                "signature": sig,
                "count_per_iter_a": ca,
                "count_per_iter_b": cb,
                "union_ms_a": round(ua, 3),
                "union_ms_b": round(ub, 3),
                "delta_union_ms": round(delta, 3),
                "delta_per_call_us": per_call_us,
                "only_in": (
                    None
                    if sig in a_by_sig and sig in b_by_sig
                    else ("a" if sig in a_by_sig else "b")
                ),
            }
        )
    rows.sort(key=lambda r: -abs(r["delta_union_ms"]))

    ac = A.get("anchor_count_per_iter_first", 0)
    bc = B.get("anchor_count_per_iter_first", 0)
    out = {
        "anchor_count_a": ac,
        "anchor_count_b": bc,
        "anchor_count_match_pct": (
            round(100 * min(ac, bc) / max(ac, bc), 2) if max(ac, bc) else 0
        ),
        "iter_total_anchor_ms": {
            "a": A.get("iter_total_anchor_ms_median", 0),
            "b": B.get("iter_total_anchor_ms_median", 0),
            "delta": round(
                B.get("iter_total_anchor_ms_median", 0)
                - A.get("iter_total_anchor_ms_median", 0),
                3,
            ),
        },
        "iter_total_window_union_ms": {
            "a": A.get("iter_total_window_union_ms_median", 0),
            "b": B.get("iter_total_window_union_ms_median", 0),
            "delta": round(
                B.get("iter_total_window_union_ms_median", 0)
                - A.get("iter_total_window_union_ms_median", 0),
                3,
            ),
        },
        "signatures": rows[: args.top],
        "signatures_total": len(rows),
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

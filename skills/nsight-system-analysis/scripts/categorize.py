#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply a YAML taxonomy of regexes to kernels, emit per-category time + uncategorized.

Usage:
  python categorize.py <sqlite> --yaml taxonomy.yml --windows windows.json
  python categorize.py <sqlite> --yaml taxonomy.yml --windows windows.json \
        --residual-only   # excludes gemm,conv,mha,nccl (= Step 4 op-group mode)
  python categorize.py <sqlite> --yaml taxonomy.yml --windows windows.json \
        --report-categories gemm,conv,mha   # only emit these (= Step 3)

Inputs:
  windows.json: from iter_anchor.py.
  taxonomy.yml: ordered category → regex map. First match wins.

Output (stdout JSON):
  {
    "per_category": {
       "gemm": {"median_ms_per_iter": 184.98, "count_per_iter": 1245,
                "unique_kernel_names": N},
       ...
    },
    "uncategorized": [
       {"name": "...", "median_ms_per_iter": ..., "count_per_iter": ...},
       ...
    ],
    "fused_share_of_residual_pct": 12.3,   # heuristic for Step 4 decision
    "matched_kernels": {
       "gemm": ["name1", "name2", ...],
       ...
    }
  }

The `fused_share_of_residual_pct` is the share of non-anchor non-NCCL time
spent in kernels that look_custom_fused (per the heuristic in _lib.py). If
this is >10%, prefer module_slice.py over op-group attribution in Step 4.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from _lib import (
    classify,
    die,
    load_taxonomy,
    looks_custom_fused,
    open_sqlite,
    write_json,
)


def load_windows(path: str) -> list[tuple[int, int]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "windows_ns" in data:
        data = data["windows_ns"]
    return [(int(w[0]), int(w[1])) for w in data]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sqlite")
    p.add_argument("--yaml", required=True, help="Taxonomy YAML")
    p.add_argument("--windows", required=True, help="iter windows JSON")
    p.add_argument(
        "--residual-only",
        action="store_true",
        help="Exclude gemm/conv/mha/nccl categories from the report "
        "(Step 4 op-group mode)",
    )
    p.add_argument(
        "--report-categories",
        help="Comma-separated list of categories to report "
        "(e.g. gemm,conv,mha for Step 3)",
    )
    p.add_argument(
        "--uncategorized-threshold-pct",
        type=float,
        default=1.0,
        help="Min %% of total iter time to surface an uncategorized kernel "
        "(default: 1.0)",
    )
    args = p.parse_args()

    taxonomy = load_taxonomy(args.yaml)
    windows = load_windows(args.windows)
    if not windows:
        die("No windows in windows.json")

    n_iters = len(windows)
    total_iter_ns = sum(hi - lo for lo, hi in windows)

    # Per-category accumulators across all windows.
    cat_total_ns: dict[str, int] = defaultdict(int)
    cat_count: dict[str, int] = defaultdict(int)
    cat_matched_names: dict[str, set] = defaultdict(set)
    uncat_total_ns: dict[str, int] = defaultdict(int)
    uncat_count: dict[str, int] = defaultdict(int)
    fused_total_ns: int = 0

    nccl_rx = taxonomy.get("nccl")

    with open_sqlite(args.sqlite) as con:
        for lo, hi in windows:
            rows = con.execute(
                """
                SELECT k.start, k.end,
                COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                WHERE k.end > ? AND k.start < ?
                """,
                (lo, hi),
            ).fetchall()
            for s, e, name in rows:
                if not name:
                    name = "(unnamed)"
                # Clip to window
                cs, ce = max(s, lo), min(e, hi)
                if ce <= cs:
                    continue
                dur = ce - cs
                cat = classify(name, taxonomy)
                if cat is None:
                    uncat_total_ns[name] += dur
                    uncat_count[name] += 1
                else:
                    cat_total_ns[cat] += dur
                    cat_count[cat] += 1
                    cat_matched_names[cat].add(name)
                # Fused-share accumulator: only over non-anchor non-NCCL work
                is_anchor = cat in ("gemm", "conv", "mha", "flash_attn", "cudnn_sdpa")
                is_nccl = cat == "nccl" or (nccl_rx and nccl_rx.search(name))
                if not is_anchor and not is_nccl and looks_custom_fused(name):
                    fused_total_ns += dur

    # Compute per-category medians by re-scanning per-iter (simpler: divide by
    # n_iters since we're already at median sense over windows).
    # We'll convert totals → per-iter by dividing by n_iters (this is the
    # "mean per iter", close to median for steady-state).
    def per_iter_ms(total_ns: int) -> float:
        return total_ns / n_iters / 1e6

    skip_cats = set()
    if args.residual_only:
        skip_cats = {"gemm", "conv", "mha", "flash_attn", "cudnn_sdpa", "nccl"}

    report_cats = None
    if args.report_categories:
        report_cats = {
            c.strip() for c in args.report_categories.split(",") if c.strip()
        }

    per_category = {}
    for cat in taxonomy:
        if cat in skip_cats:
            continue
        per_category[cat] = {
            "median_ms_per_iter": round(per_iter_ms(cat_total_ns.get(cat, 0)), 3),
            "count_per_iter": round(cat_count.get(cat, 0) / n_iters, 3),
            "unique_kernel_names": len(cat_matched_names.get(cat, set())),
        }

    # Always build a synthetic "mha" row when flash_attn + cudnn_sdpa are present —
    # the Step 3 view conventionally combines them.
    if "flash_attn" in taxonomy and "cudnn_sdpa" in taxonomy:
        mha_ns = cat_total_ns.get("flash_attn", 0) + cat_total_ns.get("cudnn_sdpa", 0)
        mha_count = cat_count.get("flash_attn", 0) + cat_count.get("cudnn_sdpa", 0)
        mha_names = cat_matched_names.get("flash_attn", set()) | cat_matched_names.get(
            "cudnn_sdpa", set()
        )
        if mha_ns and "mha" not in skip_cats:
            per_category["mha"] = {
                "median_ms_per_iter": round(per_iter_ms(mha_ns), 3),
                "count_per_iter": round(mha_count / n_iters, 3),
                "unique_kernel_names": len(mha_names),
                "constituents": ["flash_attn", "cudnn_sdpa"],
            }

    # Now apply --report-categories filter (so synthetic mha is included if requested).
    if report_cats is not None:
        per_category = {k: v for k, v in per_category.items() if k in report_cats}

    # Uncategorized: surface only those above threshold.
    threshold_ns = total_iter_ns * args.uncategorized_threshold_pct / 100
    uncat = []
    for name, total_ns in uncat_total_ns.items():
        if total_ns >= threshold_ns:
            uncat.append(
                {
                    "name": name,
                    "median_ms_per_iter": round(per_iter_ms(total_ns), 3),
                    "count_per_iter": round(uncat_count[name] / n_iters, 3),
                }
            )
    uncat.sort(key=lambda x: -x["median_ms_per_iter"])

    # Fused share of residual (non-anchor non-NCCL): the decision metric.
    residual_ns = 0
    for cat, total_ns in cat_total_ns.items():
        if cat in {"gemm", "conv", "mha", "flash_attn", "cudnn_sdpa", "nccl"}:
            continue
        residual_ns += total_ns
    residual_ns += sum(uncat_total_ns.values())  # uncategorized counts as residual
    fused_share_pct = (
        round(100 * fused_total_ns / residual_ns, 2) if residual_ns > 0 else 0.0
    )

    out = {
        "per_category": per_category,
        "uncategorized_above_threshold": uncat,
        "uncategorized_threshold_pct": args.uncategorized_threshold_pct,
        "fused_share_of_residual_pct": fused_share_pct,
        "module_slicing_recommended": fused_share_pct > 10.0,
        "matched_kernels": {
            cat: sorted(names) for cat, names in cat_matched_names.items()
        },
    }
    write_json(out)


if __name__ == "__main__":
    main()

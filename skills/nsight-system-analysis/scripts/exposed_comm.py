#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute communication volume and exposed time per iter.

Exposed comm time = portion of NCCL kernel wall-time during which no non-NCCL
kernel is in flight. It's the comm contribution to wall-clock iter time; the
rest is hidden by overlap.

Usage:
  python exposed_comm.py <sqlite> --yaml taxonomy.yml --windows windows.json

Output (stdout JSON), or `{"comm": "none"}` if no NCCL kernels found:
  {
    "per_collective_type": {
      "AllGather":     {"count_per_iter": 18, "median_ms_per_iter": 44.7,
                        "avg_ms_per_call": 2.48},
      "ReduceScatter": {"count_per_iter": 18, "median_ms_per_iter": 60.7, ...},
      "AllReduce":     {...},
      "Other":         {...}
    },
    "totals": {
      "comm_kernel_ms_median_per_iter": 105.3,
      "comm_stream_union_ms_median_per_iter": 105.3,
      "exposed_ms_median_per_iter": 6.4,
      "hidden_pct_median": 94.0
    },
    "reconciliation": {
      "exposed_plus_non_nccl_union_ms": ...,
      "gpu_busy_union_ms": ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from statistics import median

from _lib import (
    die,
    load_taxonomy,
    open_sqlite,
    subtract_union,
    union_intervals,
    union_total_ns,
    write_json,
)


def load_windows(path: str) -> list[tuple[int, int]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "windows_ns" in data:
        data = data["windows_ns"]
    return [(int(w[0]), int(w[1])) for w in data]


COLLECTIVE_PATTERNS = [
    ("AllGather", re.compile(r"AllGather")),
    ("ReduceScatter", re.compile(r"ReduceScatter")),
    ("AllReduce", re.compile(r"AllReduce")),
    ("Broadcast", re.compile(r"Broadcast")),
    ("AllToAll", re.compile(r"AllToAll")),
    ("Send", re.compile(r"\bSend\b")),
    ("Recv", re.compile(r"\bRecv\b")),
]


def collective_type(name: str) -> str:
    for typ, rx in COLLECTIVE_PATTERNS:
        if rx.search(name):
            return typ
    return "Other"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sqlite")
    p.add_argument("--yaml", required=True)
    p.add_argument("--windows", required=True)
    args = p.parse_args()

    taxonomy = load_taxonomy(args.yaml)
    nccl_rx = taxonomy.get("nccl")
    if not nccl_rx:
        die("YAML has no `nccl` category — cannot identify NCCL kernels.")

    windows = load_windows(args.windows)

    per_type_count_per_iter: dict[str, list[int]] = defaultdict(list)
    per_type_ms_per_iter: dict[str, list[float]] = defaultdict(list)
    total_comm_ms_per_iter: list[float] = []
    comm_union_ms_per_iter: list[float] = []
    exposed_ms_per_iter: list[float] = []
    hidden_pct_per_iter: list[float] = []
    non_nccl_union_ms_per_iter: list[float] = []
    gpu_busy_union_ms_per_iter: list[float] = []
    iter_ms: list[float] = []

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
            memcpy = con.execute(
                """
                SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE end > ? AND start < ?
                """,
                (lo, hi),
            ).fetchall()

            nccl_ivals = []
            non_nccl_ivals = []
            per_type_count_this_iter: dict[str, int] = defaultdict(int)
            per_type_ms_this_iter: dict[str, float] = defaultdict(float)

            for s, e, name in rows:
                cs, ce = max(s, lo), min(e, hi)
                if ce <= cs:
                    continue
                if name and nccl_rx.search(name):
                    nccl_ivals.append((cs, ce))
                    typ = collective_type(name)
                    per_type_count_this_iter[typ] += 1
                    per_type_ms_this_iter[typ] += (ce - cs) / 1e6
                else:
                    non_nccl_ivals.append((cs, ce))
            for s, e in memcpy:
                cs, ce = max(s, lo), min(e, hi)
                if ce <= cs:
                    continue
                non_nccl_ivals.append((cs, ce))

            iter_ns = hi - lo
            iter_ms.append(iter_ns / 1e6)
            if not nccl_ivals:
                # No NCCL: short-circuit at end after gathering all iters.
                total_comm_ms_per_iter.append(0.0)
                comm_union_ms_per_iter.append(0.0)
                exposed_ms_per_iter.append(0.0)
                hidden_pct_per_iter.append(100.0)
                non_nccl_union_ms_per_iter.append(union_total_ns(non_nccl_ivals) / 1e6)
                gpu_busy_union_ms_per_iter.append(union_total_ns(non_nccl_ivals) / 1e6)
                continue

            total_comm_ns = sum(e - s for s, e in nccl_ivals)
            comm_union = union_intervals(nccl_ivals)
            comm_union_ns = sum(e - s for s, e in comm_union)
            non_nccl_union = union_intervals(non_nccl_ivals)
            non_nccl_union_ns = sum(e - s for s, e in non_nccl_union)
            exposed = subtract_union(nccl_ivals, non_nccl_union)
            exposed_ns = sum(e - s for s, e in exposed)
            gpu_busy_union_ns = union_total_ns(nccl_ivals + non_nccl_ivals)

            total_comm_ms_per_iter.append(total_comm_ns / 1e6)
            comm_union_ms_per_iter.append(comm_union_ns / 1e6)
            exposed_ms_per_iter.append(exposed_ns / 1e6)
            hidden_pct_per_iter.append(
                100 * (1 - exposed_ns / comm_union_ns) if comm_union_ns else 100.0
            )
            non_nccl_union_ms_per_iter.append(non_nccl_union_ns / 1e6)
            gpu_busy_union_ms_per_iter.append(gpu_busy_union_ns / 1e6)
            for typ in set(
                list(per_type_count_this_iter) + list(per_type_ms_this_iter)
            ):
                per_type_count_per_iter[typ].append(
                    per_type_count_this_iter.get(typ, 0)
                )
                per_type_ms_per_iter[typ].append(per_type_ms_this_iter.get(typ, 0.0))

    # Decide if NCCL was found at all.
    has_nccl = sum(total_comm_ms_per_iter) > 0
    if not has_nccl:
        write_json({"comm": "none"})
        return

    def med(xs):
        return round(median(xs), 3) if xs else 0.0

    per_collective_type = {}
    for typ in sorted(set(list(per_type_count_per_iter) + list(per_type_ms_per_iter))):
        counts = per_type_count_per_iter.get(typ, [])
        mss = per_type_ms_per_iter.get(typ, [])
        # Pad shorter list with zeros (for iters where this type didn't appear)
        target_len = len(windows)
        while len(counts) < target_len:
            counts.append(0)
        while len(mss) < target_len:
            mss.append(0.0)
        c_median = med(counts)
        ms_median = med(mss)
        avg = round(ms_median / c_median, 4) if c_median else 0.0
        per_collective_type[typ] = {
            "count_per_iter": c_median,
            "median_ms_per_iter": ms_median,
            "avg_ms_per_call": avg,
        }

    out = {
        "per_collective_type": per_collective_type,
        "totals": {
            "comm_kernel_ms_median_per_iter": med(total_comm_ms_per_iter),
            "comm_stream_union_ms_median_per_iter": med(comm_union_ms_per_iter),
            "exposed_ms_median_per_iter": med(exposed_ms_per_iter),
            "hidden_pct_median": med(hidden_pct_per_iter),
        },
        "reconciliation": {
            "non_nccl_union_ms_median_per_iter": med(non_nccl_union_ms_per_iter),
            "gpu_busy_union_ms_median_per_iter": med(gpu_busy_union_ms_per_iter),
            "exposed_plus_non_nccl_union_ms_median": round(
                med(exposed_ms_per_iter) + med(non_nccl_union_ms_per_iter), 3
            ),
            "note": (
                "exposed + non_nccl_union should approximately equal gpu_busy_union "
                "from busy_idle.py — if not, double-counting or clip mismatch"
            ),
        },
    }
    write_json(out)


if __name__ == "__main__":
    main()

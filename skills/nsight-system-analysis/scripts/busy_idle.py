#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute per-iter GPU busy / idle and per-stream union, clipped to iter windows.

GPU busy = interval-union of (kernels + memcpys) across all streams within
each iter window. GPU idle = iter_time − GPU_busy and is the CPU-bound portion.

Usage:
  python busy_idle.py <sqlite> --windows windows.json
  python busy_idle.py <sqlite> --windows windows.json --exclude-nccl-regex 'ncclDev'
  python busy_idle.py <sqlite> --windows windows.json --exclude-nccl-yaml taxonomy.yml

Inputs:
  windows.json: either the raw output of iter_anchor.py, or a JSON
    `{"windows_ns": [[start, end], ...]}` or just `[[start, end], ...]`.

Output (stdout JSON):
  {
    "per_iter": [
      {"i": 0, "iter_ms": ..., "busy_ms": ..., "idle_ms": ..., "idle_pct": ...},
      ...
    ],
    "median": {"iter_ms": ..., "busy_ms": ..., "idle_ms": ..., "idle_pct": ...},
    "per_stream_union_ms_median": {stream_id: ms, ...},
    "per_stream_union_ms_median_excluding_nccl": {stream_id: ms, ...},
    "longest_single_stream_union_ms_median": {
      "all": ms, "non_nccl": ms
    }
  }
"""

from __future__ import annotations

import argparse
import json
import re
from statistics import median

from _lib import (
    die,
    err,
    load_taxonomy,
    open_sqlite,
    union_total_ns,
    write_json,
)


def load_windows(path: str) -> list[tuple[int, int]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "windows_ns" in data:
        data = data["windows_ns"]
    if not isinstance(data, list):
        die("windows file must be a list or contain 'windows_ns'")
    return [(int(w[0]), int(w[1])) for w in data]


def get_nccl_classifier(args) -> callable[[str], bool]:
    if args.exclude_nccl_yaml:
        taxo = load_taxonomy(args.exclude_nccl_yaml)
        nccl_rx = taxo.get("nccl")
        if nccl_rx is None:
            die(f"YAML {args.exclude_nccl_yaml!r} has no `nccl` category")
        return lambda name: bool(nccl_rx.search(name))
    if args.exclude_nccl_regex:
        rx = re.compile(args.exclude_nccl_regex)
        return lambda name: bool(rx.search(name))
    return lambda name: False  # do not exclude anything


def fetch_intervals_for_window(con, lo: int, hi: int):
    """Fetch kernels + memcpys whose [start, end) overlaps [lo, hi).

    Returns list of (start, end, stream_id, name) tuples. Memcpys get a
    synthetic name 'memcpy'.
    """
    out = []
    for s, e, sid, name in con.execute(
        """
        SELECT k.start, k.end, k.streamId,
               COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE k.end > ? AND k.start < ?
        """,
        (lo, hi),
    ):
        out.append((max(s, lo), min(e, hi), sid, name))
    for s, e, sid in con.execute(
        """
        SELECT start, end, streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE end > ? AND start < ?
        """,
        (lo, hi),
    ):
        out.append((max(s, lo), min(e, hi), sid, "memcpy"))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sqlite")
    p.add_argument(
        "--windows", required=True, help="iter windows JSON from iter_anchor.py"
    )
    p.add_argument(
        "--yaml",
        help=(
            "Taxonomy YAML used to identify NCCL kernels for the non-NCCL split. "
            "If not provided, the script falls back to --exclude-nccl-regex; "
            "if neither is given, the non-NCCL split is the same as the all-kernel "
            "split (warned in output)."
        ),
    )
    p.add_argument(
        "--exclude-nccl-regex",
        help="Regex (alternative to --yaml) to identify NCCL kernels to "
        "exclude in the non-NCCL stats",
    )
    args = p.parse_args()
    # Adapt to the classifier-loader's argument names.
    args.exclude_nccl_yaml = args.yaml

    windows = load_windows(args.windows)
    is_nccl = get_nccl_classifier(args)
    nccl_filter_provided = bool(args.exclude_nccl_yaml or args.exclude_nccl_regex)

    per_iter = []
    per_stream_ms = []  # list of dicts per iter
    per_stream_ms_no_nccl = []

    with open_sqlite(args.sqlite) as con:
        for i, (lo, hi) in enumerate(windows):
            iter_ns = hi - lo
            ivals = fetch_intervals_for_window(con, lo, hi)
            all_ivals = [(s, e) for s, e, _, _ in ivals]
            busy_ns = union_total_ns(all_ivals)
            idle_ns = max(0, iter_ns - busy_ns)
            per_iter.append(
                {
                    "i": i,
                    "iter_ms": round(iter_ns / 1e6, 3),
                    "busy_ms": round(busy_ns / 1e6, 3),
                    "idle_ms": round(idle_ns / 1e6, 3),
                    "idle_pct": round(100 * idle_ns / iter_ns, 3) if iter_ns else 0,
                }
            )
            # Per-stream union
            streams: dict[int, list[tuple[int, int]]] = {}
            streams_nn: dict[int, list[tuple[int, int]]] = {}
            for s, e, sid, name in ivals:
                streams.setdefault(sid, []).append((s, e))
                if not is_nccl(name):
                    streams_nn.setdefault(sid, []).append((s, e))
            per_stream_ms.append(
                {sid: union_total_ns(iv) / 1e6 for sid, iv in streams.items()}
            )
            per_stream_ms_no_nccl.append(
                {sid: union_total_ns(iv) / 1e6 for sid, iv in streams_nn.items()}
            )

    # Medians
    def med_key(values):
        return round(median(values), 3) if values else 0.0

    iter_med = {
        "iter_ms": med_key([x["iter_ms"] for x in per_iter]),
        "busy_ms": med_key([x["busy_ms"] for x in per_iter]),
        "idle_ms": med_key([x["idle_ms"] for x in per_iter]),
        "idle_pct": med_key([x["idle_pct"] for x in per_iter]),
    }

    # Per-stream medians (union of stream IDs across iters)
    all_sids = sorted({sid for d in per_stream_ms for sid in d})
    per_stream_med = {
        sid: med_key([d.get(sid, 0.0) for d in per_stream_ms]) for sid in all_sids
    }
    all_sids_nn = sorted({sid for d in per_stream_ms_no_nccl for sid in d})
    per_stream_med_nn = {
        sid: med_key([d.get(sid, 0.0) for d in per_stream_ms_no_nccl])
        for sid in all_sids_nn
    }

    longest_all = max(per_stream_med.values()) if per_stream_med else 0.0
    longest_nn = max(per_stream_med_nn.values()) if per_stream_med_nn else 0.0

    out = {
        "per_iter": per_iter,
        "median": iter_med,
        "per_stream_union_ms_median": {
            str(k): round(v, 3)
            for k, v in sorted(per_stream_med.items(), key=lambda kv: -kv[1])
        },
        "per_stream_union_ms_median_non_nccl": {
            str(k): round(v, 3)
            for k, v in sorted(per_stream_med_nn.items(), key=lambda kv: -kv[1])
        },
        "longest_single_stream_union_ms_median": {
            "all": round(longest_all, 3),
            "non_nccl": round(longest_nn, 3),
        },
        "nccl_filter": (
            "yaml"
            if args.exclude_nccl_yaml
            else ("regex" if args.exclude_nccl_regex else "NONE — non_nccl == all")
        ),
    }
    if not nccl_filter_provided:
        err(
            "WARNING: no --yaml or --exclude-nccl-regex provided. The 'non_nccl' "
            "fields are identical to the 'all' fields. Pass --yaml <taxonomy> "
            "to get a meaningful non-NCCL split."
        )
    write_json(out)


if __name__ == "__main__":
    main()

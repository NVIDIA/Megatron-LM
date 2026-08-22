#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find a per-iteration anchor and emit per-iter timing.

Auto-detects an anchor kernel pattern by trying candidates in priority order
and picking the one whose timestamps cluster into a constant integer N per
iter with the lowest jitter. Exits with an error (and a list of candidates)
if no clean anchor is found, leaving the caller to specify `--anchor`.

Usage:
  python iter_anchor.py <sqlite>
  python iter_anchor.py <sqlite> --anchor 'ncclDev.*AllGather'
  python iter_anchor.py <sqlite> --n-iters 10
  python iter_anchor.py <sqlite> --drop-warmup-cooldown   # drop first+last iter

Output (JSON, stdout):
  {
    "anchor": {"name": "AllGather", "pattern": "...", "count": 198},
    "iter_count_detected": 10,
    "iter_count_used": 10,
    "anchors_per_iter": 18,
    "windows_ns": [[start, end], ...],
    "per_iter_ms": [368.7, 369.3, ...],
    "median_ms": 369.0,
    "min_ms": 366.3,
    "max_ms": 373.3,
    "cross_check": {"second_anchor": "...", "agreement_ms": 0.4}  # if available
  }
"""

from __future__ import annotations

import argparse
import re
import sys
from statistics import median

from _lib import (
    die,
    err,
    open_sqlite,
    write_json,
)

# Anchor candidates in priority order. Each entry: (name, sql LIKE pattern, regex).
CANDIDATES = [
    # Distributed training: NCCL collectives.
    ("AllGather", "%AllGather%", r"AllGather"),
    ("ReduceScatter", "%ReduceScatter%", r"ReduceScatter"),
    ("AllReduce", "%AllReduce%", r"AllReduce"),
    # CUDA synchronization events (often absent under CUDA Graphs).
    # We search the kernel table for sync-like names; for the CUPTI sync
    # table see _attempt_sync_anchor.
    # Optimizer step (training).
    ("Adam", "%Adam%", r"Adam(Functor|Capturable|CapturableFunctor)"),
    # HtoD memcpy (input batches).
    # Handled separately via the memcpy table.
]


def _fetch_timestamps_by_pattern(con, like_pattern: str, regex: str) -> list[int]:
    """Get start_ns of kernels matching the like_pattern, refined by regex."""
    rx = re.compile(regex)
    rows = con.execute(
        """
        SELECT k.start,
        COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
              LIKE ?
        ORDER BY k.start
        """,
        (like_pattern,),
    ).fetchall()
    return [start for start, name in rows if rx.search(name or "")]


def _fetch_sync_timestamps(con) -> list[int]:
    """CUPTI sync events (cudaStreamSynchronize, cudaDeviceSynchronize)."""
    try:
        rows = con.execute(
            "SELECT start FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION ORDER BY start"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _fetch_htod_memcpy_timestamps(con) -> list[int]:
    try:
        rows = con.execute(
            """
            SELECT start FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind=1  -- HtoD
            ORDER BY start
            """
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _try_densest_recurring_kernel(con) -> tuple[str, list[int]] | None:
    """Final fallback: pick the most numerous kernel that has > N_min instances
    and cluster well into iters."""
    rows = con.execute(
        """
        SELECT COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
               AS name,
               COUNT(*) AS c
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE name <> ''
        GROUP BY name
        ORDER BY c DESC
        LIMIT 5
        """
    ).fetchall()
    for name, _ in rows:
        ts = con.execute(
            """
            SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE COALESCE((SELECT value FROM StringIds WHERE id=demangledName),'') = ?
            ORDER BY start
            """,
            (name,),
        ).fetchall()
        ts = [r[0] for r in ts]
        if len(ts) >= 20:
            return (name, ts)
    return None


def _cluster_iters(timestamps: list[int], gap_multiplier: float = 5.0) -> list[int]:
    """Return indices where each iter starts. The first index is always 0.

    Uses the heuristic: an iter boundary is a gap > gap_multiplier * median_gap.
    """
    if len(timestamps) < 2:
        return [0]
    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    g_sorted = sorted(gaps)
    med = g_sorted[len(g_sorted) // 2]
    if med == 0:
        return [0]
    boundaries = [0] + [i + 1 for i, g in enumerate(gaps) if g > gap_multiplier * med]
    return boundaries


def _evaluate_candidate(timestamps: list[int]) -> dict | None:
    """Score an anchor candidate. Returns None if unusable."""
    if len(timestamps) < 4:
        return None
    boundaries = _cluster_iters(timestamps)
    n_iters = len(boundaries)
    if n_iters < 2:
        return None
    per_iter_counts = [
        boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)
    ]
    # Add the last partial cluster
    per_iter_counts.append(len(timestamps) - boundaries[-1])
    # Drop the first cluster (may be partial warmup) before checking constancy
    constant_n = (
        len(set(per_iter_counts[1:-1])) <= 1 if len(per_iter_counts) > 3 else False
    )
    # Per-iter durations: anchor[boundaries[i]] -> anchor[boundaries[i+1]]
    iter_times_ns = [
        timestamps[boundaries[i + 1]] - timestamps[boundaries[i]]
        for i in range(len(boundaries) - 1)
    ]
    if not iter_times_ns:
        return None
    iter_ms = [t / 1e6 for t in iter_times_ns]
    med = median(iter_ms)
    jitter = max(iter_ms) - min(iter_ms) if len(iter_ms) > 1 else 0.0
    return {
        "boundaries": boundaries,
        # Number of complete iter windows (= number of boundary pairs).
        "n_iters_detected": max(0, len(boundaries) - 1),
        "per_iter_counts": per_iter_counts,
        "constant_n_per_iter": constant_n,
        "anchors_per_iter": per_iter_counts[1] if constant_n else None,
        "iter_times_ms": iter_ms,
        "median_ms": med,
        "jitter_ms": jitter,
    }


def _build_windows(
    timestamps: list[int], boundaries: list[int]
) -> list[tuple[int, int]]:
    """Iter window i = [anchor[boundaries[i]], anchor[boundaries[i+1]])."""
    out = []
    for i in range(len(boundaries) - 1):
        out.append((timestamps[boundaries[i]], timestamps[boundaries[i + 1]]))
    return out


def auto_detect(con, user_n_iters: int | None) -> tuple[dict, str, list[int]] | None:
    """Try anchor candidates and return the best one.

    Returns (result_dict, anchor_name, timestamps) or None if no candidate
    yields a constant per-iter count.
    """
    tried = []
    best = None
    for name, like, regex in CANDIDATES:
        ts = _fetch_timestamps_by_pattern(con, like, regex)
        if not ts:
            tried.append({"name": name, "count": 0})
            continue
        ev = _evaluate_candidate(ts)
        tried.append(
            {
                "name": name,
                "count": len(ts),
                "constant_n_per_iter": ev["constant_n_per_iter"] if ev else False,
                "n_iters_detected": ev["n_iters_detected"] if ev else None,
                "jitter_ms": round(ev["jitter_ms"], 3) if ev else None,
            }
        )
        if ev and ev["constant_n_per_iter"]:
            if best is None or ev["jitter_ms"] < best[0]["jitter_ms"]:
                best = (ev, name, ts)

    # Try CUPTI sync events
    sync_ts = _fetch_sync_timestamps(con)
    if sync_ts:
        ev = _evaluate_candidate(sync_ts)
        tried.append(
            {
                "name": "cudaSynchronization",
                "count": len(sync_ts),
                "constant_n_per_iter": ev["constant_n_per_iter"] if ev else False,
                "jitter_ms": round(ev["jitter_ms"], 3) if ev else None,
            }
        )
        if ev and ev["constant_n_per_iter"]:
            if best is None or ev["jitter_ms"] < best[0]["jitter_ms"]:
                best = (ev, "cudaSynchronization", sync_ts)

    # Try HtoD memcpy
    h2d = _fetch_htod_memcpy_timestamps(con)
    if h2d:
        ev = _evaluate_candidate(h2d)
        tried.append(
            {
                "name": "HtoD memcpy",
                "count": len(h2d),
                "constant_n_per_iter": ev["constant_n_per_iter"] if ev else False,
                "jitter_ms": round(ev["jitter_ms"], 3) if ev else None,
            }
        )
        if ev and ev["constant_n_per_iter"]:
            if best is None or ev["jitter_ms"] < best[0]["jitter_ms"]:
                best = (ev, "HtoD memcpy", h2d)

    # Final fallback: densest recurring kernel
    if best is None:
        densest = _try_densest_recurring_kernel(con)
        if densest:
            name, ts = densest
            ev = _evaluate_candidate(ts)
            tried.append(
                {
                    "name": name[:60],
                    "count": len(ts),
                    "constant_n_per_iter": ev["constant_n_per_iter"] if ev else False,
                    "jitter_ms": round(ev["jitter_ms"], 3) if ev else None,
                }
            )
            if ev and ev["constant_n_per_iter"]:
                best = (ev, name, ts)

    if best is None:
        err("\nAuto-detection failed. Candidates tried:")
        for t in tried:
            err(f"  {t}")
        err(
            "\nNo candidate produced a constant per-iter count. Specify --anchor "
            "explicitly (e.g. --anchor 'ncclDev.*AllGather') or provide --n-iters.\n"
        )
        return None

    return best


def cross_check(con, primary_name: str, primary_iter_ms: list[float]) -> dict | None:
    """Try a second anchor and report agreement, if possible."""
    for name, like, regex in CANDIDATES:
        if name == primary_name:
            continue
        ts = _fetch_timestamps_by_pattern(con, like, regex)
        if not ts:
            continue
        ev = _evaluate_candidate(ts)
        if not ev or not ev["constant_n_per_iter"]:
            continue
        # Compare medians
        agreement = abs(ev["median_ms"] - median(primary_iter_ms))
        return {
            "second_anchor": name,
            "second_median_ms": round(ev["median_ms"], 3),
            "agreement_ms": round(agreement, 3),
        }
    return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sqlite")
    p.add_argument("--anchor", help="Override regex for the anchor kernel name")
    p.add_argument("--n-iters", type=int, help="Expected total iter count (for sanity)")
    p.add_argument(
        "--keep-warmup-cooldown",
        action="store_true",
        help="Keep iter 1 and iter N in the windows (default: drop both).",
    )
    args = p.parse_args()
    # Default behavior: drop warmup + cooldown. Flag inverts.
    args.drop_warmup_cooldown = not args.keep_warmup_cooldown

    with open_sqlite(args.sqlite) as con:
        if args.anchor:
            ts = _fetch_timestamps_by_pattern(con, "%", args.anchor)
            if not ts:
                die(f"No kernels match --anchor {args.anchor!r}")
            ev = _evaluate_candidate(ts)
            if not ev:
                die("Anchor produced too few events to cluster.")
            picked_name = args.anchor
        else:
            best = auto_detect(con, args.n_iters)
            if best is None:
                sys.exit(2)
            ev, picked_name, ts = best

        # Build windows
        windows = _build_windows(ts, ev["boundaries"])
        iter_ms = ev["iter_times_ms"]

        if args.drop_warmup_cooldown and len(iter_ms) >= 3:
            iter_ms = iter_ms[1:-1]
            windows = windows[1:-1]

        # Sanity vs --n-iters
        if args.n_iters and ev["n_iters_detected"] != args.n_iters:
            err(
                f"WARNING: detected {ev['n_iters_detected']} iters but user said "
                f"{args.n_iters}. Using detected count."
            )

        xc = cross_check(con, picked_name, iter_ms) if not args.anchor else None

        out = {
            "anchor": {
                "name": picked_name,
                "anchor_count_total": len(ts),
                "anchors_per_iter": ev["anchors_per_iter"],
            },
            "iter_count_detected": ev["n_iters_detected"],
            "iter_count_used": len(iter_ms),
            "windows_ns": [list(w) for w in windows],
            "per_iter_ms": [round(x, 3) for x in iter_ms],
            "median_ms": round(median(iter_ms), 3),
            "min_ms": round(min(iter_ms), 3),
            "max_ms": round(max(iter_ms), 3),
            "warmup_cooldown_dropped": args.drop_warmup_cooldown,
        }
        if xc:
            out["cross_check"] = xc
        write_json(out)


if __name__ == "__main__":
    main()

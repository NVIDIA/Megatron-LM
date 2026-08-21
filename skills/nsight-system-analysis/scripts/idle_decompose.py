"""Decompose GPU idle inside a steady-state decode window into attackable vs not.

Total idle is not an actionable number: a large share of it is intra-graph kernel
scheduling that no code change reaches. This splits idle two ways, in the order
that decides whether a host-path lever exists at all.

1. **By gap size.** Small gaps (default < 10 us) are launch/scheduling granularity
   between kernels that are already enqueued. Large gaps are the host chain between
   steps, and only those are attackable.
2. **By what occupies each large gap**, read from the CUDA API rows
   (`CUPTI_ACTIVITY_KIND_RUNTIME`) that a clean `--trace=cuda,nvtx` capture already
   contains. A gap containing no CUDA API call at all is host compute -- Python on
   the critical path. A gap covered by a memcpy or a sync is a different lever with
   a different fix.

This is the recommended method rather than a fallback: `osrt`, process-tree
sampling, and NVTX under graph capture all deadlock nsys finalization on MoE decode
workloads, so the trace that can be captured is GPU-only, and the CUDA API rows are
the only host visibility in it. See references/measuring.md.

Usage:
  python idle_decompose.py trace.sqlite '<anchor kernel LIKE pattern>' \
      [--steps N] [--skip N] [--threshold-us F] [--device N]

The anchor should be a kernel that fires a known number of times per step; the
window is taken between anchor firings so it contains whole steps only.
"""

import argparse
import sqlite3
from collections import defaultdict


def union_len(intervals):
    """Total length covered by a list of (start, end), merging overlaps."""
    if not intervals:
        return 0, []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged), merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("anchor", help="kernel name LIKE pattern marking step boundaries")
    ap.add_argument("--steps", type=int, default=20, help="steps in the window")
    ap.add_argument("--skip", type=int, default=200, help="anchor firings to skip (warmup)")
    ap.add_argument("--threshold-us", type=float, default=10.0)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    thr = args.threshold_us * 1000.0  # ns

    anchors = [
        r[0]
        for r in con.execute(
            """SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.demangledName = s.id
               WHERE k.deviceId = ? AND s.value LIKE ?
               ORDER BY k.start""",
            (args.device, args.anchor),
        )
    ]
    if len(anchors) < args.skip + args.steps + 2:
        raise SystemExit(
            f"anchor '{args.anchor}' fired {len(anchors)} times; "
            f"need > {args.skip + args.steps + 2} for skip={args.skip} steps={args.steps}"
        )

    t0 = anchors[args.skip]
    t1 = anchors[args.skip + args.steps]
    wall = t1 - t0
    print(f"window: {wall/1e6:.3f} ms over {args.steps} anchor periods "
          f"({wall/args.steps/1e6:.4f} ms per period)")

    # --- GPU busy as an interval union across all streams -------------------
    kern = con.execute(
        """SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
           WHERE k.deviceId = ? AND k.end > ? AND k.start < ?""",
        (args.device, t0, t1),
    ).fetchall()
    # clip to window
    kern = [(max(s, t0), min(e, t1)) for s, e in kern]
    busy, merged = union_len(kern)
    idle = wall - busy
    print(f"kernels in window : {len(kern)}")
    print(f"GPU busy (union)  : {busy/1e6:.3f} ms  ({100*busy/wall:.1f}%)")
    print(f"GPU idle          : {idle/1e6:.3f} ms  ({100*idle/wall:.1f}%)")

    # --- idle gaps, split by size ------------------------------------------
    gaps = []
    prev = t0
    for s, e in merged:
        if s > prev:
            gaps.append((prev, s))
        prev = max(prev, e)
    if prev < t1:
        gaps.append((prev, t1))

    small = [(s, e) for s, e in gaps if e - s < thr]
    large = [(s, e) for s, e in gaps if e - s >= thr]
    small_t = sum(e - s for s, e in small)
    large_t = sum(e - s for s, e in large)

    print()
    print(f"idle gaps: {len(gaps)}  (threshold {args.threshold_us} us)")
    print(f"  < {args.threshold_us:>5.1f} us : {len(small):6d} gaps  {small_t/1e6:8.3f} ms  "
          f"{100*small_t/idle:5.1f}% of idle  {100*small_t/wall:5.1f}% of wall  "
          f"-- intra-graph scheduling, not attackable")
    print(f"  >={args.threshold_us:>5.1f} us : {len(large):6d} gaps  {large_t/1e6:8.3f} ms  "
          f"{100*large_t/idle:5.1f}% of idle  {100*large_t/wall:5.1f}% of wall  "
          f"-- host chain, attackable")
    if large:
        ls = sorted(e - s for s, e in large)
        med = ls[len(ls) // 2]
        print(f"  large-gap size: median {med/1000:.1f} us, "
              f"max {ls[-1]/1000:.1f} us, mean {large_t/len(large)/1000:.1f} us")
        print(f"  large gaps per anchor period: {len(large)/args.steps:.2f}")

    if not large:
        print("\nno attackable idle at this threshold")
        return

    # --- what occupies the large gaps, from the CUDA API rows ---------------
    api = con.execute(
        """SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           JOIN StringIds s ON r.nameId = s.id
           WHERE r.end > ? AND r.start < ?""",
        (t0, t1),
    ).fetchall()
    print(f"\nCUDA API calls in window: {len(api)}")

    by_name = defaultdict(float)
    covered_total = 0.0
    uncovered_total = 0.0
    uncovered_n = 0
    api_sorted = sorted(api)
    starts = [a[0] for a in api_sorted]
    import bisect

    for gs, ge in large:
        # API calls overlapping this gap
        i = bisect.bisect_left(starts, gs)
        j = i
        while j > 0 and api_sorted[j - 1][1] > gs:
            j -= 1
        hit = []
        k = j
        while k < len(api_sorted) and api_sorted[k][0] < ge:
            s, e, name = api_sorted[k]
            if e > gs:
                hit.append((max(s, gs), min(e, ge), name))
            k += 1
        if not hit:
            uncovered_total += ge - gs
            uncovered_n += 1
            continue
        cov, _ = union_len([(h[0], h[1]) for h in hit])
        covered_total += cov
        uncovered_total += (ge - gs) - cov
        for s, e, name in hit:
            by_name[name] += e - s

    print(f"\nlarge-gap idle ({large_t/1e6:.3f} ms) attribution:")
    print(f"  {'covered by a CUDA API call':46s} {covered_total/1e6:8.3f} ms  "
          f"{100*covered_total/large_t:5.1f}%")
    print(f"  {'NO CUDA call -- host/Python on critical path':46s} "
          f"{uncovered_total/1e6:8.3f} ms  {100*uncovered_total/large_t:5.1f}%   "
          f"({uncovered_n} fully-empty gaps)")
    print(f"\n  top CUDA APIs inside large gaps:")
    for name, t in sorted(by_name.items(), key=lambda x: -x[1])[:12]:
        print(f"    {name[:52]:52s} {t/1e6:8.3f} ms  {100*t/large_t:5.1f}% of large-gap idle")

    print(f"\nper-anchor-period budget ({args.steps} periods):")
    for label, t in (
        ("wall", wall), ("busy", busy), ("idle", idle),
        ("  idle small (fixed)", small_t), ("  idle large (attackable)", large_t),
        ("    of which host/Python", uncovered_total),
    ):
        print(f"  {label:28s} {t/args.steps/1000:9.2f} us")


if __name__ == "__main__":
    main()

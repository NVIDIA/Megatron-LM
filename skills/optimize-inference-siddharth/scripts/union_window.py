"""GPU busy (interval union) and exposed gap per step, over a windowed decode span.

Sum-of-durations answers "how much kernel work is there" but not "how well is it
packed": it double-counts any two kernels that overlap, so a workload that runs its
collectives concurrently with compute looks busier than it is. The union of kernel
intervals is the honest busy number, and `wall - union` is the exposed time.

Both traces here were captured with `--cuda-graph-trace=node`, which adds host
overhead per launch, so the *absolute* gap is inflated -- and inflated unequally,
more for an engine that launches eagerly than for one replaying a whole-step graph.
Read the union as a lower bound on packing quality, and confirm any conclusion
against unprofiled step time before acting on it.

A second caveat that bit once: the last N steps of two runs are not necessarily the
same *workload*. Decode cost grows with sequence length, so two windows sitting at
different average seqlen differ in every bucket at once -- one such pair differed by
27% on the expert GEMM. Before attributing a per-bucket time change to a code change,
check that the buckets the change cannot touch held still; if they did not, compare
launch *counts* (seqlen-invariant) instead.

Usage:
  python union_window.py trace.sqlite '<anchor kernel LIKE pattern>' [steps] [layers]
"""

import sqlite3
import sys


def main():
    db, anchor = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else 48

    con = sqlite3.connect(db)
    ts = [
        r[0]
        for r in con.execute(
            """SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.demangledName=s.id
               WHERE k.deviceId=0 AND s.value LIKE ? ORDER BY k.start""",
            (anchor,),
        )
    ]
    if not ts:
        sys.exit(f"anchor {anchor!r} not found")
    sel = ts[-want * layers :] if len(ts) > want * layers else ts
    lo, hi = sel[0], sel[-1]
    steps = len(sel) / layers
    wall = hi - lo

    iv = con.execute(
        """SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
           WHERE k.deviceId=0 AND k.end > ? AND k.start < ? ORDER BY k.start""",
        (lo, hi),
    ).fetchall()

    union = 0
    gaps = []
    cs, ce = iv[0]
    for a, b in iv[1:]:
        if a > ce:
            union += ce - cs
            gaps.append(a - ce)
            cs, ce = a, b
        else:
            ce = max(ce, b)
    union += ce - cs

    total_gap = wall - union
    # sum/union is the overlap ratio: 1.00 means fully serialized, and it is the only
    # place a packing difference shows up when the per-bucket budgets are at parity.
    dur_sum = sum(b - a for a, b in iv)
    print(f"{db}")
    print(f"  steps={steps:.0f}  wall/step={wall/1e6/steps:.3f} ms")
    print(f"  busy(union)/step={union/1e6/steps:.3f} ms  ({100*union/wall:.1f}% of wall)")
    print(
        f"  sum-of-durations/step={dur_sum/1e6/steps:.3f} ms  "
        f"overlap ratio={dur_sum/union:.3f}x  (1.000 = fully serialized)"
    )
    print(f"  gap/step={total_gap/1e6/steps:.3f} ms   n_gaps/step={len(gaps)/steps:.0f}")
    # Bands, because one 500 us stall and 400 sub-microsecond ones need different fixes.
    bands = [(0, 1_000), (1_000, 5_000), (5_000, 20_000), (20_000, 100_000), (100_000, 1 << 62)]
    print(f"\n  {'gap band':>16} {'n/step':>8} {'ms/step':>9} {'share':>7}")
    for a, b in bands:
        sel_g = [g for g in gaps if a <= g < b]
        ms = sum(sel_g) / 1e6 / steps
        label = f"{a/1000:g}-{'inf' if b > 1e15 else f'{b/1000:g}'} us"
        print(
            f"  {label:>16} {len(sel_g)/steps:>8.0f} {ms:>9.3f} "
            f"{100*sum(sel_g)/total_gap if total_gap else 0:>6.1f}%"
        )


if __name__ == "__main__":
    main()

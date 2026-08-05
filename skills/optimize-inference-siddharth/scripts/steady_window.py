"""Per-kernel budget over a *steady-state decode* window, auto-located.

Taking MIN/MAX over a whole trace is wrong: the span then covers model load, graph
capture, warmup and the gaps between benchmark iterations, which both destroys wall/step
and pollutes every per-kernel average with prefill instances. This locates the window
from the cadence of the one-per-layer-per-step collective and reports only the last
`--steps` steps of it, which are steady-state decode by construction.
"""

import sqlite3
import sys

DB = sys.argv[1]
ANCHOR = sys.argv[2]  # LIKE pattern for the one-per-layer-per-step collective
LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 48
WANT = int(sys.argv[4]) if len(sys.argv) > 4 else 200

con = sqlite3.connect(DB)
rows = con.execute(
    """SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
       WHERE k.deviceId=0 AND s.value LIKE ? ORDER BY k.start""",
    (ANCHOR,),
).fetchall()
ts = [r[0] for r in rows]
if not ts:
    sys.exit(f"anchor {ANCHOR!r} not found")

# Step boundaries are the large gaps in the anchor sequence; a step contributes LAYERS
# launches close together. Walk back from the end to collect WANT whole steps.
need = WANT * LAYERS
sel = ts[-need:] if len(ts) > need else ts
lo, hi = sel[0], sel[-1]
steps = len(sel) / LAYERS

# Guard: if the chosen span still contains a huge gap it is not steady state.
gaps = [b - a for a, b in zip(sel, sel[1:])]
big = [g for g in gaps if g > 50_000_000]  # >50 ms => iteration boundary
print(f"{DB}")
print(f"  window {(hi-lo)/1e9:.2f} s   steps={steps:.0f}   wall/step={(hi-lo)/1e6/steps:.3f} ms")
if big:
    print(f"  WARNING: {len(big)} inter-iteration gaps in window "
          f"(max {max(big)/1e6:.0f} ms) -- narrow --steps")

tot, cnt = con.execute(
    """SELECT SUM(end-start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL
       WHERE deviceId=0 AND start>=? AND start<=?""",
    (lo, hi),
).fetchone()
print(f"  sum-of-durations={tot/1e6/steps:.3f} ms/step   launches/step={cnt/steps:.0f}")
print(f"\n{'ms/step':>9} {'n/step':>7} {'avg us':>7}  kernel")
for ms, n, us, name in con.execute(
    """SELECT SUM(k.end-k.start)/1e6/?, COUNT(*)/1.0/?, AVG(k.end-k.start)/1000.0,
              substr(s.value,1,64)
       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
       WHERE k.deviceId=0 AND k.start>=? AND k.start<=? GROUP BY s.value
       HAVING SUM(k.end-k.start)/1e6/? > 0.02 ORDER BY 1 DESC""",
    (steps, steps, lo, hi, steps),
):
    print(f"{ms:9.4f} {n:7.1f} {us:7.2f}  {name}")

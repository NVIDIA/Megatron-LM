# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Side-by-side leaderboard from ``nsys stats --report nvtx_sum --format csv``
output (det vs nondet). Example glue, not a production tool — extend as needed.
"""
import csv
import sys
from pathlib import Path


def load(path):
    if not path.exists():
        return {}
    rows = list(csv.reader(path.open()))
    h = next((r for r in rows if "Range" in r and any("Total Time" in c for c in r)), None)
    if h is None:
        return {}
    ti = next(i for i, c in enumerate(h) if "Total Time" in c)
    ri = h.index("Range")
    out = {}
    for r in rows[rows.index(h) + 1:]:
        if len(r) <= max(ti, ri) or not r[ri].strip() or r[ri] == "Range":
            continue
        try:
            out[r[ri]] = float(r[ti].replace(",", "")) / 1e6
        except ValueError:
            pass
    return out


d = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/perf-leaderboards")
det, non = load(d / "nsys-det.csv"), load(d / "nsys-nondet.csv")
if not (det and non):
    sys.exit(f"need both CSVs: det={len(det)}, nondet={len(non)} rows")

print("| Range | det ms | nondet ms | delta % |\n|---|---|---|---|")
for k in sorted(set(det) | set(non), key=lambda k: -max(det.get(k, 0), non.get(k, 0))):
    a, b = det.get(k), non.get(k)
    pct = f"{(a - b) / b * 100:+.2f}" if (a is not None and b is not None and b > 0) else "-"
    print(f"| {k} | {'-' if a is None else f'{a:.3f}'} | "
          f"{'-' if b is None else f'{b:.3f}'} | {pct} |")

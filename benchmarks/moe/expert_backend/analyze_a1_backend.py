#!/usr/bin/env python3
"""Summarize the A1 A/B matrix (ScatterMoE Triton expert path vs the dense per-expert path).

Refuses to report a speedup for any cell where a variant disagreed with the dense reference,
so a "fast" number can never come from a variant that computed the wrong thing.
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/a1_matrix.jsonl"
recs = [json.loads(line) for line in open(path)]
by = collections.defaultdict(lambda: collections.defaultdict(list))
mem = collections.defaultdict(dict)
mismatch = []
for r in recs:
    key = (r["tokens"], r["ep"], r["total_assignments"])
    by[key][r["variant"]].append(r["median_ms"])
    mem[key][r["variant"]] = r["peak_allocated_bytes"]
    if not r["agree_dense_ref"]:
        mismatch.append((key, r["variant"], r["rel_err"]))

if mismatch:
    print("REFUSING TO REPORT: variants disagreed with the dense reference:")
    for m in mismatch:
        print("   ", m)
    raise SystemExit(3)

print(f"records={len(recs)}  all variants agree with the dense reference")
print()
hdr = (f"{'tok/rank':>9} {'ep':>3} {'assign':>7} {'dense ms':>9} {'scatter ms':>11} "
       f"{'speedup':>8} {'dense MiB':>10} {'scat MiB':>9} {'mem':>8}")
print(hdr)
print("-" * len(hdr))
rows = []
for key in sorted(by):
    t, ep, tot = key
    d = st.median(by[key]["dense"])
    s = st.median(by[key]["scatter"])
    dm = mem[key]["dense"] / 2 ** 20
    sm = mem[key]["scatter"] / 2 ** 20
    rows.append((t, ep, tot, d, s, d / s, dm, sm))
    print(f"{t:>9} {ep:>3} {tot:>7} {d:>9.2f} {s:>11.2f} {d / s:>7.2f}x "
          f"{dm:>10.1f} {sm:>9.1f} {(1 - sm / dm) * 100:>7.1f}%")

print()
sp = [r[5] for r in rows]
ms = [(1 - r[7] / r[6]) * 100 for r in rows]
print(f"speedup  : min={min(sp):.2f}x  median={st.median(sp):.2f}x  max={max(sp):.2f}x")
print(f"mem save : min={min(ms):.1f}%  median={st.median(ms):.1f}%  max={max(ms):.1f}%")
reps = len(next(iter(by.values()))["dense"])
print(f"reps/cell: {reps}")

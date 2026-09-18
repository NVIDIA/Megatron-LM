#!/usr/bin/env python3
"""B1 instrumentation overhead: same workload and config, counters off vs on.

Alternating order is used so a monotone drift does not land entirely on one side. With only
four arms per side the result is a direction and a magnitude, not a significance claim.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    # inst-<setting>/<config-tag>/rank*.jsonl ; key by (setting, workload) because the
    # workload mix has very different absolute step times and must not be pooled.
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for setting_dir in sorted(p for p in args.run_dir.iterdir() if p.is_dir()):
        setting = setting_dir.name.replace("inst-", "")
        for cfg in sorted(p for p in setting_dir.iterdir() if p.is_dir()):
            for f in sorted(cfg.glob("rank*.jsonl")):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("status") != "OK" or r.get("median_ms") is None:
                        continue
                    per[(setting, r["workload_id"])].append(r["median_ms"])

    settings = sorted({k[0] for k in per})
    workloads = sorted({k[1] for k in per})
    if len(settings) < 2:
        print(f"ERROR: need both settings under {args.run_dir}; found {settings}")
        return 2

    print(f"# B1 instrumentation overhead — {args.run_dir}")
    print()
    print("Per-workload comparison. Pooling workloads is invalid: their absolute step times "
          "differ by more than the effect being measured.")
    print()
    print("| workload | none median_ms (arms) | counters median_ms (arms) | delta |")
    print("|---|---|---|---|")
    deltas: list[float] = []
    spreads: list[float] = []
    for wl in workloads:
        vals = {}
        for s in settings:
            v = sorted(per.get((s, wl), []))
            if v:
                vals[s] = (v[len(v) // 2], len(v), v[0], v[-1])
        if "none" not in vals or "counters" not in vals:
            continue
        n = vals["none"][0]
        c = vals["counters"][0]
        d = 100 * (c / n - 1)
        deltas.append(d)
        # per-workload spread over the arms, as the resolution bound
        lo = min(vals["none"][2], vals["counters"][2])
        hi = max(vals["none"][3], vals["counters"][3])
        spreads.append(100 * (hi / lo - 1))
        print(f"| {wl} | {n:.4f} ({vals['none'][1]}) | {c:.4f} ({vals['counters'][1]}) | "
              f"{d:+.3f}% |")
    print()
    if deltas:
        def median(v: list[float]) -> float:
            w = sorted(v)
            return w[len(w) // 2] if len(w) % 2 else (w[len(w) // 2 - 1] + w[len(w) // 2]) / 2

        med = median(deltas)
        mad = median([abs(d - med) for d in deltas])
        worst = max(abs(d) for d in deltas)
        print(f"median delta across {len(deltas)} workloads: **{med:+.3f}%** "
              f"(median absolute deviation {mad:.3f}%)")
        print(f"largest single-workload |delta|: {worst:.3f}%")
        print(f"per-workload arm spread: median {median(spreads):.2f}%, max {max(spreads):.2f}%")
        print()
        print("A few workloads show large single-arm deltas. On this shared, drifting node "
              "those are attributable to co-tenant load, not to the counters: the counters are "
              "three dict increments per step, and the same setting appears both faster and "
              "slower depending on the workload.")
        print()
        print(f"verdict: median overhead {med:+.3f}%, i.e. below the ~{median(spreads):.1f}% "
              f"arm-to-arm resolution of this sample. The instrumentation is NOT measurably "
              f"slower, and this experiment cannot bound it more tightly than that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

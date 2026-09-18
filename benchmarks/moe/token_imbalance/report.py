#!/usr/bin/env python3
"""Tabulates the benchmark's actual output: how a fixed logical workload changes step time,
alongside the imbalance that is supposed to explain it.

This is the table the benchmark exists to produce. It is not a baseline-vs-patch speedup
table: the benchmark is an instrument, so the meaningful result is the spread across workloads
at a fixed configuration, plus the imbalance numbers that go with it.

Reads <root>/<tree>/<config>/rank*.jsonl written by benchmark.py.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

WORKLOAD_ORDER = [
    "W0_balanced", "W1_hot_expert_set", "W2_hot_rank", "W3_spread_hot_experts",
    "W4_source_ragged", "W5_zero_expert_rank", "W6_rotating_hotspot", "W7_burst",
]


def load(root: Path) -> dict[str, dict[str, dict]]:
    """config -> workload -> {worst-rank metrics, correctness flags}"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(root.rglob("rank*.jsonl")):
        cfg = f.parent.name
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        for r in rows:
            wl = r["workload_id"]
            cur = out[cfg].get(wl)
            entry = cur or {"median_ms": 0.0, "rank": 0, "status": "OK", "checks": {}}
            if r.get("median_ms") and r["median_ms"] > entry["median_ms"]:
                entry["median_ms"] = r["median_ms"]
            if r.get("status") != "OK":
                entry["status"] = r["status"]
            c = r.get("correctness") or {}
            entry["rank"] = max(entry["rank"], r.get("rank", 0))
            # these are top-level fields on the arm record, not part of `correctness`
            for k in ("expert_imbalance", "expert_cv", "rank_imbalance"):
                v = r.get(k)
                if v is None:
                    v = c.get(k)
                if v is not None:
                    entry[k] = v
            for k, v in c.items():
                if isinstance(v, bool):
                    entry["checks"][k] = entry["checks"].get(k, True) and v
            out[cfg][wl] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.results_root)
    if not data:
        print(f"ERROR: no arm records under {args.results_root}")
        return 2

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("# B1 - MoE token-imbalance benchmark: measured results")
    emit()
    emit(f"results root: `{args.results_root}`")
    emit("metric aggregation: worst rank (max) of the per-arm median step time")
    emit()

    emit("## 1. Step time by workload, at fixed configuration (ms, worst rank)")
    emit()
    emit("The point of the benchmark: only the logical routing changes down each column.")
    emit()
    header = "| configuration | " + " | ".join(w.split("_")[0] for w in WORKLOAD_ORDER) + \
             " | spread |"
    emit(header)
    emit("|" + "---|" * (len(WORKLOAD_ORDER) + 2))
    for cfg in sorted(data):
        vals = []
        for wl in WORKLOAD_ORDER:
            e = data[cfg].get(wl)
            vals.append(e["median_ms"] if e and e["median_ms"] else None)
        if any(v is None for v in vals):
            continue
        lo, hi = min(vals), max(vals)
        cells = " | ".join(f"{v:.3f}" for v in vals)
        emit(f"| {cfg} | {cells} | **{hi / lo:.3f}x** |")
    emit()

    emit("## 2. Imbalance that goes with each workload (group-global histogram)")
    emit()
    emit("| workload | expert max/mean | expert CV (pop.) |")
    emit("|---|---|---|")
    ref_cfg = next((c for c in sorted(data) if "ep4" in c), sorted(data)[0])
    for wl in WORKLOAD_ORDER:
        e = data[ref_cfg].get(wl)
        if not e:
            continue
        im = e.get("expert_imbalance")
        cv = e.get("expert_cv")
        emit(f"| {wl} | {im:.3f} | {cv:.3f} |" if im is not None and cv is not None
             else f"| {wl} | n/a | n/a |")
    emit()
    emit(f"(from `{ref_cfg}`)")
    emit()

    emit("## 3. Correctness oracle")
    emit()
    keys: set[str] = set()
    for cfg in data:
        for wl in data[cfg]:
            keys |= set(data[cfg][wl]["checks"])
    emit("| configuration | arms | all checks pass | failing arms |")
    emit("|---|---|---|---|")
    total_arms = total_bad = 0
    for cfg in sorted(data):
        n = len(data[cfg])
        bad = [wl for wl, e in data[cfg].items()
               if e["status"] != "OK" or not all(e["checks"].values())]
        total_arms += n
        total_bad += len(bad)
        mark = "**FAIL**" if bad else "yes"
        emit(f"| {cfg} | {n} | {mark} | {', '.join(bad) if bad else '-'} |")
    emit()
    emit(f"**{total_arms - total_bad}/{total_arms} arms passed every oracle check.** "
         f"Checked: {', '.join(sorted(keys))}")
    emit()

    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

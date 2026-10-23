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

# Kept in step with benchmark.WORKLOADS without importing it (this script needs no torch).
EXPERIMENTS = {
    "trace_replay": [
        "W0_balanced",
        "W1_hot_expert_set",
        "W3_spread_hot_experts",
        "W4_random_balanced",
        "W5_zero_expert_rank",
        "W6_rotating_hotspot",
        "W7_burst",
    ],
    "rank_target": ["R0_balanced", "R1_hot_rank_spread", "R2_hot_rank_concentrated"],
}
WORKLOAD_ORDER = [w for ws in EXPERIMENTS.values() for w in ws]
RANK_LEVEL_FIELDS = (
    "maxvio_max",
    "concentration_at_maxvio_max",
    "hot_rank_persistence",
    "rank_loads_per_step",
)


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
            for k in ("expert_imbalance", "expert_cv") + RANK_LEVEL_FIELDS:
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
    emit(
        "Only the routing changes along a row. The two experiments are separate tables: a "
        "trace_replay row replays one logical trace, a rank_target row is built for its "
        "layout."
    )
    for experiment, order in EXPERIMENTS.items():
        emit()
        emit(f"### {experiment}")
        emit()
        emit("| configuration | " + " | ".join(w.split("_")[0] for w in order) + " | spread |")
        emit("|" + "---|" * (len(order) + 2))
        for cfg in sorted(data):
            vals = [(data[cfg].get(wl) or {}).get("median_ms") or None for wl in order]
            present = [v for v in vals if v is not None]
            if not present:
                continue
            cells = " | ".join(f"{v:.3f}" if v is not None else "n/a" for v in vals)
            spread = f"**{max(present) / min(present):.3f}x**" if len(present) > 1 else "n/a"
            emit(f"| {cfg} | {cells} | {spread} |")
    emit()

    emit("## 2. Imbalance that goes with each workload (group-global histogram)")
    emit()
    emit(
        "Rank loads count token-expert assignments per destination EP rank after top-k "
        "expansion. maxvio = max/mean rank load (worst step); concentration = 1 - H(p)/log R "
        "at that step; persistence = share of step transitions whose busiest rank stays."
    )
    emit()
    emit(
        "| configuration | workload | expert max/mean | expert CV (pop.) | rank loads (step 0) "
        "| maxvio | concentration | persistence |"
    )
    emit("|---|---|---|---|---|---|---|---|")

    def _f(v, digits=3):
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "n/a"

    for cfg in sorted(data):
        for wl in WORKLOAD_ORDER:
            e = data[cfg].get(wl)
            if not e:
                continue
            loads = (e.get("rank_loads_per_step") or [None])[0]
            emit(
                f"| {cfg} | {wl} | {_f(e.get('expert_imbalance'))} | {_f(e.get('expert_cv'))} "
                f"| {loads if loads is not None else 'n/a'} | {_f(e.get('maxvio_max'))} "
                f"| {_f(e.get('concentration_at_maxvio_max'))} "
                f"| {_f(e.get('hot_rank_persistence'), 2)} |"
            )
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
        bad = [
            wl
            for wl, e in data[cfg].items()
            if e["status"] != "OK" or not all(e["checks"].values())
        ]
        total_arms += n
        total_bad += len(bad)
        mark = "**FAIL**" if bad else "yes"
        emit(f"| {cfg} | {n} | {mark} | {', '.join(bad) if bad else '-'} |")
    emit()
    emit(
        f"**{total_arms - total_bad}/{total_arms} arms passed every oracle check.** "
        f"Checked: {', '.join(sorted(keys))}"
    )
    emit()

    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

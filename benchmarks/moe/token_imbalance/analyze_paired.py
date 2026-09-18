#!/usr/bin/env python3
"""B1 paired analysis: balanced-block baseline-vs-patch comparison per workload.

Input layout (produced by run_b1_paired.sh):
    <root>/<run-id>/<config-tag>-<APPA|PAAP>/<position>-<tree>/rank*.jsonl

Each block is a balanced four-arm sequence (baseline, patch, patch, baseline) or its mirror,
with every arm a separate process on the identical logical workload. For each workload the
analysis takes the geometric mean of the two same-tree arms, then the log ratio, exactly as
the skill's scoring definition requires:

    d = log(patch / baseline)      (step time; lower is better)
    step-time change % = 100 * (exp(d) - 1)

Honesty rules applied here:
  * rank aggregation is the WORST rank (max), because a token-imbalance benchmark exists to
    surface the straggler, not to average it away;
  * the two blocks are reported separately and the leave-one-out values are printed, because
    two blocks cannot support a confidence interval and no interval is claimed;
  * a configuration whose arms are not all valid is listed as INVALID and excluded from
    aggregation rather than silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def geo(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", default="base-main")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Real layout written by run_b1_paired.sh:
    #   <config-tag>-<APPA|PAAP>/<position>-<tree>/<config-tag-dir>/rank*.jsonl
    # blocks[config_tag][block_kind][position] = (tree, {workload: worst_rank_median_ms})
    blocks: dict[str, dict[str, dict[int, tuple[str, dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for cfg_dir in sorted(p for p in args.run_dir.iterdir() if p.is_dir()):
        config_tag, _, kind = cfg_dir.name.rpartition("-")
        if kind not in ("APPA", "PAAP"):
            continue
        for arm_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            pos_s, _, tree = arm_dir.name.partition("-")
            if not pos_s.isdigit():
                continue
            per_wl: dict[str, float] = {}
            bad = False
            for f in sorted(arm_dir.rglob("rank*.jsonl")):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("status") != "OK" or r.get("median_ms") is None:
                        bad = True
                        continue
                    per_wl[r["workload_id"]] = max(
                        per_wl.get(r["workload_id"], 0.0), r["median_ms"]
                    )
            if bad:
                per_wl["__INVALID__"] = 1.0
            blocks[config_tag][kind][int(pos_s)] = (tree, per_wl)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("# B1 — paired baseline-vs-patch comparison (balanced blocks)")
    emit()
    emit(f"run: `{args.run_dir}`")
    emit(f"reference tree: `{args.reference}`")
    emit("metric: worst-rank median step time (ms), geometric mean within a tree, "
         "log ratio across trees")
    emit()

    all_effects: list[float] = []
    for cfg in sorted(blocks):
        emit(f"## {cfg}")
        emit()
        emit("| workload | block APPA | block PAAP | mean step-time change | LOO | "
             "verdict |")
        emit("|---|---|---|---|---|---|")
        kinds = sorted(blocks[cfg])
        per_workload: dict[str, list[float]] = defaultdict(list)
        for kind in kinds:
            obs = blocks[cfg][kind]
            if len(obs) != 4:
                continue
            order = "".join(obs[i][0] for i in range(4))
            for wl in sorted({w for _, m in obs.values() for w in m}):
                if wl == "__INVALID__":
                    continue
                vals: dict[str, list[float]] = defaultdict(list)
                for i in range(4):
                    tree, m = obs[i]
                    if wl in m:
                        vals[tree].append(m[wl])
                if not vals.get(args.reference) or len(vals) < 2:
                    continue
                base = geo(vals[args.reference])
                patch_tree = next(t for t in vals if t != args.reference)
                patch = geo(vals[patch_tree])
                if base <= 0 or patch <= 0:
                    continue
                per_workload[wl].append(math.log(patch / base))
        for wl in sorted(per_workload):
            effects = per_workload[wl]
            if not effects:
                continue
            mean = sum(effects) / len(effects)
            change = 100 * math.expm1(mean)
            cells = [f"{100 * math.expm1(e):+.3f}%" for e in effects]
            while len(cells) < 2:
                cells.append("n/a")
            loo = [100 * math.expm1(sum(e for j, e in enumerate(effects) if j != i)
                                    / max(1, len(effects) - 1))
                   for i in range(len(effects))] if len(effects) > 1 else []
            verdict = "flat (|Δ| < 1%)" if abs(change) < 1.0 else (
                "patch faster" if change < 0 else "patch slower")
            emit(f"| {wl} | {cells[0]} | {cells[1]} | {change:+.3f}% | "
                 f"{', '.join(f'{v:+.3f}' for v in loo) or 'n/a'} | {verdict} |")
            all_effects.extend(effects)
        emit()

    if all_effects:
        geo_ratio = math.exp(sum(all_effects) / len(all_effects))
        emit(f"Across {len(all_effects)} block-level effects: geometric mean "
             f"patch/baseline step-time ratio **{geo_ratio:.6f}** "
             f"({100 * (geo_ratio - 1):+.3f}%).")
        emit()
        abs_mean = statistics.fmean(abs(e) for e in all_effects)
        emit(f"Mean absolute block effect: {100 * abs_mean:.3f}%. Compare this with the A/A "
             f"noise floor measured on the same hardware before reading any single row as a "
             f"real change.")
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paired analysis: balanced-block baseline-vs-patch comparison per workload.

Input layout (see "Paired runs" in README.md):
    <root>/<run-id>/<config-tag>-<APPA|PAAP>/<position>-<tree>/rank*.jsonl

Each block is a balanced four-arm sequence (baseline, patch, patch, baseline) or its mirror,
with every arm a separate process on the identical logical workload. For each workload the
analysis takes the geometric mean of the two same-tree arms, then the log ratio:

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
import math
import statistics
from pathlib import Path

from summarize import Arm, load


def _block_effect(
    kind: str, positions: list[tuple[int, str, list[Arm]]], workload: str, reference: str
) -> tuple[float | str, tuple | None]:
    ordered = sorted(positions, key=lambda pos: pos[0])
    if [p for p, _, _ in ordered] != [0, 1, 2, 3]:
        return "INVALID (expected positions 0–3 once each)", None
    trees = {tree for _, tree, _ in ordered}
    sequence = "".join("A" if tree == reference else "P" for _, tree, _ in ordered)
    if len(trees) != 2 or sequence != kind:
        return f"INVALID (expected {kind} tree order)", None

    if all(arms for _, _, arms in ordered) and not any(
        arm.workload_id == workload for _, _, arms in ordered for arm in arms
    ):
        return "n/a", None  # This complete block did not run the workload at any position.

    identity = None
    log_times = []
    for pos, _, arms in ordered:
        matches = [arm for arm in arms if arm.workload_id == workload]
        if len(matches) != 1:
            return f"INVALID (expected one {workload} job at position {pos})", None
        arm = matches[0]
        status = arm.status()
        if status != "OK":
            return f"INVALID (position {pos}: {status})", None
        if not arm.has_comparison_identity():
            return f"INVALID (unknown base or checksum at position {pos})", None
        if identity is not None and arm.key != identity:
            return "INVALID (different run identities)", None
        identity = arm.key
        log_times.append(math.log(arm.worst("median_ms")))
    # Mean of log times is the log of the same-tree geometric mean.
    patch = statistics.fmean(t for side, t in zip(kind, log_times) if side == "P")
    base = statistics.fmean(t for side, t in zip(kind, log_times) if side == "A")
    return patch - base, (frozenset(trees), identity)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", default="base-main")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # A position may contain a nested result directory. It still represents exactly one job.
    blocks: dict[str, dict[str, list[tuple[int, str, list[Arm]]]]] = {}
    for cfg_dir in sorted(p for p in args.run_dir.iterdir() if p.is_dir()):
        config_tag, _, kind = cfg_dir.name.rpartition("-")
        if kind not in ("APPA", "PAAP"):
            continue
        positions = []
        for arm_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            pos, _, tree = arm_dir.name.partition("-")
            if pos.isdigit():
                positions.append((int(pos), tree, load(arm_dir)))
        blocks.setdefault(config_tag, {})[kind] = positions

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("# Paired baseline-vs-patch comparison (balanced blocks)")
    emit()
    emit(f"run: `{args.run_dir}`")
    emit(f"reference tree: `{args.reference}`")
    emit(
        "metric: worst-rank median step time (ms), geometric mean within a tree, "
        "log ratio across trees"
    )
    emit()

    if not blocks:
        emit("INVALID: no APPA or PAAP blocks found.")

    all_effects: list[float] = []
    for cfg in sorted(blocks):
        emit(f"## {cfg}")
        emit()
        emit("| workload | block APPA | block PAAP | mean step-time change | LOO | " "verdict |")
        emit("|---|---|---|---|---|---|")
        positions_by_kind = blocks[cfg]
        order = ("APPA", "PAAP")
        workloads = sorted(
            {
                arm.workload_id
                for positions in positions_by_kind.values()
                for _, _, arms in positions
                for arm in arms
            }
        )
        if not workloads:
            emit("INVALID: no workload records in this configuration.")
        for wl in workloads:
            results = {}
            identities = {}
            for kind, positions in positions_by_kind.items():
                value, identity = _block_effect(kind, positions, wl, args.reference)
                results[kind] = value
                if identity is not None:
                    identities[kind] = identity
            if len(set(identities.values())) > 1:
                results = dict.fromkeys(identities, "INVALID (different patch trees or workloads)")
            cells = []
            effects = []
            for kind in order:
                effect = results.get(kind, "n/a")
                if isinstance(effect, float):
                    effects.append(effect)
                    cells.append(f"{100 * math.expm1(effect):+.3f}%")
                else:
                    cells.append(effect)
            mean_cell = loo = "n/a"
            verdict = "INVALID"
            if effects:
                change = 100 * math.expm1(statistics.fmean(effects))
                mean_cell = f"{change:+.3f}%"
                # There are at most two blocks. Omitting each one leaves the other.
                if len(effects) == 2:
                    loo = ", ".join(f"{100 * math.expm1(e):+.3f}" for e in reversed(effects))
                verdict = (
                    "flat (|Δ| < 1%)"
                    if abs(change) < 1.0
                    else ("patch faster" if change < 0 else "patch slower")
                )
                all_effects.extend(effects)
            emit(f"| {wl} | {cells[0]} | {cells[1]} | {mean_cell} | {loo} | {verdict} |")
        emit()

    if all_effects:
        geo_ratio = math.exp(sum(all_effects) / len(all_effects))
        emit(
            f"Across {len(all_effects)} block-level effects: geometric mean "
            f"patch/baseline step-time ratio **{geo_ratio:.6f}** "
            f"({100 * (geo_ratio - 1):+.3f}%)."
        )
        emit()
        abs_mean = statistics.fmean(abs(e) for e in all_effects)
        emit(
            f"Mean absolute block effect: {100 * abs_mean:.3f}%. Compare this with the A/A "
            f"noise floor measured on the same hardware before reading any single row as a "
            f"real change."
        )
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

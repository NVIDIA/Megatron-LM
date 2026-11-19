#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Summarizer: per-arm metrics and the baseline-vs-patch speed table.

Reads the per-rank JSONL emitted by benchmark.py and never mixes runs that disagree on
base SHA, workload checksum, mode, or configuration. Failing arms are kept in the
output marked with their status; they are never silently dropped or averaged in.

Rank aggregation: an arm is one distributed job, so its step time is the WORST rank
(max median_ms). Reporting a mean would hide the straggler, which is the quantity the
token-imbalance benchmark exists to expose.

Usage:
    summarize.py --results-root <dir> [--out table.md] [--reference base-main]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional fields stay in the identity when an older record does not report them.
CONFIG_FIELDS = (
    "mode",
    "ep",
    "topk",
    "logical_tokens",
    "steps",
    "hidden_size",
    "world_size",
    "num_experts",
    "tp",
    "etp",
    "backend",
    "experiment",
    "warmup_iters",
    "measured_iters",
    "ffn_hidden_size",
    "instrument",
    "schedule",
    "schedule_period",
    "hot_maxvio",
)
IDENTITY_FIELDS = (*CONFIG_FIELDS, "workload_checksum", "base_sha")
REQUIRED_CHECKS = {
    "distinct_topk_ok",
    "id_range_ok",
    "assignments_match_T_times_k",
    "global_histogram_exact",
    "roundtrip_exact",
    "counts_len_ok",
    "local_counts_sum_ok",
    "combine_factor_is_topk",
}


@dataclass
class Arm:
    tree: str
    directory: Path
    workload_id: str
    ranks: list[dict[str, Any]]

    @property
    def key(self) -> tuple:
        return tuple(self.ranks[0].get(name) for name in IDENTITY_FIELDS)

    def has_comparison_identity(self) -> bool:
        return all(
            str(self.ranks[0].get(name) or "").upper() not in ("", "?", "UNKNOWN")
            for name in ("base_sha", "workload_checksum")
        )

    @property
    def config_label(self) -> str:
        r = self.ranks[0]
        return (
            f"{r['mode']} ep={r['ep']} k={r['topk']} T={r['logical_tokens']} "
            f"steps={r['steps']} H={r['hidden_size']} w={r['world_size']} "
            f"E={r.get('num_experts', '?')} tp={r.get('tp', '?')} etp={r.get('etp', '?')} "
            f"base={r.get('base_sha', '?')} checksum={r.get('workload_checksum', '?')}"
        )

    def worst(self, field_name: str) -> float | None:
        vals = [r.get(field_name) for r in self.ranks]
        if any(v is None or not math.isfinite(v) for v in vals):
            return None
        return (min if field_name == "logical_tokens_per_second" else max)(vals)

    def status(self) -> str:
        statuses = {str(r.get("status")) for r in self.ranks}
        if statuses != {"OK"}:
            return "/".join(sorted(statuses))
        first = self.ranks[0]
        if any(
            r.get(name) != first.get(name)
            for r in self.ranks
            for name in (*IDENTITY_FIELDS, "run_id")
        ):
            return "INVALID (inconsistent run identity)"
        world = first.get("world_size")
        ranks = [r.get("rank") for r in self.ranks]
        if not isinstance(world, int) or len(ranks) != world or set(ranks) != set(range(world)):
            return "INVALID (missing or duplicate ranks)"
        required = REQUIRED_CHECKS
        if first.get("experiment") == "rank_target":
            required = required | {"rank_target_exact"}
        if first["mode"] in ("moe_fwd_bwd", "training_step"):
            required = required | {
                "replayed_routing_exact",
                "model_step_ran",
                "model_step_loss_finite",
                "model_step_grads_flow",
            }
        for r in self.ranks:
            checks = r.get("correctness") or {}
            if any(checks.get(name) is not True for name in required):
                return "INVALID (failed or missing correctness checks)"
            if checks.get("recv_splits_assignments_ok") is False:
                return "INVALID (failed receive-split check)"
            ms = r.get("median_ms")
            if ms is None or not math.isfinite(ms) or ms <= 0:
                return "INVALID (missing or nonpositive timing)"
        return "OK"

    def checks(self) -> dict[str, bool | None]:
        checks = [r.get("correctness") or {} for r in self.ranks]
        names = {k for row in checks for k, v in row.items() if isinstance(v, bool)}
        result = {}
        for name in names:
            values = [row.get(name) for row in checks]
            if False in values:
                result[name] = False
            elif all(v is True for v in values):
                result[name] = True
            else:
                result[name] = None
        return result


def load(root: Path, run_pattern: str | None = None) -> list[Arm]:
    """Keep each result directory as a separate distributed job."""
    arms: dict[tuple[Path, str], Arm] = {}
    for f in sorted(root.rglob("rank*.jsonl")):
        if run_pattern and run_pattern not in str(f):
            continue
        directory = f.parent.relative_to(root)
        tree = directory.parts[0] if directory.parts else root.name
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                workload = r["workload_id"]
                arm = arms.setdefault((directory, workload), Arm(tree, directory, workload, []))
                arm.ranks.append(r)
    return list(arms.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--reference", default="base-main")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--run-pattern", default=None, help="only aggregate runs whose path contains this substring"
    )
    args = ap.parse_args()

    arms = load(args.results_root, args.run_pattern)
    if not arms:
        print(f"ERROR: no arm records under {args.results_root}")
        return 2

    # config key -> workload_id -> tree -> Arm. The workload level is essential: one config
    # key covers all W0..W7, so keying only by tree would silently keep the last workload.
    buckets: dict[tuple, dict[str, dict[str, Arm]]] = defaultdict(lambda: defaultdict(dict))
    for a in arms:
        by_tree = buckets[a.key][a.workload_id]
        if a.tree in by_tree:
            ap.error(
                f"Ambiguous repeated jobs for {a.workload_id}: "
                f"{by_tree[a.tree].directory} and {a.directory}. Select one with --run-pattern."
            )
        by_tree[a.tree] = a

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("# MoE token-imbalance benchmark: baseline vs patch")
    emit()
    shas = sorted({a.ranks[0].get("base_sha") or "?" for a in arms})
    emit(f"base SHA(s): {', '.join(shas)}")
    emit(f"reference tree: `{args.reference}`")
    emit(f"metric aggregation: worst rank (max over ranks) of the per-arm median")
    emit()

    # ---- 1. imbalance characterisation (config-independent view) -------------------
    emit("## 1. Workload imbalance (group-global expert histogram)")
    emit()
    emit(
        "| workload | expert max/mean | expert CV (pop.) | rank maxvio (worst step) | "
        "rank concentration | assignments |"
    )
    emit("|---|---|---|---|---|---|")
    # one representative configuration per workload: the widest EP, then the most tokens,
    # so the imbalance numbers describe the configuration where imbalance actually bites.
    for wl in sorted({a.workload_id for a in arms}):
        cands = [
            a
            for a in arms
            if a.workload_id == wl and a.tree == args.reference and a.status() == "OK"
        ]
        if not cands:
            emit(f"| {wl} | n/a | n/a | n/a | n/a | n/a |")
            continue
        a = max(
            cands,
            key=lambda x: (x.ranks[0]["ep"], x.ranks[0]["logical_tokens"], x.ranks[0]["topk"]),
        )
        r0 = a.ranks[0]
        c = r0.get("correctness") or {}
        emit(
            f"| {a.workload_id} | {_f(r0.get('expert_imbalance'))} | {_f(r0.get('expert_cv'))} "
            f"| {_f(r0.get('maxvio_max'))} | {_f(r0.get('concentration_at_maxvio_max'))} "
            f"| {(c.get('global_assignments_per_step') or [None])[0]} "
            f"(ep={r0['ep']}, T={r0['logical_tokens']}, k={r0['topk']}) |"
        )
    emit()

    # ---- 2. speed table -----------------------------------------------------------
    emit("## 2. Speed comparison (baseline vs patch)")
    emit()
    emit(
        "| configuration | workload | baseline worst-rank ms | patch worst-rank ms | delta | "
        "baseline tok/s | patch tok/s | throughput delta | peak alloc (B, worst rank) | status |"
    )
    emit("|---|---|---|---|---|---|---|---|---|---|")

    ratios: list[float] = []
    patch_trees = sorted({a.tree for a in arms} - {args.reference})
    for wl_map in buckets.values():
        for wl, by_tree in sorted(wl_map.items()):
            b = by_tree.get(args.reference)
            for patch_tree in patch_trees or ["(no patch tree)"]:
                p = by_tree.get(patch_tree)
                if b is None and p is None:
                    continue
                arm = b or p
                bms = b.worst("median_ms") if b else None
                pms = p.worst("median_ms") if p else None
                btps = b.worst("logical_tokens_per_second") if b else None
                ptps = p.worst("logical_tokens_per_second") if p else None
                st = f"{b.status() if b else 'UNPAIRED'}/{p.status() if p else 'UNPAIRED'}"
                if st == "OK/OK" and not arm.has_comparison_identity():
                    st = "UNPAIRED (unknown base or checksum)"
                delta = tdelta = "n/a"
                if st == "OK/OK":
                    ratio = pms / bms
                    ratios.append(ratio)
                    delta = f"{100 * (ratio - 1):+.2f}%"
                    if btps and ptps and btps > 0 and ptps > 0:
                        tdelta = f"{100 * (ptps / btps - 1):+.2f}%"
                peaks = [a.worst("peak_allocated_bytes") for a in (b, p) if a]
                peak = max((v for v in peaks if v is not None), default=None)
                emit(
                    f"| {patch_tree} {arm.config_label} | {wl} | {_f(bms, 4)} | {_f(pms, 4)} | "
                    f"{delta} | {_f(btps, 1)} | {_f(ptps, 1)} | {tdelta} | {_f(peak, 0)} | {st} |"
                )
    emit()
    if ratios:
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        emit(
            f"Geometric mean of patch/baseline median-time ratios across "
            f"{len(ratios)} paired arms: **{geo:.6f}** "
            f"({100 * (geo - 1):+.3f}% step time)."
        )
        emit()

    # ---- 3. correctness -----------------------------------------------------------
    emit("## 3. Correctness oracle results (all ranks)")
    emit()
    names: set[str] = set()
    for a in arms:
        names |= set(a.checks())
    emit("| configuration | workload | " + " | ".join(sorted(names)) + " | status |")
    emit("|" + "---|" * (len(names) + 3))
    for arm in arms:
        ch = arm.checks()
        mark = " | ".join(
            {True: "PASS", False: "**FAIL**", None: "NOT_EVALUATED"}[ch.get(n)]
            for n in sorted(names)
        )
        emit(
            f"| {arm.directory} {arm.config_label} | {arm.workload_id} | {mark} | {arm.status()} |"
        )
    emit()

    # ---- 4. method notes ----------------------------------------------------------
    emit("## 4. Method notes and limits")
    emit()
    emit(
        "* `dispatch_only` times dispatcher preprocess/dispatch/postprocess, a stand-in local "
        "expert computation, and combine. It does not represent GEMM cost or a training step."
    )
    emit("* `moe_fwd_bwd` runs the real MoELayer forward and backward, router included.")
    emit(
        "* Probs are applied inside the expert module upstream; this harness feeds raw expert "
        "outputs to combine, and the round trip is checked against a calibration probe."
    )
    emit(
        "* Expert weights are indexed by global expert id and drawn from one fixed seed on "
        "every rank, so changing the EP layout does not change the logical model."
    )
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())

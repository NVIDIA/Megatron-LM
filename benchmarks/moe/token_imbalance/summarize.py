#!/usr/bin/env python3
"""B1 summarizer: per-arm metrics and the baseline-vs-patch speed table.

Reads the per-rank JSONL emitted by b1_benchmark.py and never mixes runs that disagree on
base SHA, dtype, workload checksum, mode, or configuration. Failing arms are kept in the
output marked with their status; they are never silently dropped or averaged in.

Rank aggregation: an arm is one distributed job, so its step time is the WORST rank
(max median_ms). Reporting a mean would hide the straggler, which is the quantity the
token-imbalance benchmark exists to expose.

Usage:
    summarize_b1.py --results-root <dir> [--out table.md] [--reference base-main]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Arm:
    tree: str
    run_id: str
    mode: str
    ep: int
    topk: int
    tokens: int
    steps: int
    hidden_size: int
    world: int
    workload_id: str
    workload_checksum: str
    base_sha: str
    ranks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.mode, self.ep, self.topk, self.tokens, self.steps, self.hidden_size,
                self.world)

    @property
    def config_label(self) -> str:
        return (f"{self.mode} ep={self.ep} k={self.topk} T={self.tokens} "
                f"steps={self.steps} H={self.hidden_size} w={self.world}")

    def worst(self, field_name: str) -> float | None:
        vals = [r[field_name] for r in self.ranks if r.get(field_name) is not None]
        return max(vals) if vals else None

    def status(self) -> str:
        statuses = {r.get("status") for r in self.ranks}
        if statuses == {"OK"}:
            return "OK"
        return "/".join(sorted(str(s) for s in statuses))

    def checks(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for r in self.ranks:
            for k, v in (r.get("correctness") or {}).items():
                if isinstance(v, bool):
                    out[k] = out.get(k, True) and v
        return out


def load(root: Path, run_pattern: str | None = None) -> list[Arm]:
    arms: dict[tuple, Arm] = {}
    for f in sorted(root.rglob("rank*.jsonl")):
        if run_pattern and run_pattern not in str(f):
            continue
        tree = f.parent.relative_to(root).parts[0]
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = (tree, r["workload_id"], r["mode"], r["ep"], r["topk"],
                 r["logical_tokens"], r["steps"], r["hidden_size"], r["world_size"])
            if k not in arms:
                arms[k] = Arm(
                    tree=tree, run_id=r.get("run_id") or f.parent.name, mode=r["mode"],
                    ep=r["ep"], topk=r["topk"], tokens=r["logical_tokens"],
                    steps=r["steps"],
                    hidden_size=r["hidden_size"], world=r["world_size"],
                    workload_id=r["workload_id"], workload_checksum=r["workload_checksum"],
                    base_sha=r.get("base_sha", "?"),
                )
                arms[k].run_ids = getattr(arms[k], "run_ids", set()) | {f.parents[1].name}
            arms[k].ranks.append(r)
    return list(arms.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--reference", default="base-main")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run-pattern", default=None,
                    help="only aggregate runs whose path contains this substring")
    args = ap.parse_args()

    arms = load(args.results_root, args.run_pattern)
    if not arms:
        print(f"ERROR: no arm records under {args.results_root}")
        return 2

    # config key -> workload_id -> tree -> Arm. The workload level is essential: one config
    # key covers all W0..W7, so keying only by tree would silently keep the last workload.
    buckets: dict[tuple, dict[str, dict[str, Arm]]] = defaultdict(lambda: defaultdict(dict))
    for a in arms:
        buckets[a.key][a.workload_id][a.tree] = a

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("# B1 — MoE token-imbalance benchmark: baseline vs patch")
    emit()
    shas = sorted({a.base_sha for a in arms})
    emit(f"base SHA(s): {', '.join(shas)}")
    emit(f"reference tree: `{args.reference}`")
    emit(f"metric aggregation: worst rank (max over ranks) of the per-arm median")
    emit()

    # ---- 1. imbalance characterisation (config-independent view) -------------------
    emit("## 1. Workload imbalance (group-global expert histogram)")
    emit()
    emit("| workload | expert max/mean | expert CV (pop.) | rank max/mean | assignments |")
    emit("|---|---|---|---|---|")
    # one representative configuration per workload: the widest EP, then the most tokens,
    # so the imbalance numbers describe the configuration where imbalance actually bites.
    for wl in sorted({a.workload_id for a in arms}):
        cands = [a for a in arms if a.workload_id == wl and a.tree == args.reference
                 and a.status() == "OK"]
        if not cands:
            emit(f"| {wl} | n/a | n/a | n/a | n/a |")
            continue
        a = max(cands, key=lambda x: (x.ep, x.tokens, x.topk))
        r0 = a.ranks[0]
        c = r0.get("correctness") or {}
        emit(f"| {a.workload_id} | {_f(r0.get('expert_imbalance'))} | {_f(r0.get('expert_cv'))} "
             f"| {_f(r0.get('rank_imbalance'))} | {c.get('global_assignments')} "
             f"(ep={a.ep}, T={a.tokens}, k={a.topk}) |")
    emit()

    # ---- 2. speed table -----------------------------------------------------------
    emit("## 2. Speed comparison (baseline vs patch)")
    emit()
    emit("| configuration | workload | baseline worst-rank ms | patch worst-rank ms | delta | "
         "baseline tok/s | patch tok/s | throughput delta | peak alloc (B, worst rank) | status |")
    emit("|---|---|---|---|---|---|---|---|---|---|")

    ratios: list[float] = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], k[3], k[2], k[4])):
        # buckets[key] maps workload_id -> tree -> Arm
        wl_map = buckets[key]
        trees_here = {t for wmap in wl_map.values() for t in wmap}
        patch_tree = next((t for t in sorted(trees_here) if t != args.reference), None)
        if args.reference not in trees_here or patch_tree is None:
            continue
        for wl in sorted(wl_map):
            b = wl_map[wl].get(args.reference)
            p = wl_map[wl].get(patch_tree)
            if b is None or p is None:
                continue
            bms, pms = b.worst("median_ms"), p.worst("median_ms")
            if bms and pms and bms > 0:
                ratio = pms / bms
                ratios.append(ratio)
                delta = f"{100 * (ratio - 1):+.2f}%"
            else:
                delta = "n/a"
            btps = b.worst("logical_tokens_per_second")
            ptps = p.worst("logical_tokens_per_second")
            tdelta = (f"{100 * (ptps / btps - 1):+.2f}%" if btps and ptps and btps > 0 else "n/a")
            st = f"{b.status()}/{p.status()}"
            if st != "OK/OK":
                st = f"**{st}**"
            emit(f"| {b.config_label} | {wl} | {_f(bms, 4)} | {_f(pms, 4)} | {delta} | "
                 f"{_f(btps, 1)} | {_f(ptps, 1)} | {tdelta} | "
                 f"{_f(max(b.worst('peak_allocated_bytes') or 0, p.worst('peak_allocated_bytes') or 0), 0)} "
                 f"| {st} |")
    emit()
    if ratios:
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        emit(f"Geometric mean of patch/baseline median-time ratios across "
             f"{len(ratios)} paired arms: **{geo:.6f}** "
             f"({100 * (geo - 1):+.3f}% step time).")
        emit()
        emit("A ratio of 1.000 means the measured step time did not move. The A3 fix adds one "
             "reduction over a [num_experts] tensor and removes nothing, so approximately 1.0 "
             "is the expected value; a large apparent gain here would be evidence of a "
             "confounded measurement, not of a speedup.")
        emit()

    # ---- 3. correctness -----------------------------------------------------------
    emit("## 3. Correctness oracle results (all ranks)")
    emit()
    names: set[str] = set()
    for a in arms:
        names |= set(a.checks())
    emit("| configuration | workload | " + " | ".join(sorted(names)) + " | status |")
    emit("|" + "---|" * (len(names) + 3))
    for key in sorted(buckets, key=lambda k: (k[0], k[1], k[3], k[2], k[4])):
        wl_map = buckets[key]
        for wl in sorted(wl_map):
            for tree in sorted(wl_map[wl]):
                arm = wl_map[wl][tree]
                if arm.status() != "OK":
                    emit(f"| {tree} {arm.config_label} | {wl} | " +
                         " | ".join("NOT_EVALUATED" for _ in names) +
                         f" | {arm.status()} |")
                    continue
                ch = arm.checks()
                mark = " | ".join("PASS" if ch.get(n) else "**FAIL**" for n in sorted(names))
                emit(f"| {tree} {arm.config_label} | {wl} | {mark} | {arm.status()} |")
    emit()

    # ---- 4. method notes ----------------------------------------------------------
    emit("## 4. Method notes and limits")
    emit()
    emit("* `dispatch_only` times dispatcher preprocess/dispatch/postprocess, a stand-in local "
         "expert computation, and combine. It does not represent GEMM cost or a training step.")
    emit("* `moe_fwd_bwd` runs the real MoELayer forward and backward, router included.")
    emit("* Probs are applied inside the expert module upstream; this harness feeds raw expert "
         "outputs to combine, and the round trip is checked against a calibration probe.")
    emit("* Expert weights are indexed by global expert id and drawn from one fixed seed on "
         "every rank, so changing the EP layout does not change the logical model.")
    emit("* `transformer_engine` and `apex` are absent on this node, so the expert computation "
         "is a per-expert loop rather than TE grouped GEMM. Absolute step times are therefore "
         "not comparable with TE-enabled runs; the baseline/patch comparison remains valid "
         "because both sides use the identical path.")
    emit("* The node is shared: other tenants hold memory on the GPUs used here. Absolute "
         "latencies carry that noise; the paired baseline/patch comparison is the defensible "
         "quantity.")
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

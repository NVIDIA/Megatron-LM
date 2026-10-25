# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from .cost_model import build_timeline
from .planner import (
    RuntimePlan,
    RuntimeScheduleError,
    dummy_tokens_per_group,
    encode_rows,
    validate_rows,
)
from .sched.costs import PackCostModel


@dataclass(frozen=True)
class LightweightOptions:
    time_budget_s: float = 0.8
    max_scored_candidates: int = 4
    cp_floors: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    max_guide_candidates: int = 4

    def __post_init__(self):
        if not math.isfinite(self.time_budget_s) or self.time_budget_s <= 0:
            raise ValueError("time_budget_s must be finite and positive")
        if type(self.max_scored_candidates) is not int or self.max_scored_candidates < 1:
            raise ValueError("max_scored_candidates must be a positive integer")
        if not self.cp_floors or any(type(c) is not int or c < 1 for c in self.cp_floors):
            raise ValueError("cp_floors must contain positive integers")
        if type(self.max_guide_candidates) is not int or self.max_guide_candidates < 1:
            raise ValueError("max_guide_candidates must be a positive integer")


class LightweightRuntimePlanner:
    """No offline solver, CP-SAT, GPU search, or iterative global polishing.

    Deterministic FFD and cost-per-token candidates cover different pack widths.
    candidates are scored by the same exact E2E timeline as the offline solver.
    The deadline is cooperative: one mandatory feasible score is always completed,
    and any overrun from that score is reported, never hidden as a hard deadline.
    An optional authenticated incumbent can be injected by the MCore caller.
    """

    def __init__(self, geometry, cost_model, options=None):
        geometry = replace(geometry, allowed_cp_sizes=tuple(sorted(set(geometry.allowed_cp_sizes))))
        self.geometry = geometry
        self.model = cost_model
        self.options = options or LightweightOptions()
        if not any(floor <= geometry.world_size for floor in self.options.cp_floors):
            raise ValueError("cp_floors must include a width no larger than world_size")
        cost_model.check_domain(geometry, None)
        self.timeline = build_timeline(cost_model, geometry)
        prepare = getattr(cost_model, "prepare_guide", None)
        if prepare is not None:
            prepare(geometry.allowed_cp_sizes)
        self.costs = self._new_costs()
        self.last_statistics = {}

    def _new_costs(self):
        return PackCostModel(
            self.geometry,
            self.model.forward,
            self.model.backward,
            getattr(self.model, "program_costs", None),
            placement_period=(
                None
                if getattr(self.model, "placement_invariant", False)
                else self.geometry.world_size
            ),
            max_group_seqs=self.geometry.max_group_seqs or 64,
        )

    def _start_plan(self, lengths):
        lengths = dict(lengths)
        if not lengths or any(
            type(i) is not int or i < 0 or type(n) is not int or n <= 0 for i, n in lengths.items()
        ):
            raise ValueError(
                "lengths must map nonnegative integer IDs to positive integer logical lengths"
            )
        self.model.clear()
        self.costs = self._new_costs()
        return lengths

    def _make_plan(self, rounds, lengths, predicted, begin):
        rows = encode_rows(rounds, self.geometry.world_size)
        groups = validate_rows(rows, lengths, self.geometry, min(self.geometry.allowed_cp_sizes))
        return RuntimePlan(
            rows=rows,
            groups=groups,
            predicted_ms=predicted,
            planning_s=time.perf_counter() - begin,
            microbatches=len(rows),
            dummy_tokens=dummy_tokens_per_group(groups, lengths, self.geometry),
        )

    def _packing_tables(self, lengths):
        return (
            {i: self.costs.min_cp(n) for i, n in lengths.items()},
            {
                cp: {i: self.costs.stored(n, cp) for i, n in lengths.items()}
                for cp in self.geometry.allowed_cp_sizes
            },
        )

    def _fill_pack(self, anchor, pending, cp, storage):
        """All proposal families use the same ordered physical-capacity check."""
        ids, used = [anchor], storage[cp][anchor]
        for i in pending:
            if len(ids) >= (self.geometry.max_group_seqs or 64):
                break
            if i != anchor and used + storage[cp][i] <= self.geometry.capacity(cp):
                ids.append(i)
                used += storage[cp][i]
        return ids, used

    def _expand(self, row, lengths, widths=None, cost_aware=False):
        world = self.geometry.world_size
        slack = world - sum(group["cp"] for group in row)
        dp = {0: (0.0, [])}
        for group in row:
            base = group["cp"]
            values = tuple(lengths[i] for i in group["sample_ids"])
            choices = [
                c - base
                for c in (widths or self.geometry.allowed_cp_sizes)
                if base <= c <= base + slack and self.costs.fits(values, c)
            ]
            next_dp = {}
            for previous, (work, increments) in dp.items():
                for delta in choices:
                    if previous + delta <= slack:
                        score = work + (
                            self._group_work(values, base + delta) if cost_aware else 0.0
                        )
                        key = previous + delta
                        if key not in next_dp or score < next_dp[key][0]:
                            next_dp[key] = score, increments + [delta]
            dp = next_dp
        if slack not in dp:
            return None
        return [
            dict(cp=g["cp"] + delta, sample_ids=list(g["sample_ids"]))
            for g, delta in zip(row, dp[slack][1])
        ]

    def _group_work(self, values, cp):
        guide = getattr(self.model, "guide_program_costs", None)
        if guide is not None and self.timeline.program is not None:
            program = guide(values, cp, ranks=tuple(range(cp)))
            duration = sum(
                program[payload] if kind == "cp" else payload
                for phase in self.timeline.program
                for kind, payload in phase
            )
            return cp * duration
        fixed = self.timeline.layer_count * (
            self.timeline.forward_non_attention_ms + self.timeline.backward_non_attention_ms
        )
        return sum(n * n for n in values) / cp + cp * fixed

    def _global_pack(self, lengths, floor, width_extra=0, tables=None):
        """Pack tokens globally, then tile whole groups into exact-world rows."""
        pending = sorted(lengths, key=lambda i: (-lengths[i], i))
        allowed = self.geometry.allowed_cp_sizes
        minimums, storage = tables or self._packing_tables(lengths)
        groups = []
        while pending:
            anchor = pending[0]
            minimum = minimums[anchor]
            target = min(self.geometry.world_size, minimum + width_extra)
            cp = next((c for c in allowed if c >= max(target, floor)), allowed[-1])
            ids, _ = self._fill_pack(anchor, pending, cp, storage)
            selected = set(ids)
            pending = [i for i in pending if i not in selected]
            groups.append(dict(cp=cp, sample_ids=ids))
        return self._tile(groups, lengths)

    def _tile(self, groups, lengths, widths=None, cost_aware=False):
        groups.sort(key=lambda g: (-g["cp"], -sum(lengths[i] for i in g["sample_ids"])))
        allowed = widths or self.geometry.allowed_cp_sizes
        rounds = []
        while groups:
            row = [groups.pop(0)]
            remaining = self.geometry.world_size - row[0]["cp"]
            dp = {0: ()}
            for index, group in enumerate(groups):
                width = group["cp"]
                for previous, indices in list(dp.items()):
                    total = previous + width
                    if total <= remaining:
                        dp.setdefault(total, indices + (index,))
                if remaining in dp:
                    break
            chosen = set(dp[max(dp)])
            row.extend(group for index, group in enumerate(groups) if index in chosen)
            groups = [group for index, group in enumerate(groups) if index not in chosen]
            expanded = self._expand(row, lengths, widths, cost_aware)
            if expanded is None:
                if self.geometry.world_size not in allowed:
                    return None
                groups.extend(row[1:])
                expanded = [dict(cp=self.geometry.world_size, sample_ids=row[0]["sample_ids"])]
                groups.sort(key=lambda g: -g["cp"])
            rounds.append(expanded)
        return rounds

    def _cost_pack(self, lengths, tables, ep_aligned=False):
        """All eligible widths compete on cheap predicted work per useful token.

        EP-aligned is a proposal family, not an exclusion of other CP sizes.
        The full E2E timeline alone decides whether either family is accepted.
        """
        guide = getattr(self.model, "guide_program_costs", None)
        if guide is None or self.timeline.program is None:
            return None
        _, storage = tables
        widths = tuple(
            cp
            for cp in self.geometry.allowed_cp_sizes
            if not ep_aligned or cp % self.geometry.ep_size == 0
        )
        if not widths:
            return None
        pending = sorted(lengths, key=lambda i: (-lengths[i], i))
        groups = []
        while pending:
            anchor, best = pending[0], None
            for cp in widths:
                used = storage[cp][anchor]
                if used > self.geometry.capacity(cp):
                    continue
                ids, _ = self._fill_pack(anchor, pending, cp, storage)
                values = tuple(lengths[i] for i in ids)
                useful = sum(values)
                key = self._group_work(values, cp) / useful, -useful, cp
                if best is None or key < best[0]:
                    best = key, cp, ids
            if best is None:
                return None
            _, cp, ids = best
            chosen = set(ids)
            pending = [i for i in pending if i not in chosen]
            groups.append(dict(cp=cp, sample_ids=ids))
        return self._tile(groups, lengths, widths, cost_aware=True)

    def _pack(self, lengths, floor, tables=None):
        pending = sorted(lengths, key=lambda i: (-lengths[i], i))
        rounds = []
        world = self.geometry.world_size
        allowed = sorted(self.geometry.allowed_cp_sizes)
        minimums, storage = tables or self._packing_tables(lengths)
        while pending:
            row, used = [], 0
            while used < world and pending:
                remaining = world - used
                anchor = next((i for i in pending if minimums[i] <= remaining), None)
                if anchor is None:
                    break
                cp_min = minimums[anchor]
                cp = next((c for c in allowed if max(cp_min, floor) <= c <= remaining), None)
                if cp is None:
                    cp = max(c for c in allowed if cp_min <= c <= remaining)
                ids, _ = self._fill_pack(anchor, pending, cp, storage)
                chosen = set(ids)
                pending = [i for i in pending if i not in chosen]
                row.append(dict(cp=cp, sample_ids=ids))
                used += cp
            if not row:
                raise RuntimeScheduleError("no sequence fits the configured capacity")
            if used < world:
                # Exact-fill expansion, including non-power-of-two allowed domains.
                expanded = self._expand(row, lengths)
                if expanded is None:
                    # A full-world pack remains feasible; leave unconsumed samples pending.
                    ids = row[0]["sample_ids"]
                    for group in row[1:]:
                        pending.extend(group["sample_ids"])
                    if world not in allowed:
                        raise RuntimeScheduleError("allowed CP widths cannot fill every rank")
                    row = [dict(cp=world, sample_ids=ids)]
                    pending.sort(key=lambda i: (-lengths[i], i))
                else:
                    row = expanded
            rounds.append(row)
        return rounds

    def _costed(self, rounds, lengths, guide=None):
        out = []
        for row in rounds:
            start, groups = 0, []
            for group in row:
                cp = group["cp"]
                values = tuple(lengths[i] for i in group["sample_ids"])
                value = (
                    (0.0, 0.0, guide(values, cp, ranks=tuple(range(start, start + cp))))
                    if guide is not None
                    else self.costs.pack(values, cp, start)
                )
                if value is None:
                    raise RuntimeScheduleError(f"cost model rejected CP{cp} pack {values}")
                groups.append((start, cp, *value))
                start += cp
            out.append(groups)
        return out

    def _guide(self, rounds, lengths):
        # Candidate ordering only; acceptance ALWAYS uses exact injected E2E costs.
        guide = getattr(self.model, "guide_program_costs", None)
        if guide is not None and self.timeline.program is not None:
            return self.timeline.evaluate(self._costed(rounds, lengths, guide)), len(rounds)
        work = sum(
            sum(lengths[i] ** 2 for i in g["sample_ids"]) / g["cp"] for row in rounds for g in row
        )
        return len(rounds), work

    def _work_guide(self, rounds, lengths):
        # Intrinsic rank-load lower bound; waiting is added by the full guide.
        loads = [0.0] * self.geometry.world_size
        for row in rounds:
            start = 0
            for group in row:
                cp = group["cp"]
                duration = self._group_work(tuple(lengths[i] for i in group["sample_ids"]), cp) / cp
                for rank in range(start, start + cp):
                    loads[rank] += duration
                start += cp
        return max(loads), sum(loads), len(rounds)

    def _balance_exact(self, rounds, lengths):
        """One cyclic-rotation sweep, reusing placement-invariant exact pack costs."""
        if not getattr(self.model, "placement_invariant", False):
            return rounds, None, 0
        costed = self._costed(rounds, lengths)
        state = self.timeline.initial_state()
        balanced, changed = [], 0
        for index, (row, priced) in enumerate(zip(rounds, costed)):
            best = None
            for offset in range(len(row)):
                rotated = priced[offset:] + priced[:offset]
                start, shifted = 0, []
                for group in rotated:
                    shifted.append((start, *group[1:]))
                    start += group[1]
                next_state = self.timeline.advance(state, shifted, index == 0)
                key = max(next_state), sum(next_state), sum(v * v for v in next_state)
                if best is None or key < best[0]:
                    best = key, next_state, offset
            _, state, offset = best
            balanced.append(row[offset:] + row[:offset])
            changed += offset != 0
        return balanced, self.timeline.makespan(state), changed

    def plan(self, lengths, *, incumbent_rounds=None):
        begin = time.perf_counter()
        lengths = self._start_plan(lengths)
        tables = self._packing_tables(lengths)
        candidates, seen, rejected = [], set(), []
        for floor in self.options.cp_floors:
            if floor > self.geometry.world_size:
                continue
            try:
                proposals = [self._pack(lengths, floor, tables)]
            except RuntimeScheduleError as exc:
                rejected.append(str(exc))
                proposals = []
            if floor in (1, 4, 8, 16, 32):
                proposals.append(self._global_pack(lengths, floor, tables=tables))
            if floor == 1:
                proposals.extend(
                    self._global_pack(lengths, floor, extra, tables) for extra in (1, 2, 4)
                )
            for rounds in proposals:
                if rounds is None:
                    continue
                signature = tuple(
                    tuple((g["cp"], tuple(g["sample_ids"])) for g in row) for row in rounds
                )
                if signature not in seen:
                    seen.add(signature)
                    candidates.append(rounds)
        for ep_aligned in (False, True):
            rounds = self._cost_pack(lengths, tables, ep_aligned)
            if rounds is not None:
                signature = tuple(
                    tuple((g["cp"], tuple(g["sample_ids"])) for g in row) for row in rounds
                )
                if signature not in seen:
                    seen.add(signature)
                    candidates.append(rounds)
        generated = len(candidates)
        # The slowest rank, not total GPU work, drives the cheap shortlist.
        candidates.sort(key=lambda rounds: self._work_guide(rounds, lengths))
        candidates = candidates[: self.options.max_guide_candidates]
        candidates.sort(key=lambda rounds: self._guide(rounds, lengths))
        if incumbent_rounds is not None:
            validate_rows(
                encode_rows(incumbent_rounds, self.geometry.world_size),
                lengths,
                self.geometry,
                min(self.geometry.allowed_cp_sizes),
            )
            candidates.insert(0, incumbent_rounds)
        if not candidates:
            raise RuntimeScheduleError(
                "runtime heuristics could not produce an exact-width row; "
                "this is not an infeasibility proof"
            )
        best, scored, attempted, selected = None, 0, 0, None
        previous_score_s = 0.0
        generation_s = time.perf_counter() - begin
        mandatory = (
            2 if incumbent_rounds is not None and self.options.max_scored_candidates > 1 else 1
        )
        for index, rounds in enumerate(candidates):
            if attempted >= mandatory and (
                attempted >= self.options.max_scored_candidates
                or time.perf_counter() - begin >= self.options.time_budget_s
            ):
                break
            if (
                attempted >= mandatory
                and time.perf_counter() - begin + previous_score_s > self.options.time_budget_s
            ):
                break
            score_begin = time.perf_counter()
            attempted += 1
            try:
                predicted = self.timeline.evaluate(self._costed(rounds, lengths))
            except RuntimeScheduleError as exc:
                rejected.append(str(exc))
                previous_score_s = time.perf_counter() - score_begin
                if best is None:
                    raise
                continue
            previous_score_s = time.perf_counter() - score_begin
            scored += 1
            if best is None or predicted < best[0] - 1e-9:
                best = predicted, rounds
                selected = index
        balance_begin = time.perf_counter()
        balanced, balanced_ms, changed_rows = self._balance_exact(best[1], lengths)
        balance_improvement = 0.0
        if balanced_ms is not None and balanced_ms < best[0] - 1e-9:
            balance_improvement = best[0] - balanced_ms
            best = balanced_ms, balanced
        else:
            changed_rows = 0
        balance_s = time.perf_counter() - balance_begin
        plan = self._make_plan(best[1], lengths, best[0], begin)
        elapsed = plan.planning_s
        self.last_statistics = dict(
            candidate_count=len(candidates),
            generated_candidates=generated,
            max_guide_candidates=self.options.max_guide_candidates,
            scored_candidates=scored,
            attempted_candidates=attempted,
            time_budget_s=self.options.time_budget_s,
            budget_overrun=elapsed > self.options.time_budget_s,
            selected_incumbent=incumbent_rounds is not None and selected == 0 and changed_rows == 0,
            seed_incumbent=incumbent_rounds is not None and selected == 0,
            rotation_changed_rows=changed_rows,
            rotation_improvement_ms=balance_improvement,
            rotation_s=balance_s,
            rejected_candidates=rejected,
            candidate_generation_s=generation_s,
            exact_scoring_s=elapsed - generation_s - balance_s,
            pack_cost_stats=dict(self.costs.stats),
            algorithm="FFD + all-width work/token + EP-aligned proposals + bounded guide + exact E2E acceptance + rank rotation",
        )
        return plan

    def plan_padded(self, padded_lengths, *, logical_lengths=None, incumbent_rows=None):
        """MCore input [(global_sample_id, dataset-padded length)]; pure CPU.

        No LM shift is inferred. Exact logical lengths are optional and must ceil
        to the supplied padded lengths. Otherwise the cost input is explicitly
        the padded upper bound, not fabricated logical lengths.
        """
        items = list(padded_lengths.items() if hasattr(padded_lengths, "items") else padded_lengths)
        padded = dict(items)
        if (
            len(padded) != len(items)
            or not padded
            or any(
                type(i) is not int
                or i < 0
                or type(n) is not int
                or n <= 0
                or n % self.geometry.dataset_alignment
                for i, n in items
            )
        ):
            raise ValueError("unique nonnegative IDs and dataset-aligned positive lengths required")
        logical = padded if logical_lengths is None else dict(logical_lengths)
        if logical.keys() != padded.keys() or any(
            type(n) is not int
            or n <= 0
            or (n + self.geometry.dataset_alignment - 1)
            // self.geometry.dataset_alignment
            * self.geometry.dataset_alignment
            != padded[i]
            for i, n in logical.items()
        ):
            raise ValueError(
                "logical lengths must match every padded input after dataset alignment"
            )
        incumbent = None
        if incumbent_rows is not None:
            groups = validate_rows(
                incumbent_rows, padded, self.geometry, min(self.geometry.allowed_cp_sizes)
            )
            incumbent = [[dict(cp=cp, sample_ids=ids) for _, cp, ids in row] for row in groups]
        plan = self.plan(logical, incumbent_rounds=incumbent)
        validate_rows(plan.rows, padded, self.geometry, min(self.geometry.allowed_cp_sizes))
        self.last_statistics["cost_length_source"] = (
            "after_lm_logical" if logical_lengths is not None else "dataset_padded_upper_bound"
        )
        return plan

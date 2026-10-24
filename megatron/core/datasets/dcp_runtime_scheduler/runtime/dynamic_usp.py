# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapt generic hot-switch layouts to the existing CPU cost and rank-row API."""

from __future__ import annotations

import math
import time

from .hot_switch import HotSwitchOptions, HotSwitchScheduler, InfeasibleLayout
from .lightweight import LightweightRuntimePlanner
from .planner import (
    RuntimePlan,
    RuntimeScheduleError,
    dummy_tokens_per_group,
    encode_rows,
    validate_rows,
)


class DynamicUSPRuntimePlanner(LightweightRuntimePlanner):
    """Optional solver sharing the existing cost wrapper and padded-input contract.

    Only ``plan`` is replaced. Inherited input validation, pack-cost memoization
    and timeline construction are unchanged; the lightweight search is not run.
    This adapter never selects or initializes an attention communication backend.
    """

    def __init__(self, geometry, cost_model, **options) -> None:
        super().__init__(geometry, cost_model)
        self.options = HotSwitchOptions(**options)
        sizes = tuple(
            s
            for s in self.geometry.allowed_cp_sizes
            if s & (s - 1) == 0 and geometry.world_size % s == 0
        )
        self.scheduler = HotSwitchScheduler(
            geometry.world_size, sizes, geometry.tokens_per_rank, self.options
        )

    def _rounds(self, layout):
        return [
            [dict(cp=group.size, sample_ids=list(group.sample_ids)) for group in row]
            for row in layout
        ]

    def _sample_work(self, sample_id, lengths):
        values = (lengths[sample_id],)
        size = next((s for s in self.scheduler.sizes if self.costs.fits(values, s)), None)
        if size is None:
            raise InfeasibleLayout(f"sample {sample_id} exceeds every allowed group capacity")
        value = self.costs.pack(values, size, 0)
        if value is None:
            raise RuntimeScheduleError(f"cost model rejected scheduling sample {sample_id}")
        forward, backward, program = value
        if self.timeline.program is None:
            duration = self.timeline.layer_count * (
                forward
                + backward
                + (self.timeline.forward_non_attention_ms + self.timeline.backward_non_attention_ms)
            )
        else:
            duration = sum(
                program[payload] if kind == "cp" else payload
                for phase in self.timeline.program
                for kind, payload in phase
            )
        work = size * duration
        if not math.isfinite(work) or work <= 0:
            raise ValueError("cost model must produce finite positive scheduling work")
        return work

    def plan(self, lengths, *, incumbent_rounds=None) -> RuntimePlan:
        """Search hot-switch count candidates with the injected complete-step cost."""
        begin = time.perf_counter()
        lengths = dict(lengths)
        if not lengths or any(
            type(i) is not int or i < 0 or type(n) is not int or n <= 0 for i, n in lengths.items()
        ):
            raise ValueError("lengths must map nonnegative integer IDs to positive lengths")
        self.model.clear()
        self.costs = self._new_costs()
        self.scheduler.last_candidates = []

        def score(rounds):
            validate_rows(
                encode_rows(rounds, self.geometry.world_size),
                lengths,
                self.geometry,
                min(self.geometry.allowed_cp_sizes),
            )
            value = self.timeline.evaluate(self._costed(rounds, lengths))
            if not math.isfinite(value) or value < 0:
                raise ValueError("cost model must produce a finite nonnegative step time")
            return value

        best = None if incumbent_rounds is None else (score(incumbent_rounds), incumbent_rounds)
        rejection = None
        selected_incumbent = best is not None
        alignment = self.geometry.dataset_alignment
        tokens = {i: (n + alignment - 1) // alignment * alignment for i, n in lengths.items()}
        try:
            weights = {i: self._sample_work(i, lengths) for i in lengths}
            predicted, layout = self.scheduler.solve(
                tokens,
                weights,
                lambda ids, size: self.costs.fits(tuple(lengths[i] for i in ids), size),
                lambda layout: score(self._rounds(layout)),
            )
            if best is None or predicted < best[0] - 1e-9:
                best = predicted, self._rounds(layout)
                selected_incumbent = False
        except InfeasibleLayout as error:
            rejection = str(error)
            if best is None:
                raise RuntimeScheduleError(f"hot-switch search: {error}") from error

        rows = encode_rows(best[1], self.geometry.world_size)
        groups = validate_rows(rows, lengths, self.geometry, min(self.geometry.allowed_cp_sizes))
        elapsed = time.perf_counter() - begin
        self.last_statistics = dict(
            algorithm="DynamicUSP solve_gbs Stage 1 + injected complete-step cost",
            source_commit="1ff97a4ae0beb5e088fadb9e112265bdd194b24d",
            group_sizes=self.scheduler.sizes,
            candidates=self.scheduler.last_candidates,
            selected_incumbent=selected_incumbent,
            rejection=rejection,
            pack_cost_stats=dict(self.costs.stats),
        )
        return RuntimePlan(
            rows=rows,
            groups=groups,
            predicted_ms=best[0],
            planning_s=elapsed,
            greedy_s=elapsed,
            polish_evaluations=0,
            microbatches=len(rows),
            cache_hit=False,
            dummy_tokens=dummy_tokens_per_group(groups, lengths, self.geometry),
        )

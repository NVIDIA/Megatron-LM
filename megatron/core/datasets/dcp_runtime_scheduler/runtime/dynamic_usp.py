# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapt generic hot-switch layouts to the existing CPU cost and rank-row API."""

from __future__ import annotations

import copy
import math

from ..models.solver_section import PIPELINE_SCHEDULES, sample_weight
from .hot_switch import HotSwitchOptions, HotSwitchScheduler, InfeasibleLayout
from .pipeline import step_time
from .planner import RuntimePlan, RuntimePlanner, RuntimeScheduleError
from .stage2 import ADOPTED, PROFILE_EXECUTION_ERROR, error_text, repack_layout

# Outcome of a microbatch that Stage 2 re-packed when the re-packed layout of the step could
# not be scored: the step keeps its Stage 1 layout. DynamicUSP has no such outcome.
RESCORE_REJECTED = "RESCORE_REJECTED"
# What the plan statistics keep of the Stage 2 record of a microbatch (``stage2``).
_STAGE2_RECORD = ("code", "order", "base_ms", "selected_ms", "error", "runtime_capacity")
# The statistics keep that record only for a microbatch with one of these outcomes or with
# the capacity mark, and for at most this many microbatches of a plan: the statistics travel
# with every plan, and a global batch can have thousands of microbatches.
_STAGE2_RECORDED = (ADOPTED, RESCORE_REJECTED, PROFILE_EXECUTION_ERROR)
_STAGE2_RECORD_LIMIT = 64


class DynamicUSPRuntimePlanner(RuntimePlanner):
    """Optional solver sharing the existing cost wrapper and padded-input contract.

    Both solvers share model injection, validation, pricing and rank-row output; neither
    inherits the other's search algorithm. Keyword options are those of ``HotSwitchOptions``
    except ``pp_multiple``. The cost model supplies the Stage 1 coefficients, the step time
    and the Stage 2 members (``models.base.DcpCostModel``).

    ``pipeline`` is ``(pipeline stages, virtual stages per pipeline stage)`` of the run and
    ``pipeline_schedule`` the name of its schedule (``runtime.pipeline``). None or
    ``"interleaved"`` is DynamicUSP's order, with counts that are multiples of the stages
    (``pp_multiple``), also for one chunk per stage. ``"non_interleaved"`` is Megatron's order
    without virtual pipeline stages: it takes ``(stages, 1)`` only and any count. With one
    pipeline stage neither argument is read.

    Stage 2 is switched by the cost model (``stage2_enabled``, off by default); ``_stage2``
    has the rules and ``last_statistics["stage2"]`` the outcome. Three attributes, which
    exist whether it is on or off, say what the planner does: ``stage2_enabled``,
    ``stage2_keep_threshold`` (None when off) and ``stage2_idle`` (why a Stage 2 that is on
    has no work in any plan of this run, else None).
    """

    def __init__(
        self, geometry, cost_model, *, pipeline=None, pipeline_schedule=None, **options
    ) -> None:
        super().__init__(geometry, cost_model)
        # One pipeline stage has no chunks to interleave: its virtual stages are not read,
        # and neither is the schedule. The pair and the name are checked where a plan is
        # replayed (``pipeline.step_time``).
        self.pipeline = tuple(pipeline) if pipeline is not None and pipeline[0] > 1 else (1, 1)
        stages = self.pipeline[0]
        self.pipeline_schedule = None
        if stages > 1:
            self.pipeline_schedule = pipeline_schedule or PIPELINE_SCHEDULES[0]
        non_interleaved = self.pipeline_schedule == PIPELINE_SCHEDULES[1]
        if stages > 1 and not callable(getattr(cost_model, "pipeline_group_times", None)):
            raise ValueError(
                f"the {cost_model.name!r} cost model has no step time for more than one "
                "pipeline stage (it must offer pipeline_group_times); "
                f"this run has {stages} pipeline stages"
            )
        # The count multiple (DynamicUSP's ``pp_multiple``) of the search, the incumbent and
        # the capacity fallback: the stages, or 1 on Megatron's non-interleaved schedule.
        self.count_multiple = 1 if non_interleaved else stages
        # Time-balance factors: a table given as a Python option wins over the one the cost
        # model read from its coefficient file and checked against its CP sizes; those this
        # run cannot form are never read.
        if options.get("tier_time_factors") is None:
            options["tier_time_factors"] = getattr(cost_model, "tier_time_factors", None)
        self.options = HotSwitchOptions(**options, pp_multiple=self.count_multiple)
        sizes = tuple(
            s
            for s in self.geometry.allowed_cp_sizes
            if s & (s - 1) == 0 and geometry.world_size % s == 0
        )
        self.scheduler = HotSwitchScheduler(
            geometry.world_size, sizes, geometry.tokens_per_rank, self.options
        )
        # Anything but True leaves Stage 2 off, and then no Stage 2 member is read.
        self.stage2_enabled = getattr(cost_model, "stage2_enabled", False) is True
        self.stage2_keep_threshold: float | None = None
        self.stage2_idle: str | None = None
        self._stage2_reference: int | None = None
        self._stage2_guard: float | None = None
        if self.stage2_enabled:
            # Read once (``models.base.DcpCostModel``): the plan statistics record these
            # without asking the model again. The engine checks them for every microbatch.
            self.stage2_keep_threshold = float(cost_model.stage2_keep_threshold)
            self._stage2_reference = reference = cost_model.stage2_reference_cp_size
            guard = getattr(cost_model, "stage2_guard_threshold", None)
            self._stage2_guard = None if guard is None else float(guard)
            # Stage 2 leaves a microbatch of reference groups alone: with no larger size to
            # form, it has no work in any plan of this run.
            if not any(size > reference for size in self.scheduler.sizes):
                self.stage2_idle = f"no usable CP size above the reference size {reference}"

    def _rounds(self, layout):
        return [
            [dict(cp=group.size, sample_ids=list(group.sample_ids)) for group in row]
            for row in layout
        ]

    def _sample_work(self, sample_id, lengths):
        """Stage 1 weight for a cost model without ``sample_weight_coefficients``.

        The CP size times the predicted time of one microbatch (forward and backward, without
        the iteration tail) that holds the sample alone in the smallest CP group that fits it.
        This definition is not part of DynamicUSP; it jumps where a sample needs the next CP
        size.
        """
        values = (lengths[sample_id],)
        size = next((s for s in self.scheduler.sizes if self.costs.fits(values, s)), None)
        if size is None:
            raise InfeasibleLayout(f"sample {sample_id} exceeds every allowed group capacity")
        value = self.costs.pack(values, size, 0)
        if value is None:
            raise RuntimeScheduleError(f"cost model rejected scheduling sample {sample_id}")
        program = value[2]
        duration = sum(
            program[payload] if kind in ("cp", "ep_mean") else payload
            for phase in self.timeline.program
            for kind, payload in phase
        )
        work = size * duration
        if not math.isfinite(work) or work <= 0:
            raise ValueError("cost model must produce finite positive scheduling work")
        return work

    def _pipeline_step_time(self, rounds, lengths):
        """Step time with more than one pipeline stage, iteration tail included.

        Every pack is priced through the pack costs first, as with one stage: a pack that
        the cost model rejects ends the candidate the same way, and the pack statistics
        count it. The replay runs the order of ``pipeline_schedule``.
        """
        rows = []
        for row, costed in zip(rounds, self._costed(rounds, lengths)):
            groups = []
            for group, (start, cp, *_) in zip(row, costed):
                values = tuple(lengths[i] for i in group["sample_ids"])
                groups.append((start, cp, *self.model.pipeline_group_times(values, cp)))
            rows.append(groups)
        overhead = self.model.pipeline_terms["per_mb_overhead_ms"]
        world, schedule = self.geometry.world_size, self.pipeline_schedule
        return (
            step_time(rows, world, *self.pipeline, overhead, schedule=schedule)
            + self.timeline.iteration_tail_ms
        )

    def _stage2(self, layout, predicted, lengths, tokens, score):
        """Stage 2 on the layout of the chosen count: ``(step time, rounds, statistics)``.

        Every microbatch is re-packed on its own (``stage2.repack_layout``). When one was
        re-packed the whole layout is scored again and is then the result of the search, as
        in DynamicUSP: its step time can be higher than ``predicted``, that of the Stage 1
        layout. An exception while the re-packed layout is scored keeps the Stage 1 layout
        of the step; the re-packed microbatches are then recorded as ``RESCORE_REJECTED``.
        The statistics count every outcome and hold the records that ``_STAGE2_RECORDED``
        selects or that carry the capacity mark, at most ``_STAGE2_RECORD_LIMIT`` of them,
        each with its ``index`` in the layout.
        """
        rounds = self._rounds(layout)
        repacked, records = repack_layout(
            [[(group.start, group.size, group.sample_ids) for group in row] for row in layout],
            lengths,
            tokens,
            self.model,
            self.stage2_keep_threshold,
            lambda ids, size: self.costs.fits(tuple(lengths[i] for i in ids), size),
        )
        statistics = dict(stage1_predicted_ms=predicted, stage2_predicted_ms=predicted)
        result, source = (predicted, rounds), "stage1"
        if any(record["code"] == ADOPTED for record in records):
            after = [
                [dict(cp=size, sample_ids=list(ids)) for _, size, ids in row] for row in repacked
            ]
            try:
                value = score(after)
            except Exception as error:
                statistics["rescore_error"] = f"{type(error).__name__}: {error_text(error)}"
                for record in records:
                    if record["code"] == ADOPTED:
                        record.update(code=RESCORE_REJECTED, order=None)
                        record["selected_ms"] = record["base_ms"]
            else:
                statistics["stage2_predicted_ms"] = value
                result, source = (value, after), "stage2"
        counts = {}
        for record in records:
            counts[record["code"]] = counts.get(record["code"], 0) + 1
        errors = [record for record in records if "error" in record]
        if errors:
            statistics["first_error"] = f"{errors[0]['error']}: {errors[0]['reason']}"
        recorded = [
            dict(index=index, **{key: record[key] for key in _STAGE2_RECORD if key in record})
            for index, record in enumerate(records)
            if record["code"] in _STAGE2_RECORDED or record.get("runtime_capacity")
        ]
        statistics.update(
            counts=counts,
            adopted_microbatches=counts.get(ADOPTED, 0),
            planned=source,
            microbatches=recorded[:_STAGE2_RECORD_LIMIT],
        )
        if len(recorded) > _STAGE2_RECORD_LIMIT:
            statistics["microbatches_omitted"] = len(recorded) - _STAGE2_RECORD_LIMIT
        return (*result, statistics)

    def plan(self, lengths, *, incumbent_rounds=None) -> RuntimePlan:
        """Search hot-switch count candidates with the injected complete-step cost.

        The layout of the default heuristic (``incumbent_rounds``) is scored with the same
        step time and kept unless the search result is predicted faster; an incumbent whose
        count of microbatches is no multiple of ``count_multiple`` is dropped. When the
        search yields no layout and no incumbent was priced, a capacity-only layout is used.
        With Stage 2 on, the layout of the search is re-packed first (``_stage2``); Stage 2
        is not applied to the other two layouts. ``last_statistics`` marks which layout the
        plan is (``selected_incumbent``, ``capacity_fallback``, ``stage2["planned"]``); the
        README of the package ("Trace records") lists its main fields.
        """
        lengths = self._start_plan(lengths)
        self.scheduler.last_candidates = []

        def score(rounds):
            self._validated(rounds, lengths)
            if self.pipeline[0] > 1:
                value = self._pipeline_step_time(rounds, lengths)
            else:
                value = self.timeline.evaluate(self._costed(rounds, lengths))
            if not math.isfinite(value) or value < 0:
                raise ValueError("cost model must produce a finite nonnegative step time")
            return value

        best, rejection, incumbent_unpriced = None, None, None
        if incumbent_rounds is not None:
            self._validated(incumbent_rounds, lengths)
            # The incumbent is a seed: one the cost model cannot price is dropped, and so is
            # one with a count of microbatches that the pipeline schedule cannot run.
            if len(incumbent_rounds) % self.count_multiple:
                incumbent_unpriced = (
                    f"{len(incumbent_rounds)} microbatches, not a multiple of "
                    f"{self.count_multiple}"
                )
            else:
                try:
                    best = score(incumbent_rounds), incumbent_rounds
                except RuntimeScheduleError as error:
                    incumbent_unpriced = str(error)
        selected_incumbent = best is not None
        capacity_fallback = False
        stage2: dict | None = None
        if self.stage2_enabled:
            stage2 = dict(
                enabled=True,
                keep_threshold=self.stage2_keep_threshold,
                reference_cp_size=self._stage2_reference,
                skipped=self.stage2_idle or "the count search gave no layout",
            )
            if self._stage2_guard is not None:
                stage2["guard_threshold"] = self._stage2_guard
        alignment = self.geometry.dataset_alignment
        tokens = {i: (n + alignment - 1) // alignment * alignment for i, n in lengths.items()}
        # Stage 1 weight: DynamicUSP's A, B, C form when the cost model carries coefficients.
        coefficients = getattr(self.model, "sample_weight_coefficients", None)
        try:
            if coefficients is None:
                weights = {i: self._sample_work(i, lengths) for i in lengths}
            else:
                weights = {i: sample_weight(n, coefficients) for i, n in lengths.items()}
            predicted, layout = self.scheduler.solve(
                tokens,
                weights,
                lambda ids, size: self.costs.fits(tuple(lengths[i] for i in ids), size),
                lambda layout: score(self._rounds(layout)),
                # Most sequences one microbatch holds under the per-group sequence cap.
                sequences_per_microbatch=max(
                    self.geometry.world_size // size * self.sequence_cap
                    for size in self.scheduler.sizes
                ),
            )
            if stage2 is not None and self.stage2_idle is None:
                predicted, rounds, ran = self._stage2(layout, predicted, lengths, tokens, score)
                stage2.update(ran, skipped=None)
            else:
                rounds = self._rounds(layout)
            if best is None or predicted < best[0] - 1e-9:
                best = predicted, rounds
                selected_incumbent = False
            elif stage2 is not None and stage2["skipped"] is None:
                stage2["planned"] = "default"
        except (InfeasibleLayout, RuntimeScheduleError) as error:
            # No better layout from the search: keep the incumbent, else the fallback. A
            # cost-model rejection of one candidate ends the search.
            rejection = str(error)
            if best is None:
                rounds = self._capacity_fallback(lengths, self.scheduler.sizes)
                if rounds is None:
                    raise self._no_layout(lengths, f"hot-switch search: {error}") from error
                best = score(rounds), rounds
                capacity_fallback = True

        self.last_statistics = dict(
            source_commit="1ff97a4ae0beb5e088fadb9e112265bdd194b24d",
            group_sizes=self.scheduler.sizes,
            stage1_weight=(
                "cost_model_sample_work"
                if coefficients is None
                else dict(
                    zip("ABC", coefficients),
                    default=bool(getattr(self.model, "sample_weight_is_default", False)),
                )
            ),
            candidates=self.scheduler.last_candidates,
            selected_incumbent=selected_incumbent,
            capacity_fallback=capacity_fallback,
            rejection=rejection,
            # Why the incumbent has no score: a pack the cost model rejected, or a count of
            # microbatches that the pipeline schedule cannot run.
            incumbent_unpriced=incumbent_unpriced,
            pack_cost_stats=dict(self.costs.stats),
        )
        if self.pipeline[0] > 1:
            # What scored the counts: the replayed schedule and the step terms of the cost
            # model, copied so that the statistics share no object with it.
            self.last_statistics["pipeline"] = dict(
                stages=self.pipeline[0],
                virtual_stages=self.pipeline[1],
                schedule=self.pipeline_schedule,
                count_multiple=self.count_multiple,
                **copy.deepcopy(dict(self.model.pipeline_terms)),
            )
        if stage2 is not None:
            self.last_statistics["stage2"] = stage2
        return self._make_plan(best[1], lengths, best[0])

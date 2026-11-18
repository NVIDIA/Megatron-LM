# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapt generic hot-switch layouts to the existing CPU cost and rank-row API."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from typing import Any

from ..models.solver_section import PIPELINE_SCHEDULES, check_factor_sizes, sample_weight
from .hot_switch import HotSwitchOptions, HotSwitchScheduler, InfeasibleLayout
from .pipeline import step_time
from .planner import RuntimePlan, RuntimePlanner, RuntimeScheduleError, encode_rows, validate_rows
from .stage2 import ADOPTED, PROFILE_EXECUTION_ERROR, _finite, error_text, repack_layout

# The members of a cost model that Stage 2 reads (``models.base.DcpCostModel``). Two more are
# optional, those of the placement guard: ``stage2_guard_threshold`` and, for a model whose
# threshold is not None, ``stage2_guard_margin``.
_STAGE2_HOOKS = (
    "stage2_reference_cp_size",
    "stage2_global_constant_ms",
    "stage2_sample_loads",
    "stage2_empty_group_load",
    "stage2_group_time",
)
# Outcome of a microbatch that Stage 2 re-packed when the re-packed layout of the step could
# not be scored: the step keeps its Stage 1 layout. DynamicUSP has no such outcome.
RESCORE_REJECTED = "RESCORE_REJECTED"
# What the plan statistics keep of the Stage 2 record of a microbatch: the outcome, the queue
# of an adopted layout and the slowest group time, plus the global constant, before and
# after; the type of a contained exception; and the mark of a placement that the capacity
# predicate of the runtime rejected.
_STAGE2_RECORD = ("code", "order", "base_ms", "selected_ms", "error", "runtime_capacity")
# The statistics keep that record only for a microbatch with one of these outcomes or with
# the capacity mark, and for at most this many microbatches of a plan: the statistics travel
# with every plan, and a global batch can have thousands of microbatches.
_STAGE2_RECORDED = (ADOPTED, RESCORE_REJECTED, PROFILE_EXECUTION_ERROR)
_STAGE2_RECORD_LIMIT = 64


def _offers_pipeline_step_time(cost_model) -> bool:
    """True when the cost model has what the step time of several pipeline stages reads.

    ``pipeline_group_times`` must be callable, and ``pipeline_terms`` a mapping whose
    ``per_mb_overhead_ms`` is a finite nonnegative number (the ``DcpCostModel`` contract).
    """
    if not callable(getattr(cost_model, "pipeline_group_times", None)):
        return False
    terms = getattr(cost_model, "pipeline_terms", None)
    if not isinstance(terms, Mapping):
        return False
    overhead = terms.get("per_mb_overhead_ms")
    if isinstance(overhead, bool) or not isinstance(overhead, (int, float)):
        return False
    try:
        return math.isfinite(overhead) and overhead >= 0
    except OverflowError:  # an integer beyond the float range
        return False


class DynamicUSPRuntimePlanner(RuntimePlanner):
    """Optional solver sharing the existing cost wrapper and padded-input contract.

    Both solvers share model injection, validation, pricing and rank-row output.
    Neither solver inherits the other's search algorithm.
    This adapter never selects or initializes an attention communication backend.

    Keyword options are those of ``HotSwitchOptions``. The cost model supplies the rest:
    ``sample_weight_coefficients`` for the Stage 1 sample weight and ``tier_time_factors``
    for Stage 1 time balance, both optional, and the step time that scores a candidate
    count through its timeline.

    ``pipeline`` is ``(pipeline stages, virtual stages per pipeline stage)`` of the training
    run. With one pipeline stage, whatever the second number, everything is as without it.
    With more, the count of microbatches is a multiple of the stages (``pp_multiple``), and
    a candidate is scored with the step time of the interleaved pipeline schedule
    (``pipeline.step_time``) on the group times of the cost model's
    ``pipeline_group_times``, plus its ``pipeline_terms["per_mb_overhead_ms"]`` per
    microbatch; a cost model that does not offer both is rejected.

    ``pipeline_schedule`` names the schedule of a run with more than one pipeline stage.
    None or ``"interleaved"`` is the above: DynamicUSP's order, also for one chunk per
    stage, ``(stages, 1)``. ``"non_interleaved"`` is the order of Megatron without virtual
    pipeline stages (``pipeline.non_interleaved_plan``), which DynamicUSP does not have.
    It takes ``(stages, 1)`` only. It runs any count of microbatches, so the count
    multiple is 1 and the count search is DynamicUSP's with ``pp_multiple`` 1; the group
    times, the step terms and everything else are as for the interleaved schedule.
    ``pipeline_schedule`` (the attribute) is the name, or None with one pipeline stage,
    where the argument is checked and not read further.

    Stage 2 (``stage2``) is switched by the cost model: ``stage2_enabled``, which the model
    reads from its coefficient file, with ``stage2_keep_threshold``. It is off by default.
    When on, the layout of the chosen count is re-packed microbatch by microbatch and
    scored again; ``_stage2`` has the rules and ``last_statistics["stage2"]`` the outcome.
    A cost model with a placement guard (``stage2_guard_threshold``) has it applied inside
    Stage 2, and the statistics then record the threshold. The threshold is validated once, here,
    and read again for every microbatch: every plan uses the validated one, and a cost model that
    then gives another, or none, fails Stage 2 for the microbatch, as any error of its hooks.
    The planner says what it does in three attributes, which exist whether Stage 2 is on or
    off: ``stage2_enabled``, ``stage2_keep_threshold`` (None when off) and ``stage2_idle``
    (why a Stage 2 that is on has no work in any plan of this run, else None).
    """

    def __init__(
        self, geometry, cost_model, *, pipeline=None, pipeline_schedule=None, **options
    ) -> None:
        super().__init__(geometry, cost_model)
        given = (1, 1) if pipeline is None else pipeline
        if (
            not isinstance(given, (tuple, list))
            or len(given) != 2
            or any(type(number) is not int or number < 1 for number in given)
        ):
            raise ValueError(
                "pipeline must be (pipeline stages, virtual stages per pipeline stage), two "
                f"positive integers; got {pipeline!r}"
            )
        if pipeline_schedule is not None and pipeline_schedule not in PIPELINE_SCHEDULES:
            raise ValueError(
                f"pipeline_schedule must be one of {list(PIPELINE_SCHEDULES)} or None; got "
                f"{pipeline_schedule!r}"
            )
        # One pipeline stage has no chunks to interleave: its virtual stages are not read,
        # and neither is the schedule.
        self.pipeline = tuple(given) if given[0] > 1 else (1, 1)
        stages = self.pipeline[0]
        self.pipeline_schedule = None
        if stages > 1:
            self.pipeline_schedule = pipeline_schedule or PIPELINE_SCHEDULES[0]
        non_interleaved = self.pipeline_schedule == PIPELINE_SCHEDULES[1]
        if non_interleaved and self.pipeline[1] != 1:
            raise ValueError(
                "the non-interleaved pipeline schedule has one model chunk per pipeline stage; "
                f"got pipeline={pipeline!r}"
            )
        if stages > 1 and not _offers_pipeline_step_time(cost_model):
            raise ValueError(
                f"the {cost_model.name!r} cost model has no step time for more than one "
                "pipeline stage (it must offer pipeline_group_times and pipeline_terms); "
                f"this run has {stages} pipeline stages"
            )
        # The count multiple is the number of pipeline stages (DynamicUSP's ``pp_multiple``),
        # for the search, the incumbent and the capacity fallback alike. Megatron's
        # non-interleaved schedule runs any count: its multiple is 1, with which the search
        # is DynamicUSP's for a count that is left free.
        multiple = 1 if non_interleaved else stages
        if options.setdefault("pp_multiple", multiple) != multiple:
            if non_interleaved:
                raise ValueError(
                    "pp_multiple is 1 on the non-interleaved pipeline schedule, which runs "
                    f"any count of microbatches; got {options['pp_multiple']!r}"
                )
            raise ValueError(
                f"pp_multiple is the number of pipeline stages, {stages}; got "
                f"{options['pp_multiple']!r}"
            )
        self.count_multiple = multiple
        # Time-balance factors: a table given as a Python option wins over the one the cost
        # model read from its coefficient file. Keys are CP sizes of the cost model; those
        # this run cannot form are never read.
        if options.get("tier_time_factors") is None:
            options["tier_time_factors"] = getattr(cost_model, "tier_time_factors", None)
        self.options = HotSwitchOptions(**options)
        supported = getattr(cost_model, "supported_cp_sizes", None)
        if supported is not None:
            check_factor_sizes(self.options.tier_time_factors or (), supported)
        sizes = tuple(
            s
            for s in self.geometry.allowed_cp_sizes
            if s & (s - 1) == 0 and geometry.world_size % s == 0
        )
        self.scheduler = HotSwitchScheduler(
            geometry.world_size, sizes, geometry.tokens_per_rank, self.options
        )
        # Stage 2 is switched by the cost model, which reads the switch from its coefficient
        # file. Anything but True leaves it off, and then no Stage 2 member is read. The
        # three public attributes exist either way (the runtime reports them at start-up).
        self.stage2_enabled = getattr(cost_model, "stage2_enabled", False) is True
        self.stage2_keep_threshold: float | None = None
        self.stage2_idle: str | None = None
        self._stage2_reference: int | None = None
        self._stage2_guard: float | None = None
        if self.stage2_enabled:
            missing = [name for name in _STAGE2_HOOKS if not hasattr(cost_model, name)]
            # The placement guard is optional: a model with a threshold also has the margin.
            guard: Any = getattr(cost_model, "stage2_guard_threshold", None)
            if guard is not None and not hasattr(cost_model, "stage2_guard_margin"):
                missing.append("stage2_guard_margin")
            if missing:
                raise ValueError(
                    f"the {cost_model.name!r} cost model switches Stage 2 on without "
                    f"offering {missing}"
                )
            threshold: Any = getattr(cost_model, "stage2_keep_threshold", None)
            try:
                valid = (
                    type(threshold) in (int, float) and math.isfinite(threshold) and threshold > 0
                )
            except OverflowError:  # an integer beyond the float range
                valid = False
            if not valid:
                raise ValueError(
                    f"stage2_keep_threshold of the {cost_model.name!r} cost model must be a "
                    f"finite positive number; got {threshold!r}"
                )
            self.stage2_keep_threshold = float(threshold)
            reference = cost_model.stage2_reference_cp_size
            if type(reference) is not int or reference < 1:
                raise ValueError(
                    f"stage2_reference_cp_size of the {cost_model.name!r} cost model must be "
                    f"a positive integer; got {reference!r}"
                )
            # Kept: the plan statistics record it without asking the model again.
            self._stage2_reference = reference
            if guard is not None:
                # The engine's own test of a threshold: an int or a float, finite, no boolean.
                if not _finite(guard):
                    raise ValueError(
                        f"stage2_guard_threshold of the {cost_model.name!r} cost model must be "
                        f"a finite number or None; got {guard!r}"
                    )
                # Kept as well: the threshold of every plan and of the plan statistics.
                self._stage2_guard = float(guard)
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
        forward, backward, program = value
        if self.timeline.program is None:
            duration = self.timeline.layer_count * (
                forward
                + backward
                + (self.timeline.forward_non_attention_ms + self.timeline.backward_non_attention_ms)
            )
        else:
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
        stages, virtual_stages = self.pipeline
        overhead = self.model.pipeline_terms["per_mb_overhead_ms"]
        world_size = self.geometry.world_size
        return (
            step_time(
                rows, world_size, stages, virtual_stages, overhead, schedule=self.pipeline_schedule
            )
            + self.timeline.iteration_tail_ms
        )

    def _stage2(self, layout, predicted, lengths, tokens, score):
        """Stage 2 on the layout of the chosen count: ``(step time, rounds, statistics)``.

        Every microbatch is re-packed on its own (``stage2.repack_layout``): as in
        DynamicUSP the rule looks at the slowest group of the microbatch, not at the step.
        The capacity predicate of the runtime holds for every placement, and the guard
        threshold is the one read at construction (None without a guard). When a microbatch
        was re-packed the whole layout is scored again, with the step time that scored the
        counts. The re-packed layout with that step time is then the result of the search,
        as in DynamicUSP: the step time can be higher than ``predicted``, the step time of
        the Stage 1 layout, and the Stage 1 layout is not a candidate any more.

        Nothing raised in here loses the Stage 1 layout. An exception while one microbatch
        is re-packed keeps the Stage 1 layout of that microbatch (DynamicUSP's fallback
        mode). An exception while the re-packed layout is scored keeps the Stage 1 layout of
        the step; the microbatches that were re-packed are then recorded as
        ``RESCORE_REJECTED``.

        The statistics count the outcome of every microbatch. They hold a record only for a
        microbatch that was re-packed, hit an exception or had a placement rejected by the
        capacity predicate, with its ``index`` in the layout (its position in the plan,
        unless the default layout is planned), and for at most ``_STAGE2_RECORD_LIMIT`` of
        them; ``microbatches_omitted``, present when some are left out, counts the rest.
        """
        rounds = self._rounds(layout)
        repacked, records = repack_layout(
            [[(group.start, group.size, group.sample_ids) for group in row] for row in layout],
            lengths,
            tokens,
            self.model,
            self.stage2_keep_threshold,
            lambda ids, size: self.costs.fits(tuple(lengths[i] for i in ids), size),
            self._stage2_guard,
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
        step time and kept unless the search result is predicted faster. When the search
        yields no layout and no incumbent was priced, a capacity-only layout is used.
        DynamicUSP has neither; ``last_statistics`` marks such plans (``selected_incumbent``,
        ``capacity_fallback``). With more than one pipeline stage both have a count of
        microbatches that is a multiple of the stages, as every count of the search: an
        incumbent with another count is dropped, and the capacity-only layout is built
        with such a count. On the non-interleaved schedule the multiple is 1: every
        incumbent is priced.

        With Stage 2 switched on, the layout of the search is re-packed (``_stage2``)
        and the re-packed layout, with its step time, is what is compared with the default
        layout: the one rule of DynamicUSP, with or without a default layout. Stage 2 is
        not applied to the default layout or to the capacity-only layout.
        ``last_statistics["stage2"]``, which exists only then, holds the switch and the keep
        threshold, ``guard_threshold`` when the cost model has a placement guard,
        ``skipped`` (why Stage 2 did no work for this plan, else None) and, when
        it ran: the outcome ``counts`` of all microbatches, the number of re-packed
        microbatches, the predicted step time before and after, the records
        ``microbatches`` (see ``_stage2``), and ``planned``: which layout the plan is.
        ``"stage2"`` is the re-packed layout, scored again; ``"stage1"`` the Stage 1 layout,
        when no microbatch was re-packed or the re-packed layout could not be scored;
        ``"default"`` the default layout, when it was kept. ``first_error`` and
        ``rescore_error`` are present when an exception was contained.
        """
        begin = time.perf_counter()
        lengths = self._start_plan(lengths)
        self.scheduler.last_candidates = []

        def score(rounds):
            validate_rows(
                encode_rows(rounds, self.geometry.world_size),
                lengths,
                self.geometry,
                min(self.geometry.allowed_cp_sizes),
            )
            if self.pipeline[0] > 1:
                value = self._pipeline_step_time(rounds, lengths)
            else:
                value = self.timeline.evaluate(self._costed(rounds, lengths))
            if not math.isfinite(value) or value < 0:
                raise ValueError("cost model must produce a finite nonnegative step time")
            return value

        best, rejection, incumbent_unpriced = None, None, None
        if incumbent_rounds is not None:
            validate_rows(
                encode_rows(incumbent_rounds, self.geometry.world_size),
                lengths,
                self.geometry,
                min(self.geometry.allowed_cp_sizes),
            )
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
                # Most sequences one microbatch holds under the per-group sequence caps.
                sequences_per_microbatch=max(
                    self.geometry.world_size // size * self.sequence_limit(size)
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
            # The step time that scores a count is DynamicUSP's objective only when no rank
            # waits for all others after the first forward pass; else it is the model's own.
            algorithm=(
                "DynamicUSP solve_gbs Stage 1 + step time of the cost model"
                if self.timeline.first_forward_dp_cp_sync
                else "DynamicUSP solve_gbs Stage 1 + step-time objective at one pipeline stage"
            ),
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
            # Why the incumbent has no score: the cost model rejected one of its packs, or,
            # with more than one pipeline stage, its count of microbatches is no multiple of
            # the stages; such an incumbent is not priced. ``selected_incumbent`` therefore
            # marks an incumbent whose count is a multiple of the stages (of the count
            # multiple, which is 1 on the non-interleaved schedule).
            incumbent_unpriced=incumbent_unpriced,
            pack_cost_stats=dict(self.costs.stats),
            model_sequence_limits=self.model_sequence_limits(),
        )
        if self.pipeline[0] > 1:
            stages, virtual_stages = self.pipeline
            self.last_statistics.update(
                # The interleaved order is DynamicUSP's objective; the non-interleaved order
                # is Megatron's, so its step time is not called DynamicUSP's objective.
                algorithm=(
                    "DynamicUSP solve_gbs Stage 1 + step time of Megatron's non-interleaved "
                    "pipeline schedule"
                    if self.pipeline_schedule == PIPELINE_SCHEDULES[1]
                    else "DynamicUSP solve_gbs Stage 1 + step-time objective of the interleaved "
                    "pipeline schedule"
                ),
                # What scored the counts: the replayed schedule and the step terms of the
                # cost model, with the names of those left at their defaults. The terms are
                # copied with their nested values: the statistics share no object with the
                # cost model.
                pipeline=dict(
                    stages=stages,
                    virtual_stages=virtual_stages,
                    schedule=self.pipeline_schedule,
                    count_multiple=self.count_multiple,
                    **copy.deepcopy(dict(self.model.pipeline_terms)),
                ),
            )
        if stage2 is not None:
            self.last_statistics["algorithm"] += ", Stage 2 on"
            self.last_statistics["stage2"] = stage2
        return self._make_plan(best[1], lengths, best[0], begin)

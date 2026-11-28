# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-unit cost model fitted from Megatron's per-rank timers.

Per micro-batch, stage ``s`` costs ``fwd_s = sum_u n_su * fwd_u + fwd_0`` and
``bwd_s = sum_u n_su * bwd_u + bwd_0``, for units ``E``, ``t``, ``m`` and ``L`` with counts
``n_su`` from the layout and a per-stage constant. The observations are the per-rank
``forward-compute`` and ``backward-compute`` timers of profiling runs divided by the number of
micro-batches. Two profiling layouts on four stages give eight equations per direction for five
unknowns. A weak ridge prior toward FLOP ratios keeps any unit that the profiling layouts cannot
separate near its FLOP-based value.

Step time is the simulated forward-backward makespan plus a per-step overhead (optimizer,
gradient synchronization). Level-2 timers synchronize the device and inflate the overhead seen on
profiling runs, so :func:`calibrate_overhead` re-estimates it from timer-free runs.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .layout import UNITS, Stages
from .model import ModelShape
from .schedule import simulate_1f1b
from .timings import LogSummary


def _solve(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve a small dense linear system by Gaussian elimination with partial pivoting."""
    size = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-18:
            raise ValueError("Singular system; the profiling layouts do not identify the costs.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(size):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                for k in range(col, size + 1):
                    aug[row][k] -= factor * aug[col][k]
    return [aug[i][size] / aug[i][i] for i in range(size)]


def _counts(stage: List[str]) -> List[float]:
    return [float(stage.count(unit)) for unit in UNITS] + [1.0]


@dataclass
class CostModel:
    """Per-micro-batch unit costs plus the per-step overhead."""

    forward_ms: Dict[str, float] = field(default_factory=dict)
    """Forward time per micro-batch for each unit, plus the per-stage constant ``c0``."""

    backward_ms: Dict[str, float] = field(default_factory=dict)
    """Backward time per micro-batch for each unit (with recompute), plus ``c0``."""

    overhead_ms: float = 0.0
    """Per-step time outside the forward-backward pass at ``overhead_ref_params``."""

    overhead_ref_params: float = 0.0
    """Largest stage parameter count at which ``overhead_ms`` was measured."""

    optimizer_ms_per_param: float = 0.0
    """Optimizer time per parameter on the largest stage, used to adjust the overhead."""

    p2p_ms: float = 0.2
    """Latency of one activation or gradient transfer between adjacent stages."""

    source: str = "flops"
    """How the costs were obtained."""

    fit_report: Dict = field(default_factory=dict)
    """Diagnostics from the fit."""

    def stage_ms(self, stage: List[str]) -> Tuple[float, float]:
        """Return (forward, backward) time per micro-batch for one stage."""
        counts = _counts(stage)
        keys = list(UNITS) + ["c0"]
        forward = sum(n * self.forward_ms.get(k, 0.0) for n, k in zip(counts, keys))
        backward = sum(n * self.backward_ms.get(k, 0.0) for n, k in zip(counts, keys))
        return forward, backward

    def overhead_for(self, max_stage_params: float) -> float:
        """Return the per-step overhead for a layout whose largest stage has these parameters."""
        delta = max_stage_params - self.overhead_ref_params
        return self.overhead_ms + self.optimizer_ms_per_param * delta

    def to_json(self) -> str:
        """Return the model as JSON."""
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(path: str) -> "CostModel":
        """Load a cost model written by :meth:`to_json`."""
        with open(path) as handle:
            return CostModel(**json.load(handle))

    @staticmethod
    def from_flops(shape: ModelShape, tflops: float, p2p_ms: float = 0.2) -> "CostModel":
        """Return a cost model from FLOPs alone, at an assumed effective throughput.

        Args:
            shape: Model shape.
            tflops: Effective TFLOP/s per GPU.
            p2p_ms: Point-to-point latency between stages.

        Returns:
            A cost model with no per-stage constant and no overhead.
        """
        flops_per_ms = tflops * 1e9
        forward = {u: shape.unit_flops_per_microbatch(u)[0] / flops_per_ms for u in UNITS}
        backward = {u: shape.unit_flops_per_microbatch(u)[1] / flops_per_ms for u in UNITS}
        forward["c0"] = backward["c0"] = 0.0
        return CostModel(forward, backward, p2p_ms=p2p_ms, source=f"flops@{tflops}TFLOP/s")


@dataclass
class ProfilingRun:
    """Per-stage timings of one profiling layout."""

    stages: Stages
    """The profiled layout."""

    num_microbatches: int
    """Micro-batches per step."""

    forward_ms: List[float]
    """Per-stage ``forward-compute`` time per iteration."""

    backward_ms: List[float]
    """Per-stage ``backward-compute`` time per iteration."""

    step_ms: Optional[float] = None
    """Median elapsed time per iteration."""

    forward_backward_ms: Optional[float] = None
    """Median ``forward-backward`` time per iteration (largest rank)."""

    optimizer_ms: Optional[List[float]] = None
    """Per-stage ``optimizer`` time per iteration, if logged."""


def profiling_run_from_summary(
    stages: Stages, summary: LogSummary, num_microbatches: int, ranks_per_stage: int = 1
) -> ProfilingRun:
    """Build a :class:`ProfilingRun` from a summarized log.

    Stage ``s`` is taken as the slowest of ranks ``s * k`` to ``s * k + k - 1`` (``k`` =
    ``ranks_per_stage``), which holds when pipeline parallelism is the outermost dimension of the
    rank order (Megatron's default ``tp-cp-ep-dp-pp``) and there is no virtual pipelining.

    Args:
        stages: The profiled layout.
        summary: Summary of a log run with ``--timing-log-level 2 --timing-log-option all``.
        num_microbatches: Micro-batches per step.
        ranks_per_stage: Data, context and tensor parallel ranks per pipeline stage.

    Returns:
        The per-stage timings.
    """

    def per_stage(name: str) -> List[Optional[float]]:
        values = summary.timers.get(name, {})
        out = []
        for stage_idx in range(len(stages)):
            ranks = range(stage_idx * ranks_per_stage, (stage_idx + 1) * ranks_per_stage)
            found = [values[r] for r in ranks if r in values]
            out.append(max(found) if found else None)
        return out

    forward, backward = per_stage("forward-compute"), per_stage("backward-compute")
    if None in forward or None in backward:
        raise ValueError(
            "Missing forward/backward-compute timers; profile with "
            "--timing-log-level 2 --timing-log-option all."
        )
    optimizer = per_stage("optimizer")
    forward_backward = summary.timers.get("forward-backward", {})
    return ProfilingRun(
        stages=stages,
        num_microbatches=num_microbatches,
        forward_ms=forward,
        backward_ms=backward,
        step_ms=summary.step_ms,
        forward_backward_ms=max(forward_backward.values()) if forward_backward else None,
        optimizer_ms=None if None in optimizer else optimizer,
    )


def fit(
    shape: ModelShape, runs: Sequence[ProfilingRun], prior_weight: float = 0.05, p2p_ms: float = 0.2
) -> CostModel:
    """Fit unit costs to the per-stage timings of profiling runs.

    Args:
        shape: Model shape, used for the FLOP prior and per-stage parameters.
        runs: Profiling runs; two layouts that vary decoder counts per stage and place MTP on
            its own stage identify every unit.
        prior_weight: Strength of the ridge prior toward FLOP ratios.
        p2p_ms: Point-to-point latency between stages.

    Returns:
        The fitted cost model.
    """
    flops = {unit: shape.unit_flops_per_microbatch(unit) for unit in UNITS}
    model = CostModel(p2p_ms=p2p_ms, source="fit")
    report: Dict = {}
    for direction, index in (("forward", 0), ("backward", 1)):
        rows, targets, stage_flops = [], [], []
        for run in runs:
            measured = run.forward_ms if index == 0 else run.backward_ms
            for stage, value in zip(run.stages, measured):
                rows.append(_counts(stage))
                targets.append(value / run.num_microbatches)
                stage_flops.append(sum(stage.count(u) * flops[u][index] for u in UNITS))
        ms_per_flop = statistics.median(t / f for t, f in zip(targets, stage_flops) if f > 0)
        prior = [ms_per_flop * flops[unit][index] for unit in UNITS] + [0.0]
        weight = (prior_weight**2) * len(targets)
        size = len(prior)
        normal = [[sum(r[i] * r[j] for r in rows) for j in range(size)] for i in range(size)]
        rhs = [sum(r[i] * t for r, t in zip(rows, targets)) for i in range(size)]
        for i in range(size):
            w = weight if i < len(UNITS) else weight * 0.01  # c0 is pulled only weakly to 0
            normal[i][i] += w
            rhs[i] += w * prior[i]
        coef = _solve(normal, rhs)
        costs = {unit: coef[i] for i, unit in enumerate(UNITS)}
        costs["c0"] = coef[-1]
        setattr(model, f"{direction}_ms", costs)
        predicted = [sum(c * x for c, x in zip(coef, row)) for row in rows]
        errors = [abs(p - t) / t for p, t in zip(predicted, targets) if t > 0]
        report[direction] = {
            "mean_rel_residual": statistics.mean(errors) if errors else None,
            "max_rel_residual": max(errors) if errors else None,
            "points": len(targets),
        }

    overheads, max_params, optimizer_points = [], [], []
    for run in runs:
        first_layers = [sum(s.count("t") for s in run.stages[:i]) for i in range(len(run.stages))]
        params = [shape.stage_params(s, f) for s, f in zip(run.stages, first_layers)]
        if run.step_ms is not None and run.forward_backward_ms is not None:
            overheads.append(run.step_ms - run.forward_backward_ms)
            max_params.append(max(params))
        for p, ms in zip(params, run.optimizer_ms or []):
            optimizer_points.append((p, ms))
    if overheads:
        model.overhead_ms = statistics.mean(overheads)
        model.overhead_ref_params = statistics.mean(max_params)
    if len(optimizer_points) >= 2:
        mean_p = statistics.mean(p for p, _ in optimizer_points)
        mean_ms = statistics.mean(ms for _, ms in optimizer_points)
        var = sum((p - mean_p) ** 2 for p, _ in optimizer_points)
        if var:
            cov = sum((p - mean_p) * (ms - mean_ms) for p, ms in optimizer_points)
            model.optimizer_ms_per_param = cov / var
    report["overhead_ms"] = model.overhead_ms
    model.fit_report = report
    return model


def calibrate_overhead(
    shape: ModelShape, model: CostModel, timer_free_runs: Sequence[Tuple[Stages, float]]
) -> CostModel:
    """Re-estimate the per-step overhead from runs without level-2 timers.

    Use baseline layouts only, so predictions for planned layouts stay out of sample.

    Args:
        shape: Model shape.
        model: Fitted cost model; updated in place.
        timer_free_runs: Pairs of (layout, measured median step time in ms).

    Returns:
        The updated cost model.
    """
    overheads = []
    for stages, step_ms in timer_free_runs:
        forward, backward = zip(*(model.stage_ms(stage) for stage in stages))
        result = simulate_1f1b(forward, backward, shape.num_microbatches, model.p2p_ms)
        first_layers = [sum(s.count("t") for s in stages[:i]) for i in range(len(stages))]
        largest = max(shape.stage_params(s, f) for s, f in zip(stages, first_layers))
        adjustment = model.optimizer_ms_per_param * (largest - model.overhead_ref_params)
        overheads.append(step_ms - result.makespan_ms - adjustment)
    if overheads:
        model.fit_report["overhead_ms_profiling"] = model.overhead_ms
        model.overhead_ms = statistics.mean(overheads)
        model.fit_report["overhead_ms_timer_free"] = model.overhead_ms
        model.source += "+timer-free-overhead"
    return model

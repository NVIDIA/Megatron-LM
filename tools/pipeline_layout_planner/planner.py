# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Score pipeline layouts and search for the fastest one that fits in memory."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import layout as layouts
from .cost_model import CostModel
from .layout import Stages
from .model import ModelShape
from .schedule import simulate_1f1b


@dataclass
class LayoutScore:
    """Predicted step time and memory of one layout."""

    layout: str
    """The layout string."""

    step_ms: float
    """Predicted time per training step."""

    forward_backward_ms: float
    """Predicted forward-backward makespan."""

    stage_forward_ms: List[float]
    """Forward time per micro-batch for each stage."""

    stage_backward_ms: List[float]
    """Backward time per micro-batch for each stage."""

    stage_memory_gb: List[float]
    """Estimated peak "max allocated" per stage."""

    critical_stages: int
    """Stages within 3 percent of the slowest; several critical stages compound jitter."""

    mtp_standalone: bool
    """Whether MTP sits on a stage without the loss (predictions are less accurate there)."""

    fits: bool
    """Whether every stage fits in the memory limit."""


def score(
    shape: ModelShape, model: CostModel, stages: Stages, hbm_gb: Optional[float] = None
) -> LayoutScore:
    """Predict step time and memory for one layout without virtual pipelining.

    Args:
        shape: Model shape.
        model: Unit cost model.
        stages: The layout, one stage per pipeline rank.
        hbm_gb: Memory limit per GPU in GB, or None for no limit.

    Returns:
        The predicted score.
    """
    forward, backward = zip(*(model.stage_ms(stage) for stage in stages))
    result = simulate_1f1b(forward, backward, shape.num_microbatches, model.p2p_ms)
    first_layers = [sum(s.count("t") for s in stages[:i]) for i in range(len(stages))]
    params = [shape.stage_params(s, f) for s, f in zip(stages, first_layers)]
    memory = [shape.stage_memory_gb(s, f) for s, f in zip(stages, first_layers)]
    totals = [f + b for f, b in zip(forward, backward)]
    return LayoutScore(
        layout=layouts.to_string(stages),
        step_ms=result.makespan_ms + model.overhead_for(max(params)),
        forward_backward_ms=result.makespan_ms,
        stage_forward_ms=list(forward),
        stage_backward_ms=list(backward),
        stage_memory_gb=memory,
        critical_stages=sum(1 for t in totals if t >= 0.97 * max(totals)),
        mtp_standalone=any("m" in s and "L" not in s for s in stages),
        fits=hbm_gb is None or max(memory) <= hbm_gb,
    )


def _sort_key(candidate: LayoutScore):
    totals = [f + b for f, b in zip(candidate.stage_forward_ms, candidate.stage_backward_ms)]
    mean = sum(totals) / len(totals)
    return (
        round(candidate.step_ms, 1),
        candidate.critical_stages,
        round(max(candidate.stage_memory_gb), 1),
        sum((t - mean) ** 2 for t in totals),
        candidate.layout,
    )


@dataclass
class PlanResult:
    """Ranked layouts and the baselines they are compared against."""

    top: List[LayoutScore]
    """Best layouts, fastest first."""

    baselines: Dict[str, LayoutScore]
    """Megatron's default split (when defined) and the count-based split."""

    num_fitting: int
    """Valid layouts that fit in memory."""


def plan(
    shape: ModelShape,
    model: CostModel,
    pp_size: int,
    hbm_gb: Optional[float] = None,
    top: int = 5,
    exclude_standalone_mtp: bool = True,
) -> PlanResult:
    """Rank every valid layout without virtual pipelining by predicted step time.

    Ties go to layouts with fewer critical stages, then lower peak memory, then more even
    stages.

    Args:
        shape: Model shape.
        model: Unit cost model.
        pp_size: Pipeline-model-parallel size.
        hbm_gb: Memory limit per GPU in GB, or None for no limit.
        top: Number of layouts to return.
        exclude_standalone_mtp: Skip layouts with MTP on a stage without the loss.

    Returns:
        The ranked layouts and the baselines.
    """
    candidates = []
    for stages in layouts.enumerate_layouts(shape.num_layers, pp_size, shape.mtp_num_layers):
        candidate = score(shape, model, stages, hbm_gb)
        if candidate.fits and not (exclude_standalone_mtp and candidate.mtp_standalone):
            candidates.append(candidate)
    candidates.sort(key=_sort_key)
    baselines = {}
    if shape.num_layers % pp_size == 0:
        default = layouts.megatron_default(shape.num_layers, pp_size, shape.mtp_num_layers)
        baselines["megatron_default"] = score(shape, model, default, hbm_gb)
    count = layouts.count_based(shape.num_layers, pp_size, shape.mtp_num_layers)
    baselines["count_based"] = score(shape, model, count, hbm_gb)
    return PlanResult(top=candidates[:top], baselines=baselines, num_fitting=len(candidates))


def profiling_layouts(num_layers: int, pp_size: int, mtp_num_layers: int = 0) -> List[str]:
    """Return two layouts whose timings identify every unit cost.

    The first is Megatron's default split (or the count-based split when the default is not
    defined). The second spreads decoder layers unevenly over the first ``pp_size - 1`` stages,
    puts MTP alone at the end of the second-to-last stage and the loss alone on the last stage,
    so the MTP and loss costs separate.

    Args:
        num_layers: Number of decoder layers.
        pp_size: Pipeline-model-parallel size.
        mtp_num_layers: Number of MTP layers.

    Returns:
        Two layout strings.
    """
    if num_layers % pp_size == 0:
        first = layouts.megatron_default(num_layers, pp_size, mtp_num_layers)
    else:
        first = layouts.count_based(num_layers, pp_size, mtp_num_layers)
    if pp_size < 3 or not mtp_num_layers:
        second = layouts.count_based(num_layers, pp_size, mtp_num_layers)
    else:
        base, remainder = divmod(num_layers, pp_size - 1)
        sizes = [base + (1 if i < remainder else 0) for i in range(pp_size - 1)]
        if sizes[-1] > 1:
            sizes[0] += 1
            sizes[-1] -= 1
        second = [["t"] * size for size in sizes]
        second[0].insert(0, "E")
        second[-1].extend(["m"] * mtp_num_layers)
        second.append(["L"])
    return [layouts.to_string(first), layouts.to_string(second)]

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Discrete-event simulation of the non-interleaved 1F1B pipeline schedule.

Per stage ``s`` the op order follows ``forward_backward_pipelining_without_interleaving`` in
``megatron/core/pipeline_parallel/schedules.py``: ``min(PP - s - 1, M)`` warmup forwards, then
steady forward/backward pairs, then the remaining backwards. ``F(s, i)`` waits for ``F(s - 1, i)``
plus one point-to-point hop; ``B(s, i)`` waits for ``B(s + 1, i)`` plus one hop, and the last
stage's ``B`` waits for its own ``F``.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class ScheduleResult:
    """Outcome of a simulated 1F1B forward-backward pass."""

    makespan_ms: float
    """Time from the first forward to the last backward, as Megatron's forward-backward timer."""

    bottleneck_stage: int
    """Stage with the largest forward plus backward time per micro-batch."""

    idle_fraction: List[float]
    """Per-stage fraction of the makespan spent waiting."""


def simulate_1f1b(
    forward_ms: Sequence[float],
    backward_ms: Sequence[float],
    num_microbatches: int,
    p2p_ms: float = 0.0,
) -> ScheduleResult:
    """Simulate one forward-backward pass of the non-interleaved 1F1B schedule.

    Args:
        forward_ms: Forward time per micro-batch for each stage.
        backward_ms: Backward time per micro-batch for each stage, including recompute.
        num_microbatches: Micro-batches per step.
        p2p_ms: Latency of one activation or gradient transfer between adjacent stages.

    Returns:
        The simulated makespan and per-stage utilization.
    """
    num_stages = len(forward_ms)
    if num_stages == 0 or len(backward_ms) != num_stages:
        raise ValueError("forward_ms and backward_ms need one entry per stage.")
    orders: List[List[Tuple[str, int]]] = []
    for stage in range(num_stages):
        warmup = min(num_stages - stage - 1, num_microbatches)
        ops = [("F", mb) for mb in range(warmup)]
        for k in range(num_microbatches - warmup):
            ops.append(("F", warmup + k))
            ops.append(("B", k))
        ops.extend(("B", mb) for mb in range(num_microbatches - warmup, num_microbatches))
        orders.append(ops)

    done: Dict[Tuple[str, int, int], float] = {}
    free = [0.0] * num_stages
    position = [0] * num_stages
    remaining = sum(len(ops) for ops in orders)
    while remaining:
        progress = False
        for stage in range(num_stages):
            while position[stage] < len(orders[stage]):
                kind, mb = orders[stage][position[stage]]
                if kind == "F":
                    upstream = 0.0 if stage == 0 else done.get(("F", stage - 1, mb))
                    if upstream is None:
                        break
                    ready = upstream + (p2p_ms if stage else 0.0)
                    duration = forward_ms[stage]
                else:
                    own_forward = done.get(("F", stage, mb))
                    if stage < num_stages - 1:
                        downstream = done.get(("B", stage + 1, mb))
                    else:
                        downstream = own_forward
                    if own_forward is None or downstream is None:
                        break
                    hop = p2p_ms if stage < num_stages - 1 else 0.0
                    ready = max(own_forward, downstream + hop)
                    duration = backward_ms[stage]
                end = max(free[stage], ready) + duration
                done[(kind, stage, mb)] = end
                free[stage] = end
                position[stage] += 1
                remaining -= 1
                progress = True
        if not progress:
            raise RuntimeError("1F1B simulation deadlocked; this is a bug in the simulator.")
    makespan = max(free)
    busy = [num_microbatches * (forward_ms[s] + backward_ms[s]) for s in range(num_stages)]
    return ScheduleResult(
        makespan_ms=makespan,
        bottleneck_stage=max(range(num_stages), key=lambda s: forward_ms[s] + backward_ms[s]),
        idle_fraction=[1.0 - b / makespan for b in busy],
    )

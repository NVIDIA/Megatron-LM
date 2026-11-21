# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from DynamicUSP at commit
# 1ff97a4ae0beb5e088fadb9e112265bdd194b24d (src/hot_switch_solver: objectives/pipeline.py),
# BSD-3-Clause license.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license text.

"""Step time of a global batch with more than one pipeline stage.

``interleaved_plan`` and the replay of ``step_time`` port the schedule and the replay of
the DynamicUSP step-time objective. The model is cut into ``stages * virtual_stages``
chunks, ``virtual_stages`` of them on every pipeline stage. ``interleaved_plan`` lists the
forward and backward chunk operations of every stage in an order in which they can run.
``step_time`` replays them on the DPxCP ranks: an operation takes the forward or backward
time of its CP group divided by the number of chunks, starts when the previous operation
of its stage and the operation that produces its input have ended, and starts at the same
time on all ranks of one CP group.

Differences from upstream: times are kept per DPxCP rank instead of per smallest group
size, which gives the same result for the layouts upstream accepts and also scores a
layout whose groups are not aligned to one size; the uncoupled replay and the imbalance
diagnostics are not ported. The arguments of ``interleaved_plan`` are checked: a count of
microbatches, of stages or of virtual stages that is not a positive integer raises
ValueError, where upstream returns a plan or fails in another way. Upstream limits the
warm-up of a stage to the number of its forward chunks, ``min(warm-up, total)``; that is
not needed here, because the limit never applies when the count of microbatches is a
multiple of the stages. At most 32 plans are kept; upstream keeps every plan it has built.

``non_interleaved_plan`` is not part of DynamicUSP, which has the one plan builder above:
it lists the order in which Megatron runs a training step with more than one pipeline
stage and no virtual pipeline stages (``forward_backward_pipelining_without_interleaving``
of ``megatron/core/pipeline_parallel/schedules.py``), built by ``_plan`` from its warm-up rule. It
has one model chunk per stage. Against upstream's order for one chunk per stage
(``interleaved_plan`` with ``virtual_stages`` 1) it is one-forward-one-backward with the
same dependencies as well, and differs in two points. The warm-up: stage ``r`` first runs
``min(stages - r - 1, microbatches)`` forward microbatches, where upstream's formula gives
``2 * (stages - r - 1)``, and all of them when the count equals the stages. The counts:
any count of microbatches runs, also one below the number of stages, where upstream's
plan takes multiples of the stages only; the ``min`` is therefore needed in this builder.
For identical microbatches the two orders give the same step time; for others they
differ in general. ``step_time`` replays either plan with the same rules (``schedule``).
In both replays an operation waits only for its input and for the previous operation of
its stage: a stage that Megatron holds in a pure send, until the next stage takes the
activation or the gradient, is not modelled. ``non_interleaved_plan`` is the order of a
training step: an evaluation step of Megatron's function runs the forward passes only,
in the order this plan has them. The batches of an evaluation iterator are planned with
the order of a training step all the same, as they are with one stage and with virtual
stages.

All three functions are pure: no clock and no random source.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence


def _execution_order(sequences, predecessor) -> tuple:
    """Merge the per-stage operation sequences into one order in which they can run.

    ``sequences[stage]`` lists ``(direction, microbatch, chunk)`` in the order the stage
    runs them. ``predecessor(direction, microbatch, chunk, stage)`` names the operation
    whose output the operation needs, or None. Every entry of the result is
    ``(operation, predecessor, has_previous)`` with ``operation = (direction, microbatch,
    chunk, stage)``; ``has_previous`` is False for the first operation of a stage.
    """
    order, done, position = [], set(), [0] * len(sequences)
    remaining = sum(len(sequence) for sequence in sequences)
    while remaining:
        before = remaining
        for stage, sequence in enumerate(sequences):
            while position[stage] < len(sequence):
                operation = (*sequence[position[stage]], stage)
                needed = predecessor(*operation)
                if needed is not None and needed not in done:
                    break
                order.append((operation, needed, position[stage] > 0))
                done.add(operation)
                position[stage] += 1
                remaining -= 1
        if remaining == before:
            raise RuntimeError("the pipeline schedule cannot run: its operations wait in a cycle")
    return tuple(order)


def _positive(**counts) -> None:
    for name, value in counts.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _plan(microbatches, stages, virtual_stages, warm_up) -> tuple:
    """The one-forward-one-backward order of every stage, merged by ``_execution_order``.

    Stage ``r`` first runs ``warm_up(r)`` forward chunks; then it alternates one forward and
    one backward chunk, then runs the backward chunks that are left. Forward chunks come in
    blocks of ``stages`` microbatches: chunk 0 of each of them, then chunk 1, and so on;
    backward chunks walk the chunks of a block in reverse. A forward chunk needs the same
    chunk on the previous stage, or the previous chunk on the last stage; a backward chunk
    needs the same chunk on the next stage, or the next chunk on the first stage, or, for
    the last chunk on the last stage, its own forward chunk.
    """
    total, block = microbatches * virtual_stages, stages * virtual_stages

    def chunk_of(index, direction):
        """Microbatch and chunk of the ``index``-th forward or backward chunk of a stage."""
        first, offset = divmod(index, block)
        chunk = offset // stages
        if direction == "B":
            chunk = virtual_stages - 1 - chunk
        return direction, first * stages + offset % stages, chunk

    sequences = []
    for stage in range(stages):
        first = warm_up(stage)
        sequence = [chunk_of(index, "F") for index in range(first)]
        for index in range(total - first):
            sequence.append(chunk_of(first + index, "F"))
            sequence.append(chunk_of(index, "B"))
        sequence.extend(chunk_of(index, "B") for index in range(total - first, total))
        sequences.append(sequence)

    def predecessor(direction, microbatch, chunk, stage):
        if direction == "F":
            if stage > 0:
                return ("F", microbatch, chunk, stage - 1)
            return ("F", microbatch, chunk - 1, stages - 1) if chunk > 0 else None
        if stage < stages - 1:
            return ("B", microbatch, chunk, stage + 1)
        if chunk < virtual_stages - 1:
            return ("B", microbatch, chunk + 1, 0)
        return ("F", microbatch, chunk, stage)

    return _execution_order(sequences, predecessor)


# At most 32 plans are kept: a plan has 2 * microbatches * stages * virtual_stages entries.
# Typed, so that a kept plan never answers for arguments that only compare equal (True, 2.0).
@lru_cache(maxsize=32, typed=True)
def interleaved_plan(microbatches: int, stages: int, virtual_stages: int) -> tuple:
    """Operations of the interleaved one-forward-one-backward schedule, in a runnable order.

    The order is that of ``_plan``, where stage ``r`` first runs ``2 * (stages - r - 1) +
    (virtual_stages - 1) * stages`` forward chunks, or all of them when ``microbatches ==
    stages``. The entries are those of ``_execution_order``. The count of microbatches must
    be a multiple of ``stages``.
    """
    _positive(microbatches=microbatches, stages=stages, virtual_stages=virtual_stages)
    if microbatches % stages:
        raise ValueError(
            "the interleaved schedule needs a microbatch count that is a multiple of the "
            f"pipeline-parallel size ({microbatches} vs {stages})"
        )

    def warm_up(stage):
        if microbatches == stages:
            return microbatches * virtual_stages
        return 2 * (stages - stage - 1) + (virtual_stages - 1) * stages

    return _plan(microbatches, stages, virtual_stages, warm_up)


# A cache of its own, with the parameters of the one above.
@lru_cache(maxsize=32, typed=True)
def non_interleaved_plan(microbatches: int, stages: int) -> tuple:
    """Operations of Megatron's non-interleaved schedule, in a runnable order.

    The order of a Megatron training step without virtual pipeline stages, which is not
    part of DynamicUSP (see the module docstring): that of ``_plan`` with one model chunk
    per stage, where stage ``r`` first runs ``min(stages - r - 1, microbatches)`` forward
    microbatches. The entries are those of ``_execution_order``, with chunk 0. Any count
    of microbatches runs.
    """
    _positive(microbatches=microbatches, stages=stages)
    return _plan(microbatches, stages, 1, lambda stage: min(stages - stage - 1, microbatches))


def _tiles(groups, world_size) -> bool:
    """True when ``(first rank, end rank, ...)`` groups hold each of the ranks exactly once."""
    position = 0
    for first, end, *_ in sorted(groups):
        if first != position or end <= first:
            return False
        position = end
    return position == world_size


def step_time(
    rows: Sequence[Sequence[tuple]],
    world_size: int,
    stages: int,
    virtual_stages: int,
    per_mb_overhead_ms: float = 0.0,
    *,
    schedule: str = "interleaved",
) -> float:
    """Milliseconds until the last DPxCP rank has finished the step.

    ``rows[j]`` lists the CP groups of microbatch ``j`` as ``(start, size, forward_ms,
    backward_ms)``: the first rank, the number of ranks and the forward and backward time
    of the whole model for that group. The groups of a microbatch must hold each of the
    ``world_size`` ranks exactly once, in any order: as upstream, a microbatch whose
    groups overlap or leave a rank without a group raises ValueError, and so does a group
    that ends past the last rank. ``per_mb_overhead_ms`` is added once per microbatch.

    ``schedule`` names the plan builder whose order is replayed: ``"interleaved"``
    (``interleaved_plan``) or ``"non_interleaved"`` (``non_interleaved_plan``, which has one
    chunk per stage: ``virtual_stages`` must be 1).
    """
    if schedule == "interleaved":
        plan = interleaved_plan(len(rows), stages, virtual_stages)
    elif schedule == "non_interleaved":
        if type(virtual_stages) is not int or virtual_stages != 1:
            raise ValueError(
                "the non-interleaved schedule has one model chunk per pipeline stage; got "
                f"virtual_stages={virtual_stages!r}"
            )
        plan = non_interleaved_plan(len(rows), stages)
    else:
        raise ValueError(f"schedule must be 'interleaved' or 'non_interleaved'; got {schedule!r}")
    scale = 1.0 / (stages * virtual_stages)
    costs = [
        [
            (start, start + size, forward * scale, backward * scale)
            for start, size, forward, backward in row
        ]
        for row in rows
    ]
    for microbatch, row in enumerate(costs):
        if not _tiles(row, world_size):
            raise ValueError(
                f"the CP groups of microbatch {microbatch} do not hold each of the {world_size} "
                "DPxCP ranks exactly once; (first rank, number of ranks) of the groups: "
                f"{[(first, end - first) for first, end, *_ in row]}"
            )
    ended: dict = {}  # operation -> the time each rank ends it
    last: list = [None] * stages  # per stage: the end times of its latest operation
    for operation, needed, has_previous in plan:
        direction, microbatch, _, stage = operation
        ready = list(last[stage]) if has_previous else [0.0] * world_size
        if needed is not None:
            ready = [max(own, other) for own, other in zip(ready, ended[needed])]
        cost = 2 if direction == "F" else 3
        for group in costs[microbatch]:
            first, end = group[0], group[1]
            # A chunk of a CP group is one operation of all its ranks: they start together.
            finish = max(ready[first:end]) + group[cost]
            ready[first:end] = [finish] * (end - first)
        ended[operation] = last[stage] = ready
    return max(max(times) for times in last) + len(rows) * per_mb_overhead_ms

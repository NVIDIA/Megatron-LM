# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .planner import RuntimePlanGeometry


@dataclass(frozen=True)
class TimelineModel:
    """Simulate CP/EP synchronization dependencies across a complete training step."""

    world_size: int
    ep_size: int
    iteration_tail_ms: float = 0.0
    first_forward_dp_cp_sync: bool = True
    program: tuple = ()  # (forward_ops, backward_ops)

    def evaluate(self, rounds: Sequence[Sequence[tuple]]):
        """Step time of the microbatches, forward then backward each, with the iteration tail."""
        forward, backward = self.program
        state = [0.0] * self.world_size  # completion time of every rank
        for index, groups in enumerate(rounds):
            state = self._program_phase(state, groups, forward)
            if index == 0 and self.first_forward_dp_cp_sync:
                state = [max(state)] * len(state)
            state = self._program_phase(state, groups, backward)
        return max(state) + self.iteration_tail_ms

    def _program_phase(self, state, groups, ops):
        w, e = self.world_size, self.ep_size
        t = list(state)
        for kind, payload in ops:
            if kind == 'local':
                t = [v + payload for v in t]
            elif kind == 'ep':
                for s in range(0, w, e):
                    m = max(t[s : s + e]) + payload
                    t[s : s + e] = [m] * e
            elif kind == 'ep_mean':
                # One per-rank contribution comes from its current CP group's pack.
                # Rank weighting preserves the EP block's valid-token population
                # when that block contains different CP widths and pack lengths.
                contributions = [None] * w
                for g in groups:
                    start, cp = g[0], g[1]
                    contributions[start : start + cp] = [g[4][payload]] * cp
                for s in range(0, w, e):
                    duration = sum(contributions[s : s + e]) / e
                    m = max(t[s : s + e]) + duration
                    t[s : s + e] = [m] * e
            else:  # 'cp' with named cost
                for g in groups:
                    start, cp = g[0], g[1]
                    m = max(t[start : start + cp]) + g[4][payload]
                    t[start : start + cp] = [m] * cp
        return t


def _freeze_op(op):
    if "cost" in op:
        if op.get("sync") == "ep":
            if op.get("aggregation") != "mean":
                raise ValueError("Named EP costs require an explicit mean aggregation")
            return ("ep_mean", op["cost"])
        return ("cp", op["cost"])
    return (op["sync"], float(op["duration_ms"]))


def layer_program(layers, *, recompute_forward: bool = False) -> dict:
    """Execution program of a stack of layers on one pipeline stage.

    ``layers`` lists, in model order, ``(forward_cost, forward_ms, backward_cost,
    backward_ms)``: the names of a layer's CP-group attention costs and its fixed
    non-attention times. A forward layer runs its attention and then the non-attention work
    that ends at the EP rendezvous; the backward pass takes the layers in reverse order and
    each layer in reverse. With ``recompute_forward`` every backward layer first replays
    its forward.
    """
    forward, backward = [], []
    for forward_cost, forward_ms, _, _ in layers:
        forward += [dict(sync="cp", cost=forward_cost), dict(sync="ep", duration_ms=forward_ms)]
    for forward_cost, forward_ms, backward_cost, backward_ms in reversed(layers):
        if recompute_forward:
            backward += [
                dict(sync="cp", cost=forward_cost),
                dict(sync="ep", duration_ms=forward_ms),
            ]
        backward += [dict(sync="ep", duration_ms=backward_ms), dict(sync="cp", cost=backward_cost)]
    return dict(forward=forward, backward=backward)


def build_timeline(model, geometry: "RuntimePlanGeometry") -> TimelineModel:
    """Timeline of the execution program that ``model.timing()`` gives.

    A model may declare ``first_forward_dp_cp_sync`` in ``timing()``: False when only the
    ranks of one CP group wait for each other. Without it all DPxCP ranks are synchronised
    after the forward pass of the first microbatch.
    """
    timing = model.timing()
    (program,) = timing["execution_program"]  # one program: one virtual stage
    ops = tuple(tuple(_freeze_op(op) for op in program[phase]) for phase in ("forward", "backward"))
    sync = timing.get("first_forward_dp_cp_sync", True)
    if type(sync) is not bool:
        raise ValueError("timing()['first_forward_dp_cp_sync'] must be a boolean")
    return TimelineModel(
        geometry.world_size,
        geometry.ep_size,
        iteration_tail_ms=float(timing.get("iteration_tail_ms", 0.0)),
        first_forward_dp_cp_sync=sync,
        program=ops,
    )

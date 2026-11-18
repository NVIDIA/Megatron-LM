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
    layer_count: int
    forward_non_attention_ms: float = 0.0
    backward_non_attention_ms: float = 0.0
    iteration_tail_ms: float = 0.0
    first_forward_dp_cp_sync: bool = True
    program: tuple | None = None  # (forward_ops, backward_ops) for program mode

    def initial_state(self):
        """Completion times of every EP block (scalar costs) or of every rank (program)."""
        blocks = self.world_size // self.ep_size
        return [0.0] * (blocks if self.program is None else self.world_size)

    def makespan(self, state):
        """Return step completion time including the configured iteration tail."""
        return max(state) + self.iteration_tail_ms

    def advance(self, state, groups: Sequence[tuple], first: bool):
        """Apply one microbatch (forward then backward) to a state; returns the new state."""
        run = self._scalar_phase if self.program is None else self._program_phase
        forward, backward = (0, 1) if self.program is None else self.program
        state = run(state, groups, forward)
        if first and self.first_forward_dp_cp_sync:
            state = [max(state)] * len(state)
        return run(state, groups, backward)

    def evaluate(self, rounds: Sequence[Sequence[tuple]]):
        """Score all microbatches with the shared forward/backward timeline."""
        state = self.initial_state()
        for index, groups in enumerate(rounds):
            state = self.advance(state, groups, index == 0)
        return self.makespan(state)

    def _scalar_phase(self, state, groups, phase):
        e = self.ep_size
        na = self.forward_non_attention_ms if phase == 0 else self.backward_non_attention_ms
        spans = []
        for g in groups:
            start, cp = g[0], g[1]
            b0, b1 = start // e, (start + cp - 1) // e
            spans.append((b0, b1 + 1, g[2 + phase] + na))
        t = list(state)
        nb = len(t)
        layers = self.layer_count
        done = 0
        while done < layers:
            new = [float('-inf')] * nb
            for b0, b1, cost in spans:
                s = max(t[b0:b1]) + cost
                for b in range(b0, b1):
                    if s > new[b]:
                        new[b] = s
            done += 1
            delta = [a - b for a, b in zip(new, t)]
            t = new
            if done < layers and all(d == delta[0] for d in delta):
                # Every block advanced by the same amount, so the relative offsets are
                # unchanged and the max-plus recurrence repeats this step exactly.
                t = [v + (layers - done) * delta[0] for v in t]
                break
        return t

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
    return dict(layer_count=len(layers), forward=forward, backward=backward)


def build_timeline(model, geometry: "RuntimePlanGeometry") -> TimelineModel:
    """Program-mode timeline from ``model.timing()`` or a scalar timeline when it is None.

    A program-mode model may declare ``first_forward_dp_cp_sync`` in ``timing()``: False when
    only the ranks of one CP group wait for each other. Without it all DPxCP ranks are
    synchronised after the forward pass of the first microbatch.
    """
    timing = model.timing()
    if timing is None:
        return TimelineModel(
            geometry.world_size,
            geometry.ep_size,
            int(getattr(model, "layer_count", 1)),
            float(getattr(model, "forward_non_attention_ms", 0.0)),
            float(getattr(model, "backward_non_attention_ms", 0.0)),
        )
    (program,) = timing["execution_program"]  # one program: one virtual stage
    ops = tuple(tuple(_freeze_op(op) for op in program[phase]) for phase in ("forward", "backward"))
    sync = timing.get("first_forward_dp_cp_sync", True)
    if type(sync) is not bool:
        raise ValueError("timing()['first_forward_dp_cp_sync'] must be a boolean")
    return TimelineModel(
        geometry.world_size,
        geometry.ep_size,
        int(program["layer_count"]),
        iteration_tail_ms=float(timing.get("iteration_tail_ms", 0.0)),
        first_forward_dp_cp_sync=sync,
        program=ops,
    )

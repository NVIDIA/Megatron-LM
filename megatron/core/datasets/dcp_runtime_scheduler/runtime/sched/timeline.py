# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TimelineModel:
    world_size: int
    ep_size: int
    layer_count: int
    forward_non_attention_ms: float = 0.0
    backward_non_attention_ms: float = 0.0
    iteration_tail_ms: float = 0.0
    first_forward_dp_cp_sync: bool = True
    program: tuple | None = None  # (forward_ops, backward_ops) for program mode

    @property
    def blocks(self):
        return self.world_size // self.ep_size

    @property
    def scalar(self):
        return self.program is None

    @classmethod
    def from_evaluator(cls, evaluator):
        """Build from a dcp_scheduler TrainingEvaluator (PP1/VPP1, no token models)."""
        o = evaluator.options
        if o.get('pp_size', 1) != 1 or o.get('vpp_size', 1) != 1:
            raise ValueError('fast timeline supports PP1/VPP1 only')
        for key in (
            'forward_local_token_models',
            'backward_local_token_models',
            'forward_ep_group_token_models',
            'backward_ep_group_token_models',
        ):
            if o.get(key):
                raise ValueError('token-dependent non-attention models are not supported')
        program = o.get('execution_program')
        if program is not None:
            if len(program) != 1:
                raise ValueError('program mode supports one virtual stage')
            ops = (
                tuple(_freeze_op(op) for op in program[0]['forward']),
                tuple(_freeze_op(op) for op in program[0]['backward']),
            )
            return cls(
                o['world_size'],
                o['ep_size'],
                o['layer_count'],
                0.0,
                0.0,
                float(o.get('iteration_tail_ms', 0.0)),
                bool(o.get('first_forward_dp_cp_sync', True)),
                ops,
            )
        fwd = o.get('forward_non_attention_ms_per_layer', 0.0)
        bwd = o.get('backward_non_attention_ms_per_layer', 0.0)
        if isinstance(fwd, (list, tuple)):
            fwd = fwd[0]
        if isinstance(bwd, (list, tuple)):
            bwd = bwd[0]
        return cls(
            o['world_size'],
            o['ep_size'],
            o['layer_count'],
            float(fwd),
            float(bwd),
            float(o.get('iteration_tail_ms', 0.0)),
            bool(o.get('first_forward_dp_cp_sync', True)),
            None,
        )

    # ------------------------------------------------------------------ state
    def initial_state(self):
        n = self.blocks if self.scalar else self.world_size
        return [0.0] * n

    def makespan(self, state):
        return max(state) + self.iteration_tail_ms

    def advance(self, state, groups: Sequence[tuple], first: bool):
        """Apply one microbatch (forward then backward) to a state; returns the new state."""
        if self.scalar:
            state = self._scalar_phase(state, groups, 0)
            if first and self.first_forward_dp_cp_sync:
                m = max(state)
                state = [m] * len(state)
            return self._scalar_phase(state, groups, 1)
        state = self._program_phase(state, groups, self.program[0])
        if first and self.first_forward_dp_cp_sync:
            m = max(state)
            state = [m] * len(state)
        return self._program_phase(state, groups, self.program[1])

    def evaluate(self, rounds: Sequence[Sequence[tuple]]):
        state = self.initial_state()
        for index, groups in enumerate(rounds):
            state = self.advance(state, groups, index == 0)
        return self.makespan(state)

    def microbatch_time(self, groups, state=None):
        """Time added to the slowest block by one microbatch from a given (or idle) state."""
        base = state if state is not None else self.initial_state()
        new = self.advance(base, groups, state is None)
        return max(new) - max(base)

    # ---------------------------------------------------------------- scalar
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

    # --------------------------------------------------------------- program
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
            else:  # 'cp' with named cost
                for g in groups:
                    start, cp = g[0], g[1]
                    m = max(t[start : start + cp]) + g[4][payload]
                    t[start : start + cp] = [m] * cp
        return t


def _freeze_op(op):
    if 'cost' in op:
        return ('cp', op['cost'])
    return (op['sync'], float(op['duration_ms']))


def costed_groups(rounds):
    """Convert rebuilt dcp_scheduler rounds (with cost fields) into the tuple form."""
    result = []
    for row in rounds:
        groups, start = [], 0
        for g in row:
            program = g.get('program_costs_ms_by_stage')
            groups.append(
                (
                    start,
                    int(g['cp']),
                    float(g['forward_cost_ms']),
                    float(g.get('backward_cost_ms', 0.0)),
                    dict(program[0]) if program else None,
                )
            )
            start += int(g['cp'])
        result.append(groups)
    return result

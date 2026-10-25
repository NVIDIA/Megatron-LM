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

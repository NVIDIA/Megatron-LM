# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Two-stream forward timeline used by TE P2P context parallel attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class PipelineRound:
    step: int
    prepare_ms: float
    attention_ms: float
    lse_or_init_ms: float
    outgoing_communication_ms: float

    def validate(self, cp_size: int) -> None:
        if not 0 <= self.step < cp_size:
            raise ValueError(f"invalid step {self.step} for CP={cp_size}")
        for name in (
            "prepare_ms",
            "attention_ms",
            "lse_or_init_ms",
            "outgoing_communication_ms",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class RoundTimeline:
    step: int
    stream: int
    start_ms: float
    communication_end_ms: float
    prepare_end_ms: float
    attention_end_ms: float
    correction_end_ms: float


@dataclass(frozen=True)
class PipelineTimeline:
    rounds: tuple[RoundTimeline, ...]
    output_correction_start_ms: float
    output_correction_end_ms: float
    total_ms: float

    def to_dict(self) -> dict:
        return {
            "rounds": [asdict(item) for item in self.rounds],
            "output_correction_start_ms": self.output_correction_start_ms,
            "output_correction_end_ms": self.output_correction_end_ms,
            "total_ms": self.total_ms,
        }


def simulate_two_stream_pipeline(
    rounds: Sequence[PipelineRound],
    output_correction_ms: Sequence[float],
    *,
    fixed_overhead_ms: float = 0.0,
) -> PipelineTimeline:
    """Simulate the device dependency graph in TE's P2P forward loop.

    Communication for step ``i + 1`` starts immediately before step ``i``'s
    prepare/attention work. Step ``i + 1`` waits for that receive. Attention
    and its LSE correction share alternating CUDA streams, while LSE
    corrections are also globally ordered by TE's correction event. Output
    corrections run serially after both streams drain.

    Durations passed here must already include any measured kernel slowdown
    caused by concurrent communication or peer-stream kernels.
    """
    cp_size = len(rounds)
    if cp_size <= 1:
        raise ValueError("the CP pipeline requires at least two rounds")
    if len(output_correction_ms) != cp_size:
        raise ValueError(
            f"expected {cp_size} output-correction durations, "
            f"got {len(output_correction_ms)}"
        )
    if fixed_overhead_ms < 0 or any(value < 0 for value in output_correction_ms):
        raise ValueError("pipeline durations must be non-negative")

    ordered = sorted(rounds, key=lambda item: item.step)
    if [item.step for item in ordered] != list(range(cp_size)):
        raise ValueError("round steps must be exactly 0..CP-1")
    for item in ordered:
        item.validate(cp_size)
    if ordered[-1].outgoing_communication_ms != 0:
        raise ValueError("the final CP round must not start another KV exchange")

    stream_ready = [0.0, 0.0]
    prior_communication_end = 0.0
    prior_correction_end = 0.0
    timeline = []
    for item in ordered:
        stream = item.step % 2
        start = max(stream_ready[stream], prior_communication_end)
        communication_end = start + item.outgoing_communication_ms
        prepare_end = start + item.prepare_ms
        attention_end = prepare_end + item.attention_ms
        correction_end = (
            max(attention_end, prior_correction_end) + item.lse_or_init_ms
        )
        stream_ready[stream] = correction_end
        prior_correction_end = correction_end
        prior_communication_end = communication_end
        timeline.append(
            RoundTimeline(
                step=item.step,
                stream=stream,
                start_ms=start,
                communication_end_ms=communication_end,
                prepare_end_ms=prepare_end,
                attention_end_ms=attention_end,
                correction_end_ms=correction_end,
            )
        )

    output_start = max(stream_ready)
    output_end = output_start + sum(float(value) for value in output_correction_ms)
    total = output_end + fixed_overhead_ms
    return PipelineTimeline(tuple(timeline), output_start, output_end, total)

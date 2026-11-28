# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parse step times, per-rank timers and peak memory from a Megatron training log.

Expected lines (``--log-interval 1``):

- ``iteration N/M | ... | elapsed time per iteration (ms): X | ...``
- with ``--timing-log-level 2 --timing-log-option all``, after each iteration line::

      times across ranks (ms):
        forward-compute:
           rank  0: 1234.56

  Values are milliseconds per iteration, summed over micro-batches.
- ``[Rank N] (after K iterations) memory (MB) | ... | max allocated: X | ...``
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

_STEP_RE = re.compile(r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)")
_TIMER_HEAD_RE = re.compile(r"^\s{2}([a-z][a-z0-9-]*):\s*$")
_TIMER_RANK_RE = re.compile(r"^\s+rank\s+(\d+):\s*([\d.]+)\s*$")
_MEMORY_RE = re.compile(r"\[Rank (\d+)\].*?memory \(MB\).*?max allocated:\s*([\d.]+)")


@dataclass
class TrainingLog:
    """Measurements parsed from one training log."""

    step_ms: Dict[int, float] = field(default_factory=dict)
    """Elapsed time per iteration, by iteration number."""

    timers: Dict[int, Dict[str, Dict[int, float]]] = field(default_factory=dict)
    """Per-rank timer values in ms, by iteration and timer name."""

    max_allocated_mb: Dict[int, float] = field(default_factory=dict)
    """Largest "max allocated" reported by each rank, in MB."""


def parse_log(lines: Iterable[str]) -> TrainingLog:
    """Parse the lines of a Megatron training log.

    Args:
        lines: Log lines, for example an open file.

    Returns:
        The parsed measurements.
    """
    log = TrainingLog()
    iteration: Optional[int] = None
    timer: Optional[str] = None
    in_timer_block = False
    for line in lines:
        step = _STEP_RE.search(line)
        if step:
            iteration = int(step.group(1))
            log.step_ms[iteration] = float(step.group(2))
            in_timer_block = False
            continue
        if "times across ranks (ms):" in line:
            in_timer_block, timer = True, None
            continue
        memory = _MEMORY_RE.search(line)
        if memory:
            rank, value = int(memory.group(1)), float(memory.group(2))
            log.max_allocated_mb[rank] = max(log.max_allocated_mb.get(rank, 0.0), value)
            continue
        if not in_timer_block or iteration is None:
            continue
        head = _TIMER_HEAD_RE.match(line)
        if head:
            timer = head.group(1)
            continue
        rank_value = _TIMER_RANK_RE.match(line)
        if rank_value and timer:
            per_timer = log.timers.setdefault(iteration, {}).setdefault(timer, {})
            per_timer[int(rank_value.group(1))] = float(rank_value.group(2))
        elif line.strip():
            in_timer_block = False
    return log


@dataclass
class LogSummary:
    """Medians over the iterations after warmup."""

    num_iterations: int
    """Iterations included."""

    step_ms: Optional[float]
    """Median elapsed time per iteration."""

    timers: Dict[str, Dict[int, float]]
    """Median per-rank value of each timer."""

    max_allocated_mb: Dict[int, float]
    """Largest "max allocated" per rank."""


def summarize(log: TrainingLog, warmup: int = 5) -> LogSummary:
    """Summarize a parsed log, skipping the first ``warmup`` iterations.

    Args:
        log: Parsed log.
        warmup: Number of leading iterations to drop.

    Returns:
        Medians of step time and per-rank timers.
    """
    iterations = [it for it in sorted(log.step_ms) if it > warmup]
    per_timer: Dict[str, Dict[int, list]] = {}
    for it in iterations:
        for name, ranks in log.timers.get(it, {}).items():
            for rank, value in ranks.items():
                per_timer.setdefault(name, {}).setdefault(rank, []).append(value)
    return LogSummary(
        num_iterations=len(iterations),
        step_ms=statistics.median(log.step_ms[it] for it in iterations) if iterations else None,
        timers={
            name: {rank: statistics.median(values) for rank, values in sorted(ranks.items())}
            for name, ranks in per_timer.items()
        },
        max_allocated_mb=dict(log.max_allocated_mb),
    )

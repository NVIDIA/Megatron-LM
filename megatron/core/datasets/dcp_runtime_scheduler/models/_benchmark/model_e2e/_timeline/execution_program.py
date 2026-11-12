# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Serializable mixed-layer operations inside the common PP/VPP timeline.

Predictors own kernel costs; the program owns dependency order. Neither the
search nor this executor contains model names or benchmark coefficients.
"""

from __future__ import annotations

import math

from .rank_timeline import rank_intervals

def validate_program(program, layers_per_virtual_stage):
    if not isinstance(program, (list, tuple)) or len(program) != len(layers_per_virtual_stage):
        raise ValueError("execution_program needs one entry per chunk-major virtual stage")
    for entry, layers in zip(program, layers_per_virtual_stage):
        if set(entry) != {"layer_count", "forward", "backward"} or entry["layer_count"] != layers:
            raise ValueError("program layer_count disagrees with pipeline layer partition")
        for phase in ("forward", "backward"):
            if not isinstance(entry[phase], (list, tuple)):
                raise ValueError("program phases must contain operation lists")
            for op in entry[phase]:
                if op.get("sync") not in ("cp", "ep", "local"):
                    raise ValueError("program sync must be cp, ep, or local")
                if set(op) not in ({"sync", "cost"}, {"sync", "duration_ms"}):
                    raise ValueError("program operation needs exactly one cost or duration_ms")
                if "cost" in op:
                    if op["sync"] != "cp" or not isinstance(op["cost"], str) or not op["cost"]:
                        raise ValueError("named group costs require CP synchronization")
                elif not math.isfinite(float(op["duration_ms"])) or float(op["duration_ms"]) < 0:
                    raise ValueError("program durations must be finite and nonnegative")

def execute_program(input_times, groups, *, world_size, ep_size, stage, operations):
    """Execute a stage phase without a cross-CP microbatch barrier."""
    intervals = rank_intervals(groups, world_size)
    times = list(input_times)
    cp_wait = ep_wait = work = 0.0
    for op in operations:
        sync = op["sync"]
        if sync == "local":
            duration = float(op["duration_ms"])
            times = [value + duration for value in times]
            work += world_size * duration
            continue
        domains = intervals if sync == "cp" else (
            range(start, start + ep_size) for start in range(0, world_size, ep_size))
        for index, ranks in enumerate(domains):
            if "cost" in op:
                try:
                    duration = float(groups[index]["program_costs_ms_by_stage"][stage][op["cost"]])
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError("missing per-stage program cost: " + op["cost"]) from exc
            else:
                duration = float(op["duration_ms"])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid program duration")
            start = max(times[r] for r in ranks)
            wait = sum(start - times[r] for r in ranks)
            if sync == "cp":
                cp_wait += wait
            else:
                ep_wait += wait
            work += len(ranks) * duration
            for rank in ranks:
                times[rank] = start + duration
    return times, cp_wait, ep_wait, work, work

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Source-positive normal/checked loop prices; an unapproved CP1 hypothesis.

This preserves the existing source jobs and dispatch hypothesis, but removes
the assumption that the checked path costs at least the normal path. Loop
prices fitted to whole calls are not measurements of isolated instructions.
"""

from functools import lru_cache

import math

from .prospective_backward_worker_cp1 import source_stages

from .prospective_forward_worker_cp1 import source_jobs

@lru_cache(maxsize=512)
def path_stages(lengths, geometry, hardware, phase):
    lengths, geometry = tuple(lengths), tuple(geometry)
    if phase == 'forward':
        jobs, workers = source_jobs(lengths, geometry, hardware)
        return ((tuple((c - m, m, 1) for c, m in jobs), workers, 'fifo'),)
    if phase != 'backward' or hardware not in ('h100', 'gb200'):
        raise ValueError('only independently sourced backward CP1 paths; no CP extrapolation')
    if hardware == 'h100':
        from .prospective_h100_backward_mask_work import split
        paths = split(lengths, geometry)
        original, = source_stages(lengths, geometry, hardware)
        if tuple(n + m for n, m, _ in paths) != original[0]:
            raise ValueError('H100 disjoint paths must preserve all original source dispatch units')
        return ((paths, original[1], original[2]),)
    from .backward_phase_work import phase_work
    result = []
    for stage, (jobs, workers, dispatch) in enumerate(source_stages(lengths, geometry, hardware)):
        weight = 10 if geometry[2] == 128 else (6, 4, 6)[stage]
        work = phase_work(lengths, geometry, 'backward_core_family_stage_%d' % stage)
        if any(c % weight for c in jobs):
            raise ValueError('actual backward stage matrix weight must divide every original source job')
        separated = tuple((c - weight * min(c // weight, work['mask_limit']),
                           weight * min(c // weight, work['mask_limit']), 1) for c in jobs)
        if (sum(n + m for n, m, _ in separated) != sum(jobs)
                or sum(m for _, m, _ in separated) != weight * work['masked_steps']):
            raise ValueError('disjoint paths must conserve all independently authenticated loop work')
        result.append((separated, workers, dispatch))
    return tuple(result)

def reduced_coordinates(hardware, phase, geometry):
    return phase == 'forward' and hardware == 'h100' or phase == 'backward' and hardware == 'gb200' and geometry[2] == 128

def path_prices(coordinates, hardware, phase, geometry):
    if (len(coordinates) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates)
            or reduced_coordinates(hardware, phase, geometry) and coordinates[1] != 1):
        raise ValueError('two nonnegative source-price simplex coordinates; redundant setup fixed to zero')
    a, b = coordinates
    return a, (1 - a) * b, (1 - a) * (1 - b)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Unapproved per-job matrix/scalar resource envelope; not whole attention.

Source loop counts, kernel stages and dispatch are unchanged. Asynchronous
matrix instructions motivate testing overlap; their presence does NOT prove
this envelope is an exact latency law or recover actual worker ownership.
"""

import math

from .backward_worker_work import fifo_span

from .prospective_disjoint_path_cp1 import path_prices, path_stages

def resource_prices(coordinates, hardware, phase, geometry):
    if (len(coordinates) != 3 or type(coordinates[2]) not in (int, float)
            or not math.isfinite(coordinates[2]) or not 0 <= coordinates[2] <= 1):
        raise ValueError('original two source coordinates and one nonnegative matrix-resource fraction required')
    original = path_prices(tuple(coordinates[:2]), hardware, phase, tuple(geometry))
    matrix = coordinates[2]
    return tuple((1 - matrix) * p for p in original) + (matrix,)

def job_service(normal, checked, prices):
    if (type(normal) is not int or type(checked) is not int or normal < 0 or checked < 0
            or normal + checked < 1 or len(prices) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in prices)):
        raise ValueError('positive original loop work and four nonnegative resource prices required')
    scalar = prices[0] * normal + prices[1] * checked + prices[2]
    matrix = prices[3] * (normal + checked)
    return max(scalar, matrix)

def resource_stage_span(jobs, workers, dispatch, prices):
    if (not jobs or type(workers) is not int or workers < 1 or dispatch not in ('fifo', 'static_modulo')
            or any(len(job) != 3 or type(job[2]) is not int or job[2] != 1 for job in jobs)):
        raise ValueError('the same actual native jobs and explicit existing dispatch hypothesis required')
    weights = [job_service(n, m, prices) for n, m, _ in jobs]
    if dispatch == 'fifo':
        return fifo_span(weights, workers)
    ready = [0.] * workers
    for index, value in enumerate(weights):
        ready[index % workers] += value
    return max(ready)

def resource_service_span(lengths, geometry, hardware, phase, coordinates):
    prices = resource_prices(coordinates, hardware, phase, geometry)
    return math.fsum(resource_stage_span(j, w, d, prices) for j, w, d in path_stages(
        tuple(lengths), tuple(geometry), hardware, phase))

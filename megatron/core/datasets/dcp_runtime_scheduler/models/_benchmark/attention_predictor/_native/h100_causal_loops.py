# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Captured H100 causal consumer-loop bounds, not issued MMA or latency."""

from .geometry import GEOMETRIES, positive_int

from .h100_backward_dispatch import job_counts

def consumer_loop_ranges(length, phase):
    positive_int(length, "logical length")
    if phase == "forward":
        return tuple((0, (min(length, origin + 128) + 127) // 128) for origin in range(0, length, 128))
    if phase == "backward":
        return tuple((origin // 64, (length + 63) // 64) for origin in range(0, length, 128))
    raise ValueError("captured forward or backward consumer required")

def work(lengths, geometry, phase):
    lengths = tuple(lengths)
    if tuple(geometry) not in GEOMETRIES or not lengths:
        raise ValueError("captured geometry and nonempty real sequences required")
    for length in lengths:
        positive_int(length, "logical length")
    hq, _, dim = geometry
    tiles = [(length + 127) // 128 for length in lengths]
    if phase == "forward":
        jobs = hq * sum(tiles)
        loops = hq * sum(n * (n + 1) // 2 for n in tiles)
    elif phase == "backward":
        jobs = sum(job_counts(lengths, hq, dim))
        loops = hq * (dim // 128) * sum(n * ((length + 63) // 64) - n * (n - 1)
            for length, n in zip(lengths, tiles))
    else:
        raise ValueError("captured forward or backward consumer required")
    return dict(consumer_loop_iterations=loops, source_dispatch_units=jobs,
        logical_job_extent=128, inner_loop_extent=128 if phase == "forward" else 64,
        loop_counts_are_not_issued_mma_or_useful_flops=True)

def consumer_jobs(lengths, geometry, phase):
    """Explicit input-table/head-major hypothesis; never select order by error."""
    lengths = tuple(lengths)
    expected = work(lengths, geometry, phase)
    hq, _, dim = geometry
    partitions = dim // 128 if phase == "backward" else 1
    jobs = tuple(end - begin for length in lengths for _ in range(hq)
        for begin, end in consumer_loop_ranges(length, phase) for _ in range(partitions))
    if (len(jobs) != expected["source_dispatch_units"]
            or sum(jobs) != expected["consumer_loop_iterations"] or min(jobs) < 1):
        raise ValueError("explicit jobs must conserve every captured real work unit and loop")
    return jobs

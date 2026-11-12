# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Static H100 backward mask-path bounds; not counters or latency accuracy."""

from .export_cp1_program_bodies import require

from .h100_causal_loops import consumer_jobs

def split(lengths, geometry):
    jobs = consumer_jobs(tuple(lengths), tuple(geometry), 'backward')
    # Source interior fast path is strictly begin+2 < k < end-1.
    # Its complement contains up to three prefix panels plus one tail panel.
    paths = tuple((max(c - 4, 0), min(c, 4), 1) for c in jobs)
    require(all(n + m == c for (n, m, _), c in zip(paths, jobs, strict=True)), 'all original backward loop work conserved')
    return paths

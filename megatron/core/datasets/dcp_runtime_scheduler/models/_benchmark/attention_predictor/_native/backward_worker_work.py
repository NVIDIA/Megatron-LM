# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Backward critical-work hypotheses from captured GB200 jobs; no timing data.

D128 completion-FIFO and simultaneous availability of all eligible clusters
are modeling assumptions, not recovered physical GPU scheduling.
"""

import heapq

import math

def fifo_span(weights, workers):
    if type(workers) is not int or workers < 1:
        raise ValueError('positive source-derived cluster count required')
    ready = [(0., i) for i in range(workers)]
    for value in weights:
        if not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative work required')
        stamp, worker = ready[0]
        heapq.heapreplace(ready, (stamp+value, worker))
    return max(stamp for stamp, _ in ready)

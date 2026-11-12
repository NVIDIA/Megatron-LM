# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Finite-worker forward service hypothesis, not an exact CUDA scheduler.

Work sizes and eligible workers are source-backed. Completion-order FIFO,
sequence-table order and omission of producer prefetch remain explicit
modeling assumptions, to be tested rather than selected by timing error.
"""

from functools import lru_cache

from .geometry import GEOMETRIES

from .native_sdpa_cta_work import consumer_jobs

from .prospective_forward_mask_work import panel_split

@lru_cache(maxsize=256)
def source_jobs(lengths,geometry,hardware):
    lengths=tuple(lengths);geometry=tuple(geometry)
    if (geometry not in GEOMETRIES or any(type(v) is not int for v in geometry)
            or hardware not in ('h100','gb200') or not 1<=len(lengths)<=64
            or any(type(n) is not int or n<1 for n in lengths) or sum(lengths)>16384):
        raise ValueError('one captured hardware/geometry and positive CP1 logical pack up to16K required')
    panels,workers,_=consumer_jobs(lengths,lengths,geometry,'padding_causal',hardware)
    limit=1 if hardware=='h100' else 2
    jobs=tuple((n,min(n,limit)) for n in panels)
    expected=panel_split(lengths,geometry,hardware)
    if (len(jobs)!=expected['jobs'] or sum(c for c,_ in jobs)!=expected['panels']
            or sum(m for _,m in jobs)!=expected['checked_panels']):
        raise ValueError('job descriptors must conserve every authenticated source work count')
    return jobs,workers

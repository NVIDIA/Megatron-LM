# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Source-sized forward consumer work for an explicit provisional CTA hypothesis.

H100128x128 consumer bounds are from h100_causal_loops.py;GB200256x128
effective work and76/152 worker counts are from the closed source/resource
audit. Rectangular extension,completion-order FIFO and whole-call
service normalization are hypotheses,not a validated primitive predictor.
No storage padding is turned into logical attention work.
"""

from functools import lru_cache

from .geometry import GEOMETRIES

@lru_cache(maxsize=8192)
def consumer_jobs(q,k,geometry,mask,hardware):
    if tuple(geometry) not in GEOMETRIES or hardware not in ("h100","gb200") or len(q)!=len(k):
        raise ValueError("one of six captured geometries and matching logical fragment lists required")
    if any(type(value) is not int or value<0 for value in q+k) or not q:
        raise ValueError("nonnegative real logical fragment sizes required")
    if mask not in ("padding","padding_causal") or mask=="padding_causal" and q!=k:
        raise ValueError("actual native causal diagonal or rectangular padding mask required")
    hq,_,dim = geometry
    query_extent = 128 if hardware=="h100" else 256
    width = 1 if hardware=="h100" else dim//128
    workers = (132 if hardware=="h100" else 152)//width
    fragments = list(zip(q,k))
    if hardware=="gb200":
        # Descending Q is observed for CP1;its rectangular extension is provisional.
        fragments.sort(key=lambda pair:pair[0],reverse=True)
    jobs = []
    for qlen,klen in fragments:
        for _ in range(hq):
            for origin in range(0,qlen,query_extent):
                loops = ((min(klen,origin+query_extent)+127)//128 if mask=="padding_causal"
                    else (klen+127)//128)
                jobs.append(loops)
    return tuple(jobs),workers,width

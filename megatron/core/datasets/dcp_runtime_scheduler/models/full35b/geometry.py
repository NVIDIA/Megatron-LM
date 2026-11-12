# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen full-model prediction source; preserve numerical operations.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Native ceil64 then2CP storage, extend-last padding, and fresh fit features."""
from ._checks import require
import math

def pack_geometry(lengths,cp):
    lengths=tuple(lengths)
    require(type(cp) is int and 1<=cp<=32 and 1<=len(lengths)<=64
        and all(type(n) is int and 0<n<=524288 for n in lengths),'unsupported logical pack/CP')
    stored=[((n+63)//64*64+2*cp-1)//(2*cp)*(2*cp) for n in lengths]
    require(sum(stored)<=cp*16384,'pack exceeds true16K/rank capacity')
    stored[-1]+=cp*16384-sum(stored)
    return dict(cp=cp,logical_lengths=list(lengths),storage_lengths=stored,
        density=sum(lengths)/(cp*16384),declared_max_seqlen=cp*16384,static_endpoints=65)

def fa_features(point):
    logical,stored,cp=point['logical_lengths'],point['storage_lengths'],point['cp']
    require(sum(stored)==cp*16384 and all(s%(2*cp)==0 and 0<n<=s for n,s in zip(logical,stored,strict=True)), 'invalid FA geometry')
    if cp==1:critical=round_critical=sum(n*n for n in logical)
    else:
        from .cp_round_geometry import build_round_geometries,critical_rank_work
        pieces=build_round_geometries(logical,stored,cp);work=[[0]*cp for _ in range(cp)]
        for item in pieces:work[item.step][item.rank]+=item.cp1_equivalent_sum_squared_tokens
        round_critical=sum(max(row) for row in work);critical=max(critical_rank_work(pieces).values())
    return dict(bias=1.,round_critical_work_m=round_critical/1e6,critical_rank_work_m=critical/1e6,
        valid_tokens_per_rank_k=sum(logical)/cp/1e3,sequence_count=float(len(logical)))

def dot(coefficients,features):
    require(set(coefficients)==set(features),'fresh feature/coefficient names differ')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in coefficients.values()),
        'fresh NNLS coefficients must be finite and nonnegative')
    value=sum(coefficients[name]*x for name,x in features.items())
    require(value>0,'fresh predictor produced nonpositive service')
    return value

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Pack layout and dot product of the full-model prediction source (numerical operations
# preserved); fa_features returns the features of the one form shared by every CP size.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Native ceil64 then2CP storage, extend-last padding, and the full-attention features."""
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
    """One feature vector for every CP size: a ring of cp ranks pays a set-up once
    (cp_enabled) and one more round per further rank (extra_rounds); the slowest rank of
    each round sets its duration (round_critical_work_m, 1e6 CP1-equivalent token pairs)."""
    logical,stored,cp=point['logical_lengths'],point['storage_lengths'],point['cp']
    require(sum(stored)==cp*16384 and all(s%(2*cp)==0 and 0<n<=s for n,s in zip(logical,stored,strict=True)), 'invalid FA geometry')
    if cp==1:round_critical=sum(n*n for n in logical)
    else:
        from .cp_round_geometry import build_round_geometries
        pieces=build_round_geometries(logical,stored,cp);work=[[0]*cp for _ in range(cp)]
        for item in pieces:work[item.step][item.rank]+=item.cp1_equivalent_sum_squared_tokens
        round_critical=sum(max(row) for row in work)
    return dict(bias=1.,cp_enabled=float(cp>1),extra_rounds=float(cp-1),
        round_critical_work_m=round_critical/1e6,valid_tokens_k=sum(logical)/1e3,
        sequences_x_cp=float(len(logical)*cp))

def dot(coefficients,features):
    require(set(coefficients)==set(features),'fresh feature/coefficient names differ')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in coefficients.values()),
        'fresh NNLS coefficients must be finite and nonnegative')
    value=sum(coefficients[name]*x for name,x in features.items())
    require(value>0,'fresh predictor produced nonpositive service')
    return value

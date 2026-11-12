# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen full-model prediction source; preserve numerical operations.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Prospective physical GDN feature definition, independently frozen before v2 GPU data."""
import math
FEATURE_SCHEMA='GDN_CP_BRANCH_ACTIVATION_V2'
BASE_FEATURES=('bias','cp_enabled','summary_hops','max_prefix_chain','boundary_summary_chunks_per_64','extra_local_sequences')
FEATURES=BASE_FEATURES+('active_cross_rank_kernel',)

def require(ok,message):
 if not ok:raise ValueError(message)

def features(point,phase):
 require(phase in ('forward','backward'),'GDN phase must be forward/backward')
 cp=point['cp'];stored=point['storage_lengths']
 require(type(cp) is int and 1<=cp<=32 and 1<=len(stored)<=64 and all(type(n) is int and n>0 and n%(2*cp)==0 for n in stored) and sum(stored)==cp*16384,'native full-pack16K/rank aligned physical geometry required')
 cu=[0]
 for length in stored:cu.append(cu[-1]+length)
 difference=[0]*(cp+1);chain=boundary=0
 for begin,end in zip(cu,cu[1:]):
  first,last=begin//16384,(end-1)//16384;difference[first]+=1;difference[last+1]-=1
  chain=max(chain,last-first)
  if last>first:
   chunks=((16384-begin%16384+63)//64 if phase=='forward' else ((end-1)%16384+64)//64)
   if last-first>1:chunks=16384//64
   boundary=max(boundary,chunks)
 active=maximum=0
 for delta in difference[:-1]:active+=delta;maximum=max(maximum,active)
 return dict(bias=1.,cp_enabled=float(cp>1),summary_hops=float(cp-1),max_prefix_chain=float(chain),boundary_summary_chunks_per_64=boundary/64,extra_local_sequences=float(maximum-1),active_cross_rank_kernel=float(chain>0))

def predict(coefficients,point,phase):
 f=features(point,phase);require(set(coefficients)==set(FEATURES),'v2 coefficients must contain exactly frozen7 features')
 require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in coefficients.values()),'finite nonnegative v2 coefficients required')
 value=sum(coefficients[k]*f[k] for k in FEATURES);require(value>0,'positive v2 prediction required');return value

#!/usr/bin/env bash
# One arm of the residual-add+RMSNorm fusion A/B at the real throughput regime.
# s1base is the tag,  the gates under test on top of the standing set.
set -uo pipefail

GATES="MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 \
MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1 \
MCORE_INFER_INCR_ATTN_STATE=1 MCORE_INFER_VEC_UPDATE_REQS=1 MCORE_INFER_FAST_POST_PROCESS=1 \
MCORE_FUSED_QK_NORM=1 MCORE_FUSED_ADD_NORM=1 MCORE_FUSED_ADD_NORM_QKV=1 MCORE_FLASH_ATTN_VERSION=2"

env $GATES  \
    TAG=s1base OSL=1024 BS=256 NITERS=5 NWARMUP=2 \
    bash dev/moe_fused/run_e2e_cfg.sh 2>&1 | tee /tmp/s1base_raw.log

echo "================= RESULT s1base ================="
echo "extra gates: "
grep -iE 'throughput=|throughput_tok_per_sec|avg_latency|TPOT|COHERENCE|coherent' /tmp/s1base_raw.log | tail -25

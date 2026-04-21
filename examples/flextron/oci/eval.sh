#!/bin/bash

# Import common vars
. ${DIR}/examples/flextron/oci/common.sh

NAME="nemotron5_eval"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "$SLURM_JOB_USER" ]; then
    # for SLURM environments
    LOG_DIR="/lustre/fsw/portfolios/coreai/users/$SLURM_JOB_USER/logs/${NAME}"
else
    LOG_DIR="/workspace/logs"
fi
mkdir -p ${LOG_DIR}

HEADS=${1:-32}
MLP=${2:-1856}
HIDDEN=${3:-2688}
MAMBA_HEAD_DIM=${4:-64}
MAMBA_NHEAD=${5:-64}
MAMBA_STATE_DIM=${6:-128}

CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/Convert_DIR_GA_Long_Context_FLEX_reasoning_LR7e-5_BUDGET_1.0_0.78_0.577_SAMPLE_PROBS_5.0_3.0_2.0_GBS1024/Pruned_FFN_1856_HIDDEN_2688_iteration_3200_fix"

NLAYERS=${7:-16}
# HYBRID_PATTERN=${8:-"M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"}
HYBRID_PATTERN=${8:-"MEMEM*EMEMEM*EME"}

ARGS="
    ${COMMON_ARGS} \
    --skip-train \
    --finetune \
    --sequence-parallel \
    --ddp-num-buckets 8 \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --per-split-data-args-path ${BLEND_PATH} \
    --num-layers ${NLAYERS} \
    --mamba-num-heads ${MAMBA_NHEAD} \
    --mamba-head-dim ${MAMBA_HEAD_DIM} \
    --mamba-state-dim ${MAMBA_STATE_DIM} \
    --hybrid-layer-pattern=${HYBRID_PATTERN} \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
    --hidden-size ${HIDDEN} \
    --num-attention-heads ${HEADS} \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size ${MLP} \
    --kv-channels 128 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --tensor-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --eval-iters 1 \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --attention-backend flash \
    --load ${CHECKPOINT_PATH} \
"

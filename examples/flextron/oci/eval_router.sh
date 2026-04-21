#!/bin/bash

TP=2
PP=1
EP=8
ETP=1
DIR=`pwd`

# Add repo root to PYTHONPATH
export PYTHONPATH="${DIR}:${PYTHONPATH}"

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
NUM_EXPERTS=${7:-128}
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/3b_hybrid_moe/checkpoints/phase1"
# NLAYERS=${7:-16}
# HYBRID_PATTERN=${8:-"MEMEM*EMEMEM*EME"}

# HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# # CHECKPOINT_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/3b_hybrid_moe/checkpoints/phase2_lc"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/mean_batch_mean_seq_112"
# NLAYERS=52

HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/femb_2176_LR5e-5_MLR1e-5/checkpoints/femb_2176_LR5e-5_MLR1e-5"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/sorted_all_dims"
CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/moe_hetero_0.75_LR7.5e-5_MLR1e-5/checkpoints/moe_hetero_0.75_LR7.5e-5_MLR1e-5"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/sorted_all_dims_pruned_expert_96"
# NUM_EXPERTS=${8:-96}

NLAYERS=52
MLP_PER_LIST=${8:-1.0}
EMB_PER_LIST=${8:-1.0}
MOE_EXPERT_PER_LIST="1.0 0.875 0.75 0.625 0.5"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/pruned/EMB_2176_LR_5e-5_MLR_1e-5"
# NUM_EXPERTS=${7:-128}
# HIDDEN=${3:-2176}
# EMB_PER_LIST=${8:-1.0}

# HYBRID_PATTERN="MEMEM*EMMEM*EMEMM*MEMEM*EMEMEM*EMEMMEM*EMEMEMEME"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/layer_48"
# NLAYERS=48

# NLAYERS=${7:-16}
# # HYBRID_PATTERN=${8:-"M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"}
# # HYBRID_PATTERN=${8:-"MEME|M*EME|MEM*E|ME"}
# HYBRID_PATTERN=${8:-"MEMEM*EMEMEM*EME"}
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/3b_hybrid_moe/checkpoints/phase1"
# TP=1
# PP=1
# EP=1
# ETP=1

ARGS="
    ${COMMON_ARGS} \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk 6 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
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
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --pipeline-model-parallel-size ${PP} \
    --eval-iters 4 \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --attention-backend flash \
    --load ${CHECKPOINT_PATH} \
"

if [ ! -d ${CHECKPOINT_PATH} ]; then
    echo "Checkpoint dir ${CHECKPOINT_PATH} does not exist."
    exit 1
fi

echo "Evaluating checkpoint: ${CHECKPOINT_PATH}"
BUDGET="1.0"
OVERRIDE_BUDGET="1.0"
FLEXTON_ARGS="
    --flextron \
    --is-flex-eval \
    --enable-router \
    --binary-mask \
    --hard-sample-th 0.996 \
    --router-beta 1.0 \
    --loss-alpha 1.0 \
    --router-inter-dim 256 \
    --linear-scaler-start 10.0 \
    --linear-scaler-end 10.0 \
    --tau-init 0.05 \
    --slice \
    --budget-list ${BUDGET} \
    --budget-type param \
    --mlp-per-list ${MLP_PER_LIST} \
    --emb-per-list ${EMB_PER_LIST} \
    --flex-hetero-moe-expert \
    --moe-expert-per-list ${MOE_EXPERT_PER_LIST} \
    --mamba-per-list 1.0 \
    --head-per-list 1.0 \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --override-selected-budget ${OVERRIDE_BUDGET} \
    --mem-batch-size 1
    "

torchrun --nproc_per_node=8 ${DIR}/examples/flextron/oci/evaluate_router.py ${ARGS} ${FLEXTON_ARGS}
echo $CHECKPOINT_PATH

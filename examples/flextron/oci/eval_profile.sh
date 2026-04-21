#!/bin/bash
# eval_profile.sh — identical to eval.interactive.sh but calls evaluate_profile.py
# instead of evaluate.py.  Prints a parameter count and memory footprint report
# (actual vs utility-function estimates) before running normal evaluation.
#
# Usage: bash eval_profile.sh [HEADS] [MLP] [HIDDEN] [MAMBA_HEAD_DIM] [MAMBA_NHEAD] [MAMBA_STATE_DIM] [NUM_EXPERTS]

TP=1
PP=1
EP=8
ETP=1
DIR=`pwd`

export PYTHONPATH="${DIR}:${PYTHONPATH}"

. ${DIR}/examples/flextron/oci/common.sh

NAME="nemotron5_profile"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "$SLURM_JOB_USER" ]; then
    LOG_DIR="/lustre/fsw/portfolios/coreai/users/$SLURM_JOB_USER/logs/${NAME}"
else
    LOG_DIR="/workspace/logs"
fi
mkdir -p ${LOG_DIR}

BLEND_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/GA_no_mask_70SFT_30PT.json"

HEADS=${1:-32}
MLP=${2:-1856}
HIDDEN=${3:-2688}
MAMBA_HEAD_DIM=${4:-64}
MAMBA_NHEAD=${5:-64}
MAMBA_STATE_DIM=${6:-128}
NUM_EXPERTS=${7:-128}

CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/Convert_DIR_GA_Long_Context_FLEX_reasoning_LR7e-5_BUDGET_1.0_0.78_0.577_SAMPLE_PROBS_5.0_3.0_2.0_GBS1024/Pruned_FFN_1856_HIDDEN_2688_iteration_3200_fix"

HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

NLAYERS=52
MLP_PER_LIST=${8:-1.0}
EMB_PER_LIST=${8:-1.0}
MOE_EXPERT_PER_LIST=${8:-1.0}

TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/soumyes/nano-v3/nano-v3-sft-tokenizer"

ARGS="
    ${COMMON_ARGS} \
    --train-samples 10000 \
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
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
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
    --eval-iters 8 \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --attention-backend flash \
    --load ${CHECKPOINT_PATH} \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --sft-tokenizer-prompt-format identity
"

if [ ! -d ${CHECKPOINT_PATH} ]; then
    echo "Checkpoint dir ${CHECKPOINT_PATH} does not exist."
    exit 1
fi

echo "Profiling checkpoint: ${CHECKPOINT_PATH}"

BUDGET="1.0 0.577"
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
    --mlp-per-list 1.0 \
    --emb-per-list 1.0 \
    --moe-expert-per-list 1.0 \
    --mamba-per-list 1.0 \
    --head-per-list 1.0 \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --override-selected-budget ${OVERRIDE_BUDGET} \
    --mem-batch-size 1
    "

torchrun --nproc_per_node=8 ${DIR}/examples/flextron/oci/evaluate_profile.py ${ARGS} ${FLEXTON_ARGS}
echo $CHECKPOINT_PATH

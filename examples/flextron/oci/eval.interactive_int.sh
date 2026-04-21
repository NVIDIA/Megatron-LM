#!/bin/bash

TP=1
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


BLEND_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/GA_no_mask_70SFT_30PT.json"


HEADS=${1:-32}
MLP=${2:-1856}
HIDDEN=${3:-2688}
MAMBA_HEAD_DIM=${4:-64}
MAMBA_NHEAD=${5:-64}
MAMBA_STATE_DIM=${6:-128}
NUM_EXPERTS=${7:-128}

# NLAYERS=${7:-16}
# HYBRID_PATTERN=${8:-"MEMEM*EMEMEM*EME"}

# HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/3b_hybrid_moe/checkpoints/phase2_lc"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_best_ckpt"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_best_ckpt"
# NLAYERS=52

# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_PRUNED_15p_emb_blend_${BLEND_NAME}_seqlen_8192_iteration_32"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_best_ckpt"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.537_SAMPLE_PROBS_1.0_1.0/checkpoints/FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.537_SAMPLE_PROBS_1.0_1.0"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/pruned_check_FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.537_SAMPLE_PROBS_1.0_1.0"
CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/Convert_DIR_GA_Long_Context_FLEX_reasoning_LR7e-5_BUDGET_1.0_0.78_0.577_SAMPLE_PROBS_5.0_3.0_2.0_GBS1024/Pruned_FFN_1856_HIDDEN_2688_iteration_3200_fix"
# CHECKPOINT_PATH="//lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/check_pruned_hidden_1920_current_best_ckpt"
# CHECKPOINT_PATH="//lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/check_pruned_ffn_960_hidden_1920_current_best_ckpt"
# CHECKPOINT_PATH="//lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/check_pruned_ffn_960_current_best_ckpt"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.577_SAMPLE_PROBS_1.0_1.0/checkpoints/FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.577_SAMPLE_PROBS_1.0_1.0"
HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/femb_2176_LR5e-5_MLR1e-5/checkpoints/femb_2176_LR5e-5_MLR1e-5"
# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_best_ckpt"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/method_REAP_sftbranch_FINAL"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/sorted_all_dims_pruned_expert_96"
# NUM_EXPERTS=${8:-96}

NLAYERS=52
MLP_PER_LIST=${8:-1.0}
EMB_PER_LIST=${8:-1.0}
MOE_EXPERT_PER_LIST=${8:-1.0}
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
    --eval-iters 2 \
    --auto-detect-ckpt-format \
    --attention-backend flash \
    --load ${CHECKPOINT_PATH} \
    --tokenizer-type SFTTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --sft-tokenizer-prompt-format identity
"

if [ "$BLEND_NAME" == "does_mask_100r_0p" ]; then
    PROMPT_FORMAT=identity
    ARGS="${ARGS} \
    --sft
"
fi

if [ ! -d ${CHECKPOINT_PATH} ]; then
    echo "Checkpoint dir ${CHECKPOINT_PATH} does not exist."
    exit 1
fi

echo "Evaluating checkpoint: ${CHECKPOINT_PATH}"
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
    --mlp-int-list ${MLP}\
    --emb-int-list 2016 \
    --moe-expert-int-list ${NUM_EXPERTS}\
    --mamba-int-list ${MAMBA_NHEAD}\
    --head-int-list ${HEADS}\
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --override-selected-budget ${OVERRIDE_BUDGET} \
    --mem-batch-size 1
    "

torchrun --nproc_per_node=8 ${DIR}/examples/flextron/oci/evaluate.py ${ARGS} ${FLEXTON_ARGS}
echo $CHECKPOINT_PATH

#!/bin/bash

TP=1
PP=1
EP=1
ETP=1
DIR=`pwd`

# Add repo root to PYTHONPATH
export PYTHONPATH="${DIR}:${PYTHONPATH}"

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


# NLAYERS=${7:-16}
# # HYBRID_PATTERN=${8:-"M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"}
# # HYBRID_PATTERN=${8:-"MEME|M*EME|MEM*E|ME"}
# HYBRID_PATTERN=${8:-"MEMEM*EMEMEM*EME"}
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/3b_hybrid_moe/checkpoints/phase1"


# HYBRID_PATTERN="MEMEM*EMEMEM*|EMEMEM*EMEMEM|*EMEMEM*EMEME|MEM*EMEMEMEME"
# # CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/flex/checkpoints/flex"
# NLAYERS=${7:-52}

# HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/pruned_emb_31_LR1e-4_MLR1e-5_GBS3072_rGBS256_rLR1e-2_A1.0/checkpoints/pruned_emb_31_LR1e-4_MLR1e-5_GBS3072_rGBS256_rLR1e-2_A1.0"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/sorted_all_dims"
# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/megatron_artifacts/flx_LR1e-4_MLR1e-5_GBS3072_rGBS256_rLR1e-2_A1.0/checkpoints/flx_LR1e-4_MLR1e-5_GBS3072_rGBS256_rLR1e-2_A1.0"
NLAYERS=${7:-52}

# Things to change:

BLEND_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/GA_no_mask_70SFT_30PT.json"

NUM_EXPERTS_PRUNED=${2:-128}
HIDDEN_PRUNED=${3:-2688}
FFN_HIDDEN_PRUNED=${4:-1856}
MAMBA_HEAD_DIM_PRUNED=${5:-64}
MAMBA_NUM_HEADS_PRUNED=${6:-64}
MOE_SHARED_INTERMEDIATE_SIZE_PRUNED=${7:-3712}

GBS=${8:-32}
MICRO_BATCH_SIZE=${9:-1}
EVAL_ITERS=${10:-32}
SEQUENCE_LENGTH=${11:-8192}

# Things to change end
# export MOE_SORTING_METRIC="VOTING"
export MOE_SORTING_METRIC="REAP"

CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/ga_nanov3"
SAVE_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/ga_nanov3_sorted"

# CHECKPOINT_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_best_ckpt" #"/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_sorted_blend_${BLEND_NAME}_seqlen_${SEQUENCE_LENGTH}_iteration_${EVAL_ITERS}"
# SAVE_PATH="/lustre/fs1/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/reasoning_current_PRUNED_15p_emb_blend_${BLEND_NAME}_seqlen_${SEQUENCE_LENGTH}_iteration_${EVAL_ITERS}"

# CHECKPOINT_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.537_SAMPLE_PROBS_1.0_1.0/checkpoints/FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.537_SAMPLE_PROBS_1.0_1.0"
# SAVE_PATH="/lustre/fsw/portfolios/coreai/users/ataghibakhsh/megatron_artifacts/pruned_check_FLEX_reasoning_BLEND_no_mask_70r_30p_PP1_LR1e-4_BUDGET_1.0_0.537_SAMPLE_PROBS_1.0_1.0"

# Final pretrained checkpoint: /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/3b_hybrid_moe/checkpoints/phase2_lc
# --prune-skip-sorting: doesn't sort and doesn't apply hooks, just prunes
# --prune-scores-path: save the scores to a file, WARNING: if set, it will override your scores file

PRUNING_ARGS="
    --prune-hidden-size ${HIDDEN_PRUNED} \
    --prune-num-moe-experts ${NUM_EXPERTS_PRUNED} \
    --prune-moe-shared-expert-intermediate-size ${MOE_SHARED_INTERMEDIATE_SIZE_PRUNED} \
    --prune-moe-ffn-hidden-size ${FFN_HIDDEN_PRUNED} \
    --prune-mamba-head-dim ${MAMBA_HEAD_DIM_PRUNED} \
    --prune-mamba-num-heads ${MAMBA_NUM_HEADS_PRUNED}"

TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/soumyes/nano-v3/nano-v3-sft-tokenizer"

ARGS="
    ${COMMON_ARGS} \
    --train-samples 100000 \
    --moe-token-dispatcher-type alltoall \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 128 \
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
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings 8192 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GBS} \
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --pipeline-model-parallel-size ${PP} \
    --eval-iters ${EVAL_ITERS} \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness ignore_all \
    --attention-backend flash \
    --load ${CHECKPOINT_PATH} \
    --save ${SAVE_PATH} \
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
echo "Using blend: ${BLEND_NAME} at path: ${BLEND_PATH}"
torchrun --nproc_per_node=${PP} ${DIR}/examples/flextron/oci/evaluate_mtp.py ${ARGS} ${PRUNING_ARGS}
echo $CHECKPOINT_PATH

#!/bin/bash

# Launch script for Qwen3.5-397B-A17B VLM training via MIMO.
#
# Usage (from the Megatron-LM repo root):
#   ./examples/mimo/scripts/run_qwen35_vlm_train.sh /path/to/dataset [/path/to/llm/checkpoint]
#
# NOTE: MIMO currently requires PP=1 and CP=1. For the full 397B model,
# use large TP and EP to distribute across GPUs.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export NVTE_FUSED_ATTN=1

DRY_RUN=${DRY_RUN:-false}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NNODES:-1}

DATASET_PATH=${1:-"mock"}
PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH=${2:-"None"}

LANGUAGE_MODEL_CKPT_ARG=()
if [ "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" != "None" ]; then
    LANGUAGE_MODEL_CKPT_ARG=(--language-model-checkpoint "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH")
fi

# Batch sizes
MBS=${MBS:-1}
GBS=${GBS:-64}

# Parallelism — MIMO requires PP=1, CP=1
TP=${TP:-8}
EP=${EP:-32}

WANDB_PROJECT='mimo-qwen35-vlm'
EXP_NAME="qwen35_397b_vlm_mbs_${MBS}_gbs_${GBS}_tp${TP}_ep${EP}"

ROOT_DIR='./local/'
CHECKPOINT_STORE_PATH="${ROOT_DIR}${EXP_NAME}"
mkdir -p "$CHECKPOINT_STORE_PATH"

TENSORBOARD_LOGS_PATH='./logs'
mkdir -p "$TENSORBOARD_LOGS_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
)

if [ "$NUM_NODES" -gt 1 ]; then
    DISTRIBUTED_ARGS+=(
        --master_addr "${MASTER_ADDR:-localhost}"
        --master_port "${MASTER_PORT:-6000}"
    )
fi

# --- Parallelism ---
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size "$EP"
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

# --- Training ---
TRAINING_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --train-iters 2000
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 1.2e-4
    --min-lr 1.2e-5
    --lr-decay-style cosine
    --lr-warmup-iters 100
    --lr-decay-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --auto-detect-ckpt-format
    --accumulate-allreduce-grads-in-fp32
    --model-provider qwen35_vlm
    --bf16
    --use-mcore-models
    --use-flash-attn
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --enable-experimental
    --manual-gc
    --manual-gc-interval 5
)

# --- Logging & Checkpointing ---
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 500
    --eval-interval 500
    --save "$CHECKPOINT_STORE_PATH"
    --eval-iters 10
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$CHECKPOINT_STORE_PATH"
    --log-throughput
    "${LANGUAGE_MODEL_CKPT_ARG[@]}"
)

# --- Tokenizer ---
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model 'Qwen/Qwen3.5-397B-A17B'
)

# --- Dataset ---
if [ "$DATASET_PATH" = "mock" ]; then
    DATASET_ARGS=(
        --dataset-provider qwen35_mock
        --image-token-id 248056
    )
else
    echo "ERROR: Real dataset is not supported yet. Only mock dataset is available." >&2
    exit 1
fi

# --- Qwen3-Next Decoder Architecture ---
# These must match configs/qwen35_vlm.py::get_qwen35_language_model_config
GPT_MODEL_ARGS=(
    # Network size
    --num-layers 60
    --hidden-size 4096
    --ffn-hidden-size 10240
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --max-position-embeddings 262144
    --seq-length 4096

    # Normalization & activation
    --normalization RMSNorm
    --apply-layernorm-1p
    --norm-epsilon 1e-06
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights

    # Position embeddings
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --rotary-seq-len-interpolation-factor 1

    # Attention
    --qk-layernorm
    --attention-output-gate
    --attention-dropout 0.0
    --hidden-dropout 0.0

    # Gated Delta Net (hybrid linear + full attention)
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 64

    # MoE
    --num-experts 512
    --moe-ffn-hidden-size 1024
    --moe-shared-expert-intermediate-size 1024
    --moe-shared-expert-gate
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 10
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type flex
    --moe-router-dtype fp32

    # MTP (Multi-Token Prediction)
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1

    # Vocab
    --make-vocab-size-divisible-by 485
)

# --- Recompute (for memory savings) ---
RECOMPUTE_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act shared_experts layernorm
)

echo "================================================================"
echo "Qwen3.5-397B-A17B VLM MIMO Training"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Num nodes:     $NUM_NODES"
echo "  TP=$TP  EP=$EP  PP=1  CP=1"
echo "  MBS=$MBS  GBS=$GBS"
echo "  Dataset:       $DATASET_PATH"
echo "================================================================"

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN ==="
    echo "torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${TOKENIZER_ARGS[@]}" \
        "${GPT_MODEL_ARGS[@]}" \
        "${DATASET_ARGS[@]}" \
        "${RECOMPUTE_ARGS[@]}"
    echo "=== End of DRY RUN ==="
else
    torchrun "${DISTRIBUTED_ARGS[@]}" examples/mimo/train.py \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${TOKENIZER_ARGS[@]}" \
        "${GPT_MODEL_ARGS[@]}" \
        "${DATASET_ARGS[@]}" \
        "${RECOMPUTE_ARGS[@]}"
fi

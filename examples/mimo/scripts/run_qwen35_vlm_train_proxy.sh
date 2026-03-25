#!/bin/bash

# Launch script for Qwen3.5-397B-A17B VLM training via MIMO.
#
# Usage (from the Megatron-LM repo root):
#   ./examples/mimo/scripts/run_qwen35_vlm_train.sh /path/to/dataset [/path/to/llm/checkpoint]
#
# NOTE: MIMO currently requires PP=1 and CP=1. For the full 397B model,
# use large TP and EP to distribute across GPUs.

set -euo pipefail

# Install megatron-energon if not installed
# pip install megatron-energon
# cd ../transformers && pip install '.[torch]'

export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_IB_SL=1
export NVTE_FUSED_ATTN=1

DRY_RUN=${DRY_RUN:-false}
PROFILE=${PROFILE:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-10}
PROFILE_STEP_END=${PROFILE_STEP_END:-12}
PROFILE_RANKS=${PROFILE_RANKS:-0}
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
TRAIN_ITERS=${TRAIN_ITERS:-100}

# Parallelism — MIMO requires PP=1, CP=1
TP=${TP:-1}
EP=${EP:-2}

NUM_LAYERS=${NUM_LAYERS:-4}
NUM_EXPERTS=${NUM_EXPERTS:-4}
SEQ_LEN=${SEQ_LEN:-4096}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-2}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}

WANDB_PROJECT='mimo-qwen35-vlm'
EXP_NAME="qwen35_proxy_vlm_mbs_${MBS}_gbs_${GBS}_tp${TP}_ep${EP}"

ROOT_DIR='./local/'
CHECKPOINT_STORE_PATH="${ROOT_DIR}${EXP_NAME}"
mkdir -p "$CHECKPOINT_STORE_PATH"

# Nsight Systems profiling (enabled when PROFILE=1)
NSYS_OUTPUT=${NSYS_OUTPUT:-"${CHECKPOINT_STORE_PATH}/nsys_qwen35_vlm_train_proxy_$(date +%Y%m%d_%H%M%S)"}
NSYS_TRACE=${NSYS_TRACE:-"cuda,nvtx,cudnn,cublas"}
NSYS_CAPTURE_RANGE=${NSYS_CAPTURE_RANGE:-"cudaProfilerApi"}

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
    --train-iters "$TRAIN_ITERS"
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
    --log-timers-to-tensorboard
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
        --total-seq-length ${SEQ_LEN}
    )
else
    echo "ERROR: Real dataset is not supported yet. Only mock dataset is available." >&2
    exit 1
fi

# --- Qwen3-Next Decoder Architecture ---
# These must match configs/qwen35_vlm.py::get_qwen35_language_model_config
GPT_MODEL_ARGS=(
    # Network size
    --num-layers ${NUM_LAYERS}
    --hidden-size 4096
    --ffn-hidden-size 10240
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --max-position-embeddings 262144
    --seq-length ${SEQ_LEN}

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
    --num-experts ${NUM_EXPERTS}
    --moe-ffn-hidden-size 1024
    --moe-shared-expert-intermediate-size 1024
    --moe-shared-expert-gate
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk ${MOE_ROUTER_TOPK}
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE}
    --moe-router-dtype fp32
    --moe-router-force-load-balancing

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

if [ "$PROFILE" = "1" ]; then
    read -r -a PROFILE_RANKS_ARR <<< "$PROFILE_RANKS"
    PROFILE_ARGS=(
        --profile
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks "${PROFILE_RANKS_ARR[@]}"
    )
else
    PROFILE_ARGS=()
fi

USE_FSDP=${USE_FSDP:-1}
if [ "$USE_FSDP" -eq 1 ]; then
    FSDP_ARGS=(        
        --use-megatron-fsdp
        --data-parallel-sharding-strategy optim_grads_params
        --no-gradient-accumulation-fusion
        --init-model-with-meta-device
        --use-distributed-optimizer
        --ckpt-format fsdp_dtensor
    )
else
    FSDP_ARGS=()
fi

USE_CUDA_GRAPH=${USE_CUDA_GRAPH:-0}
if [ "$USE_CUDA_GRAPH" -eq 1 ]; then
    CUDA_GRAPH_ARGS=(
        --cuda-graph-impl transformer_engine
        --cuda-graph-scope attn moe_router moe_preprocess
    )
else
    CUDA_GRAPH_ARGS=()
fi

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
    base_cmd="torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TOKENIZER_ARGS[@]} \
        ${GPT_MODEL_ARGS[@]} \
        ${DATASET_ARGS[@]} \
        ${RECOMPUTE_ARGS[@]} \
        ${FSDP_ARGS[@]} \
        ${PROFILE_ARGS[@]} \
        ${CUDA_GRAPH_ARGS[@]}"
    if [ "$PROFILE" = "1" ]; then
        echo "nsys profile -t $NSYS_TRACE -s none --capture-range=$NSYS_CAPTURE_RANGE --capture-range-end=stop-shutdown --force-overwrite true -o $NSYS_OUTPUT $base_cmd"
    else
        echo "$base_cmd"
    fi
    echo "=== End of DRY RUN ==="
else
    base_cmd="torchrun ${DISTRIBUTED_ARGS[@]} examples/mimo/train.py \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TOKENIZER_ARGS[@]} \
        ${GPT_MODEL_ARGS[@]} \
        ${DATASET_ARGS[@]} \
        ${RECOMPUTE_ARGS[@]} \
        ${FSDP_ARGS[@]} \
        ${PROFILE_ARGS[@]} \
        ${CUDA_GRAPH_ARGS[@]}"
    if [ "$PROFILE" = "1" ]; then
        cmd="nsys profile -t $NSYS_TRACE -s none --capture-range=$NSYS_CAPTURE_RANGE --capture-range-end=stop-shutdown --force-overwrite true -o $NSYS_OUTPUT $base_cmd"
    else
        cmd="$base_cmd"
    fi
    echo "Running command: $cmd"
    eval $cmd
fi

# AssertionError: FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one
#!/bin/bash

# Launch script for Kimi K2.5 VL training via multimodal_dev (FSDP + EP).
#
# Usage (from the Megatron-LM repo root):
#   # Quick proxy test (2 GPUs, no real data needed):
#   MODEL_VARIANT=proxy GPUS_PER_NODE=2 EP=2 \
#       ./examples/multimodal_dev/scripts/run_kimi_k25_vlm_train.sh
#
#   # Full production run:
#   MODEL_VARIANT=full TP=8 EP=32 \
#       ./examples/multimodal_dev/scripts/run_kimi_k25_vlm_train.sh
#
# MODEL_VARIANT choices:
#   proxy   (default) 4 layers, 16 experts — single-node testing
#   full    production: 61 layers, 256 experts
#
# Environment variables:
#   KIMI_K25_HF_MODEL_PATH  Path or HF hub ID for Kimi K2.5 VL model
#                            (default: moonshotai/Kimi-K2.5)
#   MODEL_VARIANT: proxy (default), full
#   TP, EP, PP: parallelism sizes
#   MBS, GBS: micro/global batch sizes
#   LAUNCHER: torchrun (default) or python
#   PROFILE: set to 1 to enable Nsight Systems profiling (default: 0)
#   PROFILE_STEP_START/PROFILE_STEP_END: profiled iteration window (default: 4-5)

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DRY_RUN=${DRY_RUN:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
if [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
    NUM_NODES="$SLURM_JOB_NUM_NODES"
else
    NUM_NODES=${NNODES:-1}
fi
PROFILE=${PROFILE:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-4}
PROFILE_STEP_END=${PROFILE_STEP_END:-5}
PROFILE_RANKS=${PROFILE_RANKS:-0}
LAUNCHER=${LAUNCHER:-torchrun}

MODEL_VARIANT=${MODEL_VARIANT:-proxy}

# Batch sizes
MBS=${MBS:-1}
GBS=${GBS:-64}
SEQ_LEN=${SEQ_LEN:-4096}

# Parallelism — PP=1, CP=1 required
if [ "$MODEL_VARIANT" = "proxy" ]; then
    TP=${TP:-1}
    EP=${EP:-2}
else
    TP=${TP:-8}
    EP=${EP:-32}
fi

WANDB_PROJECT='multimodal-v2-kimi-k25-vl'
EXP_NAME="kimi_k25_${MODEL_VARIANT}_tp${TP}_ep${EP}"

RECOMPUTE=${RECOMPUTE:-0}
if [ "$RECOMPUTE" -eq 1 ]; then
    EXP_NAME+="_recompute_decoder"
fi

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-${MEGATRON_LM_PATH}/local/}"
CHECKPOINT_STORE_PATH="${ROOT_DIR}${EXP_NAME}"
mkdir -p "$CHECKPOINT_STORE_PATH"

TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-${MEGATRON_LM_PATH}/logs}"
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
    --bf16
    --use-mcore-models
    --use-flash-attn
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --enable-experimental
    --manual-gc
    --manual-gc-interval 5
    --accumulate-allreduce-grads-in-fp32
)

PROFILE_ARGS=()
NSYS_CMD=()
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
        --profile
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks "$PROFILE_RANKS"
    )

    NSYS_OUTPUT_DIR="${CHECKPOINT_STORE_PATH}/nsys"
    mkdir -p "$NSYS_OUTPUT_DIR"
    NSYS_CMD=(
        nsys profile
        --sample=none
        --cpuctxsw=none
        --trace=cuda,nvtx,cublas,cudnn
        --force-overwrite=true
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        -o "${NSYS_OUTPUT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
    )
fi

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
)

# --- Tokenizer ---
TOKENIZER_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 163840
)

# --- Multimodal-specific ---
MULTIMODAL_ARGS=(
    --model-arch kimi_k25
    --model-variant "$MODEL_VARIANT"
    --dataset-provider mock
    --image-token-id 163605
    --image-size 224
    --total-seq-length "$SEQ_LEN"
    --image-seq-length 64
)

# --- Kimi K2.5 Decoder Architecture (CLI args for MCore-level flags) ---
# Architecture values are built from configuration.py variants inside
# the model factory, but Megatron also uses these CLI args internally
# (PP splits, param counting).
if [ "$MODEL_VARIANT" = "proxy" ]; then
    GPT_MODEL_ARGS=(
        --num-layers 4
        --hidden-size 4096
        --ffn-hidden-size 12288
        --num-attention-heads 32
        --group-query-attention
        --num-query-groups 32
        --max-position-embeddings 4096
        --seq-length "$SEQ_LEN"
        --normalization RMSNorm
        --norm-epsilon 1e-06
        --swiglu
        --disable-bias-linear
        --untie-embeddings-and-output-weights
        --position-embedding-type rope
        --qk-layernorm
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --num-experts 16
        --moe-ffn-hidden-size 1536
        --moe-shared-expert-intermediate-size 1536
        --moe-router-load-balancing-type seq_aux_loss
        --moe-router-topk 4
        --moe-grouped-gemm
        --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
        --moe-router-dtype fp32
        --make-vocab-size-divisible-by 1280
    )
else
    GPT_MODEL_ARGS=(
        --num-layers 61
        --hidden-size 4096
        --ffn-hidden-size 12288
        --num-attention-heads 32
        --group-query-attention
        --num-query-groups 32
        --max-position-embeddings 4096
        --seq-length "$SEQ_LEN"
        --normalization RMSNorm
        --norm-epsilon 1e-06
        --swiglu
        --disable-bias-linear
        --untie-embeddings-and-output-weights
        --position-embedding-type rope
        --qk-layernorm
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --num-experts 256
        --moe-ffn-hidden-size 1536
        --moe-shared-expert-intermediate-size 1536
        --moe-router-load-balancing-type seq_aux_loss
        --moe-router-topk 8
        --moe-grouped-gemm
        --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
        --moe-router-dtype fp32
        --make-vocab-size-divisible-by 1280
    )
fi

# --- Recompute ---
RECOMPUTE=${RECOMPUTE:-0}
if [ "$RECOMPUTE" -eq 1 ]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity selective
        --recompute-modules moe_act shared_experts layernorm
    )
else
    RECOMPUTE_ARGS=()
fi

# --- FSDP ---
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
    export CUDA_DEVICE_MAX_CONNECTIONS=8
else
    FSDP_ARGS=()
fi

echo "================================================================"
echo "Kimi K2.5 VL Multimodal Training (multimodal_dev)"
echo "  Variant:       $MODEL_VARIANT"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Num nodes:     $NUM_NODES"
echo "  TP=$TP  EP=$EP  PP=1  CP=1"
echo "  MBS=$MBS  GBS=$GBS  SEQ=$SEQ_LEN"
echo "  Launcher:      $LAUNCHER"
echo "  PROFILE:       $PROFILE"
if [ "$PROFILE" = "1" ]; then
    echo "  Profile steps: ${PROFILE_STEP_START}-${PROFILE_STEP_END}"
    echo "  Profile ranks: $PROFILE_RANKS"
fi
echo "================================================================"

if [ "$LAUNCHER" = "python" ]; then
    LAUNCH_CMD=( python $MEGATRON_LM_PATH/examples/multimodal_dev/pretrain_multimodal.py )
elif [ "$LAUNCHER" = "torchrun" ]; then
    LAUNCH_CMD=( torchrun "${DISTRIBUTED_ARGS[@]}" $MEGATRON_LM_PATH/examples/multimodal_dev/pretrain_multimodal.py )
else
    echo "Unsupported LAUNCHER=$LAUNCHER (expected torchrun or python)" >&2
    exit 1
fi

cmd=( "${NSYS_CMD[@]}" "${LAUNCH_CMD[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${TOKENIZER_ARGS[@]}" \
    "${MULTIMODAL_ARGS[@]}" \
    "${GPT_MODEL_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FSDP_ARGS[@]}" )

echo "${cmd[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== DRY RUN ==="
    exit 0
else
    "${cmd[@]}"
fi

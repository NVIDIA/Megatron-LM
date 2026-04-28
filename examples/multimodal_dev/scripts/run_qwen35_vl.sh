#!/bin/bash

# Launch script for Qwen3.5-VL training via multimodal_dev (FSDP + EP).
#
# Usage (from the Megatron-LM repo root):
#   ./examples/multimodal_dev/scripts/run_qwen35_vl.sh
#
# Environment variables:
#   MODEL_VARIANT: proxy (default), 0.8b, 2b, 4b, 9b, 27b, 35b_a3b, 122b_a10b, 397b_a17b, 35b_a3b_light
#   CKPT_LOAD: path to a pre-converted checkpoint to load (enables --load + --finetune)
#   CKPT_FORMAT: checkpoint format override (e.g. torch_dist); auto-detected when empty
#   TP, EP, PP: parallelism sizes
#   MBS, GBS: micro/global batch sizes
#   NUM_LAYERS, NUM_EXPERTS: override for proxy testing
#   LAUNCHER: torchrun (default) or python
#   PROFILE: set to 1 to enable Nsight Systems profiling (default: 0)
#   PROFILE_STEP_START/PROFILE_STEP_END: profiled iteration window (default: 4-5)

# example script: 
# WANDB_PROJECT=qwen35-cp-test  WANDB_MODE=online CP=2 GPUS_PER_NODE=8 CKPT_LOAD=/lustre/fs1/portfolios/coreai/users/lit/workspace/dev-project/models/Qwen/Qwen3.5-0.8B-fsdp-0420/ USE_FSDP=1 EP=1  GBS=16 MODEL_VARIANT=0.8b   SAVE_INTERVAL=10000 CKPT_RESUME=0 DRY_RUN=0  USE_PACKED_SEQUENCE=1  bash ./examples/multimodal_dev/scripts/run_qwen35_vl.sh

# WANDB_PROJECT=qwen35-cp-test  WANDB_MODE=online CP=1 GPUS_PER_NODE=4 CKPT_LOAD=/lustre/fs1/portfolios/coreai/users/lit/workspace/dev-project/models/Qwen/Qwen3.5-0.8B-fsdp-0420/ USE_FSDP=1 EP=1  GBS=16 MODEL_VARIANT=0.8b   SAVE_INTERVAL=10000 CKPT_RESUME=0 DRY_RUN=0  USE_PACKED_SEQUENCE=1  bash ./examples/multimodal_dev/scripts/run_qwen35_vl.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
# export NVTE_FUSED_ATTN=1
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
VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-}

# Batch sizes
MBS=${MBS:-2}
GBS=${GBS:-16}

# Parallelism
TP=${TP:-1}
EP=${EP:-2}
PP=${PP:-1}

# Variant-aware architecture defaults.
# The model provider builds configs from the variant dict in
# multimodal_dev/models/qwen35_vl/configuration.py, but Megatron also
# uses these CLI args internally (PP splits, param counting).
case "$MODEL_VARIANT" in
    0.8b)
        NUM_LAYERS=${NUM_LAYERS:-24}
        NUM_EXPERTS=${NUM_EXPERTS:-0}
        HIDDEN_SIZE=1024
        FFN_HIDDEN_SIZE=3584
        NUM_ATTN_HEADS=8
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=16
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-12}
        ;;
    2b)
        NUM_LAYERS=${NUM_LAYERS:-24}
        NUM_EXPERTS=${NUM_EXPERTS:-0}
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=6144
        NUM_ATTN_HEADS=8
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=16
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-24}
        ;;
    4b)
        NUM_LAYERS=${NUM_LAYERS:-32}
        NUM_EXPERTS=${NUM_EXPERTS:-0}
        HIDDEN_SIZE=2560
        FFN_HIDDEN_SIZE=9216
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=4
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-24}
        ;;
    proxy)
        NUM_LAYERS=${NUM_LAYERS:-4}
        NUM_EXPERTS=${NUM_EXPERTS:-16}
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=10240
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=64
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-2}
        ;;
    9b)
        NUM_LAYERS=${NUM_LAYERS:-32}
        NUM_EXPERTS=${NUM_EXPERTS:-0}
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=12288
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=4
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    27b)
        NUM_LAYERS=${NUM_LAYERS:-64}
        NUM_EXPERTS=${NUM_EXPERTS:-0}
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=17408
        NUM_ATTN_HEADS=24
        NUM_QUERY_GROUPS=4
        LINEAR_NUM_VALUE_HEADS=48
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    35b_a3b)
        NUM_LAYERS=${NUM_LAYERS:-40}
        NUM_EXPERTS=${NUM_EXPERTS:-256}
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=4096
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    35b_a3b_light)
        NUM_LAYERS=${NUM_LAYERS:-12}
        NUM_EXPERTS=${NUM_EXPERTS:-128}
        HIDDEN_SIZE=2048
        FFN_HIDDEN_SIZE=4096
        NUM_ATTN_HEADS=16
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=32
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-7}
        ;;
    122b_a10b)
        NUM_LAYERS=${NUM_LAYERS:-48}
        NUM_EXPERTS=${NUM_EXPERTS:-256}
        HIDDEN_SIZE=3072
        FFN_HIDDEN_SIZE=8192
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=64
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    397b_a17b)
        NUM_LAYERS=${NUM_LAYERS:-60}
        NUM_EXPERTS=${NUM_EXPERTS:-512}
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=10240
        NUM_ATTN_HEADS=32
        NUM_QUERY_GROUPS=2
        LINEAR_NUM_VALUE_HEADS=64
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
    *)
        : "${NUM_LAYERS:?NUM_LAYERS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${NUM_EXPERTS:?NUM_EXPERTS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${HIDDEN_SIZE:?HIDDEN_SIZE must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${FFN_HIDDEN_SIZE:?FFN_HIDDEN_SIZE must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${NUM_ATTN_HEADS:?NUM_ATTN_HEADS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${NUM_QUERY_GROUPS:?NUM_QUERY_GROUPS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${LINEAR_NUM_VALUE_HEADS:?LINEAR_NUM_VALUE_HEADS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-27}
        ;;
esac
SEQ_LEN=${SEQ_LEN:-4096}

WANDB_PROJECT=${WANDB_PROJECT:-'qwen35-vl-0524'}
EXP_NAME="qwen35vl_${MODEL_VARIANT}_tp${TP}_ep${EP}_pp${PP}"

RECOMPUTE_VISION=${RECOMPUTE_VISION:-0}
if [ "$RECOMPUTE_VISION" -eq 1 ]; then
    EXP_NAME+="_recompute_encoder"
fi
RECOMPUTE=${RECOMPUTE:-0}
if [ "$RECOMPUTE" -eq 1 ]; then
    EXP_NAME+="_recompute_decoder"
fi

USE_PACKED_SEQUENCE=${USE_PACKED_SEQUENCE:-0}
if [ "$USE_PACKED_SEQUENCE" -eq 1 ]; then
    EXP_NAME+="_thd"
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
    --pipeline-model-parallel-size "$PP"
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
    --train-iters "${TRAIN_ITERS:-500}"
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
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --sft
    --attention-backend flash
    # --calculate-per-token-loss
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
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval "$SAVE_INTERVAL"
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
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen3.5-397B-A17B}
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
)

# --- Multimodal-specific ---
MULTIMODAL_ARGS=(
    --model-arch qwen35_vl
    --model-variant "$MODEL_VARIANT"
    --dataset-provider cord_v2
    --hf-processor-path Qwen/Qwen3.5-397B-A17B
    --use-vanilla-collate-fn
    --image-token-id 248056
    --image-size 224
    --total-seq-length "$SEQ_LEN"
    --image-seq-length 256
    --vision-num-layers "$VISION_NUM_LAYERS"
)

if [ "$USE_PACKED_SEQUENCE" -eq 1 ]; then
    MULTIMODAL_ARGS+=( --use-packed-sequence )
fi

# --- Qwen3.5 Decoder Architecture (variant-specific dims set above) ---
# These must match examples/multimodal_dev/models/qwen35_vl/configuration.py
GPT_MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTN_HEADS"
    --group-query-attention
    --num-query-groups "$NUM_QUERY_GROUPS"
    --kv-channels 256
    --max-position-embeddings 262144
    --seq-length "$SEQ_LEN"
    --normalization RMSNorm
    --apply-layernorm-1p
    --norm-epsilon 1e-06
    --swiglu
    --disable-bias-linear
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --rotary-seq-len-interpolation-factor 1
    --qk-layernorm
    --attention-output-gate
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads "$LINEAR_NUM_VALUE_HEADS"
    --make-vocab-size-divisible-by 485
    --moe-router-force-load-balancing
)

# --- Tied / untied embeddings ---
# 0.8B, 2B, 4B use tied embeddings; all other variants untie them.
case "$MODEL_VARIANT" in
    0.8b|2b|4b) ;;
    *)           GPT_MODEL_ARGS+=( --untie-embeddings-and-output-weights ) ;;
esac

# --- MoE args (MoE variants only) ---
MOE_ARGS=()
case "$MODEL_VARIANT" in
    proxy)
        MOE_TOPK=2; MOE_FFN_HIDDEN=1024; MOE_SHARED_HIDDEN=1024
        ;;
    35b_a3b|35b_a3b_light)
        MOE_TOPK=8; MOE_FFN_HIDDEN=512;  MOE_SHARED_HIDDEN=512
        ;;
    122b_a10b)
        MOE_TOPK=8; MOE_FFN_HIDDEN=1024; MOE_SHARED_HIDDEN=1024
        ;;
    397b_a17b)
        MOE_TOPK=10; MOE_FFN_HIDDEN=1024; MOE_SHARED_HIDDEN=1024
        ;;
    0.8b|2b|4b|9b|27b)
        ;;
esac
if [ "${NUM_EXPERTS:-0}" -gt 0 ]; then
    MOE_ARGS=(
        --num-experts "$NUM_EXPERTS"
        --moe-ffn-hidden-size "$MOE_FFN_HIDDEN"
        --moe-shared-expert-intermediate-size "$MOE_SHARED_HIDDEN"
        --moe-shared-expert-gate
        --moe-router-load-balancing-type aux_loss
        --moe-router-topk "$MOE_TOPK"
        --moe-grouped-gemm
        --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type alltoall
        --moe-router-dtype fp32
    )
fi

# --- Recompute ---
if [ "$RECOMPUTE" -eq 1 ]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
    # RECOMPUTE_ARGS=(
    #     --recompute-granularity selective
    #     --recompute-modules moe_act shared_experts layernorm moe
    # )
else
    RECOMPUTE_ARGS=()
fi
if [ "$RECOMPUTE_VISION" -eq 1 ]; then
    RECOMPUTE_ARGS+=( --recompute-vision )
fi

# --- Checkpoint loading ---
# CKPT_LOAD: path to checkpoint directory
# CKPT_FORMAT: override checkpoint format (default: auto-detect)
# CKPT_RESUME: set to 1 to resume training (keep iteration, optimizer, rng);
#              default 0 = finetune mode (reset iteration, skip optim/rng)
CKPT_LOAD=${CKPT_LOAD:-}
CKPT_FORMAT=${CKPT_FORMAT:-}
CKPT_RESUME=${CKPT_RESUME:-0}
CKPT_OVERRIDE_SCHEDULER=${CKPT_OVERRIDE_SCHEDULER:-0}
CKPT_ARGS=()
if [ -n "$CKPT_LOAD" ]; then
    CKPT_ARGS+=( --load "$CKPT_LOAD" )
    if [ "$CKPT_RESUME" -eq 0 ]; then
        CKPT_ARGS+=( --finetune --no-load-optim --no-load-rng )
    fi
    if [ -n "$CKPT_FORMAT" ]; then
        CKPT_ARGS+=( --ckpt-format "$CKPT_FORMAT" )
    fi
    if [ "$CKPT_OVERRIDE_SCHEDULER" -eq 1 ]; then
        CKPT_ARGS+=( --override-opt-param-scheduler )
    fi
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
echo "Qwen3.5-VL Multimodal Training (multimodal_dev)"
echo "  Variant:       $MODEL_VARIANT"
echo "  Vision layers: $VISION_NUM_LAYERS"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Num nodes:     $NUM_NODES"
echo "  TP=$TP  EP=$EP  PP=$PP  CP=1"
echo "  MBS=$MBS  GBS=$GBS"
echo "  Launcher:      $LAUNCHER"
echo "  FSDP:          $USE_FSDP"
echo "  PROFILE:       $PROFILE"
if [ -n "$CKPT_LOAD" ]; then
    echo "  CKPT_LOAD:     $CKPT_LOAD"
    echo "  CKPT_FORMAT:   ${CKPT_FORMAT:-auto}"
    echo "  CKPT_RESUME:   $CKPT_RESUME"
fi
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
    "${MOE_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FSDP_ARGS[@]}" \
    "${CKPT_ARGS[@]}" )

echo "${cmd[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== DRY RUN ==="
    exit 0
else
    "${cmd[@]}"
fi

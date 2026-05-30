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
#   MTP_NUM_LAYERS: number of MTP layers (default: 1, set 0 to disable)
#   LINEAR_ATTENTION_FREQ: every Nth decoder layer uses standard attention (default: 4; set 1 to force all standard attention)
#   DATASET_PROVIDER: cord_v2 (default) or mock
#   TOKENIZER_TYPE: HuggingFaceTokenizer (default) or NullTokenizer
#   NO_ROPE_FUSION: set to 1 to pass --no-rope-fusion for baseline profiling
#   SAVE_CHECKPOINTS: set to 0 to skip checkpoint saves in short profiling runs
#   LAUNCHER: torchrun (default) or python
#   TORCHRUN_PYTHON: Python executable for LAUNCHER=torchrun (default: python)
#   PROFILE: set to 1 to enable Nsight Systems profiling (default: 0)
#   NVTX_RANGES: set to 1 to emit Megatron custom NVTX ranges when PROFILE=1 (default: 1)
#   PROFILE_STEP_START/PROFILE_STEP_END: profiled iteration window (default: 4-5)

# example script: 
# DRY_RUN=0 MODEL_VARIANT=proxy USE_PACKED_SEQUENCE=1  bash ./examples/multimodal_dev/scripts/run_qwen35_vl.sh

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
NVTX_RANGES=${NVTX_RANGES:-1}
PROFILE_STEP_START=${PROFILE_STEP_START:-4}
PROFILE_STEP_END=${PROFILE_STEP_END:-5}
PROFILE_RANKS=${PROFILE_RANKS:-0}
LAUNCHER=${LAUNCHER:-torchrun}
TORCHRUN_PYTHON=${TORCHRUN_PYTHON:-python}
NO_ROPE_FUSION=${NO_ROPE_FUSION:-0}

MODEL_VARIANT=${MODEL_VARIANT:-proxy}
VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-}

# Batch sizes
MBS=${MBS:-2}
GBS=${GBS:-16}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}
LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-4}

# Parallelism
TP=${TP:-1}
EP=${EP:-2}
PP=${PP:-1}
CP=${CP:-1}

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
EXP_NAME="qwen35vl_${MODEL_VARIANT}_tp${TP}_ep${EP}_pp${PP}_cp${CP}"

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
if [ "$NO_ROPE_FUSION" -eq 1 ]; then
    EXP_NAME+="_no_rope_fusion"
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
    --context-parallel-size "$CP"
    --cp-comm-type "a2a"
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
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --enable-experimental
    --manual-gc
    --manual-gc-interval 50
    --sft
    --use-flash-attn
    # --attention-backend flash
    --calculate-per-token-loss
)
if [ "$MTP_NUM_LAYERS" -gt 0 ]; then
    TRAINING_ARGS+=(
        --mtp-num-layers "$MTP_NUM_LAYERS"
        --mtp-loss-scaling-factor 0.1
    )
fi

PROFILE_ARGS=()
NSYS_CMD=()
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
        --profile
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks "$PROFILE_RANKS"
    )
    if [ "$NVTX_RANGES" -eq 1 ]; then
        PROFILE_ARGS+=( --nvtx-ranges )
    fi

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
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 500
    --eval-iters 10
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$CHECKPOINT_STORE_PATH"
    --log-throughput
    --log-timers-to-tensorboard
    --log-params-norm
)
if [ "$SAVE_CHECKPOINTS" -eq 1 ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --save-interval "$SAVE_INTERVAL"
        --save "$CHECKPOINT_STORE_PATH"
    )
fi

# --- Tokenizer ---
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen3.5-397B-A17B}
TOKENIZER_TYPE=${TOKENIZER_TYPE:-HuggingFaceTokenizer}
VOCAB_SIZE=${VOCAB_SIZE:-248320}
TOKENIZER_ARGS=(
    --tokenizer-type "$TOKENIZER_TYPE"
)
if [ "$TOKENIZER_TYPE" = "NullTokenizer" ]; then
    TOKENIZER_ARGS+=( --vocab-size "$VOCAB_SIZE" )
else
    TOKENIZER_ARGS+=( --tokenizer-model "$TOKENIZER_MODEL" )
fi

# --- Multimodal-specific ---
DATASET_PROVIDER=${DATASET_PROVIDER:-cord_v2}
HF_PROCESSOR_PATH=${HF_PROCESSOR_PATH-Qwen/Qwen3.5-397B-A17B}
IMAGE_SEQ_LENGTH=${IMAGE_SEQ_LENGTH:-256}
MULTIMODAL_ARGS=(
    --model-arch qwen35_vl
    --model-variant "$MODEL_VARIANT"
    --dataset-provider "$DATASET_PROVIDER"
    --use-vanilla-collate-fn
    --image-token-id 248056
    --image-size 224
    --total-seq-length "$SEQ_LEN"
    --image-seq-length "$IMAGE_SEQ_LENGTH"
    --vision-num-layers "$VISION_NUM_LAYERS"
)
if [ -n "$HF_PROCESSOR_PATH" ]; then
    MULTIMODAL_ARGS+=( --hf-processor-path "$HF_PROCESSOR_PATH" )
fi

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
    --linear-attention-freq "$LINEAR_ATTENTION_FREQ"
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads "$LINEAR_NUM_VALUE_HEADS"
    --make-vocab-size-divisible-by 485
    --moe-router-force-load-balancing
)
if [ "$NO_ROPE_FUSION" -eq 1 ]; then
    GPT_MODEL_ARGS+=( --no-rope-fusion )
fi

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
        --moe-permute-fusion
        --moe-router-fusion
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
echo "  TP=$TP  EP=$EP  PP=$PP  CP=$CP"
echo "  MBS=$MBS  GBS=$GBS"
echo "  MTP layers:    $MTP_NUM_LAYERS"
echo "  Linear attn freq: $LINEAR_ATTENTION_FREQ"
echo "  Launcher:      $LAUNCHER"
if [ "$LAUNCHER" = "torchrun" ]; then
    echo "  Torchrun py:   $TORCHRUN_PYTHON"
fi
echo "  FSDP:          $USE_FSDP"
echo "  PROFILE:       $PROFILE"
echo "  RoPE fusion:   $([ "$NO_ROPE_FUSION" -eq 1 ] && echo off || echo on)"
echo "  Dataset:       $DATASET_PROVIDER"
echo "  Tokenizer:     $TOKENIZER_TYPE"
echo "  Checkpoints:   $([ "$SAVE_CHECKPOINTS" -eq 1 ] && echo on || echo off)"
if [ -n "$CKPT_LOAD" ]; then
    echo "  CKPT_LOAD:     $CKPT_LOAD"
    echo "  CKPT_FORMAT:   ${CKPT_FORMAT:-auto}"
    echo "  CKPT_RESUME:   $CKPT_RESUME"
fi
if [ "$PROFILE" = "1" ]; then
    echo "  Profile steps: ${PROFILE_STEP_START}-${PROFILE_STEP_END}"
    echo "  Profile ranks: $PROFILE_RANKS"
    echo "  NVTX ranges:   $([ "$NVTX_RANGES" -eq 1 ] && echo on || echo off)"
fi
echo "================================================================"

if [ "$LAUNCHER" = "python" ]; then
    LAUNCH_CMD=( python $MEGATRON_LM_PATH/examples/multimodal_dev/pretrain_multimodal.py )
elif [ "$LAUNCHER" = "torchrun" ]; then
    LAUNCH_CMD=(
        "$TORCHRUN_PYTHON" -m torch.distributed.run
        "${DISTRIBUTED_ARGS[@]}"
        $MEGATRON_LM_PATH/examples/multimodal_dev/pretrain_multimodal.py
    )
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

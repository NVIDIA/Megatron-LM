#!/bin/bash

# Launch DeepSeek-V4-Flash-Vision through multimodal_dev + HybridModel.
# Phase one intentionally omits MTP. DRY_RUN=1 is the default.
#
# Full architecture:
#   MODEL_VARIANT=flash NNODES=8 GPUS_PER_NODE=8 NUM_LAYERS=43 \
#     VISION_NUM_LAYERS=32 EP=64 DRY_RUN=0 \
#     bash examples/multimodal_dev/scripts/run_deepseek_v4_vision.sh
# Small construction/smoke proxy:
#   MODEL_VARIANT=proxy DRY_RUN=0 \
#     bash examples/multimodal_dev/scripts/run_deepseek_v4_vision.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}

DRY_RUN=${DRY_RUN:-1}
MODEL_VARIANT=${MODEL_VARIANT:-proxy}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
MBS=${MBS:-1}
GBS=${GBS:-8}
SEQ_LEN=${SEQ_LEN:-1024}
IMAGE_SIZE=${IMAGE_SIZE:-224}
TRAIN_ITERS=${TRAIN_ITERS:-10}
USE_FSDP=${USE_FSDP:-1}
USE_PACKED_SEQUENCE=${USE_PACKED_SEQUENCE:-0}
USE_EXTERNAL_VISION_EMBEDDINGS=${USE_EXTERNAL_VISION_EMBEDDINGS:-0}

case "$MODEL_VARIANT" in
    flash)
        NUM_LAYERS=${NUM_LAYERS:-43}
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-32}
        NUM_EXPERTS=${NUM_EXPERTS:-256}
        ROUTER_TOPK=${ROUTER_TOPK:-6}
        HASH_LAYERS=${HASH_LAYERS:-3}
        EP=${EP:-64}
        DSA_KERNEL_BACKEND=${DSA_KERNEL_BACKEND:-cudnn}
        ;;
    proxy)
        NUM_LAYERS=${NUM_LAYERS:-3}
        VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-2}
        NUM_EXPERTS=${NUM_EXPERTS:-8}
        ROUTER_TOPK=${ROUTER_TOPK:-2}
        HASH_LAYERS=${HASH_LAYERS:-3}
        EP=${EP:-1}
        DSA_KERNEL_BACKEND=${DSA_KERNEL_BACKEND:-none}
        ;;
    *)
        echo "Unsupported MODEL_VARIANT=$MODEL_VARIANT (expected flash or proxy)" >&2
        exit 1
        ;;
esac

if [ "$TP" -ne 1 ]; then
    echo "DeepSeek-V4 hybrid attention currently requires TP=1." >&2
    exit 1
fi
if [ "$PP" -ne 1 ]; then
    echo "multimodal_dev currently requires PP=1." >&2
    exit 1
fi
if [ "$HASH_LAYERS" -gt "$NUM_LAYERS" ]; then
    echo "HASH_LAYERS=$HASH_LAYERS exceeds NUM_LAYERS=$NUM_LAYERS." >&2
    exit 1
fi

# Official main-decoder cadence: W, W, C, then H/C alternating. Each decoder
# block is represented as one HybridModel attention layer followed by one MoE layer.
HYBRID_LAYER_PATTERN=""
COMPRESS_RATIOS="["
for ((layer = 0; layer < NUM_LAYERS; layer++)); do
    if [ "$layer" -lt 2 ]; then
        symbol="W"
        ratio=0
    elif [ "$layer" -eq 2 ]; then
        symbol="C"
        ratio=4
    elif [ $(((layer - 3) % 2)) -eq 0 ]; then
        symbol="H"
        ratio=128
    else
        symbol="C"
        ratio=4
    fi
    HYBRID_LAYER_PATTERN+="${symbol}E"
    if [ "$layer" -gt 0 ]; then
        COMPRESS_RATIOS+=","
    fi
    COMPRESS_RATIOS+="$ratio"
done
COMPRESS_RATIOS+="]"

MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-$(cd "$(dirname "$0")/../../.." && pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-${MEGATRON_LM_PATH}/local/dsv4_vision_${MODEL_VARIANT}}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --model-arch deepseek_v4_vision
    --model-variant "$MODEL_VARIANT"
    --hybrid-layer-pattern "$HYBRID_LAYER_PATTERN"
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --kv-channels 512
    --max-position-embeddings "$SEQ_LEN"
    --seq-length "$SEQ_LEN"
    --normalization RMSNorm
    --norm-epsilon 1e-20
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-scaling-factor 16
    --original-max-position-embeddings 65536
    --mscale 1.0
    --mscale-all-dim 1.0
    --multi-latent-attention
    --q-lora-rank 1024
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --qk-layernorm
    --o-groups 8
    --o-lora-rank 1024
    --experimental-attention-variant dsv4_hybrid
    --csa-window-size 128
    --csa-compress-ratios "$COMPRESS_RATIOS"
    --csa-compress-rotary-base 160000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512
    --dsa-indexer-loss-coeff 1e-2
    --dsa-indexer-use-sparse-loss
    --dsa-kernel-backend "$DSA_KERNEL_BACKEND"
    --num-experts "$NUM_EXPERTS"
    --moe-n-hash-layers "$HASH_LAYERS"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk "$ROUTER_TOPK"
    --moe-aux-loss-coeff 1e-4
    --moe-router-topk-scaling-factor 1.5
    --moe-router-score-function sqrtsoftplus
    --moe-router-dtype fp32
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --activation-func-clamp-value 10.0
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --use-fused-mhc
    --vision-num-layers "$VISION_NUM_LAYERS"
    --make-vocab-size-divisible-by 3232
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

DATA_ARGS=(
    --dataset-provider mock
    --use-vanilla-collate-fn
    --image-size "$IMAGE_SIZE"
    --total-seq-length "$SEQ_LEN"
    --dataloader-type cyclic
    --num-workers 0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --context-parallel-size "$CP"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --train-iters "$TRAIN_ITERS"
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --lr-warmup-iters 1
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --use-mcore-models
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
    --calculate-per-token-loss
    --enable-experimental
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
    --save-interval 1000
    --save "$OUTPUT_DIR"
)

EXTRA_ARGS=()
if [ "$USE_PACKED_SEQUENCE" -eq 1 ]; then
    EXTRA_ARGS+=(--use-packed-sequence)
fi
if [ "$USE_EXTERNAL_VISION_EMBEDDINGS" -eq 1 ]; then
    EXTRA_ARGS+=(--use-external-vision-embeddings)
fi
if [ "$USE_FSDP" -eq 1 ]; then
    EXTRA_ARGS+=(
        --use-megatron-fsdp
        --data-parallel-sharding-strategy optim_grads_params
        --init-model-with-meta-device
        --ckpt-format fsdp_dtensor
    )
fi

cmd=(
    torchrun "${DISTRIBUTED_ARGS[@]}"
    "$MEGATRON_LM_PATH/examples/multimodal_dev/pretrain_multimodal.py"
    "${MODEL_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "DeepSeek-V4-Flash-Vision (MTP disabled)"
echo "  hybrid pattern: $HYBRID_LAYER_PATTERN"
echo "  compress ratios: $COMPRESS_RATIOS"
echo "${cmd[@]}"

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    "${cmd[@]}"
fi

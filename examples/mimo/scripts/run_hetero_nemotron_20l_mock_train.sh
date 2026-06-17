#!/bin/bash

# Run a heterogeneous mock-data loop with the Nemotron6-MoE VLM 20L architecture
# on the STOCK megatron/training train() loop (NMFW-516 PR-E5 entry).

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPUS_PER_NODE=8
TRAIN_ITERS=${TRAIN_ITERS:-1}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-4}
NUM_IMAGE_TILES=${NUM_IMAGE_TILES:-12}
TRAINING_STAGE=${TRAINING_STAGE:-stage2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LLM_DP=2
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NUM_MICROBATCHES * LLM_DP))}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-0}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-}
IMAGE_TOKEN_ID=${IMAGE_TOKEN_ID:-511}
PYTHON_BIN=${PYTHON_BIN:-python}

TOKENIZER_ARGS=()
if [[ -n "${TOKENIZER_MODEL}" ]]; then
  TOKENIZER_ARGS+=(--tokenizer-model "${TOKENIZER_MODEL}")
else
  TOKENIZER_ARGS+=(--image-token-id "${IMAGE_TOKEN_ID}")
fi

case "${TRAINING_STAGE}" in
  stage1|stage2|stage3)
    ;;
  *)
    echo "ERROR: Unknown TRAINING_STAGE='${TRAINING_STAGE}'. Use stage1, stage2, or stage3." >&2
    exit 1
    ;;
esac

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${GPUS_PER_NODE}" \
  examples/mimo/pretrain_mimo.py \
  --model-provider nemotron-moe-vlm \
  --training-stage "${TRAINING_STAGE}" \
  --num-layers 20 \
  --hybrid-layer-pattern "MEMEM*EMEMEM*EMEMEM*" \
  --hidden-size 2688 \
  --num-attention-heads 32 \
  --group-query-attention \
  --num-query-groups 8 \
  --ffn-hidden-size 1856 \
  --kv-channels 128 \
  --squared-relu \
  --disable-bias-linear \
  --normalization RMSNorm \
  --init-method-std 0.0173 \
  --num-experts 128 \
  --moe-router-topk 6 \
  --moe-grouped-gemm \
  --moe-ffn-hidden-size 1856 \
  --moe-router-score-function sigmoid \
  --moe-router-topk-scaling-factor 2.5 \
  --moe-router-enable-expert-bias \
  --moe-router-dtype fp32 \
  --moe-router-load-balancing-type seq_aux_loss \
  --moe-router-fusion \
  --moe-aux-loss-coeff 1e-4 \
  --moe-shared-expert-intermediate-size 3712 \
  --moe-shared-expert-overlap \
  --moe-permute-fusion \
  --use-fused-weighted-squared-relu \
  --mamba-num-heads 64 \
  --mamba-head-dim 64 \
  --mamba-num-groups 8 \
  --mamba-state-dim 128 \
  --linear-conv-kernel-dim 4 \
  --position-embedding-type none \
  --calculate-per-token-loss \
  --cross-entropy-loss-fusion \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --bf16 \
  --encoder-tp 2 \
  --encoder-pp 1 \
  --encoder-dp 2 \
  --llm-offset 4 \
  --llm-tp 2 \
  --llm-pp 1 \
  --llm-dp "${LLM_DP}" \
  --llm-ep 4 \
  --llm-expt-tp 1 \
  --llm-expt-dp 1 \
  --vocab-size 131072 \
  --num-image-tiles "${NUM_IMAGE_TILES}" \
  "${TOKENIZER_ARGS[@]}" \
  --tokenizer-prompt-format nemotron6-moe \
  --image-token "<image>" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --num-microbatches "${NUM_MICROBATCHES}" \
  --lr 2e-4 \
  --min-lr 2e-6 \
  --lr-decay-style cosine \
  --lr-warmup-iters "${LR_WARMUP_ITERS}" \
  --lr-decay-iters 10 \
  --weight-decay 0.05 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --ddp-bucket-size 0 \
  --train-iters "${TRAIN_ITERS}" \
  "$@"

#!/bin/bash

# Run an eight-rank heterogeneous mock training loop with Nemotron6-MoE VLM 20L.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

TRAIN_ITERS=${TRAIN_ITERS:-20}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
EVAL_ITERS=${EVAL_ITERS:-0}
MICRO_BATCH_SIZE=1
LLM_DP=2
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NUM_MICROBATCHES * LLM_DP))
TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR:-"${PWD}/logs/torchrun-$(date +%Y%m%d_%H%M%S)-$$"}
mkdir -p "${TORCHRUN_LOG_DIR}"

TORCHRUN_ARGS=(
  --standalone
  --nproc-per-node 8
  --log-dir "${TORCHRUN_LOG_DIR}"
  --redirects 3
  --tee 3
)

uv run --extra ssm python -m torch.distributed.run \
  "${TORCHRUN_ARGS[@]}" \
  -m examples.mimo.pretrain_mimo \
  --model-provider nemotron-moe-vlm \
  --dataset-provider mock \
  --image-token-id 511 \
  --dynamic-resolution \
  --pixel-shuffle \
  --disable-vision-class-token \
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
  --moe-token-dispatcher-type alltoall \
  --moe-permute-fusion \
  --use-fused-weighted-squared-relu \
  --mamba-num-heads 64 \
  --mamba-head-dim 64 \
  --mamba-num-groups 8 \
  --mamba-state-dim 128 \
  --linear-conv-kernel-dim 4 \
  --position-embedding-type none \
  --attention-backend flash \
  --calculate-per-token-loss \
  --cross-entropy-loss-fusion \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --bf16 \
  --encoder-tp 2 \
  --encoder-dp 2 \
  --llm-offset 4 \
  --llm-tp 2 \
  --llm-cp 1 \
  --llm-pp 1 \
  --llm-dp "${LLM_DP}" \
  --llm-ep 4 \
  --llm-expt-tp 1 \
  --vocab-size 131072 \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --lr 2e-4 \
  --min-lr 2e-6 \
  --lr-decay-style cosine \
  --lr-warmup-iters 0 \
  --lr-decay-iters 10 \
  --weight-decay 0.05 \
  --override-opt-param-scheduler \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --encoder-ddp-overlap \
  --train-iters "${TRAIN_ITERS}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-iters "${EVAL_ITERS}" \
  --log-interval 1 \
  --rerun-mode disabled \
  "$@"

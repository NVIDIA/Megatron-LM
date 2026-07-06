#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-m1}"
if [[ "${MODE}" != "m1" && "${MODE}" != "m1m2" ]]; then
  echo "Usage: $0 [m1|m1m2]" >&2
  exit 2
fi

if ! uv run python pretrain_gpt.py --help 2>/dev/null | grep -q -- "--report-theoretical-flops"; then
  echo "pretrain_gpt.py --help does not contain --report-theoretical-flops; check the synced commit." >&2
  exit 1
fi

export NVTE_DEBUG="${NVTE_DEBUG:-1}"
export NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-2}"

TRAIN_ITERS=2
PROFILE_ARGS=()
if [[ "${MODE}" == "m1m2" ]]; then
  TRAIN_ITERS=6
  PROFILE_ARGS=(
    --log-throughput
    --profile
    --use-pytorch-profiler
    --pytorch-profiler-collect-shapes
    --profile-step-start 2
    --profile-step-end 4
    --profile-ranks 0
  )
fi

uv run python -m torch.distributed.run \
  --nproc-per-node 8 \
  --nnodes 1 \
  --node-rank 0 \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" \
  --master-port "${MASTER_PORT:-29500}" \
  pretrain_gpt.py \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --report-theoretical-flops \
  --theoretical-flops-output-dir ./flops_analysis \
  --tensorboard-dir ./flops_analysis/tensorboard \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --num-layers 4 \
  --hidden-size 2048 \
  --ffn-hidden-size 6144 \
  --num-attention-heads 16 \
  --group-query-attention \
  --num-query-groups 8 \
  --kv-channels 128 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --position-embedding-type rope \
  --swiglu \
  --normalization RMSNorm \
  --disable-bias-linear \
  --micro-batch-size 2 \
  --global-batch-size 16 \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --bf16 \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --log-interval 1 \
  --eval-interval 1000 \
  --eval-iters 0 \
  "${PROFILE_ARGS[@]}" \
  --train-iters "${TRAIN_ITERS}"

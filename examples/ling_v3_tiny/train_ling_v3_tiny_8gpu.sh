#!/usr/bin/env bash

# Full Ling-V3 Tiny fresh-initialization smoke run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
MASTER_PORT="${MASTER_PORT:-29501}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
TRAIN_ITERS="${TRAIN_ITERS:-2}"
SAVE_ARGS=()

if [[ -n "${SAVE_DIR:-}" ]]; then
  mkdir -p "${SAVE_DIR}"
  SAVE_ARGS+=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000000000}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  pretrain_hybrid.py \
  --spec examples.ling_v3_tiny.model_spec hybrid_stack_spec \
  --hybrid-layer-pattern "K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E/+E" \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 157184 \
  --seq-length "${SEQ_LENGTH}" \
  --max-position-embeddings "${SEQ_LENGTH}" \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --train-iters "${TRAIN_ITERS}" \
  --eval-interval 1000000000 \
  --eval-iters 0 \
  --log-interval 1 \
  --seed 1234 \
  --bf16 \
  --transformer-impl transformer_engine \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --swiglu \
  --disable-bias-linear \
  --hidden-size 1536 \
  --num-attention-heads 16 \
  --kv-channels 128 \
  --ffn-hidden-size 4608 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --rotary-percent 0.5 \
  --rotary-base 6000000 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --context-parallel-size 2 \
  --linear-cp-mode headwise \
  --cp-comm-type p2p \
  --num-experts 128 \
  --moe-ffn-hidden-size 512 \
  --moe-shared-expert-intermediate-size 512 \
  --moe-router-topk 8 \
  --moe-router-score-function sigmoid \
  --moe-router-dtype fp32 \
  --moe-router-topk-scaling-factor 2.5 \
  --moe-router-num-groups 8 \
  --moe-router-group-topk 4 \
  --moe-router-enable-expert-bias \
  --moe-router-bias-update-rate 0.0 \
  --moe-router-load-balancing-type none \
  --moe-z-loss-coeff 2.9e-6 \
  --moe-token-dispatcher-type alltoall \
  --moe-grouped-gemm \
  --linear-conv-kernel-dim 4 \
  --linear-key-head-dim 128 \
  --linear-value-head-dim 128 \
  --linear-num-key-heads 16 \
  --linear-num-value-heads 16 \
  --kda-safe-gate \
  --kda-lower-bound -5.0 \
  --multi-latent-attention \
  --q-lora-rank 256 \
  --kv-lora-rank 512 \
  --qk-head-dim 128 \
  --qk-pos-emb-head-dim 64 \
  --v-head-dim 128 \
  --qk-layernorm \
  --attention-output-gate \
  --gated-attention-proj-granularity headwise \
  --mtp-num-layers 1 \
  --mtp-loss-scaling-factor 0.1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --lr 1e-4 \
  --min-lr 1e-4 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --distributed-backend nccl \
  "${SAVE_ARGS[@]}"

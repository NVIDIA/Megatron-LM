#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON="${MEGATRON:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
HF_TOKENIZER="${HF_TOKENIZER:-openai-community/gpt2}"
TB_ROOT="${TB_ROOT:-${MEGATRON}/tensorboard_runs}"
RUN_GROUP="${RUN_GROUP:-gpt_mock_baseline}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}_${RUN_STAMP}}"

mkdir -p "${TB_ROOT}"
cd "${MEGATRON}"

export PYTHONPATH="${MEGATRON}:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

echo "Writing TensorBoard logs to ${TB_ROOT}/${RUN_NAME}"

COMMON_ARGS=(
  --transformer-impl local
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-layers 2
  --hidden-size 256
  --ffn-hidden-size 1024
  --num-attention-heads 4
  --seq-length 128
  --max-position-embeddings 128
  --micro-batch-size 1
  --global-batch-size 1
  --train-iters 100
  --lr 1e-4
  --min-lr 1e-5
  --lr-decay-style cosine
  --weight-decay 0.01
  --clip-grad 1.0
  --mock-data
  --seed 1234
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${HF_TOKENIZER}"
  --eval-iters 0
  --eval-interval 1000
  --log-interval 1
  --tensorboard-dir "${TB_ROOT}/${RUN_NAME}"
)

torchrun --standalone --nproc_per_node=1 "${MEGATRON}/pretrain_gpt.py" "${COMMON_ARGS[@]}"

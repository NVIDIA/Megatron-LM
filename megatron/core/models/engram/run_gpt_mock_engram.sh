#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON="${MEGATRON:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
HF_TOKENIZER="${HF_TOKENIZER:-openai-community/gpt2}"
TB_ROOT="${TB_ROOT:-${MEGATRON}/tensorboard_runs}"
RUN_GROUP="${RUN_GROUP:-gpt_mock_engram}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_GROUP}_${RUN_STAMP}}"

mkdir -p "${TB_ROOT}"
cd "${MEGATRON}"

export PYTHONPATH="${MEGATRON}:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

echo "Writing TensorBoard logs to ${TB_ROOT}/${RUN_NAME}"

PAD_ID="$(python - <<PY
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("${HF_TOKENIZER}", trust_remote_code=True)
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
if pad_id is None:
    raise SystemExit("Tokenizer must define pad_token_id or eos_token_id for Engram.")
print(pad_id)
PY
)"

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

ENGRAM_ARGS=(
  --use-engram
  --engram-tokenizer-name-or-path "${HF_TOKENIZER}"
  --engram-vocab-size 31 37
  --engram-max-ngram-size 3
  --engram-n-embed-per-ngram 64
  --engram-n-head-per-ngram 4
  --engram-layer-ids 1 2
  --engram-pad-id "${PAD_ID}"
  --engram-seed 17
  --engram-kernel-size 5
  --engram-hc-mult 2
)

torchrun --standalone --nproc_per_node=1 "${MEGATRON}/pretrain_gpt.py" \
  "${COMMON_ARGS[@]}" \
  "${ENGRAM_ARGS[@]}"

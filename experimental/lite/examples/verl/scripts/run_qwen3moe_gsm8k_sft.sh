#!/usr/bin/env bash
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"

DATASET_DIR="${DATASET_DIR:-${HOME}/data/gsm8k_sft}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
export TRAIN_FILES="${TRAIN_FILES:-${DATASET_DIR}/train.parquet}"
export VAL_FILES="${VAL_FILES:-${DATASET_DIR}/test.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/qwen35_gsm8k_sft}"
export PROJECT_NAME="${PROJECT_NAME:-verl-mlite-qwen35-gsm8k-sft}"
export RUN_NAME="${RUN_NAME:-qwen35_gsm8k_sft_mlite}"

export TOTAL_STEPS="${TOTAL_STEPS:-100}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"

exec bash "${SCRIPT_DIR}/run_qwen3moe_sft.sh" "$@"

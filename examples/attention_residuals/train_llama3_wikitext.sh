#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
case "$MODE" in
    baseline)
        ATTENTION_RESIDUALS_VALUE=0
        ;;
    attnres)
        ATTENTION_RESIDUALS_VALUE=1
        ;;
    *)
        echo "Usage: $0 baseline|attnres" >&2
        exit 1
        ;;
esac

TOKENIZER_MODEL=${TOKENIZER_MODEL:-}
DATA_PREFIX=${DATA_PREFIX:-}
if [[ -z "$TOKENIZER_MODEL" ]]; then
    echo "Set TOKENIZER_MODEL=/path/to/tokenizer before running." >&2
    exit 1
fi
if [[ -z "$DATA_PREFIX" ]]; then
    echo "Set DATA_PREFIX=/path/to/megatron_text_document before running." >&2
    exit 1
fi

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints/attention_residuals}
LOG_ROOT=${LOG_ROOT:-tensorboard_logs/attention_residuals}
RUN_NAME=${RUN_NAME:-"${MODE}_layers${NUM_LAYERS:-16}_seq${SEQ_LENGTH:-1024}_$(date +'%y-%m-%d_%H-%M-%S')"}
CHECKPOINT_PATH="${CHECKPOINT_ROOT}/${RUN_NAME}"
TENSORBOARD_LOGS_PATH="${LOG_ROOT}/${RUN_NAME}"
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"benchmark_cache_attention_residuals/${RUN_NAME}"}

mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_LOGS_PATH" "$DATA_CACHE_PATH"

export ATTENTION_RESIDUALS="$ATTENTION_RESIDUALS_VALUE"
export ATTENTION_RESIDUAL_TYPE=${ATTENTION_RESIDUAL_TYPE:-full}
export ATTENTION_RESIDUAL_NUM_BLOCKS=${ATTENTION_RESIDUAL_NUM_BLOCKS:-8}
export ATTENTION_RESIDUAL_IMPLEMENTATION=${ATTENTION_RESIDUAL_IMPLEMENTATION:-triton_bwd}
export ATTENTION_RESIDUAL_RMSNORM=${ATTENTION_RESIDUAL_RMSNORM:-1}
export ATTENTION_RESIDUAL_LOG_WEIGHTS=${ATTENTION_RESIDUAL_LOG_WEIGHTS:-0}
export DATA_CACHE_PATH

bash examples/llama/train_llama3_8b_h100_fp8_2gpu_smoke.sh \
  "$CHECKPOINT_PATH" \
  "$TENSORBOARD_LOGS_PATH" \
  "$TOKENIZER_MODEL" \
  "$DATA_PREFIX"

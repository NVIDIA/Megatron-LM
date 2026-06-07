#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3.5 model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-1}
DRY_RUN=${DRY_RUN:-1}

COMMON_ARGS=(
  --hf-path "${HF_PATH}"
  --model-name qwen3_5
  --steps "${STEPS:-2}"
  --warmup "${WARMUP:-0}"
  --num-microbatches "${NUM_MICROBATCHES:-1}"
  --seq-len "${SEQ_LEN:-2048}"
  --truncate-layers "${TRUNCATE_LAYERS:-2}"
  --disable-mtp
)

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend mlite "${COMMON_ARGS[@]}" --dry-run
  "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend bridge "${COMMON_ARGS[@]}" --dry-run
else
  torchrun --nproc_per_node "${NPROC}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend mlite "${COMMON_ARGS[@]}" \
    --output-json "${REPO_ROOT}/experimental/lite/examples/bench/outputs/qwen35_mlite.json"
  torchrun --nproc_per_node "${NPROC}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend bridge "${COMMON_ARGS[@]}" \
    --output-json "${REPO_ROOT}/experimental/lite/examples/bench/outputs/qwen35_bridge.json"
fi

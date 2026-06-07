#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3.5 model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-1}
DRY_RUN=${DRY_RUN:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs"}

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  --hf-path "${HF_PATH}"
  --model-name qwen3_5
  --tp "${TP:-1}"
  --etp "${ETP:-1}"
  --ep "${EP:-1}"
  --pp "${PP:-1}"
  --cp "${CP:-1}"
  --steps "${STEPS:-2}"
  --warmup "${WARMUP:-0}"
  --num-microbatches "${NUM_MICROBATCHES:-1}"
  --seq-len "${SEQ_LEN:-2048}"
  --truncate-layers "${TRUNCATE_LAYERS:-2}"
  --disable-mtp
)

if [[ -n "${KEEP_EXPERTS:-}" ]]; then
  COMMON_ARGS+=(--keep-experts "${KEEP_EXPERTS}")
fi
if [[ "${SAME_DATA_ACROSS_DP:-0}" == "1" ]]; then
  COMMON_ARGS+=(--same-data-across-dp)
fi
if [[ "${SKIP_LOAD_HF_WEIGHTS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-load-hf-weights)
fi
if [[ "${SKIP_OPTIMIZER_BUILD:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-optimizer-build --no-optimizer)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend mlite "${COMMON_ARGS[@]}" --dry-run
  "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend bridge "${COMMON_ARGS[@]}" --dry-run
else
  mkdir -p "${OUTPUT_DIR}"
  MLITE_TORCHRUN_ARGS=(--nproc_per_node "${NPROC}")
  BRIDGE_TORCHRUN_ARGS=(--nproc_per_node "${NPROC}")
  if [[ -n "${MASTER_PORT:-}" ]]; then
    MLITE_TORCHRUN_ARGS+=(--master_port "${MASTER_PORT}")
  fi
  if [[ -n "${MASTER_PORT_BRIDGE:-}" ]]; then
    BRIDGE_TORCHRUN_ARGS+=(--master_port "${MASTER_PORT_BRIDGE}")
  fi
  torchrun "${MLITE_TORCHRUN_ARGS[@]}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend mlite "${COMMON_ARGS[@]}" \
    --output-json "${OUTPUT_DIR}/qwen35_mlite.json" \
    2>&1 | tee "${OUTPUT_DIR}/qwen35_mlite.log"
  torchrun "${BRIDGE_TORCHRUN_ARGS[@]}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    --backend bridge "${COMMON_ARGS[@]}" \
    --output-json "${OUTPUT_DIR}/qwen35_bridge.json" \
    2>&1 | tee "${OUTPUT_DIR}/qwen35_bridge.log"
fi

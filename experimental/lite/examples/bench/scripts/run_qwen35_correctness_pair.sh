#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3.5 model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs/correctness"}
REFERENCE_BACKEND=${REFERENCE_BACKEND:-bridge}

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MEGATRON_LITE_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --hf-path "${HF_PATH}"
  --model-name qwen3_5
  --tp "${TP:-1}"
  --etp "${ETP:-1}"
  --ep "${EP:-1}"
  --pp "${PP:-1}"
  --cp "${CP:-1}"
  --steps "${STEPS:-2}"
  --num-microbatches "${NUM_MICROBATCHES:-1}"
  --seq-len "${SEQ_LEN:-128}"
  --seed "${SEED:-42}"
  --truncate-layers "${TRUNCATE_LAYERS:-2}"
  --disable-mtp
  --same-data-across-dp
)

if [[ -n "${KEEP_EXPERTS:-}" ]]; then
  COMMON_ARGS+=(--keep-experts "${KEEP_EXPERTS}")
fi
if [[ "${SKIP_LOAD_HF_WEIGHTS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-load-hf-weights)
fi
if [[ "${SKIP_WEIGHT_HASH:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-weight-hash)
fi
if [[ -n "${ACTIVATION_PROBES_JSON:-}" ]]; then
  COMMON_ARGS+=(--activation-probes-json "${ACTIVATION_PROBES_JSON}")
fi

MLITE_TORCHRUN_ARGS=(--nproc_per_node "${NPROC}")
BRIDGE_TORCHRUN_ARGS=(--nproc_per_node "${NPROC}")
if [[ -n "${MASTER_PORT:-}" ]]; then
  MLITE_TORCHRUN_ARGS+=(--master_port "${MASTER_PORT}")
fi
if [[ -n "${MASTER_PORT_BRIDGE:-}" ]]; then
  BRIDGE_TORCHRUN_ARGS+=(--master_port "${MASTER_PORT_BRIDGE}")
fi

torchrun "${MLITE_TORCHRUN_ARGS[@]}" \
  "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" run \
  --backend mlite "${COMMON_ARGS[@]}" \
  --output-json "${OUTPUT_DIR}/qwen35_mlite_correctness.json" \
  2>&1 | tee "${OUTPUT_DIR}/qwen35_mlite_correctness.log"

torchrun "${BRIDGE_TORCHRUN_ARGS[@]}" \
  "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" run \
  --backend "${REFERENCE_BACKEND}" "${COMMON_ARGS[@]}" \
  --output-json "${OUTPUT_DIR}/qwen35_${REFERENCE_BACKEND}_correctness.json" \
  2>&1 | tee "${OUTPUT_DIR}/qwen35_${REFERENCE_BACKEND}_correctness.log"

"${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" compare \
  "${OUTPUT_DIR}/qwen35_mlite_correctness.json" \
  "${OUTPUT_DIR}/qwen35_${REFERENCE_BACKEND}_correctness.json" \
  --output-json "${OUTPUT_DIR}/qwen35_correctness_compare.json" \
  --fail-on-mismatch \
  2>&1 | tee "${OUTPUT_DIR}/qwen35_correctness_compare.log"

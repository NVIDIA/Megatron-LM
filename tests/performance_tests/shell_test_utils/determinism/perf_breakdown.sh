#!/bin/bash
# Repeated unprofiled timing, with an optional separate Nsight diagnostic run.
set -euo pipefail

OUT="${1:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
LOG_DIR="${2:?usage: $0 LEADERBOARD_DIR LOG_DIR}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

ARGS=(--output "$OUT/benchmark" --recipe "${DETERMINISM_PERF_RECIPE:-dense}"
      --gpus "${DETERMINISM_PERF_GPUS:-8}" --pairs "${DETERMINISM_PERF_PAIRS:-3}"
      --warmup "${DETERMINISM_PERF_WARMUP:-20}" --steps "${DETERMINISM_PERF_STEPS:-50}")
if [[ -n "${DETERMINISM_PERF_BASE_CHECKOUT:-}" ]]; then
    ARGS+=(--base-checkout "$DETERMINISM_PERF_BASE_CHECKOUT")
fi
uv run --no-sync python "$SCRIPT_DIR/benchmark.py" "${ARGS[@]}"

if [[ "${DETERMINISM_PERF_PROFILE:-0}" == "1" ]]; then
    # nvtx_sum is a host-range diagnostic, not the timing gate.
    env -u LOG_DIR DETERMINISM_PERF_LOG_DIR="$LOG_DIR/profile" \
      DETERMINISM_PERF_TRAIN_ITERS=8 DETERMINISM_PERF_PROFILE=1 \
      bash "$SCRIPT_DIR/run_nsys_breakdown.sh" "$OUT/profile" -- \
      uv run --no-sync python "$SCRIPT_DIR/run_training.py"
fi

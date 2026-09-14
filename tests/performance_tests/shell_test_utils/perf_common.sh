# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Shared helpers for the inference perf test drivers (run_perf_test.sh for
# model-serving cases, run_frontend_perf_test.sh for the model-free frontend
# capacity case). Sourced, not executed.

# Echo the platform key whose baseline subtree this run reads or writes.
#
# baseline_values.json is a {platform: {batch_key: {metrics}}} mapping, so this
# decides which numbers a run is compared against. Honours a caller-provided
# value; otherwise identifies the node from its first GPU.
#
# Args:
#   $1: caller-provided PLATFORM, or empty to auto-detect.
#   $2: optional fallback used when no GPU is visible. Pass a value only for
#       tests that do not need a GPU; without it, undetectable is an error,
#       since silently comparing an H100 baseline on a GB200 node is worse than
#       failing.
detect_platform() {
    local PROVIDED="${1:-}"
    local FALLBACK="${2:-}"

    if [[ -n "$PROVIDED" ]]; then
        echo "[perf] using caller-provided PLATFORM=$PROVIDED" >&2
        echo "$PROVIDED"
        return 0
    fi

    local GPU_NAME
    GPU_NAME=$(nvidia-smi -L 2>/dev/null | head -1 || true)
    local DETECTED=""
    case "$GPU_NAME" in
        *GB200*|*"Grace Blackwell"*) DETECTED=gb200 ;;
        *B200*)                      DETECTED=b200  ;;
        *H100*)                      DETECTED=h100  ;;
        *A100*)                      DETECTED=a100  ;;
    esac

    if [[ -n "$DETECTED" ]]; then
        echo "[perf] auto-detected PLATFORM=$DETECTED from \"$GPU_NAME\"" >&2
        echo "$DETECTED"
        return 0
    fi

    if [[ -n "$FALLBACK" ]]; then
        echo "[perf] no GPU detected; falling back to PLATFORM=$FALLBACK" >&2
        echo "$FALLBACK"
        return 0
    fi

    echo "[perf] error: could not auto-detect PLATFORM from nvidia-smi (\"$GPU_NAME\")." >&2
    echo "       Pass PLATFORM=<h100|gb200|b200|a100> explicitly." >&2
    return 2
}

# Compare a results.json against the checked-in baseline, or record it as the
# new baseline for this platform.
#
# Honours RECORD_BASELINE=1 (merge results into the baseline's platform subtree,
# leaving other platforms intact) and SKIP_COMPARE=1 (do neither).
#
# Args:
#   $1: path to results.json produced by this run.
#   $2: path to the test case's model_config.yaml.
#   $3: platform key.
compare_or_record() {
    local RESULTS_JSON="$1"
    local CONFIG_PATH="$2"
    local PLATFORM="$3"
    local CASE_DIR BASELINE_PATH PERF_DIR
    CASE_DIR="$(dirname "$CONFIG_PATH")"
    BASELINE_PATH="$CASE_DIR/baseline_values.json"
    PERF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    if [[ "${RECORD_BASELINE:-0}" == "1" ]]; then
        echo "[perf] RECORD_BASELINE=1 → merging results.json into $BASELINE_PATH under key '$PLATFORM'"
        uv run --no-sync python - "$RESULTS_JSON" "$BASELINE_PATH" "$PLATFORM" <<'PY'
import json
import sys
from pathlib import Path

results_path, baseline_path, platform = sys.argv[1], sys.argv[2], sys.argv[3]
results = json.loads(Path(results_path).read_text())
baseline = {}
if Path(baseline_path).exists():
    baseline = json.loads(Path(baseline_path).read_text())
# Merge: overwrite only the current platform's subtree, leave others intact.
baseline[platform] = results
Path(baseline_path).write_text(json.dumps(baseline, indent=2) + "\n")
print(f"[perf] wrote {len(results)} entries under '{platform}' "
      f"({sorted(baseline.keys())} platforms recorded total)")
PY
        return 0
    fi

    if [[ "${SKIP_COMPARE:-0}" == "1" ]]; then
        echo "[perf] SKIP_COMPARE=1 → not running baseline comparison"
        return 0
    fi

    if [[ ! -f "$BASELINE_PATH" ]]; then
        echo "[perf] error: no baseline_values.json at $BASELINE_PATH." >&2
        echo "       Run once with RECORD_BASELINE=1 to bootstrap." >&2
        return 3
    fi

    uv run --no-sync python "$PERF_DIR/shell_test_utils/compare_to_baseline.py" \
        --results "$RESULTS_JSON" \
        --baseline "$BASELINE_PATH" \
        --config "$CONFIG_PATH" \
        --platform "$PLATFORM"
}

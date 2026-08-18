#!/usr/bin/env bash
# Run the model-free frontend/coordinator capacity test.
#
# Unlike run_perf_test.sh this launches no server, loads no checkpoint and needs
# no GPU: frontend_capacity_benchmark.py starts its own coordinator process and
# fake engines. What it measures is how many requests per second the HTTP
# frontend, the InferenceClient and the coordinator can move, so a regression
# here is a regression in that code and nothing else.
#
# Invoked by `cog submit` (or locally) with KEY=VALUE positional args.
#
# Required:
#   CONFIG_PATH=tests/performance_tests/test_cases/frontend/<case>/model_config.yaml
#   RESULTS_ROOT=/path/where/results.json/goes
#
# Optional:
#   RECORD_BASELINE=1   (merge results.json into baseline_values.json under PLATFORM)
#   SKIP_COMPARE=1      (skip the comparison step entirely)
#   PLATFORM=<name>     (override the baseline key; defaults to GPU-based detection,
#                        falling back to cpu_<arch> when no GPU is visible. The GPU
#                        does not participate, but its name identifies the node type,
#                        and CPU and NIC differences between node types do move these
#                        numbers.)
#
# Expects /usr/local/bin/yq (present in mcore_ci_dev image).

set -euo pipefail

for ARG in "$@"; do
    if [[ "$ARG" != *=* ]]; then
        echo "[run_frontend_perf_test] error: arg '$ARG' is not KEY=VALUE" >&2
        exit 2
    fi
    KEY="${ARG%%=*}"
    VAL="${ARG#*=}"
    export "$KEY"="$VAL"
done

: "${CONFIG_PATH:?CONFIG_PATH (path to model_config.yaml) is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"

source "$(dirname "${BASH_SOURCE[0]}")/perf_common.sh"

PLATFORM=$(detect_platform "${PLATFORM:-}" "cpu_$(uname -m)")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PERF_DIR="$ROOT_DIR/tests/performance_tests"
YQ=/usr/local/bin/yq

mkdir -p "$RESULTS_ROOT"
RESULTS_JSON="$RESULTS_ROOT/results.json"
# launch_jet_workload.py retries unless torchrun's per-rank std*.log assets exist.
ASSETS_ROOT="$(dirname "$RESULTS_ROOT")"
mkdir -p "$ASSETS_ROOT/logs/1"
rm -f "$RESULTS_JSON"

# ── Read model_config.yaml ────────────────────────────────────────────────────

CONCURRENCY=$("$YQ" '.CONCURRENCY // "1,8,32,128,512"' "$CONFIG_PATH")
SECONDS_PER_LEVEL=$("$YQ" '.SECONDS_PER_LEVEL // 5' "$CONFIG_PATH")
WARMUP_SECONDS=$("$YQ" '.WARMUP_SECONDS // 2' "$CONFIG_PATH")
NUM_INPUT_TOKENS=$("$YQ" '.NUM_INPUT_TOKENS // 512' "$CONFIG_PATH")
NUM_OUTPUT_TOKENS=$("$YQ" '.NUM_OUTPUT_TOKENS // 64' "$CONFIG_PATH")
NUM_ENGINES=$("$YQ" '.NUM_ENGINES // 1' "$CONFIG_PATH")
ENGINE_LATENCY_MS=$("$YQ" '.ENGINE_LATENCY_MS // 10' "$CONFIG_PATH")
MAX_REQUESTS=$("$YQ" '.MAX_REQUESTS // 1024' "$CONFIG_PATH")
# Each entry is "<mode>[:stream]" — e.g. "client", "http", "http:stream".
mapfile -t PATHS < <("$YQ" '.PATHS[]' "$CONFIG_PATH")

echo "[run_frontend_perf_test] PLATFORM=$PLATFORM  paths=${PATHS[*]}"
echo "[run_frontend_perf_test] concurrency=$CONCURRENCY  ISL=$NUM_INPUT_TOKENS  OSL=$NUM_OUTPUT_TOKENS"
echo "[run_frontend_perf_test] engines=$NUM_ENGINES  engine_latency=${ENGINE_LATENCY_MS}ms"

# ── Sweep every requested path ────────────────────────────────────────────────

for PATH_SPEC in "${PATHS[@]}"; do
    MODE="${PATH_SPEC%%:*}"
    STREAM_ARGS=()
    if [[ "$PATH_SPEC" == *:stream ]]; then
        STREAM_ARGS=(--streaming)
    fi
    echo "[run_frontend_perf_test] === path $PATH_SPEC ==="
    (
        cd "$ROOT_DIR"
        uv run --no-sync python "$PERF_DIR/client/frontend_capacity_benchmark.py" \
            --mode "$MODE" \
            "${STREAM_ARGS[@]}" \
            --concurrency "$CONCURRENCY" \
            --seconds-per-level "$SECONDS_PER_LEVEL" \
            --warmup-seconds "$WARMUP_SECONDS" \
            --num-input-tokens "$NUM_INPUT_TOKENS" \
            --num-output-tokens "$NUM_OUTPUT_TOKENS" \
            --num-engines "$NUM_ENGINES" \
            --engine-latency-ms "$ENGINE_LATENCY_MS" \
            --max-requests "$MAX_REQUESTS" \
            --output-json "$RESULTS_JSON"
    ) 2>&1 | tee -a "$RESULTS_ROOT/benchmark.log"
done

echo "[run_frontend_perf_test] benchmark complete. Results written to $RESULTS_JSON"

# ── Baseline comparison or recording ──────────────────────────────────────────

compare_or_record "$RESULTS_JSON" "$CONFIG_PATH" "$PLATFORM"

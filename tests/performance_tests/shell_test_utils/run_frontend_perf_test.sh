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
# Expects PyYAML in the active Megatron-LM Python environment.

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

yaml_scalar() {
    python -c 'import sys, yaml; value = yaml.safe_load(open(sys.argv[1])).get(sys.argv[2]); print(sys.argv[3] if value is None else value)' \
        "$CONFIG_PATH" "$1" "$2"
}

mkdir -p "$RESULTS_ROOT"
RESULTS_JSON="$RESULTS_ROOT/results.json"
# launch_jet_workload.py retries unless torchrun's per-rank std*.log assets exist.
ASSETS_ROOT="$(dirname "$RESULTS_ROOT")"
mkdir -p "$ASSETS_ROOT/logs/1"
rm -f "$RESULTS_JSON"

# ── Read model_config.yaml ────────────────────────────────────────────────────

CONCURRENCY=$(yaml_scalar CONCURRENCY "1,8,32,128,512")
SECONDS_PER_LEVEL=$(yaml_scalar SECONDS_PER_LEVEL 5)
WARMUP_SECONDS=$(yaml_scalar WARMUP_SECONDS 2)
NUM_INPUT_TOKENS=$(yaml_scalar NUM_INPUT_TOKENS 512)
NUM_OUTPUT_TOKENS=$(yaml_scalar NUM_OUTPUT_TOKENS 64)
NUM_ENGINES=$(yaml_scalar NUM_ENGINES 1)
ENGINE_LATENCY_MS=$(yaml_scalar ENGINE_LATENCY_MS 10)
MAX_REQUESTS=$(yaml_scalar MAX_REQUESTS 1024)
LONG_SEQUENCE_LENGTHS=$(yaml_scalar LONG_SEQUENCE_LENGTHS "")
LONG_SEQUENCE_CONCURRENCY=$(yaml_scalar LONG_SEQUENCE_CONCURRENCY 1)
LONG_SEQUENCE_ENGINE_LATENCY_MS=$(yaml_scalar LONG_SEQUENCE_ENGINE_LATENCY_MS 0)
# Each entry is "<mode>[:stream]" — e.g. "client", "http", "http:stream".
mapfile -t PATHS < <(
    python -c 'import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["PATHS"]))' \
        "$CONFIG_PATH"
)

# Each entry is "input_tokens|concurrency|engine_latency_ms". The regular
# capacity sweep retains high concurrency and a simulated engine delay. Long
# prompts run one at a time with no fake compute so the measured time isolates
# frontend/coordinator payload processing.
SWEEPS=("$NUM_INPUT_TOKENS|$CONCURRENCY|$ENGINE_LATENCY_MS")
if [[ -n "$LONG_SEQUENCE_LENGTHS" ]]; then
    IFS=',' read -r -a LONG_INPUT_LENGTHS <<< "$LONG_SEQUENCE_LENGTHS"
    for ISL in "${LONG_INPUT_LENGTHS[@]}"; do
        SWEEPS+=("$ISL|$LONG_SEQUENCE_CONCURRENCY|$LONG_SEQUENCE_ENGINE_LATENCY_MS")
    done
fi

echo "[run_frontend_perf_test] PLATFORM=$PLATFORM  paths=${PATHS[*]}"
echo "[run_frontend_perf_test] sweeps=${SWEEPS[*]}  OSL=$NUM_OUTPUT_TOKENS"
echo "[run_frontend_perf_test] engines=$NUM_ENGINES"

# ── Sweep every requested path and input length ───────────────────────────────

for PATH_SPEC in "${PATHS[@]}"; do
    MODE="${PATH_SPEC%%:*}"
    STREAM_ARGS=()
    if [[ "$PATH_SPEC" == *:stream ]]; then
        STREAM_ARGS=(--streaming)
    fi
    for SWEEP in "${SWEEPS[@]}"; do
        IFS='|' read -r ISL SWEEP_CONCURRENCY SWEEP_ENGINE_LATENCY_MS <<< "$SWEEP"
        echo "[run_frontend_perf_test] === path $PATH_SPEC  ISL $ISL  concurrency $SWEEP_CONCURRENCY ==="
        (
            cd "$ROOT_DIR"
            uv run --no-sync python "$PERF_DIR/client/frontend_capacity_benchmark.py" \
                --mode "$MODE" \
                "${STREAM_ARGS[@]}" \
                --concurrency "$SWEEP_CONCURRENCY" \
                --seconds-per-level "$SECONDS_PER_LEVEL" \
                --warmup-seconds "$WARMUP_SECONDS" \
                --num-input-tokens "$ISL" \
                --num-output-tokens "$NUM_OUTPUT_TOKENS" \
                --num-engines "$NUM_ENGINES" \
                --engine-latency-ms "$SWEEP_ENGINE_LATENCY_MS" \
                --max-requests "$MAX_REQUESTS" \
                --output-json "$RESULTS_JSON"
        ) 2>&1 | tee -a "$RESULTS_ROOT/benchmark.log"
    done
done

echo "[run_frontend_perf_test] benchmark complete. Results written to $RESULTS_JSON"

# ── Baseline comparison or recording ──────────────────────────────────────────

compare_or_record "$RESULTS_JSON" "$CONFIG_PATH" "$PLATFORM"

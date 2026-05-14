#!/usr/bin/env bash
# Run an inference performance test.
#
# Invoked by `cog submit` (or local equivalent) with KEY=VALUE positional args,
# mirroring tests/functional_tests/shell_test_utils/run_ci_test.sh.
#
# Required:
#   CONFIG_PATH=tests/performance_tests/test_cases/<model>/<case>/model_config.yaml
#   CHECKPOINT_LOAD_PATH=/lustre/.../mcore_ci
#   RESULTS_ROOT=/path/where/results.json/and/server-logs/go
#
# Optional:
#   RECORD_BASELINE=1   (skip the comparison; copy results.json over baseline_values.json)
#   SKIP_COMPARE=1      (skip the comparison step entirely)
#
# Expects /usr/local/bin/yq (present in mcore_ci_dev image, NOT in bare NGC PyTorch).

set -euo pipefail

# ── Parse KEY=VALUE positional args ───────────────────────────────────────────

for ARG in "$@"; do
    if [[ "$ARG" != *=* ]]; then
        echo "[run_perf_test] error: arg '$ARG' is not KEY=VALUE" >&2
        exit 2
    fi
    KEY="${ARG%%=*}"
    VAL="${ARG#*=}"
    export "$KEY"="$VAL"
done

: "${CONFIG_PATH:?CONFIG_PATH (path to model_config.yaml) is required}"
: "${CHECKPOINT_LOAD_PATH:?CHECKPOINT_LOAD_PATH is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PERF_DIR="$ROOT_DIR/tests/performance_tests"
YQ=/usr/local/bin/yq

mkdir -p "$RESULTS_ROOT"
RESULTS_JSON="$RESULTS_ROOT/results.json"
SERVER_LOG_DIR="$RESULTS_ROOT/server_logs"
mkdir -p "$SERVER_LOG_DIR"
# Clean any pre-existing results.json so this run starts fresh.
rm -f "$RESULTS_JSON"

# ── Read model_config.yaml ────────────────────────────────────────────────────

MODEL=$("$YQ" '.MODEL' "$CONFIG_PATH")
TP=$("$YQ" '.TP // 1' "$CONFIG_PATH")
PP=$("$YQ" '.PP // 1' "$CONFIG_PATH")
DP=$("$YQ" '.DP // 1' "$CONFIG_PATH")
NUM_INPUT_TOKENS=$("$YQ" '.NUM_INPUT_TOKENS' "$CONFIG_PATH")
NUM_OUTPUT_TOKENS=$("$YQ" '.NUM_OUTPUT_TOKENS' "$CONFIG_PATH")
NUM_WARMUP_ITERS=$("$YQ" '.NUM_WARMUP_ITERS // 2' "$CONFIG_PATH")
NUM_TIMED_ITERS=$("$YQ" '.NUM_TIMED_ITERS // 5' "$CONFIG_PATH")
mapfile -t BATCH_SIZES < <("$YQ" '.BATCH_SIZES[]' "$CONFIG_PATH")

WORLD_SIZE=$((TP * PP * DP))
ARGS_FILE="$PERF_DIR/server/model_args/${MODEL}.args"
if [[ ! -f "$ARGS_FILE" ]]; then
    echo "[run_perf_test] error: model args file $ARGS_FILE not found" >&2
    exit 2
fi

echo "[run_perf_test] MODEL=$MODEL  TP=$TP PP=$PP DP=$DP  world_size=$WORLD_SIZE"
echo "[run_perf_test] ISL=$NUM_INPUT_TOKENS  OSL=$NUM_OUTPUT_TOKENS"
echo "[run_perf_test] batch sizes: ${BATCH_SIZES[*]}"

# ── Build model args (substituting ${CHECKPOINT_LOAD_PATH}) ───────────────────

MODEL_ARGS=()
while IFS= read -r LINE; do
    # Skip comments and blanks.
    [[ -z "$LINE" || "$LINE" =~ ^[[:space:]]*# ]] && continue
    # Expand variable references like ${CHECKPOINT_LOAD_PATH}.
    EXPANDED=$(eval echo "$LINE")
    for TOK in $EXPANDED; do
        MODEL_ARGS+=("$TOK")
    done
done < "$ARGS_FILE"

# Override TP/PP from config (the args file ships defaults; config wins).
MODEL_ARGS+=(--tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP")

# ── Make image-bundled extras (mamba-ssm) visible to the cog venv ─────────────
# cog's auto-managed venv uses `uv sync --extra dev --extra mlm` and inherits
# from /usr/lib/python3.12/dist-packages — but mamba-ssm + causal-conv1d are
# bundled in the image's prebuilt /opt/venv, NOT in system dist-packages.
# For hybrid/mamba models, prepend /opt/venv's site-packages to PYTHONPATH so
# the server can import mamba_ssm without re-installing.

EXTRA_PYTHONPATH=""
if [[ "$MODEL" == hybrid_* || "$MODEL" == mamba_* ]]; then
    OPT_VENV_SITE=/opt/venv/lib/python3.12/site-packages
    if [[ -d "$OPT_VENV_SITE" ]]; then
        echo "[run_perf_test] adding $OPT_VENV_SITE to PYTHONPATH (mamba-ssm shim)"
        EXTRA_PYTHONPATH="$OPT_VENV_SITE"
    else
        echo "[run_perf_test] warning: $OPT_VENV_SITE not present — hybrid model may fail to load" >&2
    fi
fi

# ── Launch the inference server in the background ─────────────────────────────

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
SERVER_PORT=${SERVER_PORT:-5000}
SERVER_LOG="$SERVER_LOG_DIR/server.log"

echo "[run_perf_test] starting server: torchrun (--nproc-per-node $WORLD_SIZE) tools.run_dynamic_text_generation_server"
echo "[run_perf_test] server log: $SERVER_LOG"

# Server-side args common to all models.
SERVER_COMMON_ARGS=(
    --micro-batch-size 1
    --inference-rng-tracker
    --inference-dynamic-batching
    --inference-dynamic-batching-buffer-size-gb 20
    --inference-max-seq-length $((NUM_INPUT_TOKENS + NUM_OUTPUT_TOKENS + 32))
    --inference-max-requests 256
    --port "$SERVER_PORT"
    --host 0.0.0.0
)

(
    cd "$ROOT_DIR"
    if [[ -n "$EXTRA_PYTHONPATH" ]]; then
        export PYTHONPATH="$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
    fi
    uv run --no-sync python -m torch.distributed.run \
        --nproc-per-node "$WORLD_SIZE" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        -m tools.run_dynamic_text_generation_server \
        "${MODEL_ARGS[@]}" \
        "${SERVER_COMMON_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PGID=$!

cleanup() {
    local EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 && -s "$SERVER_LOG" ]]; then
        echo ""
        echo "========================================================================"
        echo "[run_perf_test] non-zero exit ($EXIT_CODE) — dumping last 200 lines of server.log:"
        echo "========================================================================"
        tail -200 "$SERVER_LOG" || true
        echo "========================================================================"
    fi
    if kill -0 "$SERVER_PGID" 2>/dev/null; then
        echo "[run_perf_test] killing server pid=$SERVER_PGID"
        kill -- -"$SERVER_PGID" 2>/dev/null || kill "$SERVER_PGID" 2>/dev/null || true
        sleep 5
        kill -9 -- -"$SERVER_PGID" 2>/dev/null || kill -9 "$SERVER_PGID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Wait for server readiness ─────────────────────────────────────────────────

echo "[run_perf_test] waiting for server on http://localhost:$SERVER_PORT/v1/health ..."
DEADLINE=$(( $(date +%s) + 900 ))   # 15 min
until curl -fsS "http://localhost:$SERVER_PORT/v1/health" >/dev/null 2>&1; do
    if (( $(date +%s) > DEADLINE )); then
        echo "[run_perf_test] error: server did not become ready within 15 min" >&2
        echo "[run_perf_test] last 100 lines of server log:" >&2
        tail -100 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if ! kill -0 "$SERVER_PGID" 2>/dev/null; then
        echo "[run_perf_test] error: server process died before becoming ready" >&2
        tail -200 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 5
done
echo "[run_perf_test] server is up."

# ── Benchmark sweep ───────────────────────────────────────────────────────────

for BS in "${BATCH_SIZES[@]}"; do
    echo "[run_perf_test] === batch size $BS ==="
    uv run --no-sync python "$PERF_DIR/client/static_benchmark.py" \
        --server-url "http://localhost:$SERVER_PORT/v1" \
        --model "$MODEL" \
        --batch-size "$BS" \
        --num-input-tokens "$NUM_INPUT_TOKENS" \
        --num-output-tokens "$NUM_OUTPUT_TOKENS" \
        --num-warmup-iters "$NUM_WARMUP_ITERS" \
        --num-iters "$NUM_TIMED_ITERS" \
        --output-json "$RESULTS_JSON" \
        2>&1 | tee -a "$RESULTS_ROOT/benchmark.log"
done

echo "[run_perf_test] benchmark complete. Results written to $RESULTS_JSON"

# ── Baseline comparison or recording ──────────────────────────────────────────

CASE_DIR="$(dirname "$CONFIG_PATH")"
BASELINE_PATH="$CASE_DIR/baseline_values.json"

if [[ "${RECORD_BASELINE:-0}" == "1" ]]; then
    echo "[run_perf_test] RECORD_BASELINE=1 → copying results.json over $BASELINE_PATH"
    cp "$RESULTS_JSON" "$BASELINE_PATH"
    exit 0
fi

if [[ "${SKIP_COMPARE:-0}" == "1" ]]; then
    echo "[run_perf_test] SKIP_COMPARE=1 → not running baseline comparison"
    exit 0
fi

if [[ ! -f "$BASELINE_PATH" ]]; then
    echo "[run_perf_test] error: no baseline_values.json at $BASELINE_PATH." >&2
    echo "                  Run once with RECORD_BASELINE=1 to bootstrap." >&2
    exit 3
fi

uv run --no-sync python "$PERF_DIR/shell_test_utils/compare_to_baseline.py" \
    --results "$RESULTS_JSON" \
    --baseline "$BASELINE_PATH" \
    --config "$CONFIG_PATH"

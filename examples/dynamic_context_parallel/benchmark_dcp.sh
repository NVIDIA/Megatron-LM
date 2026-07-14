#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
PYTHON=${PYTHON:-python}

TP_SIZE=${TP_SIZE:-1}
CP_SIZE=${CP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}

NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-8192}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-16}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-8}
TRAIN_ITERS=${TRAIN_ITERS:-30}
WARMUP_ITERS=${WARMUP_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-1234}

MAX_SEQLEN_PER_DP_CP_RANK=${MAX_SEQLEN_PER_DP_CP_RANK:-2048}
MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE=${MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE:-1}

VOCAB_SIZE=${VOCAB_SIZE:-131072}
NULL_TOKENIZER_PAD_ID=${NULL_TOKENIZER_PAD_ID:-0}

OUTPUT_DIR=${OUTPUT_DIR:-"${PWD}/dcp_benchmark_output"}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${OUTPUT_DIR}/data_cache"}
DEFAULT_VARLEN_DATASET_JSON='{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":128,"max_seq_len":8192,"mean_seq_len":1024,"lognormal_sigma":1.5}'
VARLEN_DATASET_JSON=${VARLEN_DATASET_JSON:-$DEFAULT_VARLEN_DATASET_JSON}

DATASET_PATH=${DATASET_PATH:-}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen3-30B-A3B}
LOAD_PATH=${LOAD_PATH:-}
LOSS_ATOL=${LOSS_ATOL:-1e-3}
LOSS_RTOL=${LOSS_RTOL:-1e-3}
CHECK_LOSS_PARITY=${CHECK_LOSS_PARITY:-1}
ANALYZE_ONLY=${ANALYZE_ONLY:-0}
CASE_ORDER=${CASE_ORDER:-baseline_first}

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"
MANIFEST_PATH="${OUTPUT_DIR}/reproducibility_manifest.txt"

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_positive_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer, got '${value}'."
}

require_nonnegative_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer, got '${value}'."
}

require_nonnegative_finite_number() {
    local name=$1
    local value=$2
    "$PYTHON" - "$name" "$value" <<'PY'
import math
import sys

name, raw_value = sys.argv[1:]
try:
    value = float(raw_value)
except ValueError:
    value = math.nan
if not math.isfinite(value) or value < 0:
    print(f"Error: {name} must be a finite non-negative number, got {raw_value!r}.", file=sys.stderr)
    raise SystemExit(1)
PY
}

is_power_of_two() {
    local value=$1
    (( value > 0 && (value & (value - 1)) == 0 ))
}

describe_path() {
    local value=$1
    local resolved
    local details

    if [[ -z "$value" ]]; then
        printf 'none'
    elif [[ -e "$value" ]]; then
        resolved=$(realpath -- "$value" 2>/dev/null || printf '%s' "$value")
        details=$(stat -c 'size=%s,mtime=%y' -- "$value" 2>/dev/null || printf 'metadata=unavailable')
        printf '%s (%s)' "$resolved" "$details"
    else
        # Hub identifiers are recorded verbatim. Do not record credentials or the process environment.
        printf '%s' "${value%%\?*}"
    fi
}

write_reproducibility_manifest() {
    local git_sha="unavailable"
    local git_dirty="unknown"
    local dataset_identity
    local checkpoint_identity
    local gpu_info

    git_sha=$(git rev-parse --verify HEAD 2>/dev/null || printf 'unavailable')
    if git diff --quiet --ignore-submodules HEAD -- 2>/dev/null; then
        git_dirty="false"
    else
        git_dirty="true"
    fi

    if [[ -n "$DATASET_PATH" ]]; then
        dataset_identity=$(describe_path "$DATASET_PATH")
    else
        dataset_identity="mock:${VARLEN_DATASET_JSON}"
    fi
    if [[ -n "$LOAD_PATH" ]]; then
        checkpoint_identity=$(describe_path "$LOAD_PATH")
    else
        checkpoint_identity="seeded-random-initialization(seed=${SEED})"
    fi

    {
        printf 'git_sha=%s\n' "$git_sha"
        printf 'git_dirty=%s\n' "$git_dirty"
        printf 'system=%s\n' "$(uname -srmo 2>/dev/null || printf 'unavailable')"
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu_info=$(
                nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
                    2>/dev/null | paste -sd ';' - || true
            )
            printf 'gpus=%s\n' "${gpu_info:-unavailable}"
        else
            printf 'gpus=unavailable\n'
        fi
        "$PYTHON" - <<'PY'
import importlib.metadata
import platform

print(f"python={platform.python_version()}")
for package in ("torch", "transformer-engine", "transformers", "datasets"):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"{package}={version}")
try:
    import torch

    print(f"torch_cuda={torch.version.cuda}")
except ImportError:
    print("torch_cuda=unavailable")
PY
        printf 'topology=world:%s,dp:%s,cp:%s,tp:%s,pp:%s\n' \
            "$WORLD_SIZE" "$DP_SIZE" "$CP_SIZE" "$TP_SIZE" "$PP_SIZE"
        printf 'model=layers:%s,hidden:%s,ffn:%s,heads:%s,seq:%s,max_position:%s,bf16:true\n' \
            "$NUM_LAYERS" "$HIDDEN_SIZE" "$FFN_HIDDEN_SIZE" "$NUM_ATTENTION_HEADS" \
            "$SEQ_LENGTH" "$MAX_POSITION_EMBEDDINGS"
        printf 'training=micro_batch:%s,global_batch:%s,microbatches:%s,iters:%s,warmup:%s,seed:%s\n' \
            "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$EFFECTIVE_NUM_MICROBATCHES" \
            "$TRAIN_ITERS" "$WARMUP_ITERS" "$SEED"
        printf 'packing=max_tokens_per_rank:%s,min_dcp_size:%s,cuda_graph_impl:none\n' \
            "$MAX_SEQLEN_PER_DP_CP_RANK" "$MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE"
        printf 'schedulers=baseline:dp_balanced,dcp:default_dynamic_cp\n'
        printf 'dataset=%s\n' "$dataset_identity"
        printf 'tokenizer=%s\n' "$TOKENIZER_MODEL"
        printf 'checkpoint=%s\n' "$checkpoint_identity"
        printf 'comparison=order:%s,loss_atol:%s,loss_rtol:%s,nvte_nondeterministic:%s\n' \
            "$CASE_ORDER" "$LOSS_ATOL" "$LOSS_RTOL" "$NVTE_ALLOW_NONDETERMINISTIC_ALGO"
        printf 'runtime=cuda_device_max_connections:%s\n' "$CUDA_DEVICE_MAX_CONNECTIONS"
    } > "$MANIFEST_PATH"

    echo "Reproducibility manifest: ${MANIFEST_PATH}"
}

require_positive_integer GPUS_PER_NODE "$GPUS_PER_NODE"
require_positive_integer NUM_NODES "$NUM_NODES"
require_nonnegative_integer NODE_RANK "$NODE_RANK"
require_positive_integer TP_SIZE "$TP_SIZE"
require_positive_integer CP_SIZE "$CP_SIZE"
require_positive_integer PP_SIZE "$PP_SIZE"
require_positive_integer NUM_LAYERS "$NUM_LAYERS"
require_positive_integer HIDDEN_SIZE "$HIDDEN_SIZE"
require_positive_integer FFN_HIDDEN_SIZE "$FFN_HIDDEN_SIZE"
require_positive_integer NUM_ATTENTION_HEADS "$NUM_ATTENTION_HEADS"
require_positive_integer SEQ_LENGTH "$SEQ_LENGTH"
require_positive_integer MAX_POSITION_EMBEDDINGS "$MAX_POSITION_EMBEDDINGS"
require_positive_integer MICRO_BATCH_SIZE "$MICRO_BATCH_SIZE"
require_positive_integer NUM_MICROBATCHES "$NUM_MICROBATCHES"
require_positive_integer TRAIN_ITERS "$TRAIN_ITERS"
require_nonnegative_integer WARMUP_ITERS "$WARMUP_ITERS"
require_positive_integer LOG_INTERVAL "$LOG_INTERVAL"
require_nonnegative_integer NUM_WORKERS "$NUM_WORKERS"
require_nonnegative_integer SEED "$SEED"
require_positive_integer MAX_SEQLEN_PER_DP_CP_RANK "$MAX_SEQLEN_PER_DP_CP_RANK"
require_positive_integer MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE "$MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE"
require_positive_integer VOCAB_SIZE "$VOCAB_SIZE"
require_nonnegative_finite_number LOSS_ATOL "$LOSS_ATOL"
require_nonnegative_finite_number LOSS_RTOL "$LOSS_RTOL"
if [[ -n "$GLOBAL_BATCH_SIZE" ]]; then
    require_positive_integer GLOBAL_BATCH_SIZE "$GLOBAL_BATCH_SIZE"
fi

if (( NUM_NODES != 1 || NODE_RANK != 0 )); then
    fail "this benchmark currently supports only NUM_NODES=1 and NODE_RANK=0; multi-node log collection is not yet race-free."
fi

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
MODEL_PARALLEL_SIZE=$((TP_SIZE * CP_SIZE * PP_SIZE))

if (( MAX_POSITION_EMBEDDINGS < SEQ_LENGTH )); then
    fail "MAX_POSITION_EMBEDDINGS must be at least SEQ_LENGTH."
fi

if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
    fail "$PRETRAIN_SCRIPT_PATH not found. Run this script from the Megatron-LM repo root."
fi

if (( WORLD_SIZE < MODEL_PARALLEL_SIZE )); then
    echo "Error: need at least TP_SIZE * CP_SIZE * PP_SIZE GPUs."
    echo "Got GPUS_PER_NODE=${GPUS_PER_NODE}, NUM_NODES=${NUM_NODES}, TP_SIZE=${TP_SIZE}, CP_SIZE=${CP_SIZE}, PP_SIZE=${PP_SIZE}."
    exit 1
fi

if (( WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
    echo "Error: total GPUs must be divisible by TP_SIZE * CP_SIZE * PP_SIZE."
    echo "Got WORLD_SIZE=${WORLD_SIZE}, TP_SIZE=${TP_SIZE}, CP_SIZE=${CP_SIZE}, PP_SIZE=${PP_SIZE}."
    exit 1
fi

DP_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))
DP_CP_SIZE=$((DP_SIZE * CP_SIZE))
BASELINE_PACKED_CAPACITY=$((CP_SIZE * MAX_SEQLEN_PER_DP_CP_RANK))
DCP_PACKED_CAPACITY=$((DP_CP_SIZE * MAX_SEQLEN_PER_DP_CP_RANK))
MICRO_BATCH_TIMES_DP=$((MICRO_BATCH_SIZE * DP_SIZE))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_TIMES_DP * NUM_MICROBATCHES))}

if (( DP_CP_SIZE <= 1 )); then
    fail "dynamic context parallelism requires DP_SIZE * CP_SIZE > 1, got ${DP_CP_SIZE}."
fi

if (( MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE > DP_CP_SIZE )); then
    fail "MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE must be in [1, ${DP_CP_SIZE}], got ${MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE}."
fi

if (( MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE != DP_CP_SIZE )); then
    if ! is_power_of_two "$DP_CP_SIZE"; then
        fail "DP_SIZE * CP_SIZE must be a power of two unless MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE selects the full domain (${DP_CP_SIZE})."
    fi
    if ! is_power_of_two "$MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE"; then
        fail "MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE must be a power of two or the full DP x CP domain (${DP_CP_SIZE})."
    fi
fi

if (( BASELINE_PACKED_CAPACITY < SEQ_LENGTH )); then
    fail "baseline packed capacity CP_SIZE * MAX_SEQLEN_PER_DP_CP_RANK (${BASELINE_PACKED_CAPACITY}) must be at least SEQ_LENGTH (${SEQ_LENGTH})."
fi

if (( DCP_PACKED_CAPACITY < SEQ_LENGTH )); then
    fail "DCP packed capacity DP_SIZE * CP_SIZE * MAX_SEQLEN_PER_DP_CP_RANK (${DCP_PACKED_CAPACITY}) must be at least SEQ_LENGTH (${SEQ_LENGTH})."
fi

if (( MICRO_BATCH_SIZE != 1 )); then
    echo "Error: this variable-length THD benchmark expects MICRO_BATCH_SIZE=1."
    echo "Increase effective batch size with NUM_MICROBATCHES and data-parallel ranks."
    exit 1
fi

if (( GLOBAL_BATCH_SIZE % MICRO_BATCH_TIMES_DP != 0 )); then
    echo "Error: GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * DP_SIZE."
    echo "Got GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}, DP_SIZE=${DP_SIZE}."
    exit 1
fi
EFFECTIVE_NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_TIMES_DP))

if (( TRAIN_ITERS <= WARMUP_ITERS )); then
    echo "Error: TRAIN_ITERS must be greater than WARMUP_ITERS."
    exit 1
fi

if [[ "$CHECK_LOSS_PARITY" != "0" && "$CHECK_LOSS_PARITY" != "1" ]]; then
    echo "Error: CHECK_LOSS_PARITY must be 0 or 1."
    exit 1
fi

if [[ "$NVTE_ALLOW_NONDETERMINISTIC_ALGO" != "0" && "$NVTE_ALLOW_NONDETERMINISTIC_ALGO" != "1" ]]; then
    echo "Error: NVTE_ALLOW_NONDETERMINISTIC_ALGO must be 0 or 1."
    exit 1
fi

if [[ "$ANALYZE_ONLY" != "0" && "$ANALYZE_ONLY" != "1" ]]; then
    echo "Error: ANALYZE_ONLY must be 0 or 1."
    exit 1
fi

if [[ "$CASE_ORDER" != "baseline_first" && "$CASE_ORDER" != "dcp_first" ]]; then
    echo "Error: CASE_ORDER must be baseline_first or dcp_first."
    exit 1
fi

if (( LOG_INTERVAL != 1 )); then
    echo "Error: LOG_INTERVAL must be 1 so every training loss is compared."
    exit 1
fi

if [[ -n "$DATASET_PATH" && -z "$TOKENIZER_MODEL" ]]; then
    echo "Error: TOKENIZER_MODEL is required when DATASET_PATH is set."
    exit 1
fi

if [[ "$ANALYZE_ONLY" == "0" && -n "$LOAD_PATH" && ! -d "$LOAD_PATH" ]]; then
    echo "Error: LOAD_PATH does not exist or is not a directory: $LOAD_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$DATA_CACHE_PATH"

if [[ "$ANALYZE_ONLY" == "0" ]]; then
    write_reproducibility_manifest
elif [[ -f "$MANIFEST_PATH" ]]; then
    echo "Reusing reproducibility manifest: ${MANIFEST_PATH}"
else
    echo "Warning: no reproducibility manifest found at ${MANIFEST_PATH}" >&2
fi

echo "WORLD_SIZE=${WORLD_SIZE} DP_SIZE=${DP_SIZE} CP_SIZE=${CP_SIZE} TP_SIZE=${TP_SIZE} PP_SIZE=${PP_SIZE}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} NUM_MICROBATCHES=${EFFECTIVE_NUM_MICROBATCHES}"
if [[ -n "$DATASET_PATH" ]]; then
    echo "DATASET_PATH=${DATASET_PATH} TOKENIZER_MODEL=${TOKENIZER_MODEL}"
    if [[ -z "$LOAD_PATH" && "$ANALYZE_ONLY" == "0" ]]; then
        echo "Note: LOAD_PATH is unset; both runs start from the same seeded random initialization."
    fi
else
    echo "DATASET_PATH=mock"
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --transformer-impl transformer_engine
    --attention-backend flash
)

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr-decay-iters "$TRAIN_ITERS"
    --lr 1.5e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --weight-decay 1.0e-2
    --clip-grad 1.0
    --bf16
    --calculate-per-token-loss
    --no-gradient-accumulation-fusion
    --seed "$SEED"
    --cuda-graph-impl none
)

CHECKPOINT_ARGS=()
if [[ -n "$LOAD_PATH" ]]; then
    CHECKPOINT_ARGS+=(
        --load "$LOAD_PATH"
        --finetune
        --exit-on-missing-checkpoint
    )
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
)

if (( TP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

DATA_ARGS=(--use-varlen-dataset)
if [[ -n "$DATASET_PATH" ]]; then
    DATA_ARGS+=(
        --data-path "$DATASET_PATH"
        --tokenizer-type SFTTokenizer
        --tokenizer-model "$TOKENIZER_MODEL"
        --sft-tokenizer-prompt-format default
    )
else
    DATA_ARGS+=(
        --mock-data
        --varlen-mock-dataset-config-json "$VARLEN_DATASET_JSON"
        --tokenizer-type NullTokenizer
        --vocab-size "$VOCAB_SIZE"
        --null-tokenizer-pad-id "$NULL_TOKENIZER_PAD_ID"
    )
fi
DATA_ARGS+=(
    --split 99,1,0
    --data-cache-path "$DATA_CACHE_PATH"
    --dataloader-type single
    --num-workers "$NUM_WORKERS"
    --no-create-attention-mask-in-dataloader
)

LOGGING_ARGS=(
    --log-interval "$LOG_INTERVAL"
    --log-throughput
    --timing-log-level 0
    --eval-interval 1000000
    --eval-iters 1
    --save-interval 1000000
    --distributed-backend nccl
    --distributed-timeout-minutes 60
)

extract_iteration_stats_ms() {
    local log_file=$1
    "$PYTHON" - "$log_file" "$WARMUP_ITERS" "$TRAIN_ITERS" <<'PY'
import math
import re
import statistics
import sys

log_file = sys.argv[1]
warmup = int(sys.argv[2])
expected_train_iters = int(sys.argv[3])
number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
pattern = re.compile(r"elapsed time per iteration \(ms\): " + number)
iteration_pattern = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|.*?consumed samples:\s*(\d+)\s*\|"
)

records = []
try:
    with open(log_file, errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iteration_match = iteration_pattern.search(line)
                if not iteration_match:
                    raise ValueError("iteration-time record is missing its iteration number")
                iteration = int(iteration_match.group(1))
                reported_train_iters = int(iteration_match.group(2))
                consumed_samples = int(iteration_match.group(3))
                if reported_train_iters != expected_train_iters:
                    raise ValueError(
                        f"expected {expected_train_iters} train iterations, "
                        f"log reports {reported_train_iters}"
                    )
                value = float(match.group(1))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"non-finite or non-positive iteration time: {value}")
                records.append((iteration, consumed_samples, value))
    logged_iterations = [iteration for iteration, _, _ in records]
    expected_iterations = list(range(1, expected_train_iters + 1))
    if logged_iterations != expected_iterations:
        raise ValueError(
            "incomplete or duplicate iteration-time records: "
            f"expected {expected_iterations}, got {logged_iterations}"
        )
    consumed_samples = [samples for _, samples, _ in records]
    sample_deltas = [consumed_samples[0]] + [
        current - previous for previous, current in zip(consumed_samples, consumed_samples[1:])
    ]
    if any(delta <= 0 for delta in sample_deltas) or len(set(sample_deltas)) != 1:
        raise ValueError(
            "consumed-sample records do not describe a constant positive batch: "
            f"samples={consumed_samples}, deltas={sample_deltas}"
        )
    samples_per_iteration = sample_deltas[0]
    values = [value for _, _, value in records]
    values = values[warmup:]
    if not values:
        raise ValueError("no measured iteration times remain after warmup")
except (OSError, ValueError) as exc:
    print(f"Timing analysis failed for {log_file}: {exc}", file=sys.stderr)
    raise SystemExit(1)

average = sum(values) / len(values)
median = statistics.median(values)
trim_count = int(len(values) * 0.1)
trimmed = sorted(values)
if len(trimmed) - 2 * trim_count > 0:
    trimmed = trimmed[trim_count : len(trimmed) - trim_count]
trimmed_average = sum(trimmed) / len(trimmed)

print(f"{average:.3f} {median:.3f} {trimmed_average:.3f} {samples_per_iteration}")
PY
}

calc_speedup() {
    local base=$1
    local dcp=$2
    awk -v base="$base" -v dcp="$dcp" 'BEGIN {
        if (base > 0 && dcp > 0) {
            printf "%.2f", base / dcp
        } else {
            exit 1
        }
    }'
}

calc_samples_per_second() {
    local batch=$1
    local milliseconds=$2
    awk -v batch="$batch" -v milliseconds="$milliseconds" 'BEGIN {
        if (batch > 0 && milliseconds > 0) {
            printf "%.3f", batch * 1000.0 / milliseconds
        } else {
            exit 1
        }
    }'
}

compare_loss_trajectories() {
    local baseline_log=$1
    local dcp_log=$2

    "$PYTHON" - \
        "$baseline_log" \
        "$dcp_log" \
        "$TRAIN_ITERS" \
        "$LOSS_ATOL" \
        "$LOSS_RTOL" \
        "$CHECK_LOSS_PARITY" <<'PY'
import math
import re
import sys

baseline_log, dcp_log, train_iters_arg, atol_arg, rtol_arg, enforce_arg = sys.argv[1:]
expected_train_iters = int(train_iters_arg)
atol = float(atol_arg)
rtol = float(rtol_arg)
enforce = enforce_arg == "1"
iteration_pattern = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|.*?consumed samples:\s*(\d+)\s*\|"
)
number_pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
loss_pattern = re.compile(r"lm loss:\s*" + number_pattern)
skipped_pattern = re.compile(r"number of skipped iterations:\s*(\d+)")
nan_pattern = re.compile(r"number of nan iterations:\s*(\d+)")


def read_records(path):
    records = []
    with open(path, errors="ignore") as stream:
        for line in stream:
            iteration_match = iteration_pattern.search(line)
            if not iteration_match:
                continue

            iteration = int(iteration_match.group(1))
            reported_train_iters = int(iteration_match.group(2))
            samples = int(iteration_match.group(3))
            if reported_train_iters != expected_train_iters:
                raise ValueError(
                    f"{path}: expected {expected_train_iters} train iterations, "
                    f"log reports {reported_train_iters}"
                )

            loss_match = loss_pattern.search(line)
            skipped_match = skipped_pattern.search(line)
            nan_match = nan_pattern.search(line)
            if not loss_match:
                raise ValueError(f"{path}: iteration {iteration} has no finite lm loss")
            if not skipped_match or not nan_match:
                raise ValueError(f"{path}: iteration {iteration} is missing health counters")

            loss = float(loss_match.group(1))
            if not math.isfinite(loss):
                raise ValueError(f"{path}: iteration {iteration} has non-finite lm loss")
            if int(skipped_match.group(1)) != 0 or int(nan_match.group(1)) != 0:
                raise ValueError(f"{path}: iteration {iteration} was skipped or produced NaN")
            records.append((iteration, samples, loss))
    if not records:
        raise ValueError(f"no iteration/loss records found in {path}")
    logged_iterations = [iteration for iteration, _, _ in records]
    expected_iterations = list(range(1, expected_train_iters + 1))
    if logged_iterations != expected_iterations:
        raise ValueError(
            f"{path}: incomplete or duplicate iterations: "
            f"expected {expected_iterations}, got {logged_iterations}"
        )
    return records


try:
    baseline = read_records(baseline_log)
    dcp = read_records(dcp_log)
except (OSError, ValueError) as exc:
    print(f"Loss comparison failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

baseline_keys = [(iteration, samples) for iteration, samples, _ in baseline]
dcp_keys = [(iteration, samples) for iteration, samples, _ in dcp]
if baseline_keys != dcp_keys:
    print("Loss comparison failed: iteration or consumed-sample records differ.", file=sys.stderr)
    print(f"Baseline records: {baseline_keys}", file=sys.stderr)
    print(f"DCP records:      {dcp_keys}", file=sys.stderr)
    raise SystemExit(1)

comparisons = []
for (iteration, samples, baseline_loss), (_, _, dcp_loss) in zip(baseline, dcp):
    absolute_difference = abs(baseline_loss - dcp_loss)
    relative_difference = absolute_difference / max(abs(baseline_loss), 1e-12)
    comparisons.append(
        (iteration, samples, baseline_loss, dcp_loss, absolute_difference, relative_difference)
    )

worst_absolute = max(comparisons, key=lambda record: record[4])
worst_relative = max(comparisons, key=lambda record: record[5])
failures = [
    record
    for record in comparisons
    if not math.isclose(record[2], record[3], rel_tol=rtol, abs_tol=atol)
]
status = "PASS" if not failures else ("FAIL" if enforce else "WARN")

print(f"Loss records compared:                 {len(comparisons)}")
print("Iteration/sample alignment:            PASS")
print(
    "Maximum absolute loss difference:     "
    f"{worst_absolute[4]:.6E} (iteration {worst_absolute[0]})"
)
print(
    "Maximum relative loss difference:     "
    f"{worst_relative[5]:.6E} (iteration {worst_relative[0]})"
)
print(f"Loss parity (atol={atol:g}, rtol={rtol:g}):       {status}")

if failures and enforce:
    first = failures[0]
    print(
        "First mismatch: iteration "
        f"{first[0]}, samples {first[1]}, baseline {first[2]:.6E}, DCP {first[3]:.6E}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

run_case() {
    local name=$1
    shift

    local tensorboard_dir="${OUTPUT_DIR}/tensorboard_${name}"
    local log_file="${OUTPUT_DIR}/${name}.log"
    rm -rf "$tensorboard_dir"
    mkdir -p "$tensorboard_dir"

    echo
    echo "=== Running ${name} ==="
    echo "Log: ${log_file}"

    "$PYTHON" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        "$PRETRAIN_SCRIPT_PATH" \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${PARALLEL_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${CHECKPOINT_ARGS[@]}" \
        "${LOGGING_ARGS[@]}" \
        --tensorboard-dir "$tensorboard_dir" \
        "$@" 2>&1 | tee "$log_file"
}

run_baseline() {
    run_case baseline \
        --sequence-packing-scheduler dp_balanced \
        --max-seqlen-per-dp-cp-rank "$MAX_SEQLEN_PER_DP_CP_RANK"
}

run_dcp() {
    run_case dcp \
        --dynamic-context-parallel \
        --sequence-packing-scheduler default_dynamic_cp \
        --min-dynamic-context-parallel-size "$MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE" \
        --max-seqlen-per-dp-cp-rank "$MAX_SEQLEN_PER_DP_CP_RANK"
}

if [[ "$ANALYZE_ONLY" == "0" ]]; then
    if [[ "$CASE_ORDER" == "baseline_first" ]]; then
        run_baseline
        run_dcp
    else
        run_dcp
        run_baseline
    fi
else
    echo "ANALYZE_ONLY=1; reusing ${OUTPUT_DIR}/baseline.log and ${OUTPUT_DIR}/dcp.log"
    for log_file in "${OUTPUT_DIR}/baseline.log" "${OUTPUT_DIR}/dcp.log"; do
        if [[ ! -f "$log_file" ]]; then
            echo "Error: ANALYZE_ONLY=1 requires $log_file"
            exit 1
        fi
    done
fi

baseline_stats=$(extract_iteration_stats_ms "${OUTPUT_DIR}/baseline.log")
dcp_stats=$(extract_iteration_stats_ms "${OUTPUT_DIR}/dcp.log")
read -r baseline_avg_ms baseline_median_ms baseline_trimmed_ms baseline_samples_per_iteration <<< "$baseline_stats"
read -r dcp_avg_ms dcp_median_ms dcp_trimmed_ms dcp_samples_per_iteration <<< "$dcp_stats"

if [[ "$baseline_samples_per_iteration" != "$dcp_samples_per_iteration" ]]; then
    fail "baseline and DCP logs use different per-iteration sample counts: ${baseline_samples_per_iteration} vs ${dcp_samples_per_iteration}."
fi
if [[ "$ANALYZE_ONLY" == "0" && "$baseline_samples_per_iteration" != "$GLOBAL_BATCH_SIZE" ]]; then
    fail "logs report ${baseline_samples_per_iteration} samples per iteration, expected GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}."
fi

avg_speedup=$(calc_speedup "$baseline_avg_ms" "$dcp_avg_ms")
median_speedup=$(calc_speedup "$baseline_median_ms" "$dcp_median_ms")
trimmed_speedup=$(calc_speedup "$baseline_trimmed_ms" "$dcp_trimmed_ms")
baseline_samples_per_second=$(
    calc_samples_per_second "$baseline_samples_per_iteration" "$baseline_avg_ms"
)
dcp_samples_per_second=$(calc_samples_per_second "$dcp_samples_per_iteration" "$dcp_avg_ms")

echo
echo "=== Dynamic CP throughput summary ==="
echo "Iteration-time statistics exclude the first ${WARMUP_ITERS} logged iterations."
echo "Baseline dp_balanced average:        ${baseline_avg_ms} ms"
echo "Dynamic CP average:                  ${dcp_avg_ms} ms"
echo "Baseline average throughput:         ${baseline_samples_per_second} samples/s"
echo "Dynamic CP average throughput:       ${dcp_samples_per_second} samples/s"
echo "Average sample-throughput speedup:   ${avg_speedup}x"
echo "Baseline dp_balanced median:         ${baseline_median_ms} ms"
echo "Dynamic CP median:                   ${dcp_median_ms} ms"
echo "Median sample-throughput speedup:    ${median_speedup}x"
echo "Baseline dp_balanced 10% trimmed avg: ${baseline_trimmed_ms} ms"
echo "Dynamic CP 10% trimmed avg:           ${dcp_trimmed_ms} ms"
echo "10% trimmed throughput speedup:       ${trimmed_speedup}x"
echo
echo "=== Dynamic CP loss comparison ==="
compare_loss_trajectories "${OUTPUT_DIR}/baseline.log" "${OUTPUT_DIR}/dcp.log"
echo
echo "Logs and TensorBoard data are under ${OUTPUT_DIR}"

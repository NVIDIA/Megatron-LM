#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
PYTHON=${PYTHON:-python}

TP_SIZE=${TP_SIZE:-1}
CP_SIZE=${CP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
MODEL_PARALLEL_SIZE=$((TP_SIZE * CP_SIZE * PP_SIZE))

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

MAX_SEQLEN_PER_DP_CP_RANK=${MAX_SEQLEN_PER_DP_CP_RANK:-2048}
MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE=${MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE:-1}

VOCAB_SIZE=${VOCAB_SIZE:-131072}
NULL_TOKENIZER_PAD_ID=${NULL_TOKENIZER_PAD_ID:-0}

OUTPUT_DIR=${OUTPUT_DIR:-"${PWD}/dcp_benchmark_output"}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${OUTPUT_DIR}/data_cache"}
DEFAULT_VARLEN_DATASET_JSON='{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":128,"max_seq_len":8192,"mean_seq_len":1024,"lognormal_sigma":1.5}'
VARLEN_DATASET_JSON=${VARLEN_DATASET_JSON:-$DEFAULT_VARLEN_DATASET_JSON}

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
    echo "Error: $PRETRAIN_SCRIPT_PATH not found. Run this script from the Megatron-LM repo root."
    exit 1
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
MICRO_BATCH_TIMES_DP=$((MICRO_BATCH_SIZE * DP_SIZE))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_TIMES_DP * NUM_MICROBATCHES))}

if (( MICRO_BATCH_SIZE != 1 )); then
    echo "Error: this variable-length THD mock benchmark expects MICRO_BATCH_SIZE=1."
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

mkdir -p "$OUTPUT_DIR" "$DATA_CACHE_PATH"

echo "WORLD_SIZE=${WORLD_SIZE} DP_SIZE=${DP_SIZE} CP_SIZE=${CP_SIZE} TP_SIZE=${TP_SIZE} PP_SIZE=${PP_SIZE}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} NUM_MICROBATCHES=${EFFECTIVE_NUM_MICROBATCHES}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
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
    --moe-token-dispatcher-type "$MOE_TOKEN_DISPATCHER_TYPE"
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
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
)

if (( TP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

DATA_ARGS=(
    --use-varlen-dataset
    --mock-data
    --varlen-mock-dataset-config-json "$VARLEN_DATASET_JSON"
    --tokenizer-type NullTokenizer
    --vocab-size "$VOCAB_SIZE"
    --null-tokenizer-pad-id "$NULL_TOKENIZER_PAD_ID"
    --split 99,1,0
    --data-cache-path "$DATA_CACHE_PATH"
    --dataloader-type single
    --num-workers "$NUM_WORKERS"
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
    "$PYTHON" - "$log_file" "$WARMUP_ITERS" <<'PY'
import re
import statistics
import sys

log_file = sys.argv[1]
warmup = int(sys.argv[2])
pattern = re.compile(r"elapsed time per iteration \(ms\): ([0-9.]+)")

values = []
with open(log_file, errors="ignore") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))

values = values[warmup:]
if not values:
    print("nan nan nan")
    raise SystemExit

average = sum(values) / len(values)
median = statistics.median(values)
trim_count = int(len(values) * 0.1)
trimmed = sorted(values)
if len(trimmed) - 2 * trim_count > 0:
    trimmed = trimmed[trim_count : len(trimmed) - trim_count]
trimmed_average = sum(trimmed) / len(trimmed)

print(f"{average:.3f} {median:.3f} {trimmed_average:.3f}")
PY
}

calc_speedup() {
    local base=$1
    local dcp=$2
    awk -v base="$base" -v dcp="$dcp" 'BEGIN {
        if (base > 0 && dcp > 0) {
            printf "%.2f", base / dcp
        } else {
            printf "nan"
        }
    }'
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
        "${LOGGING_ARGS[@]}" \
        --tensorboard-dir "$tensorboard_dir" \
        "$@" 2>&1 | tee "$log_file"
}

run_case baseline \
    --sequence-packing-scheduler dp_balanced \
    --max-seqlen-per-dp-cp-rank "$MAX_SEQLEN_PER_DP_CP_RANK"

run_case dcp \
    --dynamic-context-parallel \
    --sequence-packing-scheduler default_dynamic_cp \
    --min-dynamic-context-parallel-size "$MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE" \
    --max-seqlen-per-dp-cp-rank "$MAX_SEQLEN_PER_DP_CP_RANK"

read -r baseline_avg_ms baseline_median_ms baseline_trimmed_ms < <(extract_iteration_stats_ms "${OUTPUT_DIR}/baseline.log")
read -r dcp_avg_ms dcp_median_ms dcp_trimmed_ms < <(extract_iteration_stats_ms "${OUTPUT_DIR}/dcp.log")

avg_speedup=$(calc_speedup "$baseline_avg_ms" "$dcp_avg_ms")
median_speedup=$(calc_speedup "$baseline_median_ms" "$dcp_median_ms")
trimmed_speedup=$(calc_speedup "$baseline_trimmed_ms" "$dcp_trimmed_ms")

echo
echo "=== Dynamic CP benchmark summary ==="
echo "Iteration-time statistics exclude the first ${WARMUP_ITERS} logged iterations."
echo "Baseline dp_balanced average:        ${baseline_avg_ms} ms"
echo "Dynamic CP average:                  ${dcp_avg_ms} ms"
echo "Average speedup:                     ${avg_speedup}x"
echo "Baseline dp_balanced median:         ${baseline_median_ms} ms"
echo "Dynamic CP median:                   ${dcp_median_ms} ms"
echo "Median speedup:                      ${median_speedup}x"
echo "Baseline dp_balanced 10% trimmed avg: ${baseline_trimmed_ms} ms"
echo "Dynamic CP 10% trimmed avg:           ${dcp_trimmed_ms} ms"
echo "10% trimmed mean speedup:             ${trimmed_speedup}x"
echo
echo "Logs and TensorBoard data are under ${OUTPUT_DIR}"

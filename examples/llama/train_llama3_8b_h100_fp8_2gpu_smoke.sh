#!/bin/bash
set -euo pipefail

# Two-GPU smoke test for the Llama-3 8B FP8 example.
# It keeps the 8B model shape, but uses TP=2, mock data, a short sequence
# length, and only a few iterations so the example can validate the stack.

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}

# Triton/Inductor sometimes needs the unversioned libcuda.so linker name.
# Apptainer --nv commonly exposes only libcuda.so.1 from the host driver.
if [[ -e /usr/local/cuda/compat/lib/libcuda.so.1 ]]; then
    TRITON_LIBCUDA_DIR=${TRITON_LIBCUDA_DIR:-"${PWD}/.triton_libcuda"}
    mkdir -p "$TRITON_LIBCUDA_DIR"
    ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 "$TRITON_LIBCUDA_DIR/libcuda.so"
    ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 "$TRITON_LIBCUDA_DIR/libcuda.so.1"
    ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so 2>/dev/null || true
    export TRITON_LIBCUDA_PATH="$TRITON_LIBCUDA_DIR"
    export LD_LIBRARY_PATH="${TRITON_LIBCUDA_DIR}:/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${TRITON_LIBCUDA_DIR}:/usr/local/cuda/compat/lib:${LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp8_2gpu_smoke"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp8_2gpu_smoke"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_LOGS_PATH"

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
TRAIN_ITERS=${TRAIN_ITERS:-5}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
if [[ -z "${LR_WARMUP_ITERS+x}" ]]; then
    if (( TRAIN_ITERS <= 1 )); then
        LR_WARMUP_ITERS=0
    else
        LR_WARMUP_ITERS=1
    fi
fi
SEQ_LENGTH=${SEQ_LENGTH:-8192}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-8192}
NUM_LAYERS=${NUM_LAYERS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-14336}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-8}
KV_CHANNELS=${KV_CHANNELS:-128}
VOCAB_SIZE=${VOCAB_SIZE:-128256}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
ATTENTION_RESIDUALS=${ATTENTION_RESIDUALS:-0}
ATTENTION_RESIDUAL_TYPE=${ATTENTION_RESIDUAL_TYPE:-full}
ATTENTION_RESIDUAL_NUM_BLOCKS=${ATTENTION_RESIDUAL_NUM_BLOCKS:-8}
ATTENTION_RESIDUAL_RMSNORM=${ATTENTION_RESIDUAL_RMSNORM:-1}
ATTENTION_RESIDUAL_IMPLEMENTATION=${ATTENTION_RESIDUAL_IMPLEMENTATION:-torch}
ATTENTION_RESIDUAL_LOG_WEIGHTS=${ATTENTION_RESIDUAL_LOG_WEIGHTS:-0}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${PWD}/benchmark_cache_llama3_8b_fp8_2gpu_smoke"}
mkdir -p "$DATA_CACHE_PATH"

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
    --group-query-attention
    --num-query-groups "$NUM_QUERY_GROUPS"
    --kv-channels "$KV_CHANNELS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

if [[ "$ATTENTION_RESIDUALS" == "1" ]]; then
    MODEL_ARGS+=(
        --attention-residuals
        --attention-residual-type "$ATTENTION_RESIDUAL_TYPE"
        --attention-residual-num-blocks "$ATTENTION_RESIDUAL_NUM_BLOCKS"
        --attention-residual-implementation "$ATTENTION_RESIDUAL_IMPLEMENTATION"
    )
    if [[ "$ATTENTION_RESIDUAL_RMSNORM" != "1" ]]; then
        MODEL_ARGS+=(--no-attention-residual-rmsnorm)
    fi
    if [[ "$ATTENTION_RESIDUAL_LOG_WEIGHTS" == "1" ]]; then
        MODEL_ARGS+=(--attention-residual-log-weights)
    fi
fi

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4
    --decoupled-min-lr 4.5e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
    --recompute-activations
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

if [[ -n "$EXIT_DURATION_IN_MINS" ]]; then
    TRAINING_ARGS+=(--exit-duration-in-mins "$EXIT_DURATION_IN_MINS")
fi

DTYPE_ARGS=(
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --fp8-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --sequence-parallel
)

DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size "$VOCAB_SIZE"
        --data-cache-path "$DATA_CACHE_PATH"
        --tiktoken-pattern v2
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS_LIST+=(
        --data-path "$DATA_ARG"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_ARG"
        --data-cache-path "$DATA_CACHE_PATH"
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
        --vocab-size "$VOCAB_SIZE"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 1000
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
    EVAL_AND_LOGGING_ARGS+=(
        --save "$CHECKPOINT_PATH"
        --save-interval "$TRAIN_ITERS"
        --async-save
        --async-strategy mcore
    )
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    if [[ -z "$WANDB_EXP_NAME" ]]; then
        echo "Error: WANDB_EXP_NAME must be set when WANDB_PROJECT is set." >&2
        exit 1
    fi
    EVAL_AND_LOGGING_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "$WANDB_EXP_NAME"
    )
    if [[ -n "$WANDB_SAVE_DIR" ]]; then
        EVAL_AND_LOGGING_ARGS+=(--wandb-save-dir "$WANDB_SAVE_DIR")
    fi
    if [[ -n "$WANDB_ENTITY" ]]; then
        EVAL_AND_LOGGING_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Run this script from the root of the Megatron-LM repository."
    exit 1
fi

echo "Running Llama-3 FP8 smoke test on ${GPUS_PER_NODE} GPUs: layers=${NUM_LAYERS}, hidden=${HIDDEN_SIZE}, ffn=${FFN_HIDDEN_SIZE}, heads=${NUM_ATTENTION_HEADS}, query_groups=${NUM_QUERY_GROUPS}, TP=${TP_SIZE}, CP=${CP_SIZE}, PP=${PP_SIZE}, seq=${SEQ_LENGTH}, iters=${TRAIN_ITERS}, exit_mins=${EXIT_DURATION_IN_MINS:-none}, lr_decay_iters=${LR_DECAY_ITERS}, lr_warmup_iters=${LR_WARMUP_ITERS}, attn_res=${ATTENTION_RESIDUALS}, attn_res_type=${ATTENTION_RESIDUAL_TYPE}, attn_res_blocks=${ATTENTION_RESIDUAL_NUM_BLOCKS}, attn_res_impl=${ATTENTION_RESIDUAL_IMPLEMENTATION}, wandb_project=${WANDB_PROJECT:-none}"

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$PRETRAIN_SCRIPT_PATH" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DTYPE_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS_LIST[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}"

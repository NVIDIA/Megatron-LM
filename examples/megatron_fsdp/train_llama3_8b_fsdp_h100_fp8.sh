#!/bin/bash

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fsdp_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fsdp_fp8"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script 
# is run from the root of the Megatron-LM repository.
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Model & Training Parameters
USE_MEGATRON_FSDP=${USE_MEGATRON_FSDP:-1}
SHARDING_STRATEGY=${SHARDING_STRATEGY:-"optim_grads_params"}
OUTER_SHARDING_STRATEGY=${OUTER_SHARDING_STRATEGY:-"no_shard"}
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
NUM_LAYERS=32
DTYPE="fp8"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_llama3_8b_fsdp_fp8"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
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

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 1953125000
    --lr-decay-samples 1949218748
    --lr-warmup-samples 3906252
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
    --cross-entropy-loss-fusion
    --manual-gc
    --empty-unused-memory-level 1
    --exit-duration-in-mins 235
)

if [ "${USE_MEGATRON_FSDP}" = 1 ]; then
    unset CUDA_DEVICE_MAX_CONNECTIONS
    TRAINING_ARGS=(
        "${TRAINING_ARGS[@]}"
        --use-megatron-fsdp
        --data-parallel-sharding-strategy ${SHARDING_STRATEGY}
        --no-gradient-accumulation-fusion
        --calculate-per-token-loss
        --init-model-with-meta-device
        --ckpt-format fsdp_dtensor
        --grad-reduce-in-bf16
        --use-nccl-ub
        --fsdp-double-buffer
        --fsdp-manual-registration
        # To enable HFSDP, DP full-sharding of the optimizer state with
        # hierarchical data parallelism (DP-Outer=2, DP-Inner=DP//2)...
        # --num-distributed-optimizer-instances 2
        # --outer-dp-sharding-strategy ${OUTER_SHARDING_STRATEGY}
        # To further customize Megatron-FSDP data precision...
        # --megatron-fsdp-main-params-dtype fp32
        # --megatron-fsdp-main-grads-dtype auto
        # --megatron-fsdp-grad-comm-dtype auto
    )
fi

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 128256"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x
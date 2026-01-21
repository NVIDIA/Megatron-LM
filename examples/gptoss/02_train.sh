#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}


# Setup arguments with defaults
CHECKPOINT_PATH="NO_VALUE_PROVIDED"
TENSORBOARD_LOGS_PATH="./tensorboard_logs/"
TOKENIZER_ARG="MOCK"
DATA_ARG="MOCK"
DISTRIBUTED_CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --tensorboard-logs-path)
            TENSORBOARD_LOGS_PATH="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_ARG="$2"
            shift 2
            ;;
        --data)
            DATA_ARG="$2"
            shift 2
            ;;
        --distributed-config-file)
            DISTRIBUTED_CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --checkpoint-path PATH          Path to Megatron checkpoint"
            echo "  --tensorboard-logs-path PATH    Path to TensorBoard logs"
            echo "  --tokenizer PATH|MOCK           Path to tokenizer model, or 'MOCK' (default: MOCK)"
            echo "  --data PATH|MOCK                Data prefix, or 'MOCK' (default: MOCK)"
            echo "  --distributed-config-file FILE       Path to distributed training config file"
            echo "  -h, --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi
echo "Checkpoint path exists: $CHECKPOINT_PATH"

# Check if tensorboard logs path exists
if [ ! -d "$TENSORBOARD_LOGS_PATH" ]; then
    echo "Warning: TensorBoard logs path does not exist. Creating: $TENSORBOARD_LOGS_PATH"
    mkdir -p "$TENSORBOARD_LOGS_PATH"
fi
echo "TensorBoard logs path exists: $TENSORBOARD_LOGS_PATH"

# Distributed training setup - default values
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR="localhost"
MASTER_PORT=6000
NODE_RANK=0

# Load distributed config from file if provided
if [ -n "$DISTRIBUTED_CONFIG_FILE" ]; then
    if [ ! -f "$DISTRIBUTED_CONFIG_FILE" ]; then
        echo "Warning: Distributed config file does not exist: $DISTRIBUTED_CONFIG_FILE"
        echo "Continuing with default distributed training settings."
    else
        echo "Loading distributed config from: $DISTRIBUTED_CONFIG_FILE"
        source "$DISTRIBUTED_CONFIG_FILE"
    fi
fi

# Override with environment variables if set
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_gpt_oss_20b"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

# NOTE: Adjust the following model and training parameters as needed below, these are example values for openai/gpt-oss-20b
TP_SIZE=1     
EP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
NUM_LAYERS=12
DTYPE="fp8"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192
TRAIN_SAMPLES=1953125000
LR_DECAY_SAMPLES=1949218748

MODEL_ARGS=(
    --no-masked-softmax-fusion
    --transformer-impl transformer_engine
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --no-rope-fusion
    --normalization RMSNorm
    --num-layers ${NUM_LAYERS}
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --use-mcore-models
    --rotary-percent 1.0
    # --rope-type yarn
    --rope-type rope
    --position-embedding-type rope
    --rotary-base 10000
    --no-bias-gelu-fusion
    --export-force-local-attention
    --no-bias-dropout-fusion
    --quick-geglu
    --glu-linear-offset 1.0
    --softmax-type learnable
    --window-attn-skip-freq 2
    --activation-func-clamp-value 7.0
    --window-size 128,0
    --enable-gpt-oss
)

MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-ffn-hidden-size 2048
    --moe-router-dtype fp32
    --moe-z-loss-coeff 1e-3
    --moe-permute-fusion
)

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

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr 1.0e-5
    --train-samples ${TRAIN_SAMPLES}
    --lr-decay-samples ${LR_DECAY_SAMPLES}
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-fraction 0.05
    --clip-grad 1.0
    --bf16
    --use-flash-attn
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --disable-bf16-reduced-precision-matmul
    --recompute-activations
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --sequence-parallel
    --context-parallel-size 1
    --use-distributed-optimizer
    --fp8-format hybrid
    --fp8-param-gather
    --fp8-amax-compute-algo max
    --fp8-amax-history-len 1024
)
    
LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 50000000
    --eval-iters 0
    --save $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --moe-per-layer-logging
    --no-load-optim
    --no-load-rng
    --log-throughput
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} ${PRETRAIN_SCRIPT_PATH} \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
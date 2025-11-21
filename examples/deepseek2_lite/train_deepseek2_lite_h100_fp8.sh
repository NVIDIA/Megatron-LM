#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/deepseek2_lite_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/deepseek2_lite_fp8"}
# TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
TOKENIZER_ARG=${3:-"model/deepseek2_lite"} # Path to tokenizer model, or "MOCK"
# DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"
# DATA_ARG=${4:-"dataset/wikipedia_processed/wikipedia_processed_text_document"}     # Data prefix, or "MOCK"
DATA_ARG=${4:-"dataset/wikitext_processed/wikitext_processed_text_document"}     # Data prefix, or "MOCK"


# Create directories if they don't exist
rm -rf "$CHECKPOINT_PATH"
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters for DeepSeek2-Lite
TP_SIZE=8
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1  # default 1
GLOBAL_BATCH_SIZE=64 # default 128
NUM_LAYERS=16 
MOE_LAYERS=$(($NUM_LAYERS-1))
DTYPE="bf16"
# DTYPE=${5:-"fp8"}
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_deepseek2_lite_fp8"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Model architecture parameters for DeepSeek2-Lite (1.3B parameters)
# Based on typical configurations for models of this size
# TOKENIZER_MODEL="deepseek-ai/DeepSeek-V2-Lite"

# Base GPT model arguments
GPT_ARGS=(
    --num-layers $NUM_LAYERS 
    --hidden-size 2048 
    --ffn-hidden-size 10944 
    --num-attention-heads 16 
    --kv-channels 16 
    --seq-length $SEQ_LENGTH 
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --untie-embeddings-and-output-weights 
    --normalization RMSNorm 
    --norm-epsilon 1e-6 
    --swiglu 
    --no-masked-softmax-fusion 
    --attention-softmax-in-fp32 
    --use-mcore-models 
    --tokenizer-type HuggingFaceTokenizer 
    --make-vocab-size-divisible-by 3200 
    # --transformer-impl local
)

# Multi-Latent Attention (MLA) arguments
MLA_ARGS=(
    --multi-latent-attention 
    --kv-lora-rank 512 
    --v-head-dim 128 
    --qk-head-dim 128 
    --qk-layernorm 
    --qk-pos-emb-head-dim 64 
)

# Mixture of Experts (MoE) arguments
MOE_ARGS=(
    --num-experts 64 
    --moe-layer-freq "([0]+[1]*$MOE_LAYERS)" 
    --moe-ffn-hidden-size 1408 
    --moe-grouped-gemm 
    --moe-router-score-function softmax 
    --moe-router-topk 6 
    --moe-router-topk-scaling-factor 1.0 
    --moe-router-pre-softmax 
    --moe-shared-expert-intermediate-size 2816 
    --moe-aux-loss-coeff 1e-3 
    --moe-token-dispatcher-type alltoall 
    --moe-token-drop-policy probs 
    --moe-router-load-balancing-type seq_aux_loss 
)

# Rotary Position Embedding (RoPE) arguments
ROPE_ARGS=(
    --position-embedding-type rope 
    --no-rope-fusion 
    --rotary-percent 1.0 
    --rotary-base 10000 
    --rotary-scaling-factor 40 
    --mscale 0.707 
    --mscale-all-dim 0.707 
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 47340000
    --lr-decay-samples 47245280
    --lr-warmup-samples 94720
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
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
    --exit-duration-in-mins 235000000 # default 235 
    --no-persist-layer-norm
)

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
    --pipeline-model-parallel-size $PP_SIZE
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
        # "--no-mmap-bin-files"
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
    # --profile
    # --profile-step-start 4
    # --profile-step-end 6
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 120
    --save "$CHECKPOINT_PATH"
    # --load "$CHECKPOINT_PATH" 
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
    ${GPT_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${ROPE_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x

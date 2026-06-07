#!/bin/bash
# Example script for training with Bagel model and data

# This is an example script showing how to use the integrated Bagel
# model and dataset in the MIMO training pipeline
unset CUDA_DEVICE_MAX_CONNECTIONS
# expandable_segments addresses fragmentation in the SwiGLU recompute path:
# iter-2 backward allocates a (S, 2*ffn) bf16 scratch buffer (~1.8 GiB) and
# fails when it can't find a contiguous free range, even though several GiB
# are reserved-but-unallocated. Required when --recompute-granularity full
# is combined with bf16 + bias_swiglu_fusion.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTHONPATH=`pwd`/examples/mimo_bagel:$PYTHONPATH
export PYTHONPATH=`pwd`/bagel-package:$PYTHONPATH
export PYTHONPATH=`pwd`/bagel-package/bagel:$PYTHONPATH
export DATA_CONFIG_FILE=`pwd`/bagel-package/bagel/data/configs/example.yaml
# export ENERGON_DATA_CONFIG_FILE='/workspace/megatron-lm-bagel/examples/mimo_bagel/data/energon_example.yaml'
echo $PYTHONPATH

MODEL_PATH=/workspace/models/bagel
LLM_PATH="/workspace/models/bagel/"
VIT_PATH="/workspace/models/siglip-so400m-14-980-flash-attn2-navit"
VAE_PATH="/workspace/models/bagel/ae.safetensors"
GPUS_PER_NODE=8

TOKENIZER_MODEL=/workspace/models/bagel/

# Distributed training setup
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

USE_MCORE="true"
MODEL="bagel_mot"
#dMODEL="bagel"

# Model configuration
HIDDEN_SIZE=3584
NUM_LAYERS=28
NUM_ATTENTION_HEADS=28
NUM_QUERY_GROUPS=4
SEQ_LENGTH=32768
MAX_POSITION_EMBEDDINGS=32768

# Training configuration
MICRO_BATCH_SIZE=1
let GLOBAL_BATCH_SIZE=($GPUS_PER_NODE * $MICRO_BATCH_SIZE)
TRAIN_ITERS=10
LR=1e-5
MIN_LR=1e-6
LR_WARMUP_ITERS=8

# Bagel-specific configuration
IMAGE_TOKEN_ID=32000
MAX_NUM_TOKENS=36864
MAX_NUM_TOKENS_PER_SAMPLE=16384
PACKING_BUFFER_SIZE=50

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

#Parallel configuration
PARALLEL_ARGS=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --ckpt-format fsdp_dtensor
    --distributed-timeout-minutes 1
    --init-model-with-meta-device
)

    # # Precision
    # --bf16
TRAINING_ARGS=(
    --model-provider $MODEL
    --model-path $MODEL_PATH
    --llm-path $LLM_PATH
    --vit-path $VIT_PATH
    --vae-path $VAE_PATH
    --dataloader-type external
    --dataset-provider bagel
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --context-parallel-size 1

    # Model architecture
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --num-query-groups $NUM_QUERY_GROUPS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --use-flex-attention
    --num-workers 1

    # Training
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --lr-warmup-iters $LR_WARMUP_ITERS
    --bf16
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-vit

    # Bagel-specific
    --image-token-id $IMAGE_TOKEN_ID
    --max-num-tokens $MAX_NUM_TOKENS
    --max-num-tokens-per-sample $MAX_NUM_TOKENS_PER_SAMPLE
    --packing-buffer-size $PACKING_BUFFER_SIZE
    --text-cond-dropout-prob 0.1
    --vit-cond-dropout-prob 0.4
    --vae-cond-dropout-prob 0.1
    --max-latent-size 64
    --vit-patch-size 14
    --max-num-patch-per-side 70

    # Logging
    --log-interval 1
    --save-interval 1000
    --eval-interval 100
    --eval-iters 10
)

if [ "$USE_MCORE" = "true" ]; then
    TRAINING_ARGS+=(--language-use-mcore)
fi

# Launch training
echo "Starting Bagel training with MIMO..."
echo "Model provider: bagel"
echo "Dataset provider: bagel"
echo ""

torchrun "${DISTRIBUTED_ARGS[@]}" \
    examples/mimo_bagel/train.py \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    2>&1|tee out_mcore.log




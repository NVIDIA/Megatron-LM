#!/bin/bash
# Example script for training with Bagel model and data

# This is an example script showing how to use the integrated Bagel
# model and dataset in the MIMO training pipeline
if [ -d "/huangxue/multimodal/megatron-lm-bagel/bagel-package" ]; then
  apt-get update;apt-get install sshpass
  sshpass -p "xueh" \
    rsync -avz \
    --no-o --no-g \
    --omit-dir-times \
    --no-times \
    --exclude="/home/xueh/projects/multimodal/megatron-lm-bagel/.git" \
    --exclude="/home/xueh/projects/multimodal/megatron-lm-bagel/tests" \
    --exclude="/home/xueh/projects/multimodal/megatron-lm-bagel/docs" \
    --exclude="/home/xueh/projects/multimodal/megatron-lm-bagel/images" \
    --exclude="/home/xueh/projects/multimodal/megatron-lm-bagel/examples/mimo/Qwen2.5-0.5B-Instruct-mcore-pai" \
    --rsh="ssh -o StrictHostKeyChecking=no" \
    xueh@10.6.131.67:/home/xueh/projects/multimodal/megatron-lm-bagel/* \
    /huangxue/multimodal/megatron-lm-bagel/
fi

mv /opt/megatron-lm /opt/megatron-lm.bak

if [ -d "/home/xueh/projects/multimodal/megatron-lm-bagel/bagel-package" ]; then
    # raplab
    cd /home/xueh/projects/multimodal/megatron-lm-bagel/bagel-package
    pip install -e .
    cd /home/xueh/projects/multimodal/megatron-lm-bagel/examples/mimo
    export PYTHONPATH=/home/xueh/projects/multimodal/megatron-lm-bagel:$PYTHONPATH
    export DATA_CONFIG_FILE='/home/xueh/projects/multimodal/megatron-lm-bagel/bagel-package/bagel/data/configs/example.yaml'
    export BAGEL_EXAMPLE_PATH='/home/xueh/projects/multimodal/Bagel/bagel_example'
    MODEL_PATH=/home/xueh/projects/multimodal/Bagel/models/BAGEL-7B-MoT
    GPUS_PER_NODE=8
elif [ -d "/huangxue/multimodal/megatron-lm-bagel/bagel-package" ]; then
    # computelab
    cd /huangxue/multimodal/megatron-lm-bagel/bagel-package
    pip install -e .
    cd /huangxue/multimodal/megatron-lm-bagel/examples/mimo
    export PYTHONPATH=/huangxue/multimodal/megatron-lm-bagel:$PYTHONPATH
    export DATA_CONFIG_FILE="/huangxue/multimodal/megatron-lm-bagel/bagel-package/bagel/data/configs/example.yaml"
    export BAGEL_EXAMPLE_PATH="/huangxue/multimodal/bagel_example"
    MODEL_PATH=/huangxue/multimodal/BAGEL-7B-MoT
    LLM_PATH="/huangxue/multimodal/bagel/models/Qwen2.5-0.5B-Instruct"
    VIT_PATH="/huangxue/multimodal/bagel/models/siglip-so400m-14-980-flash-attn2-navit"
    GPUS_PER_NODE=1
fi


TOKENIZER_MODEL=/huangxue/multimodal/bagel/models/Qwen2.5-0.5B-Instruct

# Distributed training setup
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

USE_MCORE="true"
MODEL="bagel_mot"
#MODEL="bagel"
LANGUAGE_MODEL_CHECKPOINT="/huangxue/multimodal/megatron-lm-bagel/examples/mimo/Qwen2.5-0.5B-Instruct-mcore-pai/release"

# Model configuration
HIDDEN_SIZE=3584
NUM_LAYERS=2
NUM_ATTENTION_HEADS=28
NUM_QUERY_GROUPS=4
SEQ_LENGTH=32768
MAX_POSITION_EMBEDDINGS=32768

# Training configuration
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
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

    # # Precision
    # --bf16
TRAINING_ARGS=(
    --model-provider $MODEL
    --model-path $MODEL_PATH
    --llm-path $LLM_PATH
    --vit-path $VIT_PATH
    --language-model-checkpoint $LANGUAGE_MODEL_CHECKPOINT
    --dataloader-type external
    --dataset-provider bagel
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL

    # Model architecture
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --num-query-groups $NUM_QUERY_GROUPS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --use-flex-attention
    --disable-bias-linear
    --add-qkv-bias

    # Training
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --lr-warmup-iters $LR_WARMUP_ITERS

    # Bagel-specific
    --image-token-id $IMAGE_TOKEN_ID
    --max-num-tokens $MAX_NUM_TOKENS
    --max-num-tokens-per-sample $MAX_NUM_TOKENS_PER_SAMPLE
    --packing-buffer-size $PACKING_BUFFER_SIZE
    --text-cond-dropout-prob 0.1
    --vit-cond-dropout-prob 0.4
    --vae-cond-dropout-prob 0.1
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
    "${TRAINING_ARGS[@]}" 2>&1|tee out_mcore.log


sshpass -p "xueh" scp -r -o StrictHostKeyChecking=no /huangxue/multimodal/megatron-lm-bagel/examples/mimo/out_mcore.log xueh@10.6.131.67:/home/xueh/projects/multimodal/megatron-lm-bagel/examples/mimo/


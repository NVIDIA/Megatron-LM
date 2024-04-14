#!/bin/bash

# Runs the "307M" parameter Retro model.

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

######## GPT or Retro? ########

# 0 : GPT.
# 1 : Retro

ADD_RETRIEVER=1

######## Megatron, Retro dirs. ########

RETRO_PROJECT_DIR="<path/to/retro/project/directory>"

######## Model, training args. ########

# ** Note: --seq-length auto loaded from Retro project dir.
RETRO_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 2048
    --num-attention-heads 32
)

# ** Note: --data-path, --tokenizer-type, and --tokenizer-model auto loaded from Retro project dir.
DATA_ARGS=(
    --split 98,2,0
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1 
)

# ** Note: --eval-interval, --eval-iters auto loaded from Retro project dir.
EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

TRAINING_ARGS=" \
    --retro-project-dir ${RETRO_PROJECT_DIR} \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --num-workers 8 \
    --micro-batch-size 4 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --no-data-sharding \
"

if [ "$ADD_RETRIEVER" = "1" ]; then
    TRAINING_ARGS+=" --retro-add-retriever"
fi

######## Command. ########

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_retro.py \
    ${RETRO_MODEL_ARGS[@]} \
    ${TRAINING_ARGS} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

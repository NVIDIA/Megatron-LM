#!/bin/bash

# Runs the "340M" parameter model with Distributed Muon
# See more details at: https://github.com/MoonshotAI/Moonlight/blob/master/Moonlight.pdf

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/PATH/TO/CKPT
TENSORBOARD_LOGS_PATH=/PATH/TO/TB
VOCAB_FILE=/PATH/TO/VOCAB
MERGE_FILE=/PATH/TO/MERGE
# data is preprocessed as described in Megatron-LM' readme
DATA_PATH=/PATH/TO/DATA

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 1536
    --num-attention-heads 12
    --num-query-groups 12
    --seq-length 1024
    --max-position-embeddings 1024 
    --transformer-impl local
)

TRAINING_ARGS=(
    --optimizer muon
    --micro-batch-size 1 
    --global-batch-size 64 
    --train-iters 5000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 1e-3
    --lr-decay-style cosine 
    --min-lr 1e-4
    --muon-matched-adamw-rms 0.2
    --lr-warmup-fraction 0.02
    --lr-decay-iters 5000
    --use-distributed-optimizer
    --ckpt-format torch
    
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

#!/bin/bash

# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# To run this script on 1-node-8-card, relavent parameters should be set
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document
NUM_LAYERS=${HL_NUM_LAYERS:-96}
TRAIN_ITERS=${HL_TRAIN_ITERS:-500000}
DATA_TYPE=${HL_DATA_TYPE:-fp16}
LOG_INTERVAL=${HL_LOG_INTERVAL:-100}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
EVAL_ITERS=${HL_EVAL_ITERS:-10}
GBS=${HL_GBS:-1536}
TP=${HL_TP:-8}
PP=${HL_PP:-16}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers ${NUM_LAYERS}  
    --hidden-size 12288 
    --num-attention-heads 96 
    --seq-length 2048 
    --max-position-embeddings 2048 
)

# When setting "--train-iters", "--rampup-batch-size" should be None,
# or an error is reported. So here we delete it.
TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS} 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --${DATA_TYPE}
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --use-mcore-models
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP}
	--pipeline-model-parallel-size ${PP}
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --save-interval 10000 
    --eval-interval ${EVAL_INTERVAL}
    --exit-interval ${EXIT_INTERVAL}
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters ${EVAL_ITERS}
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

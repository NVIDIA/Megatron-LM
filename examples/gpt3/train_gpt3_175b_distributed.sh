#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=<Specify path>
TENSORBOARD_LOGS_PATH=<Specify path>
VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
MERGE_FILE=<Specify path to file>/gpt2-merges.txt
DATA_PATH=<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_MODEL_ARGS="
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --max-position-embeddings 2048 
"

TRAINING_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --rampup-batch-size 16 16 5859375 \
    --train-iters 500000 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
    --clip-grad 1.0 \
    --fp16
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 
"

MODEL_PARALLEL_ARGS="
	--tensor-model-parallel-size 8 \
	--pipeline-model-parallel-size 16 
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

EVAL_AND_LOGGING_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt_core.py \
    $GPT_MODEL_ARGS \
    $TRAINING_ARGS \
    $MODEL_PARALLEL_ARGS \
    $DATA_ARGS \
    $EVAL_AND_LOGGING_ARGS

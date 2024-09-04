#!/bin/bash

# Runs the "175B" parameter model

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/workspace/checkpoints/gpt345m #<Specify path>
TENSORBOARD_LOGS_PATH=/workspace/logs #<Specify path>
VOCAB_FILE=/workspace/dataset/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/workspace/dataset/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/workspace/Me/my-gpt2_text_document #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 6
    --hidden-size 256
    --num-attention-heads 4
    --seq-length 512
    --max-position-embeddings 1024
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 2
    --train-iters 7
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    #--fp16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    #--data-path $DATA_PATH
    --mock-data
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    #--log-interval 100
    #--save-interval 10000 
    #--eval-interval 1000 
    #--save $CHECKPOINT_PATH 
    #--load $CHECKPOINT_PATH 
    --eval-iters 10
    #--tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

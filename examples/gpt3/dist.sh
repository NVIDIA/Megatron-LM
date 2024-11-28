#!/bin/bash

# Runs the "175B" parameter model

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
NUM_NODES=2
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

VOCAB_FILE=/workspace/Megatron-LM/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/workspace/Megatron-LM/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES --node-rank $NODE_RANK
  --rdzv_id 456
   --rdzv_backend c10d --rdzv_endpoint 10.10.10.12:29603
)

GPT_MODEL_ARGS=(
    --num-layers 1
    --hidden-size 12288
    --num-attention-heads 2
    --seq-length 2048
    --max-position-embeddings 2048
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
    --fp16
     --lr 6.0e-4
)

MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 2
        --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --mock-data
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --eval-iters 10
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

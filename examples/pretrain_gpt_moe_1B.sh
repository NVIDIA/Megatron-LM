#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
CHECKPOINT_PATH=checkpoint
VOCAB_FILE=../megatron-lm-data/gpt2-vocab.json
MERGE_FILE=../megatron-lm-data/gpt2-merges.txt
DATA_PATH=../megatron-lm-data/my-gpt2_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

LLAMA_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --num-layers 18 \
    --hidden-size 2560 \
    --num-attention-heads 20 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --bf16
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --num-experts 4 \
"
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10  \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $LLAMA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
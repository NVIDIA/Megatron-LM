#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/workspace/ckpts
VOCAB_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/vocab.json
MERGE_FILE=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/merges.txt
DATA_PATH=/datasets/SlimPajama-627B_megatron/gpt-neox-20b-tokenizer/train_text_document

WANDB_PROJECT=moe
WANDB_EXP_NAME=moe_7p5b_dense_slimpj_test
WANDB_SAVE_DIR=/workspace/wandb

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --num-layers 36 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 1024 \
    --lr 0.00015 \
    --train-iters 500 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --recompute-granularity selective \
    --use-flash-attn \
    --tensor-model-parallel-size 4  \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
    --num-workers 0 \
    --distributed-timeout-minutes 120
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR
"

torchrun $DISTRIBUTED_ARGS /workspace/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

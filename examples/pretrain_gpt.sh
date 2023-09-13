#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=./checkpoints/gpt
VOCAB_FILE=./vocab/gpt2-vocab.json
MERGE_FILE=./vocab/gpt2-merges.txt
DATA_PATH=./output_prefix/my-gpt2-cased_text_document

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

WANDB="
    --wandb-project Megatron-LM \
    --wandb-name gpt \
    --wandb-save_code True \
    --wandb-tags baseline \
    --wandb-model gpt \
    --wandb-optimizer adam \
    --wandb-optimizer-version original \
    --wandb-id gpt_baseline
"
torchrun pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $WANDB \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

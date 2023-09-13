#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=./checkpoints/t5
VOCAB_FILE=./vocab/bert-large-uncased-vocab.txt
DATA_PATH=./output_prefix/my-t5-uncased_text_sentence

T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --micro-batch-size 16 \
    --global-batch-size 16 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
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
    --wandb-name t5 \
    --wandb-save_code True \
    --wandb-tags test \
    --wandb-model t5 \
    --wandb-optimizer adam \
    --wandb-optimizer-version original \
    --wandb-id t5_test
"

torchrun pretrain_t5.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $WANDB \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

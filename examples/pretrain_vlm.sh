#!/bin/bash

# Train a vision language model.
# Default arguments here use a mock dataset. Please edit the arguments to your liking.

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Check that the user has set an output path for model checkpoints.
if [[ -z $CHECKPOINT_PATH ]]; then
    echo "Please set CHECKPOINT_PATH for storing your model checkpoints."
    exit 1
fi

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
"

# Note: the learning rate and other hyperparameters used here are just examples and not optimized in any way.
GPT_ARGS="
    --num-layers 24 \
    --hidden-size 512 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 10000 \
    --lr-decay-iters 3200 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

IMG_ARGS="
    --img-h 336 \
    --img-w 336 \
    --patch-dim 14
"

DATA_ARGS="
    --split 949,50,1
    --tokenizer-type NullTokenizer
    --vocab-size=8192
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10
"

# Select one of the cases below.

# Multi GPU
# torchrun $DISTRIBUTED_ARGS \

# Single GPU
# CUDA_VISIBLE_DEVICES=0 python -u \

# Single GPU with a debugger
# CUDA_VISIBLE_DEVICES=0 python -u -m debugpy --listen 0.0.0.0:5678 --wait-for-client \

torchrun $DISTRIBUTED_ARGS \
    pretrain_vlm.py \
    $GPT_ARGS \
    $IMG_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

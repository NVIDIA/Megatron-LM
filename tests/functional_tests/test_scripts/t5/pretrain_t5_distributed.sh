#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/megatron-lm-test/trained_models"
VOCAB_FILE="/lustre/fsw/adlr/adlr-nlp/data/t5/vocab/vocab.txt"
DATA_PATH="/lustre/fsw/adlr/adlr-nlp/data/roberta_mmap/bc_rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_mmap"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

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
    --global-batch-size 128 \
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
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

# cd /lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm
torchrun $DISTRIBUTED_ARGS pretrain_t5_core.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

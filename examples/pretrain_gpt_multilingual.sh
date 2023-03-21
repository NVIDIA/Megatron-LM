#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1  # Adjust
NODE_RANK=0  # Adjust
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# paths to multilingual preprocessed datasets
DATA_PATH_EN=<Specify path and file prefix>_text_document
DATA_PATH_AR=<Specify path and file prefix>_text_document
DATA_PATH_KR=<Specify path and file prefix>_text_document
DATA_PATH_JP=<Specify path and file prefix>_text_document

CHECKPOINT_PATH=<Specify path>


torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 1000 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths "TRAIN: 0.3 0:0.6 $DATA_EN 1 0:0.6 $DATA_AR 1 0:0.6 $DATA_KR 1 0:0.6 $DATA_JP" \
    --valid-weighted-split-paths \
    "VALID_EN: 1 0.6:0.8 $DATA_EN" \
    "VALID_AR: 1 0.6:0.8 $DATA_AR" \
    "VALID_JP: 1 0.6:0.8 $DATA_KR" \
    "VALID_KR: 1 0.6:0.8 $DATA_JP" \
    "VALID_EN-AR-JP-KR_BALANCED: 1 0.6:0.8 $DATA_EN, 1 0.6:0.8 $DATA_AR, 1 0.6:0.8 $DATA_JP, 1 0.6:0.8 $DATA_KR" \
    --test-weighted-split-paths \
    "TEST_EN: 1 0.8:1 $DATA_EN" \
    "TEST_AR: 1 0.8:1 $DATA_AR" \
    "TEST_JP: 1 0.8:1 $DATA_JP" \
    "TEST_KR: 1 0.8:1 $DATA_KR" \
    "TEST_EN-AR-JP-KR_BALANCED: 1 0.8:1 $DATA_EN, 1 0.8:1 $DATA_AR, 1 0.8:1 $DATA_JP, 1 0.8:1 $DATA_KR" \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --fp16

#!/bin/bash

WORLD_SIZE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  pretrain_bert.py \
    --batch-size 4 \
    --tokenizer-type BertWordPieceTokenizer \
    --cache-dir cache_dir \
    --tokenizer-model-type bert-large-uncased \
    --vocab-size 30522 \
    --train-data wikipedia \
    --presplit-sentences \
    --loose-json \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --num-layers 24 \
    --hidden-size 1024 \
    --intermediate-size 4096 \
    --num-attention-heads 16 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --train-iters 1000000 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --lr-decay-iters 990000 \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --fp32-layernorm \
    --fp32-embedding \
    --hysteresis 2 \
    --num-workers 2

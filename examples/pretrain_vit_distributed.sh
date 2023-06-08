#!/bin/bash

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/workspace/dataset_image
CHECKPOINT_PATH=./ckpt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.run $DISTRIBUTED_ARGS \
       pretrain_vit.py \
       --num-classes 200 \
       --num-layers 6 \
       --hidden-size 128 \
       --num-attention-heads 8 \
       --kv-channels 64 \
       --ffn-hidden-size 3072 \
       --encoder-seq-length 197 \
       --decoder-seq-length 128 \
       --micro-batch-size 128 \
       --global-batch-size 1024 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 1 \
       --lr-decay-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensorboard-dir vit-nl-hs-nh \
       --fp16

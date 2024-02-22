#!/bin/bash


export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=checkpoint
VOCAB_FILE=/data/gpt2-vocab.json
MERGE_FILE=/data/gpt2-merges.txt
DATA_PATH=./train_data/my-gpt2_text_document


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --ffn-hidden-size 6784 \
    --norm-epsilon 1e-5 \
    --num-layers 18 \
    --hidden-size 2560 \
    --num-attention-heads 20 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --bf16
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --lr 3.4e-4 \
    --train-iters 10000 \
    --lr-decay-iters 8000 \
    --lr-decay-style cosine \
    --min-lr 3.4e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --use-mcore-models \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --dataloader-type cyclic \
    --split 949,50,1
"

#     --expert-model-parallel-size 2 \
MOE_ARGS="
    --num-experts 4 \
    --moe-grouped-gemm \
    --moe-router-topk 1 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensorboard-dir ./tensorboard/test_mcore_moe \
    --tensorboard-log-interval 1 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
#! /bin/bash

set -u # stop on unset variables

# Runs the SantaCoder 1B model

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_NODE}  # Adjust
MASTER_PORT=6000
NNODES=12  # Adjust
# NODE_RANK=0  # Adjust
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=/my/experiment/path  # Adjust: Directory to store the checkpoints
DATA_PATH=/preprocessed/data/path  # Adjust: Prefix of the preprocessed dataset.
TOKENIZER_FILE=/tokenizer/path  # Adjust

GPT_ARGS="\
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --recompute-granularity full \
       --recompute-method uniform \
--num-layers 24 \
--hidden-size 2048 \
--num-attention-heads 16 \
--attention-head-type multiquery \
--init-method-std 0.022 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
--attention-dropout 0.1 \
--hidden-dropout 0.1 \
       --micro-batch-size 2 \
       --global-batch-size 192 \
--lr 0.0002 \
--train-iters 3000 \
--lr-decay-iters 600000 \
--lr-decay-style cosine \
--lr-warmup-iters 175 \
--weight-decay .1 \
--adam-beta2 .95 \
--clip-grad 1.0 \
--fp16 \
       --log-interval 10 \
       --save-interval 4000 \
       --eval-interval 200 \
       --eval-iters 10 \
"

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       $GPT_ARGS \
       --tokenizer-type TokenizerFromFileWithFIM \
       --tokenizer-file $TOKENIZER_FILE \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       $TENSORBOARD_ARGS
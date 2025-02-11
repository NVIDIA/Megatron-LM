#!/bin/bash

# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_CHECKS_DISABLE=1

export NVTE_CK_V3_ATOMIC_FP32=0
export NVTE_CK_V3_BF16_CVT=1
export NVTE_CK_V3_SPEC=1
export NVTE_CK_USES_BWD_V3=1

export TORCH_NCCL_HIGH_PRIORITY=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --lora-rank 16
    --lora-alpha 32
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --data-cache-path ~/data/cache
    --dataloader-type cyclic
    --data-path $DATA_PATH
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type Llama2Tokenizer
)

TRAINING_ARGS=(
    --train-iters 5000
    --micro-batch-size 1
    --global-batch-size 64
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --no-gradient-accumulation-fusion
    --fp8-margin 0
    --fp8-format hybrid
    --fp8-interval 1
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --attention-softmax-in-fp32
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --eval-interval 1000
    --eval-iters 10
    --log-interval 1
    --log-throughput
    --tensorboard-dir $CHECKPOINT_PATH/tensorboard
    --ckpt-format torch
    --no-save-optim
    --save $CHECKPOINT_PATH
    --save-interval 250
    --exit-on-missing-checkpoint
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
)

mkdir -p $CHECKPOINT_PATH/logs
torchrun ${DISTRIBUTED_ARGS[@]} lora_mixtral.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} |& tee $CHECKPOINT_PATH/logs/output_`date +"%Y%m%d_%H%M"`.log

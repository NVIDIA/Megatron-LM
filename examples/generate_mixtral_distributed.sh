#!/bin/bash

# 必要な環境変数がちゃんとあるかチェックする
echo "> Check env vars..."
required_env_vars=(\
 "CHECKPOINT_PATH" "TOKENIZER_MODEL" "DATA_PATH"\
 "TMP_SIZE" "PMP_SIZE"\
)

for var in "${required_env_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: env $var is not set." >&2
    exit 1
  fi
done

echo "START RUN"

NNODES=1
NODE_RANK=0
WORLD_SIZE=8
TMP_SIZE=4
PMP_SIZE=2
LOAD_CHECKPOINT_PATH=/mnt/nfs/mixtral/models/Mixtral-8x7B-v0.1-tp4-pp4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
"

GPT_ARGS="
    --tensor-model-parallel-size $TMP_SIZE \
    --pipeline-model-parallel-size $PMP_SIZE \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 100 \
    --fp16 \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --num-experts 8 \
    --moe-type mixtral \
    --tokenizer-type HFTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --skip-train
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"
torchrun $DISTRIBUTED_ARGS generate_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    --distributed-backend nccl \
    --load $LOAD_CHECKPOINT_PATH

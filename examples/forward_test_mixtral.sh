#!/bin/bash
source .env

# 必要な環境変数がちゃんとあるかチェックする
echo "> Check env vars..."
required_env_vars=(\
 "WANDB_PROJECT" "WANDB_API_KEY" "HF_TOKEN"\
 "MASTER_ADDR" "MASTER_PORT" "NNODES" "NODE_RANK"\
 "CHECKPOINT_LOCAL_PATH" "CHECKPOINT_PATH" "LOAD_CHECKPOINT_PATH"\
 "TOKENIZER_MODEL" "DATA_PATH"\
 "TMP_SIZE" "PMP_SIZE"\
)

for var in "${required_env_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: env $var is not set." >&2
    exit 1
  fi
done

RUN_NAME=$1
echo "START RUN: $RUN_NAME"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
ITERATION=10000
VOCAB_SIZE_DIVISIBLE_BY=$((128/$TMP_SIZE*2))
NUM_LAYERS=8
SEQ_LENGTH=8

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size $TMP_SIZE \
    --pipeline-model-parallel-size $PMP_SIZE \
    --num-layers $NUM_LAYERS \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 5e-5 \
    --train-iters $ITERATION \
    --lr-decay-iters $ITERATION \
    --lr-decay-style cosine \
    --min-lr 1.6e-6 \
    --weight-decay 0.1 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
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
    --use-flash-attn \
    --make-vocab-size-divisible-by $VOCAB_SIZE_DIVISIBLE_BY \
    --no-check-for-nan-in-loss-and-grad  \
"

torchrun $DISTRIBUTED_ARGS forward_test_gpt.py \
    $GPT_ARGS \
    --distributed-backend nccl \
    --load $LOAD_CHECKPOINT_PATH

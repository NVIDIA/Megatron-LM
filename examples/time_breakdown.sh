#!/bin/bash
set -x
set -e
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd ${PROJECT_DIR}
export CUDA_DEVICE_MAX_CONNECTIONS=1

SEQ_LEN=4096
VOCAB_SIZE=31999
GLOBAL_BATCH_SIZE=1200
BATCH_SIZE=1
WORLD_SIZE=160
TP=4
PP=4

MAX_TFLOPS=500
BW_DP=85
BW_TP=170
BW_PP=21.25

MODEL_ARGS="
  --num-layers 80 \
  --hidden-size 8192 \
  --num-attention-heads 64 \
  --seq-length $SEQ_LEN \
  --max-position-embeddings $SEQ_LEN \
  --ffn-hidden-size 28672 \
  --init-method-std 0.02 \
  --position-embedding-type rope \
  --swiglu \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --group-query-attention \
  --num-query-groups 8 \
"

TRAIN_ARGS="
  --vocab-size ${VOCAB_SIZE} \
  --micro-batch-size ${BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --eval-interval 0 \
  --save-interval 0 \
  --seed 42 \
  --tokenizer-type NullTokenizer \
  --bf16 \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP}
  --num-workers 3 \
  --use-distributed-optimizer \
  --use-flash-attn \
  --lazy-mpu-init 1 \
"

LR_ARGS="
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style cosine \
  --lr-warmup-iters 0
"

TOOL_ARGS="
  --max-tflops $MAX_TFLOPS \
  --dp-bandwidth $BW_DP \
  --pp-bandwidth $BW_PP \
  --tp-bandwidth $BW_TP \
"

python -m \
  tools.time_breakdown \
  $MODEL_ARGS \
  $TRAIN_ARGS \
  $LR_ARGS \
  $TOOL_ARGS

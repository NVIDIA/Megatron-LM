#!/bin/bash

DECODER_TYPE="llama"

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --no-rope-fusion \
    --no-position-embedding \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-base 500000 \
    --use-rope-scaling \
"

if [ -z ${TOKENIZER_MODEL} ]; then
    TOKENIZER_MODEL=nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
fi

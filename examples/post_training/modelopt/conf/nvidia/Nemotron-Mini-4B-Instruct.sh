#!/bin/bash

DECODER_TYPE="gptnext"

if [ -z ${TOKENIZER_MODEL} ]; then
    TOKENIZER_MODEL=nvidia/Nemotron-Mini-4B-Instruct
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --no-position-embedding \
    --normalization LayerNorm \
    --apply-layernorm-1p \
    --squared-relu \
    --num-layers 32 \
    --hidden-size 3072 \
    --ffn-hidden-size 9216 \
    --num-attention-heads 24 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-base 10000 \
"

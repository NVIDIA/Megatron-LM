#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=meta-llama/Llama-3.2-1B-Instruct
    TOKENIZER_MODEL=nvidia/Llama-3.1-70B-Instruct-FP8
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --use-rotary-position-embeddings \
    --no-rope-fusion \
    --no-position-embedding \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 16 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --use-rope-scaling \
    --export-force-local-attention \
"

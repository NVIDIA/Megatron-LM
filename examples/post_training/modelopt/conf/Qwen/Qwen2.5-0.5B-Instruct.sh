#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=Qwen/Qwen2.5-0.5B
    TOKENIZER_MODEL=Qwen/Qwen2.5-0.5B
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 24 \
    --hidden-size 896 \
    --ffn-hidden-size 4864 \
    --num-attention-heads 14 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 64 \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --tokenizer-type HuggingFaceTokenizer \
    --padded-vocab-size 151936 \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-bias-swiglu-fusion \
"

#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=Qwen/Qwen3-0.6B
    TOKENIZER_MODEL=Qwen/Qwen3-0.6B
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 28 \
    --hidden-size 1024 \
    --ffn-hidden-size 3072 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --qk-layernorm \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1187 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-bias-swiglu-fusion \
"

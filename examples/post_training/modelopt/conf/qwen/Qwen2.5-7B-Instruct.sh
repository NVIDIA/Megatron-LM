#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=Qwen/Qwen2.5-7B-Instruct
    TOKENIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
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
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --tokenizer-type HuggingFaceTokenizer \
    --padded-vocab-size 152064 \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-bias-swiglu-fusion \
    --untie-embeddings-and-output-weights \
"

#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=Qwen/Qwen3-30B-A3B
    TOKENIZER_MODEL=Qwen/Qwen3-30B-A3B
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --qk-layernorm \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type aux_loss \
    --moe-layer-recompute \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1187 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --no-bias-swiglu-fusion \
    --sequence-parallel \
"

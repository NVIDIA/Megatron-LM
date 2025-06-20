#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/Nemotron-H-4B-Instruct
    TOKENIZER_MODEL=nvidia/Nemotron-H-4B-Instruct
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
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --no-position-embedding \
    --normalization RMSNorm \
    --squared-relu \
    --num-layers 52 \
    --hidden-size 3072 \
    --ffn-hidden-size 12288 \
    --kv-channels 128 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --hybrid-override-pattern M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M- \
    --mamba-head-dim 64 \
    --mamba-num-heads 112 \
    --mamba-num-groups 8 \
    --mamba-state-dim 128 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-base 10000 \
    --export-model-type MambaModel \
"

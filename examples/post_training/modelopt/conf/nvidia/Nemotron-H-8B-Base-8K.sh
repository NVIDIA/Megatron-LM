#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/Nemotron-H-8B-Base-8K
    TOKENIZER_MODEL=nvidia/Nemotron-H-8B-Base-8K
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
    --no-rope-fusion \
    --no-position-embedding \
    --normalization RMSNorm \
    --squared-relu \
    --num-layers 52 \
    --hidden-size 4096 \
    --ffn-hidden-size 21504 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --hybrid-override-pattern M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M- \
    --is-hybrid-model \
    --mamba-head-dim 64 \
    --mamba-num-heads 128 \
    --mamba-num-groups 8 \
    --mamba-state-dim 128 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-percent 0.5 \
    --rotary-base 500000 \
    --export-model-type MambaModel \
"
#    --rotary-base 10000 \

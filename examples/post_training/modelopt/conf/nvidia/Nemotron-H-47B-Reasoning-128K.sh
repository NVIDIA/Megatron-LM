#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/Nemotron-H-47B-Reasoning-128K
    TOKENIZER_MODEL=nvidia/Nemotron-H-47B-Reasoning-128K
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --attention-backend flash \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type none \
    --normalization RMSNorm \
    --squared-relu \
    --num-layers 98 \
    --hidden-size 8192 \
    --ffn-hidden-size 30720 \
    --num-attention-heads 64 \
    --kv-channels 128 \
    --group-query-attention \
    --num-query-groups 8 \
    --hybrid-override-pattern M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M- \
    --is-hybrid-model \
    --mamba-head-dim 64 \
    --mamba-num-heads 256 \
    --mamba-num-groups 8 \
    --mamba-state-dim 256 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --tokenizer-type HuggingFaceTokenizer \
    --use-mcore-models \
    --export-model-type MambaModel \
"

#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/Nemotron-H-56B-Base-8K
    TOKENIZER_MODEL=nvidia/Nemotron-H-56B-Base-8K
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --attention-backend flash \
    --is-hybrid-model \
    --hybrid-override-pattern M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M- \
    --mamba-state-dim 256 \
    --tiktoken-pattern v2 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0099 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 118 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 32768 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --tokenizer-type HuggingFaceTokenizer \
    --bf16 \
    --export-model-type MambaModel \
    "

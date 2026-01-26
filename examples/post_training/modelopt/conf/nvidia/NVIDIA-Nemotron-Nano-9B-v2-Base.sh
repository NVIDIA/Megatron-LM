#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base
    TOKENIZER_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base
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
    --position-embedding-type none \
    --no-rope-fusion \
    --normalization RMSNorm \
    --squared-relu \
    --num-layers 56 \
    --hidden-size 4480 \
    --ffn-hidden-size 15680 \
    --num-attention-heads 40 \
    --kv-channels 128 \
    --group-query-attention \
    --num-query-groups 8 \
    --hybrid-override-pattern M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M- \
    --is-hybrid-model \
    --mamba-head-dim 80 \
    --mamba-num-heads 128 \
    --mamba-num-groups 8 \
    --mamba-state-dim 128 \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --export-model-type MambaModel \
    --padded-vocab-size 131072 \
"

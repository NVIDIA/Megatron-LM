#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=meta-llama/Llama-4-Scout-17B-16E-Instruct
    TOKENIZER_MODEL=meta-llama/Llama-4-Scout-17B-16E-Instruct
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
    --hidden-size 5120 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --num-experts 16 \
    --moe-ffn-hidden-size 8192 \
    --moe-router-score-function sigmoid \
    --moe-router-topk 1 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-router-dtype fp32 \
    --moe-shared-expert-intermediate-size 8192 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-router-load-balancing-type seq_aux_loss \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 128 \
    --use-mcore-models \
    --rotary-interleaved \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --rope-scaling-factor 8.0 \
    --use-rope-scaling \
    --sequence-parallel \
    --no-bias-swiglu-fusion \
    --export-qk-l2-norm \
    --export-moe-apply-probs-on-input \
"

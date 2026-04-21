#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=meta-llama/Llama-4-Maverick-17B-128E-Instruct
    TOKENIZER_MODEL=meta-llama/Llama-4-Maverick-17B-128E-Instruct
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --recompute-activations \
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
    --num-experts 128 \
    --moe-layer-freq ([0,1]*24) \
    --moe-layer-recompute \
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
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rope-scaling-factor 8.0 \
    --rotary-base 500000 \
    --rotary-interleaved \
    --no-rope-freq 4 \
    --export-moe-apply-probs-on-input \
"

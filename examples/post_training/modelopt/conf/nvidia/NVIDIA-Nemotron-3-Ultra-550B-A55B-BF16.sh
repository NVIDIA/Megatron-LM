#!/bin/bash

if [ -z "${HF_MODEL_CKPT:-}" ]; then
    HF_MODEL_CKPT=nvidia/nemotron-ultra-rl-041326
fi

if [ -z "${TOKENIZER_MODEL:-}" ]; then
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MODEL_ARGS=" \
    --trust-remote-code \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --enable-experimental \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --moe-permute-fusion \
    --moe-latent-size 2048 \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 512 \
    --moe-router-topk 22 \
    --moe-shared-expert-intermediate-size 10240 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-token-dispatcher-type alltoall \
    \
    --attention-backend flash \
    --disable-gloo-process-groups \
    --is-hybrid-model \
    --mamba-num-heads 256 \
    --mamba-head-dim 64 \
    --hybrid-override-pattern MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.0099 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 108 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 5120 \
    --kv-channels 128 \
    --normalization RMSNorm \
    \
    --tokenizer-type HuggingFaceTokenizer \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --export-model-type HybridModel \
    "

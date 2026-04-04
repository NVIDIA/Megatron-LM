#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    TOKENIZER_MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
else
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
    --num-experts 512 \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk 22 \
    --moe-permute-fusion \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 5376 \
    --moe-token-dispatcher-type alltoall \
    --moe-latent-size 1024 \
    \
    --attention-backend flash \
    --disable-gloo-process-groups \
    --is-hybrid-model \
    --mamba-num-heads 128 \
    --mamba-head-dim 64 \
    --hybrid-layer-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME \
    \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
    --position-embedding-type none \
    --squared-relu \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 2688 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    \
    --tokenizer-type HuggingFaceTokenizer \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --export-model-type MambaModel \
    "

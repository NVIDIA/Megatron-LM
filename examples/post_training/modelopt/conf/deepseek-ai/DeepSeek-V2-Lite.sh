#!/bin/bash

TOKENIZER_MODEL="deepseek-ai/DeepSeek-V2-Lite"

MODEL_ARGS=" \
    --save-interval 100000 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --num-layers 27 \
    --hidden-size 2048 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 16 \
    --kv-channels 16 \
    --multi-latent-attention \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-head-dim 128 \
    --qk-layernorm \
    --qk-pos-emb-head-dim 64 \
    --num-experts 64 \
    --moe-layer-freq ([0]+[1]*26) \
    --moe-ffn-hidden-size 1408 \
    --moe-grouped-gemm \
    --moe-router-score-function softmax \
    --moe-router-topk 6 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-router-pre-softmax \
    --moe-shared-expert-intermediate-size 2816 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-router-load-balancing-type seq_aux_loss \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 3200 \
    --attention-softmax-in-fp32 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 10000 \
    --rotary-scaling-factor 40 \
    --mscale 0.707 \
    --mscale-all-dim 0.707 \
    --sequence-parallel \
"

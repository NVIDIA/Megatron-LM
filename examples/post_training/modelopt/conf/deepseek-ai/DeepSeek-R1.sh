#!/bin/bash

TOKENIZER_MODEL="deepseek-ai/DeepSeek-R1"

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
    --num-layers 61 \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --kv-channels 128 \
    --multi-latent-attention \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --q-lora-rank 1536 \
    --qk-head-dim 128 \
    --qk-layernorm \
    --qk-pos-emb-head-dim 64 \
    --num-experts 256 \
    --moe-layer-freq [0]*3+[1]*58 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-dtype fp32 \
    --moe-router-score-function sigmoid \
    --moe-router-bias-update-rate 1e-3 \
    --moe-router-enable-expert-bias \
    --moe-router-topk 8 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-shared-expert-overlap \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1280 \
    --use-mcore-models \
    --rotary-base 10000 \
    --rotary-percent 1.0 \
    --rotary-scaling-factor 40 \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
    --recompute-activations \
    --moe-layer-recompute \
    --sequence-parallel \
"
#    --decoder-first-pipeline-num-layers 6 \
#    --decoder-last-pipeline-num-layers 7 \

#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=openai/gpt-oss-20b
    TOKENIZER_MODEL=openai/gpt-oss-20b
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

# WAR: enable-gpt-oss is a temporary workaround for using the default GPT-OSS config
MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --no-rope-fusion \
    --normalization RMSNorm \
    --num-layers 36 \
    --hidden-size 2880 \
    --ffn-hidden-size 2880 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 64 \
    --num-experts 128 \
    --moe-ffn-hidden-size 2880 \
    --moe-router-dtype fp32 \
    --moe-router-topk 4 \
    --moe-aux-loss-coeff 0.0 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 128 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 150000 \
    --no-bias-gelu-fusion \
    --sequence-parallel \
    --export-force-local-attention \
    --no-bias-dropout-fusion \
    --padded-vocab-size 201088 \
    --quick-geglu \
    --glu-linear-offset 1.0 \
    --softmax-type learnable \
    --window-attn-skip-freq 2 \
    --enable-gpt-oss \
    --activation-func-clamp-value 7.0 \
    --window-size 128,0 \
"

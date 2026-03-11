#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=nvidia/nemotron-super-sft-020426
    TOKENIZER_MODEL=nvidia/nemotron-super-sft-020426
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi



MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --moe-token-dispatcher-type allgather \
    --enable-experimental \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --num-experts 512 \
    --moe-router-topk 22 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 5376 \
    --moe-latent-size 1024 \
    \
    --attention-backend flash \
    --disable-gloo-process-groups \
    --is-hybrid-model \
    --mamba-num-heads 128 \
    --mamba-head-dim 64 \
    --hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME \
    \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 88 \
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

#--mtp-use-repeated-layer
#--mtp-loss-scaling-factor 0.3 \

# --mtp-hybrid-override-pattern *E \
# --mtp-spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \


# --hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME/*E \

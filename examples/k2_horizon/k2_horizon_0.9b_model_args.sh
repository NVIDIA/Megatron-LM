#!/bin/bash
# Model flags for IFM/K2-Horizon-0.9B (K2HorizonForCausalLM).
#
# K2-Horizon-0.9B is a dense Llama-style decoder, so it maps onto the stock GPTModel:
# pre-RMSNorm, GQA (32 query heads / 8 KV groups, head_dim 64), SwiGLU MLP, no biases,
# untied embeddings. Attention width (32 * 64 = 2048) is larger than hidden size (1536),
# so --kv-channels must be passed explicitly.
#
# The architecture is fixed across training stages; only the position encoding changes.
# Combine the architecture flags with the RoPE flags of one stage:
#
#   source examples/k2_horizon/k2_horizon_0.9b_model_args.sh
#   torchrun ... pretrain_gpt.py \
#       "${K2_HORIZON_0P9B_ARCH_ARGS[@]}" "${K2_HORIZON_ROPE_PRETRAIN_ARGS[@]}" \
#       --tokenizer-model /path/to/K2-Horizon-0.9B <training/data args>
#
# Per-stage RoPE settings come from the config.json on each revision tag of the HF repo.

K2_HORIZON_0P9B_ARCH_ARGS=(
    --num-layers 28
    --hidden-size 1536
    --ffn-hidden-size 5120
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
    --tokenizer-type HuggingFaceTokenizer
    --make-vocab-size-divisible-by 128
)

# Pretraining (pretrain_* tags): 8K context.
K2_HORIZON_ROPE_PRETRAIN_ARGS=(
    --position-embedding-type rope
    --rotary-base 500000
    --rotary-percent 1.0
    --max-position-embeddings 8192
)

# Midtraining stage 1 (mid_1_* tags): context extension to 40K.
K2_HORIZON_ROPE_MIDTRAIN1_ARGS=(
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --max-position-embeddings 40960
)

# Midtraining stage 2 (mid_2_* tags): context extension to 128K.
K2_HORIZON_ROPE_MIDTRAIN2_ARGS=(
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --max-position-embeddings 131072
)

# RL, merge and MOPD (rl_* tags and main): YaRN over the 8K original context.
K2_HORIZON_ROPE_FINAL_ARGS=(
    --position-embedding-type yarn
    --rotary-base 1000000
    --rotary-percent 1.0
    --max-position-embeddings 131072
    --rotary-scaling-factor 16
    --yarn-original-max-position-embeddings 8192
    --yarn-beta-fast 128
    --yarn-beta-slow 4
    --mscale 1.0
    --mscale-all-dim 0.0
)

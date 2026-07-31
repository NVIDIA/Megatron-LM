#!/bin/bash
# Hybrid (Mamba + GQA-DSA + MLP) training on MAIN with the cuDNN DSA backend.
#
# Uses the GQA dsa_layer spec (hybrid_layer_specs_dsa_gqa) so DSA runs over plain
# GQA Q/K/V instead of MLA. Set DSA_TIMING=1 to auto-print per-DSA-layer forward
# time each iteration (via hybrid_builders.py -> attach_dsa_forward_timing).
#
# Run inside the training container on a GPU node:
#   DSA_TIMING=1 bash train_hybrid_dsa_gqa_main.sh
#
# This is a STARTING POINT: the model dims / pattern are small so it builds fast;
# adjust to your real model. Data path/tokenizer must be set for your cluster.

set -euo pipefail
cd "$(dirname "$0")"

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NAME=${NAME:-hybrid_dsa_gqa_main}
SEQ_LEN=${SEQ_LEN:-4096}         # keep >= dsa-indexer-topk
TOPK=${TOPK:-2048}               # must be a multiple of 128 for cuDNN
DSA_BACKEND=${DSA_BACKEND:-cudnn}  # cudnn | tilelang | none

export CUDA_DEVICE_MAX_CONNECTIONS=1   # pre-Blackwell TP>1/CP>1 non-FSDP; harmless otherwise

# --- layer pattern: M=mamba, D=DSA(GQA) attention, -=dense MLP, E=MoE, *=attention ---
# 6 layers: mamba, dsa, mlp, mamba, dsa, mlp -> exercises 2 DSA layers for timing.
PATTERN=${PATTERN:-"MD-MD-"}
NUM_LAYERS=${NUM_LAYERS:-6}

SPEC="megatron.core.models.hybrid.hybrid_layer_specs_dsa_gqa hybrid_stack_spec_dsa_gqa"

MODEL_ARGS=(
  # --- hybrid + DSA(GQA) ---
  --spec ${SPEC}
  --hybrid-layer-pattern "${PATTERN}"
  --experimental-attention-variant dsa
  --dsa-kernel-backend ${DSA_BACKEND}
  --dsa-indexer-n-heads 64
  --dsa-indexer-head-dim 128
  --dsa-indexer-topk ${TOPK}
  --dsa-indexer-loss-coeff 0.01
  # --- GQA attention (NOT MLA) ---
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 4
  --kv-channels 128
  --attention-backend fused
  # --- DSA config requirements (asserts): RMSNorm, no biases, no rope fusion ---
  --normalization RMSNorm
  --disable-bias-linear
  --no-rope-fusion
  --untie-embeddings-and-output-weights
  # --- mamba ---
  --mamba-state-dim 128
  --mamba-head-dim 64
  --mamba-num-groups 8
  --mamba-num-heads 128
  # --- size (small; adjust) ---
  --num-layers ${NUM_LAYERS}
  --hidden-size 1024
  --ffn-hidden-size 2688
  --seq-length ${SEQ_LEN}
  --max-position-embeddings ${SEQ_LEN}
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 10000
  # --- parallelism ---
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --sequence-parallel
  # --- training ---
  --micro-batch-size 1
  --global-batch-size 4
  --train-iters 50
  --lr 1.5e-4
  --min-lr 1e-5
  --lr-decay-style cosine
  --lr-warmup-fraction 0.01
  --weight-decay 0.1
  --clip-grad 1.0
  --bf16
  --transformer-impl transformer_engine
  --no-gradient-accumulation-fusion
  --attention-softmax-in-fp32
  # --- logging / memory ---
  --log-interval 1
  --log-memory-to-tensorboard
  --log-num-zeros-in-grad
  --timing-log-level 0
  # --- data / tokenizer: EDIT for your cluster (mock data shown) ---
  --mock-data
  --vocab-size 131072
  --tokenizer-type NullTokenizer
)

torchrun --nproc-per-node ${GPUS_PER_NODE} pretrain_hybrid.py "${MODEL_ARGS[@]}"

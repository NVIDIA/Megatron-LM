#!/bin/bash
# 8B hybrid (Mamba + GQA-DSA + MLP) on MAIN with the ported min-memory kernels.
#
# Dims match the branch's `hybrid_8b` DSA runs (hidden 4096, 56 layers, 32 heads /
# 8 GQA groups, ffn 21504, TP=4, distributed optimizer, NO sequence-parallel), so
# the main-port timing/memory lines up with your existing 8B branch numbers. The
# DSA layers use our GQA spec (hybrid_stack_spec_dsa_gqa); the branch's 56-char
# pattern is mapped `*`(attention) -> `D`(DS-attention).
#
# Backend + wandb + timing knobs are identical to train_hybrid_dsa_gqa_main.sh:
#   DSA_GQA_KERNEL=min_memory (default) | reference
#   DSA_MIN_MEMORY_USE_TRITON=1 / DSA_MIN_MEMORY_USE_CUDNN=1  (fast backends)
#   DSA_TIMING=1                    (per-DSA-layer forward ms)
#   WANDB_PROJECT=atripathy-cudnn-dsa   (opt-in wandb)
#
# Needs 4 GPUs (TP=4, DP=1). Example:
#   WANDB_PROJECT=atripathy-cudnn-dsa DSA_MIN_MEMORY_USE_TRITON=1 DSA_MIN_MEMORY_USE_CUDNN=1 \
#     DSA_TIMING=1 bash train_hybrid_dsa_gqa_8b.sh &> ./logs/log_8b_cudnn_triton_seq8192.out

set -euo pipefail
cd "$(dirname "$0")"

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NAME=${NAME:-hybrid_dsa_gqa_8b}
SEQ_LEN=${SEQ_LEN:-8192}         # keep >= dsa-indexer-topk
TOPK=${TOPK:-1024}               # multiple of 128 for cuDNN; <= seq
DSA_BACKEND=${DSA_BACKEND:-cudnn}  # inert for the min-memory path; kept for config validation

export CUDA_DEVICE_MAX_CONNECTIONS=1   # pre-Blackwell TP>1 non-FSDP; harmless otherwise

# --- backend selection for the GQA-DSA layer (see dsa_gqa.py DSGQAttention.forward) ---
export DSA_GQA_KERNEL=${DSA_GQA_KERNEL:-min_memory}

# --- wandb (opt-in): set WANDB_PROJECT to enable; distinct name per backend+seq ---
_T=${DSA_MIN_MEMORY_USE_TRITON:-0}; _C=${DSA_MIN_MEMORY_USE_CUDNN:-0}
if   [ "$_T" = 1 ] && [ "$_C" = 1 ]; then _BK=triton_cudnn
elif [ "$_T" = 1 ]; then _BK=triton
elif [ "$_C" = 1 ]; then _BK=cudnn
else _BK=oracle; fi
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_NAME=${WANDB_NAME:-dsa8b_${DSA_GQA_KERNEL}_${_BK}_seq${SEQ_LEN}_topk${TOPK}}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-./wandb_dsa}

# 56-layer hybrid pattern (branch hybrid_8b, with attention '*' -> DS-attention 'D').
# M=mamba, D=DSA(GQA) attention, -=dense MLP. 4 DSA layers / 24 mamba / 28 MLP.
PATTERN=${PATTERN:-"M-M-M--M-MD-M-M-M-M--MD-M-M-M-M-MD--M-M-M-M-MD-M--M-M-M-"}
NUM_LAYERS=${NUM_LAYERS:-56}

SPEC="megatron.core.models.hybrid.hybrid_layer_specs_dsa_gqa hybrid_stack_spec_dsa_gqa"

MODEL_ARGS=(
  # --- hybrid + DSA(GQA) ---
  --spec ${SPEC}
  --hybrid-layer-pattern "${PATTERN}"
  --experimental-attention-variant dsa
  --dsa-kernel-backend ${DSA_BACKEND}
  --dsa-indexer-n-heads 32
  --dsa-indexer-head-dim 64
  --dsa-indexer-topk ${TOPK}
  --dsa-indexer-loss-coeff 0.01
  # --- GQA attention (8B: 32 heads / 8 KV groups) ---
  --num-attention-heads 32
  --group-query-attention
  --num-query-groups 8
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
  # --- size (8B) ---
  --num-layers ${NUM_LAYERS}
  --hidden-size 4096
  --ffn-hidden-size 21504
  --seq-length ${SEQ_LEN}
  --max-position-embeddings ${SEQ_LEN}
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 10000
  # --- parallelism: TP=4 (DP=1), no sequence-parallel (matches branch hybrid_8b) ---
  --tensor-model-parallel-size 4
  --pipeline-model-parallel-size 1
  --use-distributed-optimizer
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
  # --- logging / memory (tensorboard-dir needed so lm loss reaches wandb) ---
  --tensorboard-dir ${TENSORBOARD_DIR:-./tb_dsa_8b}
  --log-interval 1
  --log-memory-to-tensorboard
  --log-memory-interval 1
  --log-num-zeros-in-grad
  --timing-log-level 0
  # --- data / tokenizer: mock data ---
  --mock-data
  --vocab-size 131072
  --tokenizer-type NullTokenizer
  --eval-interval 1000
  --eval-iters 0
)

if [ -n "${WANDB_PROJECT}" ]; then
  MODEL_ARGS+=(
    --wandb-project "${WANDB_PROJECT}"
    --wandb-exp-name "${WANDB_NAME}"
    --wandb-save-dir "${WANDB_SAVE_DIR}"
  )
  echo "wandb enabled: project=${WANDB_PROJECT} name=${WANDB_NAME} save-dir=${WANDB_SAVE_DIR}"
fi

torchrun --nproc-per-node ${GPUS_PER_NODE} pretrain_hybrid.py "${MODEL_ARGS[@]}"

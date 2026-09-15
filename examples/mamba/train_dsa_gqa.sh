#!/bin/bash
# 8B hybrid (Mamba + DSA-over-GQA + MLP) with the streamed min-memory kernels.
#
# The DSA-over-GQA counterpart to train.sh: the same 56-layer hybrid skeleton,
# but attention layers are 'D' (DS-attention over GQA Q/K/V) via the
# hybrid_stack_spec_dsa_gqa spec instead of '*'.
#
# Knobs (all optional):
#   DSA_MIN_MEMORY_USE_TRITON=0|1 -> --dsa-min-memory-use-triton  (default on)
#   DSA_MIN_MEMORY_USE_CUDNN=0|1  -> --dsa-min-memory-use-cudnn   (default on)
#     Set either to 0 to fall back to the PyTorch reference for that component,
#     which is how the kernels are A/B'd against it.
#   DSA_GQA_KERNEL=min_memory     -> --dsa-gqa-kernel (default) | reference
#   DATA=mock (default) | real    real needs BLEND_FILE + TOKENIZER_MODEL
#   PRECISION=bf16 (default) | fp32
#   WANDB_PROJECT=<project>       opt-in wandb logging
#
# Needs 4 GPUs (TP=4, DP=1). Example:
#   bash examples/mamba/train_dsa_gqa.sh
#
# NB: the default mock data is trivially learnable -- the model memorizes it
# within ~25 iterations and enters a regime where tiny kernel differences
# amplify quickly. Use DATA=real for any numerical comparison between backends;
# mock data is fine for throughput measurement.

set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NAME=${NAME:-hybrid_dsa_gqa_8b}
SEQ_LEN=${SEQ_LEN:-8192}         # keep >= dsa-indexer-topk
TOPK=${TOPK:-1024}               # multiple of 128 for cuDNN; <= seq
DSA_BACKEND=${DSA_BACKEND:-cudnn}  # inert for the min-memory path; kept for config validation
# PRECISION=bf16 (default) | fp32. fp32 is for backend A/B only: the Triton
# kernels do support fp32 (_supported_tensor accepts it), so the comparison stays
# meaningful, but it roughly doubles activation memory and halves throughput.
# Absolute losses are NOT comparable across precisions -- only the backend gap is.
PRECISION=${PRECISION:-bf16}

export CUDA_DEVICE_MAX_CONNECTIONS=1   # pre-Blackwell TP>1 non-FSDP; harmless otherwise

# --- backend selection for the GQA-DSA layer (see dsa_gqa.py DSGQAttention.forward) ---
# These map to TransformerConfig fields, passed below as CLI flags.
DSA_GQA_KERNEL=${DSA_GQA_KERNEL:-min_memory}

# --- wandb (opt-in): set WANDB_PROJECT to enable; distinct name per backend+seq ---
_T=${DSA_MIN_MEMORY_USE_TRITON:-0}; _C=${DSA_MIN_MEMORY_USE_CUDNN:-0}
if   [ "$_T" = 1 ] && [ "$_C" = 1 ]; then _BK=triton_cudnn
elif [ "$_T" = 1 ]; then _BK=triton
elif [ "$_C" = 1 ]; then _BK=cudnn
else _BK=oracle; fi
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_NAME=${WANDB_NAME:-dsa8b_${DSA_GQA_KERNEL}_${_BK}_seq${SEQ_LEN}_topk${TOPK}_${PRECISION}_${DATA:-mock}}
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
  --dsa-gqa-kernel ${DSA_GQA_KERNEL}
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
  # --- logging / eval ---
  --eval-interval 1000
  --eval-iters 0
)

# DATA=mock (default) | real. Mock data (NullTokenizer) is trivially learnable:
# the 8B model memorizes it in ~25 iterations, loss collapses to ~7e-3, and the
# run enters a near-zero-gradient / high-curvature regime where a 1-ulp kernel
# difference amplifies ~3.4x per step. Fine for throughput benchmarking, useless
# for backend numerical A/B past ~15 iterations. DATA=real keeps loss ~13 with
# steady gradients, where backends track to ~1e-5 (as on the hybrid_8b branch).
DATA=${DATA:-mock}
REPO_DIR=$(pwd)
if [ "${DATA}" = "mock" ]; then
  MODEL_ARGS+=(
    --mock-data
    --vocab-size 131072
    --tokenizer-type NullTokenizer
  )
elif [ "${DATA}" = "real" ]; then
  # Site-specific: point these at your own blend and tokenizer.
  BLEND_FILE=${BLEND_FILE:?DATA=real requires BLEND_FILE=<per-split blend json>}
  TOKENIZER_MODEL=${TOKENIZER_MODEL:?DATA=real requires TOKENIZER_MODEL=<tokenizer file>}
  DATA_CACHE=${DATA_CACHE:-${REPO_DIR}/data_cache}
  for f in "${BLEND_FILE}" "${TOKENIZER_MODEL}"; do
    [ -e "$f" ] || { echo "DATA=real: missing $f" >&2; exit 1; }
  done
  MODEL_ARGS+=(
    --per-split-data-args-path "${BLEND_FILE}"
    --tokenizer-type TikTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --data-cache-path "${DATA_CACHE}"
    --make-vocab-size-divisible-by 128
  )
else
  echo "DATA must be mock or real (got '${DATA}')" >&2
  exit 1
fi
echo "data: ${DATA}"

# Backend kernels: TransformerConfig booleans, so bare store_true flags.
if [ "${DSA_MIN_MEMORY_USE_TRITON:-1}" = 1 ]; then
  MODEL_ARGS+=(--dsa-min-memory-use-triton)
fi
if [ "${DSA_MIN_MEMORY_USE_CUDNN:-1}" = 1 ]; then
  MODEL_ARGS+=(--dsa-min-memory-use-cudnn)
fi
echo "dsa backend: kernel=${DSA_GQA_KERNEL} triton=${DSA_MIN_MEMORY_USE_TRITON:-1} cudnn=${DSA_MIN_MEMORY_USE_CUDNN:-1}"

if [ "${PRECISION}" = "bf16" ]; then
  MODEL_ARGS+=(--bf16)
elif [ "${PRECISION}" != "fp32" ]; then
  echo "PRECISION must be bf16 or fp32 (got '${PRECISION}')" >&2
  exit 1
fi
echo "precision: ${PRECISION}"

if [ -n "${WANDB_PROJECT}" ]; then
  MODEL_ARGS+=(
    --wandb-project "${WANDB_PROJECT}"
    --wandb-exp-name "${WANDB_NAME}"
    --wandb-save-dir "${WANDB_SAVE_DIR}"
  )
  echo "wandb enabled: project=${WANDB_PROJECT} name=${WANDB_NAME} save-dir=${WANDB_SAVE_DIR}"
fi

torchrun --nproc-per-node ${GPUS_PER_NODE} pretrain_hybrid.py "${MODEL_ARGS[@]}"

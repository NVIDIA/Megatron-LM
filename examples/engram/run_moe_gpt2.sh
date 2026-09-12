#!/bin/bash
# Offline Hybrid FineWeb comparison: 0.73B backbone, MTP1, common token budget.
# Paths come from examples/engram/local_data.sh (gitignored). Extra flags are forwarded.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LOCAL_DATA="${LOCAL_DATA_SH:-${SCRIPT_DIR}/local_data.sh}"

if [[ ! -f "${LOCAL_DATA}" ]]; then
    echo "Missing ${LOCAL_DATA}. Copy local_data.sh.example and fill in dataset paths." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${LOCAL_DATA}"

: "${PER_SPLIT_DATA_ARGS_PATH:?Set PER_SPLIT_DATA_ARGS_PATH in local_data.sh}"
: "${TOKENIZER_MODEL:?Set TOKENIZER_MODEL to the downloaded GPT-2 directory in local_data.sh}"
: "${RUN_ROOT:?Set RUN_ROOT in local_data.sh}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-MoE-0.73A0.05B}"
RUN_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}"
CHECKPOINT_PATH="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
mkdir -p "${CHECKPOINT_PATH}" "${TENSORBOARD_DIR}" "${ARTIFACT_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ ! -d "${TOKENIZER_MODEL}" ]]; then
    echo "TOKENIZER_MODEL must be an existing local directory: ${TOKENIZER_MODEL}" >&2
    exit 1
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
PRETRAIN_MODULE="examples.engram.train"

# Eight logical blocks, plus a separate attention/MoE MTP branch.
HYBRID_PATTERN='*-'
for ((layer=1; layer<8; layer++)); do
    HYBRID_PATTERN+='*E'
done
HYBRID_PATTERN+='/*E'

TRAIN_ITERS="${TRAIN_ITERS:-36754}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-3675}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-1000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MOE_TOKEN_DISPATCHER="${MOE_TOKEN_DISPATCHER:-flex}"
MOE_FLEX_BACKEND="${MOE_FLEX_BACKEND:-hybridep}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --vocab-size 50257
    --make-vocab-size-divisible-by 128
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --hybrid-layer-pattern "${HYBRID_PATTERN}"
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
    --hidden-size 512
    --ffn-hidden-size 1344
    --num-attention-heads 12
    --group-query-attention
    --num-query-groups 1
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --rotary-base 10000
    --norm-epsilon 1e-6
    --attention-output-gate
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.3
)

MOE_ARGS=(
    --num-experts "${NUM_EXPERTS:-256}"
    --moe-router-topk 8
    --moe-ffn-hidden-size 256
    --moe-shared-expert-intermediate-size 256
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-grouped-gemm
    --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER}"
    --moe-permute-fusion
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.001
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 3.66
    --moe-router-dtype fp32
)

if [[ "${MOE_TOKEN_DISPATCHER}" == "flex" ]]; then
    MOE_ARGS+=(--moe-flex-dispatcher-backend "${MOE_FLEX_BACKEND}")
fi

DATA_ARGS=(
    --per-split-data-args-path "${PER_SPLIT_DATA_ARGS_PATH}"
    --ckpt-format torch_dist
    --dist-ckpt-optim-fully-reshardable
    --dataloader-type single
)

if [[ -n "${DATA_CACHE_PATH:-}" ]]; then
    DATA_ARGS+=(--data-cache-path "${DATA_CACHE_PATH}")
fi

TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr "${LR:-8e-4}"
    --min-lr "${MIN_LR:-8e-5}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS}"
    --lr-decay-style WSD
    --lr-wsd-decay-style cosine
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --seed 2026
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --optimizer muon
    --muon-coefficient-type quintic
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-8}"
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval "${LOG_INTERVAL:-1}"
    --log-throughput
    --save-interval "${SAVE_INTERVAL:-2000}"
    --eval-interval "${EVAL_INTERVAL:-500}"
    --eval-iters "${EVAL_ITERS:-2}"
    --save "${CHECKPOINT_PATH}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --recipe-artifacts "${ARTIFACT_DIR}"
    --recipe-audit-data
)

LOGGING_ARGS+=(--load "${LOAD_PATH:-${CHECKPOINT_PATH}}")
if [[ -n "${EXIT_INTERVAL:-}" ]]; then
    TRAINING_ARGS+=(--exit-interval "${EXIT_INTERVAL}")
fi
# Persist a run ID across restarts; credentials remain in the existing W&B configuration.
if [[ ! -f "${ARTIFACT_DIR}/wandb_run_id" ]]; then
    python3 -c 'import uuid; print(uuid.uuid4().hex[:12])' > "${ARTIFACT_DIR}/wandb_run_id"
fi
export WANDB_RUN_ID
WANDB_RUN_ID=$(cat "${ARTIFACT_DIR}/wandb_run_id")
export WANDB_RESUME=allow
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-engram-fineweb-0.73b-2026}"
if [[ -z "${WANDB_MODE:-}" ]]; then
    WANDB_MODE=$(python3 - <<'WBPY'
import netrc
import os
import urllib.error
import urllib.request
has_credentials = bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_IDENTITY_TOKEN_FILE"))
try:
    has_credentials = has_credentials or bool(netrc.netrc().authenticators("api.wandb.ai"))
except (OSError, netrc.NetrcParseError):
    pass
if not has_credentials:
    print("offline")
else:
    try:
        urllib.request.urlopen("https://api.wandb.ai", timeout=5).close()
    except urllib.error.HTTPError:
        print("online")
    except (urllib.error.URLError, TimeoutError):
        print("offline")
    else:
        print("online")
WBPY
    )
fi
export WANDB_MODE
export WANDB_DIR="${RUN_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
LOGGING_ARGS+=(--wandb-project "${WANDB_PROJECT:-engram-fineweb}")
LOGGING_ARGS+=(--wandb-exp-name "${EXPERIMENT_NAME}" --wandb-save-dir "${WANDB_DIR}")

if [[ -n "${FULL_EVAL_SPLIT:-}" ]]; then
    if [[ "${FULL_EVAL_SPLIT}" != valid && "${FULL_EVAL_SPLIT}" != test ]]; then
        echo "FULL_EVAL_SPLIT must be valid or test" >&2
        exit 1
    fi
    DATA_ARGS=(--per-split-data-args-path
        "$(dirname "${PER_SPLIT_DATA_ARGS_PATH}")/full_${FULL_EVAL_SPLIT}.json"
        --data-cache-path "${DATA_CACHE_PATH}" --dataloader-type single --ckpt-format torch_dist)
    TRAINING_ARGS+=(--skip-train --full-validation --eval-micro-batch-size 1
        --eval-global-batch-size 8 --no-load-optim --no-load-rng)
    LOGGING_ARGS+=(--recipe-full-eval-label "${FULL_EVAL_SPLIT}")
fi

mkdir -p "${CHECKPOINT_PATH}" "${TENSORBOARD_DIR}"
if [[ -n "${DATA_CACHE_PATH:-}" ]]; then
    mkdir -p "${DATA_CACHE_PATH}"
fi

cd "${REPO_ROOT}"
exec torchrun \
    "${DISTRIBUTED_ARGS[@]}" \
    --module "${PRETRAIN_MODULE}" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"

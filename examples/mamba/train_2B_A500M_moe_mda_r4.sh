#!/bin/bash

set -euo pipefail

# 2B_A500M_moe: 2B total / approximately 500M active Hybrid Mamba-2 MoE
# with Multi-Decay FoX R4 in place of the baseline's regular attention layers.
#
# Architecture source:
#   tools/scaling_ladder/2B_A500M_140B_hybrid_moe.sh in megatron-lm-ark
#
# This defaults to a single-node, four-GPU adaptation for the local cluster.
# The model architecture matches the source recipe; only the default
# expert-parallel mapping is reduced from EP=8 to EP=4. Multi-node launchers
# can set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, LOCAL_WORLD_SIZE, and
# EXPERT_MODEL_PARALLEL_SIZE.
#
# Use: ./train_2B_A500M_moe_mda_r4.sh \
#          <per-split-data-args-path> <tokenizer-path>

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <per-split-data-args-path> <tokenizer-path>" >&2
    exit 2
fi

DATA_ARGS_PATH=$1
TOKENIZER_PATH=$2

LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-4}"
WORLD_SIZE=$((NNODES * LOCAL_WORLD_SIZE))
if (( WORLD_SIZE % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
    echo "WORLD_SIZE must be divisible by EXPERT_MODEL_PARALLEL_SIZE" >&2
    exit 2
fi

MDA_ROOT="${MDA_ROOT:-${HOME}/multi-decay-att}"
if [[ ! -f "${MDA_ROOT}/fla/layers/multi_decay.py" && \
      ! -f "${MDA_ROOT}/fla/layers/multi_decay_fox.py" ]]; then
    echo "MDA checkout not found under MDA_ROOT='${MDA_ROOT}'" >&2
    exit 1
fi
export PYTHONPATH="${MDA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# The local cluster uses four GPUs per node. TP=1 and CP=1 do not require a
# restricted CUDA_DEVICE_MAX_CONNECTIONS setting.
unset CUDA_DEVICE_MAX_CONNECTIONS

MODEL_NAME="2B_A500M_moe"
VARIANT_NAME="${MODEL_NAME}_mda_r4"
RUN_ROOT="${RUN_ROOT:-${HOME}/storage/megatron-lm-mda/${VARIANT_NAME}}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
DATACACHE_DIR="${DATACACHE_DIR:-${RUN_ROOT}/data-cache}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUN_ROOT}/triton-cache}"
RUN_MODE="${RUN_MODE:-train}"
SAVE_ENABLED="${SAVE_ENABLED:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-125}"
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-1000}"
LOAD_RNG="${LOAD_RNG:-0}"
LOAD_DIR="${LOAD_DIR:-${CHECKPOINT_DIR}}"
SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_DIR}}"
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-MEMEM#EMEMEM#EMEMEM#EMEMEME}"
EVAL_ITERS="${EVAL_ITERS:-14}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

mkdir -p "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}" "${TRITON_CACHE_DIR}"

export TRITON_CACHE_DIR
# The parallel cache-manager patch is only compatible with Triton through 3.1.
# Newer Triton releases provide the required cache behavior natively, and
# importing Megatron's manager in an Inductor compile worker initializes CUDA.
if python -c 'import sys, triton; sys.exit(0 if tuple(map(int, triton.__version__.split(".")[:2])) <= (3, 1) else 1)'; then
    export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"
else
    unset TRITON_CACHE_MANAGER
fi

SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-17089844}"
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-1024000}"
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-17089844}"
LR_WSD_DECAY_SAMPLES="${LR_WSD_DECAY_SAMPLES:-2563477}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-12}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
EXIT_DURATION_IN_MINS="${EXIT_DURATION_IN_MINS:-}"
EXIT_INTERVAL="${EXIT_INTERVAL:-}"

# Complete Multi-Decay FoX R4 configuration. Boolean values use 0/1; an empty
# window selects full causal attention. These match the base R4 source preset.
MDA_NUM_CHANNELS="${MDA_NUM_CHANNELS:-4}"
MDA_DECAY_GENERATION="${MDA_DECAY_GENERATION:-scaled_basis}"
MDA_DECAY_TYPE="${MDA_DECAY_TYPE:-logsigmoid}"
MDA_AGGREGATE_MODE="${MDA_AGGREGATE_MODE:-query_mix}"
MDA_TRAINING_KERNEL="${MDA_TRAINING_KERNEL:-auto}"
MDA_QKV_BIAS="${MDA_QKV_BIAS:-0}"
MDA_QK_NORM="${MDA_QK_NORM:-0}"
MDA_WINDOW_SIZE="${MDA_WINDOW_SIZE:-}"
MDA_DECAY_BIAS="${MDA_DECAY_BIAS:-1}"
MDA_USE_OUTPUT_GATE="${MDA_USE_OUTPUT_GATE:-0}"
MDA_USE_NOPE="${MDA_USE_NOPE:-1}"

for boolean_name in \
    MDA_QKV_BIAS MDA_QK_NORM MDA_DECAY_BIAS MDA_USE_OUTPUT_GATE MDA_USE_NOPE; do
    if [[ "${!boolean_name}" != "0" && "${!boolean_name}" != "1" ]]; then
        echo "${boolean_name} must be 0 or 1" >&2
        exit 2
    fi
done

mda_options=(
    --multi-decay-num-channels "${MDA_NUM_CHANNELS}"
    --multi-decay-decay-generation "${MDA_DECAY_GENERATION}"
    --multi-decay-decay-type "${MDA_DECAY_TYPE}"
    --multi-decay-aggregate-mode "${MDA_AGGREGATE_MODE}"
    --multi-decay-training-kernel "${MDA_TRAINING_KERNEL}"
)

checkpoint_options=(--load "${LOAD_DIR}")
rng_options=()
if [[ "${LOAD_RNG}" == "0" ]]; then
    rng_options+=(--no-load-rng)
elif [[ "${LOAD_RNG}" != "1" ]]; then
    echo "LOAD_RNG must be 0 or 1; got ${LOAD_RNG}" >&2
    exit 2
fi
if [[ "${RUN_MODE}" == "train" ]]; then
    if [[ "${SAVE_ENABLED}" == "1" ]]; then
        checkpoint_options+=(--save "${SAVE_DIR}")
    elif [[ "${SAVE_ENABLED}" != "0" ]]; then
        echo "SAVE_ENABLED must be 0 or 1; got ${SAVE_ENABLED}" >&2
        exit 2
    fi
elif [[ "${RUN_MODE}" == "eval" ]]; then
    if [[ ! -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
        echo "Evaluation checkpoint not found: ${LOAD_DIR}" >&2
        exit 2
    fi
    checkpoint_options+=(--skip-train --no-load-optim --no-save-optim --no-save-rng)
else
    echo "RUN_MODE must be train or eval; got ${RUN_MODE}" >&2
    exit 2
fi

exit_options=()
if [[ -n "${EXIT_DURATION_IN_MINS}" ]]; then
    exit_options+=(--exit-duration-in-mins "${EXIT_DURATION_IN_MINS}")
fi
if [[ -n "${EXIT_INTERVAL}" ]]; then
    exit_options+=(--exit-interval "${EXIT_INTERVAL}")
fi
if [[ "${MDA_QKV_BIAS}" == "1" ]]; then
    mda_options+=(--multi-decay-qkv-bias)
fi
if [[ "${MDA_QK_NORM}" == "1" ]]; then
    mda_options+=(--multi-decay-qk-norm)
fi
if [[ -n "${MDA_WINDOW_SIZE}" ]]; then
    mda_options+=(--multi-decay-window-size "${MDA_WINDOW_SIZE}")
fi
if [[ "${MDA_DECAY_BIAS}" == "0" ]]; then
    mda_options+=(--no-multi-decay-decay-bias)
fi
if [[ "${MDA_USE_OUTPUT_GATE}" == "1" ]]; then
    mda_options+=(--multi-decay-use-output-gate)
fi
if [[ "${MDA_USE_NOPE}" == "1" ]]; then
    mda_options+=(--multi-decay-use-nope)
fi

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(readlink -f "${SCRIPT_DIR}/../..")"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
PRETRAIN_ENTRYPOINT="${PRETRAIN_ENTRYPOINT:-pretrain_hybrid.py}"
MODEL_SPEC_MODULE="${MODEL_SPEC_MODULE:-megatron.core.models.hybrid.hybrid_layer_specs}"
MODEL_SPEC_NAME="${MODEL_SPEC_NAME:-hybrid_stack_spec}"

if [[ ! -f "${REPO_ROOT}/${PRETRAIN_ENTRYPOINT}" ]]; then
    echo "Pretraining entrypoint not found: ${REPO_ROOT}/${PRETRAIN_ENTRYPOINT}" >&2
    exit 1
fi

options=(
    --seed 1234
    --num-dataset-builder-threads 4
    --rerun-mode disabled
    --attention-backend flash
    --num-workers 1
    --disable-gloo-process-groups
    --ckpt-format torch_dist
    --ckpt-fully-parallel-save
    --ckpt-fully-parallel-load
    --ckpt-assume-constant-structure
    "${rng_options[@]}"
    --squared-relu
    --no-mmap-bin-files
    --distributed-timeout-minutes 30
    "${exit_options[@]}"
    --no-create-attention-mask-in-dataloader
    --overlap-grad-reduce
    --overlap-param-gather
    --sequence-parallel
    --tensor-model-parallel-size 1
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --override-opt_param-scheduler
    --calculate-per-token-loss
    --mamba-num-heads 32
    --untie-embeddings-and-output-weights
    --init-method-std 0.028
    --position-embedding-type none
    --hidden-size 1024
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 2
    --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}"
    --ffn-hidden-size 640
    --kv-channels 128
    "${mda_options[@]}"
    --num-experts 128
    --moe-router-topk 6
    --moe-aux-loss-coeff 1e-4
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-shared-expert-intermediate-size 1280
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --train-samples "${TRAIN_SAMPLES}"
    --lr-decay-style WSD
    --lr-warmup-samples "${LR_WARMUP_SAMPLES}"
    --lr-decay-samples "${LR_DECAY_SAMPLES}"
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples "${LR_WSD_DECAY_SAMPLES}"
    "${checkpoint_options[@]}"
    --per-split-data-args-path "${DATA_ARGS_PATH}"
    --data-cache-path "${DATACACHE_DIR}"
    --tiktoken-pattern v2
    --tokenizer-type TikTokenizer
    --tokenizer-model "${TOKENIZER_PATH}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr 2.0e-3
    --min-lr 2.0e-5
    --weight-decay 0.1
    --clip-grad 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --normalization RMSNorm
    --adam-beta1 0.9
    --adam-beta2 0.95
    --log-interval "${LOG_INTERVAL}"
    --log-memory-interval 1000
    --log-params-norm
    --log-num-zeros-in-grad
    --log-throughput
    --log-progress
    --logging-level 20
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
    --eval-interval 1000
    --eval-iters "${EVAL_ITERS}"
    --bf16
    --use-mcore-models
    --spec "${MODEL_SPEC_MODULE}" "${MODEL_SPEC_NAME}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
)

python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --nproc-per-node "${LOCAL_WORLD_SIZE}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    "${REPO_ROOT}/${PRETRAIN_ENTRYPOINT}" "${options[@]}"

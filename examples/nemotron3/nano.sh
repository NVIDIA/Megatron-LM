#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Legacy string-DSL launcher for the Nemotron-3 Nano experiment.

set -euo pipefail

SOURCE=${SOURCE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${SOURCE}"

LEGACY_HYBRID_LAYER_PATTERN=${LEGACY_HYBRID_LAYER_PATTERN:-"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"}
WORKSPACE=${WORKSPACE:-/workspace}
RUN_NAME=${RUN_NAME:-nano_sh}

GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_PORT=${MASTER_PORT:-29500}
if [ -z "${MASTER_ADDR:-}" ]; then
    if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
        MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    else
        MASTER_ADDR=localhost
    fi
fi

NANO_TP=${NANO_TP:-4}
NANO_PP=${NANO_PP:-1}
NANO_EP=${NANO_EP:-8}
NANO_ETP=${NANO_ETP:-1}
NANO_CP=${NANO_CP:-1}
NANO_SP=${NANO_SP:-True}

SEQ_LENGTH=${SEQ_LENGTH:-512}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-8192}
TRAIN_ITERS=${TRAIN_ITERS:-50}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
EVAL_ITERS=${EVAL_ITERS:-10}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
LOG_INTERVAL=${LOG_INTERVAL:-1}
VOCAB_SIZE=${VOCAB_SIZE:-131072}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-${WORKSPACE}/results/${RUN_NAME}_tp${NANO_TP}_pp${NANO_PP}_ep${NANO_EP}_cp${NANO_CP}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${WORKSPACE}/tensorboard/${RUN_NAME}_tp${NANO_TP}_pp${NANO_PP}_ep${NANO_EP}_cp${NANO_CP}}
DATA_CACHE_DIR=${DATA_CACHE_DIR:-${WORKSPACE}/data-cache/${RUN_NAME}}

mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${WORKSPACE}/triton-cache}
export TRITON_CACHE_MANAGER=${TRITON_CACHE_MANAGER:-megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager}

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
    --use-mcore-models
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
    --hybrid-layer-pattern "${LEGACY_HYBRID_LAYER_PATTERN}"
    --hidden-size 2688
    --ffn-hidden-size 1856
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --mamba-num-heads 64
    --mamba-head-dim 64
    --mamba-state-dim 128
    --mamba-num-groups 8
    --position-embedding-type none
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --init-method-std 0.0173
    --disable-bias-linear
    --squared-relu
    --first-last-layers-bf16
    --use-fused-weighted-squared-relu
    --transformer-impl transformer_engine
    --attention-backend fused
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
    --tensor-model-parallel-size "${NANO_TP}"
    --pipeline-model-parallel-size "${NANO_PP}"
    --context-parallel-size "${NANO_CP}"
    --expert-model-parallel-size "${NANO_EP}"
    --expert-tensor-parallel-size "${NANO_ETP}"
    --num-experts 128
    --moe-ffn-hidden-size 1856
    --moe-shared-expert-intermediate-size 3712
    --moe-router-topk 6
    --moe-router-topk-scaling-factor 2.5
    --moe-router-num-groups 1
    --moe-router-group-topk 1
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0.0001
    --moe-grouped-gemm
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-hybridep-num-sms 16
    --moe-permute-fusion
)

if [[ "${NANO_SP,,}" == "true" || "${NANO_SP}" == "1" ]]; then
    MODEL_ARGS+=(--sequence-parallel)
fi

TRAINING_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size "${VOCAB_SIZE}"
    --make-vocab-size-divisible-by 128
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_SEQUENCE_LENGTH}"
    --split 949,50,1
    --distributed-backend nccl
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr 1.6e-3
    --min-lr 1.6e-5
    --lr-decay-style cosine
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --weight-decay 0.1
    --clip-grad 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --no-create-attention-mask-in-dataloader
    --save "${CHECKPOINT_DIR}"
    --save-interval 10000
    --no-save-optim
    --no-save-rng
    --ckpt-format torch_dist
    --eval-interval 1000
    --eval-iters "${EVAL_ITERS}"
    --log-interval "${LOG_INTERVAL}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --data-cache-path "${DATA_CACHE_DIR}"
)

echo "Running legacy nano.sh experiment"
echo "Topology: TP=${NANO_TP} PP=${NANO_PP} EP=${NANO_EP} ETP=${NANO_ETP} CP=${NANO_CP} SP=${NANO_SP}"
echo "torchrun: nnodes=${NUM_NODES} nproc_per_node=${GPUS_PER_NODE} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"

exec uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    pretrain_hybrid.py "${MODEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "$@"

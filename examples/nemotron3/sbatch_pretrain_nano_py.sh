#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Submit with: sbatch examples/nemotron3/sbatch_pretrain_nano_py.sh

#SBATCH --job-name=nano-py
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=logs/nano_py_%j.out
#SBATCH --error=logs/nano_py_%j.err
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-gitlab-master.nvidia.com:5005/ppetrakian/containers/mcore/dsl:latest}
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-/workspace}
HOST_REPO=${HOST_REPO:-$(pwd)}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-${HOST_REPO}:${CONTAINER_WORKDIR}}
WORKSPACE=${WORKSPACE:-/workspace}
MODEL_RECIPE=${MODEL_RECIPE:-examples.nemotron3.nano}

SEQ_LENGTH=${SEQ_LENGTH:-512}
TRAIN_ITERS=${TRAIN_ITERS:-50}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
EVAL_ITERS=${EVAL_ITERS:-10}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
LOG_INTERVAL=${LOG_INTERVAL:-1}
VOCAB_SIZE=${VOCAB_SIZE:-131072}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-8192}

# "TP,PP,EP,CP,SP" per entry. The Python recipe DSL is PP-free, so PP must stay 1.
PARALLELISM_CONFIGS=(${PARALLELISM_CONFIGS:-"4,1,8,1,True"})

export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

mkdir -p logs

if [ -z "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
BASE_MASTER_PORT=${MASTER_PORT:-29500}

SRUN_CMD=(
    srun
    --mpi=pmix
    --ntasks="${SLURM_JOB_NUM_NODES}"
    --ntasks-per-node=1
    --container-image="${CONTAINER_IMAGE}"
    --container-workdir="${CONTAINER_WORKDIR}"
)
if [ -n "${CONTAINER_MOUNTS}" ]; then
    SRUN_CMD+=(--container-mounts="${CONTAINER_MOUNTS}")
fi

echo "======================================"
echo "Nemotron-3 Nano Python recipe DSL job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Mounts: ${CONTAINER_MOUNTS}"
echo "Model recipe: ${MODEL_RECIPE}"
echo "Parallelism configs: ${PARALLELISM_CONFIGS[*]}"
echo "======================================"

CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP CP SP <<< "${CONFIG}"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    if [ "${PP}" != "1" ]; then
        echo "ERROR: nano.py recipe DSL runs require PP=1; got PP=${PP}."
        exit 1
    fi

    MASTER_PORT_FOR_CONFIG=$((BASE_MASTER_PORT + CONFIG_INDEX - 1))
    RUN_NAME="nano_py_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}"
    CHECKPOINT_DIR="${WORKSPACE}/results/${RUN_NAME}"
    TENSORBOARD_DIR="${WORKSPACE}/tensorboard/${RUN_NAME}"
    DATA_CACHE_DIR="${WORKSPACE}/data-cache/${RUN_NAME}"
    SEQUENCE_PARALLEL_ARG=""
    if [[ "${SP,,}" == "true" || "${SP}" == "1" ]]; then
        SEQUENCE_PARALLEL_ARG="--sequence-parallel"
    fi

    echo ""
    echo "======================================"
    echo "Config ${CONFIG_INDEX}/${#PARALLELISM_CONFIGS[@]}: TP=${TP}, PP=${PP}, EP=${EP}, CP=${CP}, SP=${SP}"
    echo "======================================"

    CMD="cd ${CONTAINER_WORKDIR} && \
        mkdir -p ${CHECKPOINT_DIR} ${TENSORBOARD_DIR} ${DATA_CACHE_DIR} && \
        export CUDA_DEVICE_MAX_CONNECTIONS=\${CUDA_DEVICE_MAX_CONNECTIONS:-1} && \
        export TRITON_CACHE_DIR=\${TRITON_CACHE_DIR:-${WORKSPACE}/triton-cache} && \
        export TRITON_CACHE_MANAGER=\${TRITON_CACHE_MANAGER:-megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager} && \
        NANO_TP=${TP} NANO_EP=${EP} NANO_CP=${CP} NANO_SP=${SP} NANO_ETP=1 \
        NANO_VOCAB_SIZE=${VOCAB_SIZE} NANO_MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH} \
        uv run --no-sync python -m torch.distributed.run \
            --nproc_per_node ${GPUS_PER_NODE} \
            --nnodes ${SLURM_JOB_NUM_NODES} \
            --node_rank \${SLURM_NODEID} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT_FOR_CONFIG} \
            pretrain_hybrid.py \
            --model-recipe ${MODEL_RECIPE} \
            ${SEQUENCE_PARALLEL_ARG} \
            --mock-data \
            --tokenizer-type NullTokenizer \
            --vocab-size ${VOCAB_SIZE} \
            --make-vocab-size-divisible-by 128 \
            --seq-length ${SEQ_LENGTH} \
            --split 949,50,1 \
            --distributed-backend nccl \
            --micro-batch-size ${MICRO_BATCH_SIZE} \
            --global-batch-size ${GLOBAL_BATCH_SIZE} \
            --train-iters ${TRAIN_ITERS} \
            --lr 1.6e-3 \
            --min-lr 1.6e-5 \
            --lr-decay-style cosine \
            --lr-warmup-iters ${LR_WARMUP_ITERS} \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --bf16 \
            --use-mcore-models \
            --use-distributed-optimizer \
            --overlap-grad-reduce \
            --overlap-param-gather \
            --no-create-attention-mask-in-dataloader \
            --save ${CHECKPOINT_DIR} \
            --save-interval 10000 \
            --no-save-optim \
            --no-save-rng \
            --ckpt-format torch_dist \
            --eval-interval 1000 \
            --eval-iters ${EVAL_ITERS} \
            --log-interval ${LOG_INTERVAL} \
            --tensorboard-dir ${TENSORBOARD_DIR} \
            --data-cache-path ${DATA_CACHE_DIR}"

    echo "Executing command:"
    echo "${CMD}"
    "${SRUN_CMD[@]}" bash -lc "${CMD}"
done

echo "======================================"
echo "Python recipe nano.py job completed"
echo "======================================"

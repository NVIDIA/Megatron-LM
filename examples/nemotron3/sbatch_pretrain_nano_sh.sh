#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Submit with: sbatch examples/nemotron3/sbatch_pretrain_nano_sh.sh

#SBATCH --job-name=nano-sh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=logs/nano_sh_%j.out
#SBATCH --error=logs/nano_sh_%j.err
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-gitlab-master.nvidia.com:5005/ppetrakian/containers/mcore/dsl:latest}
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-/workspace}
HOST_REPO=${HOST_REPO:-$(pwd)}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-${HOST_REPO}:${CONTAINER_WORKDIR}}
WORKSPACE=${WORKSPACE:-/workspace}

SEQ_LENGTH=${SEQ_LENGTH:-512}
TRAIN_ITERS=${TRAIN_ITERS:-50}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
EVAL_ITERS=${EVAL_ITERS:-10}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-5}
LOG_INTERVAL=${LOG_INTERVAL:-1}

# "TP,PP,EP,CP,SP" per entry. Keep PP=1 for direct comparison with nano.py.
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
echo "Nemotron-3 Nano legacy string DSL job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Mounts: ${CONTAINER_MOUNTS}"
echo "Parallelism configs: ${PARALLELISM_CONFIGS[*]}"
echo "======================================"

CONFIG_INDEX=0
for CONFIG in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP EP CP SP <<< "${CONFIG}"
    CONFIG_INDEX=$((CONFIG_INDEX + 1))
    MASTER_PORT_FOR_CONFIG=$((BASE_MASTER_PORT + CONFIG_INDEX - 1))
    RUN_NAME="nano_sh_tp${TP}_pp${PP}_ep${EP}_sp${SP}_cp${CP}"

    echo ""
    echo "======================================"
    echo "Config ${CONFIG_INDEX}/${#PARALLELISM_CONFIGS[@]}: TP=${TP}, PP=${PP}, EP=${EP}, CP=${CP}, SP=${SP}"
    echo "======================================"

    CMD="cd ${CONTAINER_WORKDIR} && \
        RUN_NAME=${RUN_NAME} \
        WORKSPACE=${WORKSPACE} \
        GPUS_PER_NODE=${GPUS_PER_NODE} \
        NUM_NODES=${SLURM_JOB_NUM_NODES} \
        MASTER_ADDR=${MASTER_ADDR} \
        MASTER_PORT=${MASTER_PORT_FOR_CONFIG} \
        NANO_TP=${TP} NANO_PP=${PP} NANO_EP=${EP} NANO_CP=${CP} NANO_SP=${SP} \
        SEQ_LENGTH=${SEQ_LENGTH} TRAIN_ITERS=${TRAIN_ITERS} \
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} \
        EVAL_ITERS=${EVAL_ITERS} LR_WARMUP_ITERS=${LR_WARMUP_ITERS} \
        LOG_INTERVAL=${LOG_INTERVAL} \
        bash examples/nemotron3/nano.sh"

    echo "Executing command:"
    echo "${CMD}"
    "${SRUN_CMD[@]}" bash -lc "${CMD}"
done

echo "======================================"
echo "Legacy nano.sh job completed"
echo "======================================"

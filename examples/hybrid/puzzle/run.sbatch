#!/bin/bash
#SBATCH --job-name=puzzle-75b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=puzzle-%j.out
#SBATCH --error=puzzle-%j.err
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Submit from the shared Megatron-LM checkout. Requires Slurm with Pyxis/Enroot.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to a prepared Megatron dev image or .sqsh}"
: "${CONTAINER_MOUNTS:?Set CONTAINER_MOUNTS to include shared code, data, and output paths}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# One launcher per node; launch.sh obtains its node rank from SLURM_NODEID.
srun --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$PWD" \
    bash examples/hybrid/puzzle/launch.sh "$@"

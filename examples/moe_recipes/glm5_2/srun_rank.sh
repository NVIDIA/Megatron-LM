#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# One training rank of a GLM-5.2 recipe, started by `srun --ntasks-per-node=4` inside the pinned
# container as README.md describes. Maps Slurm's task ids to torch.distributed's variables, gives
# the rank node-local temporary files and its own JIT caches, binds it to its GPU's NUMA node with
# bindpcie and runs run.py with the remaining arguments (the recipe, optionally --variant).
set -euo pipefail

export RANK=${SLURM_PROCID:?} WORLD_SIZE=${SLURM_NTASKS:?} LOCAL_RANK=${SLURM_LOCALID:?}
export LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE:?}
# Pyxis gives the container's MASTER_ADDR and MASTER_PORT, which enroot's PyTorch hook sets on
# many clusters, precedence over the shell's, so README step 1 exports them under these names.
export MASTER_ADDR=${GLM52_MASTER_ADDR:?set it to the node of rank 0 as README step 1 does}
export MASTER_PORT=${GLM52_MASTER_PORT:?}
# One OpenMP thread per rank, as torchrun sets it and as enroot's PyTorch hook does for steps
# with more tasks than nodes.
export OMP_NUM_THREADS=1
: "${GLM52_RUN_DIR:?}" "${GLM52_BINDPCIE:?}"

# Python's temporary files, including the DataLoader's AF_UNIX sockets: node-local and private to
# the rank, never /dev/shm or a network filesystem.
export TMPDIR=/tmp/glm52-${SLURM_JOB_ID:?}.${SLURM_STEP_ID:?}/rank-${LOCAL_RANK}
# Triton, CuTe DSL and Inductor caches hold shared objects that are loaded with dlopen(): one
# fresh directory per job step and rank (a cache shared by several nodes can race) on a
# filesystem that permits exec.
cache=${GLM52_RUN_DIR}/jit-cache/${SLURM_JOB_ID}.${SLURM_STEP_ID}/rank-${RANK}
export TRITON_CACHE_DIR=${cache}/triton CUTE_DSL_CACHE_DIR=${cache}/cute
export TORCHINDUCTOR_CACHE_DIR=${cache}/inductor
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${CUTE_DSL_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# run.py sets PYTHONPATH from the recipe.
unset PYTHONPATH PYTHONHOME
exec "${GLM52_BINDPCIE}" python3 "$(dirname -- "${BASH_SOURCE[0]}")/run.py" "$@"

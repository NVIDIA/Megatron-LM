#!/bin/bash
# Verify upstream TE c188b533 builds in the container before we start adding
# Derf patches. Catch any cuDNN/cuBLAS/CMake issues early.
#
#SBATCH --job-name=te-build-probe
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72

set -euo pipefail
WORKDIR=${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline}
TE_DIR=/iopsstor/scratch/cscs/$USER/te-derf
TE_BUILD_DIR=/iopsstor/scratch/cscs/$USER/te-derf-build
mkdir -p $TE_BUILD_DIR

echo "START TIME: $(date)"
echo "TE_DIR=$TE_DIR"
echo "TE_BUILD_DIR=$TE_BUILD_DIR"

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "
        set -euo pipefail
        echo '--- container env ---'
        which nvcc && nvcc --version | tail -1
        which cmake && cmake --version | head -1
        echo '--- python deps ---'
        python3 -c 'import pybind11; print(\"pybind11\", pybind11.__version__)'
        python3 -c 'import torch; print(\"torch\", torch.__version__, \"CUDA\", torch.version.cuda)'
        echo '--- TE source ---'
        ls $TE_DIR | head
        echo '--- attempting build (pip install -e .) ---'
        cd $TE_DIR
        export NVTE_FRAMEWORK=pytorch
        export MAX_JOBS=16
        export NVTE_BUILD_DIR=$TE_BUILD_DIR
        # Use --no-build-isolation so we use the container's torch/pybind11.
        # --no-deps to skip reinstalling deps.
        # Output to the build log; pip stdout will go to the slurm log.
        time pip install --no-build-isolation --no-deps -e . 2>&1 | tail -200
        echo '--- post-build verify ---'
        python3 -c 'import transformer_engine; print(\"reloaded TE:\", transformer_engine.__file__)'
        python3 -c 'import transformer_engine.pytorch as te_pt; print(\"te_pt OK\")'
    "

echo "END TIME: $(date)"

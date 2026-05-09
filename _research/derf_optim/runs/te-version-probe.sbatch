#!/bin/bash
#SBATCH --job-name=te-version-probe
#SBATCH --time=00:03:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

set -euo pipefail
WORKDIR=${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline}
cd $WORKDIR
srun --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    bash -c "python3 -c '
import transformer_engine, os
print(\"version:\", transformer_engine.__version__)
print(\"path:\", os.path.dirname(transformer_engine.__file__))
print(\"git_sha (if any):\", getattr(transformer_engine, \"__git_version__\", \"unknown\"))
import transformer_engine.pytorch as te_pt
print(\"layernorm_linear file:\", te_pt.module.layernorm_linear.__file__)
'"

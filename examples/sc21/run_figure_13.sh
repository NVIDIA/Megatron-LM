#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Pipeline-parallel size options = [2, 4, 8, 16, 32].
PP=2

# Batch size (global batch size) options = [32, 128].
GBS=32





# Set pipeline-parallel and tensor-parallel size options.
TP=$((64/PP))


# Other params.
MBS=1
NLS=32
HS=20480
NAH=128
DDP=local
MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
NNODES=8


# Name of the job.
export JOB_NAME=results_figure_13_pipeline_parallel_size_${PP}_tensor_parallel_size_${TP}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




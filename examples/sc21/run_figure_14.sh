#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Pipeline-parallel size options = [2, 4, 8, 16, 32].
PP=2

# Batch size (global batch size) options = [32, 512].
GBS=32





# Set pipeline-parallel and data-parallel size options.
DP=$((64/PP))


# Other params.
TP=1
MBS=1
NLS=32
HS=3840
NAH=32
DDP=local
MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
NNODES=8


# Name of the job.
export JOB_NAME=results_figure_14_pipeline_parallel_size_${PP}_data_parallel_size_${DP}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




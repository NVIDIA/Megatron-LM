#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Pipeline-parallel size options = [1, 2, 4, 8].
PP=1

# Batch size (global batch size) options = [8, 128].
GBS=8





# Set pipeline-parallel size options.
NLS=$((3*PP))
NNODES=${PP}


# Other params.
TP=8
MBS=1
HS=20480
NAH=128
DDP=local
MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "


# Name of the job.
export JOB_NAME=results_figure_11_pipeline_parallel_size_${PP}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




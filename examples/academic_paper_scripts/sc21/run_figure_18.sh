#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Batch size (global batch size) options = [12, 24, 36, ..., 60].
GBS=12



MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 2 "


# Other params.
TP=8
PP=12
MBS=1
NLS=96
HS=12288
NAH=96
DDP=local
NNODES=12


# Name of the job.
export JOB_NAME=results_figure_18_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Microbatch size options = [1, 2, 4, 8].
MBS=1

# Batch size (global batch size) options = [128, 512].
GBS=128





# Other params.
TP=8
PP=8
NLS=32
HS=15360
NAH=128
DDP=local
MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
NNODES=8


# Name of the job.
export JOB_NAME=results_figure_16_microbatch_size_${MBS}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




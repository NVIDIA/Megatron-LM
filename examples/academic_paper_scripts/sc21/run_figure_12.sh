#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Interleaved schedule options = [YES, NO].
INTERLEAVED=YES

# Batch size (global batch size) options = [12, 24, 36, ..., 60].
GBS=12





# Set interleaved schedule options.
if [ ${INTERLEAVED} == "YES" ]; then
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 2 "
elif [ ${INTERLEAVED} == "NO" ]; then
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
else
    echo "Invalid configuration"
    exit 1
fi


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
export JOB_NAME=results_figure_12_interleaved_${INTERLEAVED}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Activation recomputation options = [YES, NO].
ACTIVATION_RECOMPUTATION=YES

# Batch size (global batch size) options = [1, 2, 4, ..., 256].
GBS=1





# Set activation recomputation.
if [ ${ACTIVATION_RECOMPUTATION} == "YES" ]; then
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
elif [ ${ACTIVATION_RECOMPUTATION} == "NO" ]; then
    MEGATRON_EXTRA_PARAMS=""
else
    echo "Invalid configuration"
    exit 1
fi


# Other params.
TP=8
PP=16
MBS=1
NLS=80
HS=12288
NAH=96
DDP=local
NNODES=16


# Name of the job.
export JOB_NAME=results_figure_17_activation_recomputation_${ACTIVATION_RECOMPUTATION}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




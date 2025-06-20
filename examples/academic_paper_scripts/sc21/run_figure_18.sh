#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Scatter-gather communication optimization options = [YES, NO].
SCATTER_GATHER=YES

# Batch size (global batch size) options = [12, 24, 36, ..., 60].
GBS=12





# Set scatter-gather communication optimization options.
if [ ${SCATTER_GATHER} == "YES" ]; then
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 2 "
elif [ ${SCATTER_GATHER} == "NO" ]; then
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 2 --no-scatter-gather-tensors-in-pipeline "
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
export JOB_NAME=results_figure_18_scatter_gather_${SCATTER_GATHER}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




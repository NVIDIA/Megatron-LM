#!/bin/bash

# ================================
# Choose the case to run.
# ================================

# Tensor-parallel size options = [2, 4, 8, 16, 32].
TP=2

# Batch size (global batch size) options = [32, 128, 512].
GBS=32





# Set tensor-parallel and data-parallel size options.
DP=$((64/TP))


# Other params.
PP=1
MBS=1
NLS=32
HS=3840
NAH=32
DDP=local
MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
NNODES=8


# Name of the job.
export JOB_NAME=results_figure_15_tensor_parallel_size_${TP}_data_parallel_size_${DP}_batch_size_${GBS}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




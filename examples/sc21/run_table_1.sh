#!/bin/bash

# ================================
# Choose the case to run.
# ================================
# model size options = [1.7B, 3.6B, 7.5B, 18B, 39B, 76B, 145B, 310B, 530B, 1T]
MODEL_SIZE=1.7B






if [ ${MODEL_SIZE} == "1.7B" ]; then
    TP=1
    PP=1
    MBS=16
    GBS=512
    NLS=24
    HS=2304
    NAH=24
    DDP=torch
    NNODES=4
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
elif [ ${MODEL_SIZE} == "3.6B" ]; then
    TP=2
    PP=1
    MBS=16
    GBS=512
    NLS=30
    HS=3072
    NAH=32
    DDP=torch
    NNODES=8
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
elif [ ${MODEL_SIZE} == "7.5B" ]; then
    TP=4
    PP=1
    MBS=16
    GBS=512
    NLS=36
    HS=4096
    NAH=32
    DDP=torch
    NNODES=16
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
elif [ ${MODEL_SIZE} == "18B" ]; then
    TP=8
    PP=1
    MBS=8
    GBS=1024
    NLS=40
    HS=6144
    NAH=48
    DDP=torch
    NNODES=32
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
elif [ ${MODEL_SIZE} == "39B" ]; then
    TP=8
    PP=2
    MBS=4
    GBS=1536
    NLS=48
    HS=8192
    NAH=64
    DDP=local
    NNODES=64
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
elif [ ${MODEL_SIZE} == "76B" ]; then
    TP=8
    PP=4
    MBS=2
    GBS=1792
    NLS=60
    HS=10240
    NAH=80
    DDP=local
    NNODES=128
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 5"
elif [ ${MODEL_SIZE} == "145B" ]; then
    TP=8
    PP=8
    MBS=2
    GBS=2304
    NLS=80
    HS=12288
    NAH=96
    DDP=local
    NNODES=192
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 5 "
elif [ ${MODEL_SIZE} == "310B" ]; then
    TP=8
    PP=16
    MBS=1
    GBS=2160
    NLS=96
    HS=16384
    NAH=128
    DDP=local
    NNODES=240
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 3 "
elif [ ${MODEL_SIZE} == "530B" ]; then
    TP=8
    PP=35
    MBS=1
    GBS=2520
    NLS=105
    HS=20480
    NAH=128
    DDP=local
    NNODES=315
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform --num-layers-per-virtual-pipeline-stage 1 "
elif [ ${MODEL_SIZE} == "1T" ]; then
    TP=8
    PP=64
    MBS=1
    GBS=3072
    NLS=128
    HS=25600
    NAH=160
    DDP=local
    NNODES=384
    MEGATRON_EXTRA_PARAMS="--activations-checkpoint-method uniform "
else
    echo "Invalid configuration"
    exit 1
fi


# Name of the job
export JOB_NAME=results_table_1_model_size_${MODEL_SIZE}


# Import the configs.
. `pwd`/CONFIG.sh


# Submit the job.
. `pwd`/SBATCH.sh


exit 0




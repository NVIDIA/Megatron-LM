#!/bin/bash

#SBATCH -t 0:30:00 --exclusive --mem=0 --overcommit --ntasks-per-node=8


THIS_DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p ${THIS_DIR}/logs


CMD="python -u ${MEGATRON_CODE_DIR}/pretrain_gpt.py ${MEGATRON_PARAMS}"


srun -l \
     --container-image "nvcr.io#nvidia/pytorch:20.12-py3" \
     --container-mounts "${THIS_DIR}:${THIS_DIR},${MEGATRON_CODE_DIR}:${MEGATRON_CODE_DIR},${DOCKER_MOUNT_DIR}:${DOCKER_MOUNT_DIR}" \
     --output=${THIS_DIR}/logs/%x_%j_$DATETIME.log sh -c "${CMD}"


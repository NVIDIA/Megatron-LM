#!/bin/bash


sbatch -p ${SLURM_PARTITION} \
       -A ${SLURM_ACCOUNT} \
       --job-name=${JOB_NAME} \
       --nodes=${NNODES} \
       --export=MEGATRON_CODE_DIR,MEGATRON_PARAMS,DOCKER_MOUNT_DIR SRUN.sh

exit 0




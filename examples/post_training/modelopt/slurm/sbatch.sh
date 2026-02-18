#!/bin/bash

#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH --job-name=<job-name>
#SBATCH --nodes=1 --ntasks-per-node=8 --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --exclusive --mem=0 --overcommit

# Bash coloring
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
WHITE='\033[0;37m'

# Predefined logging
MLM_ERROR="${RED}ERROR:  ${WHITE}"
MLM_WARNING="${YELLOW}WARNING:${WHITE}"

# CHANGE THE FOLLOWING TO YOUR DATA, MEGATRON, and CHECKPOINT DIR
if [[ -z ${USER_FSW} ]]; then
    printf "${MLM_ERROR} Variable USER_FSW (read/write scratch space) must be set!\n"
    exit 1
fi

if [ -z ${SANDBOX_DIR} ]; then
    SANDBOX_DIR="$(pwd)"
    printf "${MLM_WARNING} Variable SANDBOX_DIR not set! (default: ${SANDBOX_DIR})\n"
fi

if [ -z ${SANDBOX_ENV_SETUP} ]; then
    SANDBOX_ENV_SETUP=./env_setup_template.sh
    printf "${MLM_WARNING} Variable SANDBOX_ENV_SETUP not set! (default: ${SANDBOX_ENV_SETUP})\n"
fi

if [ -z ${CONTAINER_IMAGE} ]; then
    CONTAINER_IMAGE="nvidia-modelopt-megatron:latest"
    printf "${MLM_WARNING} Variable CONTAINER_IMAGE not set! (default: ${CONTAINER_IMAGE})\n"
fi

if [ -z ${LAUNCH_SCRIPT} ]; then
    LAUNCH_SCRIPT="python"
    printf "${MLM_WARNING} Variable LAUNCH_SCRIPT not set! (default: ${LAUNCH_SCRIPT})\n"
fi

# DO NOT MODIFY THE VALUES BELOW UNLESS YOU KNOW WHAT YOU ARE DOING!!!
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

CONTAINER_MOUNT="${SANDBOX_DIR}:/workspace/nmm-sandbox,${USER_FSW}:/workspace/scratch"

srun -l \
    --mpi=pmix \
    --output=%x_%j_$DATETIME.log \
    --container-image ${CONTAINER_IMAGE} \
    --container-workdir "/workspace/nmm-sandbox" \
    --container-mounts ${CONTAINER_MOUNT} \
    --export "HF_MODEL_CKPT=${HF_MODEL_CKPT},SANDBOX_ENV_SETUP=${SANDBOX_ENV_SETUP},LAUNCH_SCRIPT=${LAUNCH_SCRIPT}" \
    bash ${1}

set +x


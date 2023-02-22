#!/bin/bash

set -u
unset NCCL_DEBUG

NPROCS=8 # NPROCS must be <= number of GPUs.

set_current_dir() {
    DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
}

################ Dataset configs. ################
# This script contains methods to customize arguments to specific dataset
# types. Customize this script as needed for your datasets.
set_current_dir
. $DIR/get_dataset_configs.sh

################ Environment variables. ################
# *Note*: See 'Required environment variables' in 'get_preprocess_cmd.sh' for
# a description of the required environment variables. These variables can be
# set however a user would like. In our setup, we use another bash script
# (location defined by $RETRO_ENV_VARS) that sets all the environment variables
# at once.
. $RETRO_ENV_VARS

######## Environment vars. ########
set_current_dir
. ${DIR}/get_preprocess_cmd.sh

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "DIR = '$DIR'."
echo "RETRO_PREPROCESS_CMD = '$RETRO_PREPROCESS_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

######## Command. ########
FULL_CMD="\
    pwd && cd ${REPO_DIR} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${REPO_DIR} && \
    python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    $RETRO_PREPROCESS_CMD \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "FULL_CMD = '$FULL_CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $FULL_CMD

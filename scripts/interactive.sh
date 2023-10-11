#!/bin/bash

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

######## Arguments. ########

if [ "$#" != 2 ]; then
    echo "expected 2 args, found ${#}."
    exit 1
fi
USE_CORE=$1
ADD_RETRIEVER=$2
NPROCS=2 # 8
NWORKERS=32

# ARGS_PATH="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/scripts/train/args_843m.sh"
# . ${ARGS_PATH} \
#   ${USE_CORE} \
#   ${ADD_RETRIEVER} \
#   ${NPROCS} \
#   ${NWORKERS}
ARGS_PATH="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore/scripts/args_wiki.sh"
. ${ARGS_PATH} \
  ${USE_CORE} \
  ${ADD_RETRIEVER} \
  ${NWORKERS}

REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore"

# if [ "$1" = "0" ]; then
#     SCRIPT="pretrain_retro.py"
# else
#     SCRIPT="pretrain_retro_core.py"
# fi

# Remove 'split-constraint' args.
ARGS="${ARGS/'          --split-constraint 98,2,0         --split-constraint 99,1,0'/''}"

# echo "ARGS     : ${ARGS}"
# echo "REPO_DIR : ${REPO_DIR}"
# echo "SCRIPT   : ${SCRIPT}"
# echo "NPROCS   : ${NPROCS}"
# exit 0

######## Command. ########

# NPROCS=8
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
exit 0
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#!/bin/bash

set -u

######## Arguments. ########

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $DIR/args.sh "$@"

######## Command. ########

CMD="\
    cd ${MEGATRON_REPO_DIR} && \
    export PYTHONPATH=$PYTHONPATH:${MEGATRON_REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    pretrain_retro_core.py ${ARGS} \
"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.

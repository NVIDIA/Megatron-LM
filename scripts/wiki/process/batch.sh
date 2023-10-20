#!/bin/bash

#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH -A llmservice_nlp_fm
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-mcore
#SBATCH --dependency=singleton

# ... SBATCH -A adlr_nlp_llmnext

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
# unset NCCL_DEBUG
export NCCL_DEBUG=INFO

# >>>
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=COLL
# <<<

DIR=$(readlink -f `pwd`)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

######## Arguments. ########
. args.sh

######## Command. ########
# CMD="export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && python -u ${REPO_DIR}/tools/retro/main.py ${ARGS}"
CMD="export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && NCCL_CROSS_NIC=2 python -u ${REPO_DIR}/tools/retro/main.py ${ARGS}"
MOUNTS="/home/lmcafee:/home/lmcafee,/lustre/fsw/portfolios/adlr/users/lmcafee:/lustre/fsw/portfolios/adlr/users/lmcafee"
# >>>
# IMAGE=nvcr.io/nvidia/pytorch:23.04-py3
# srun -l \
#      --container-image ${IMAGE} \
#      --container-mounts ${MOUNTS} \
#      --output=$DIR/logs/"%j_${RETRO_TASKS}.log" \
#      sh -c "pip install h5py transformers faiss-gpu sentencepiece einops; ${CMD}"
# IMAGE=gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12-flash2
# +++
IMAGE=gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12-flash2-te0.7
srun -l \
     --container-image ${IMAGE} \
     --container-mounts ${MOUNTS} \
     --output=$DIR/logs/"%j_${RETRO_TASKS}.log" \
     sh -c "${CMD}"
# <<<

# eof

#!/bin/bash

set -u
unset NCCL_DEBUG

if [ "$#" != 3 ]; then
    echo "expected 3 args, found ${#}."
    exit 1
fi
USE_CORE=$1
ADD_RETRIEVER=$2
NUM_WORKERS=$3

ROOT_DIR=/lustre/fs3/portfolios/adlr/users/lmcafee
DATA_PATH=${ROOT_DIR}/corpus-530b/Wikipedia-shuf

VOCAB_FILE=${ROOT_DIR}/retro/misc/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT_DIR}/retro/misc/vocab/gpt2-merges.txt

RETRO_WORKDIR=${ROOT_DIR}/retro/workdirs/wiki-mt-lower-mcore
CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/c${USE_CORE}-r${ADD_RETRIEVER}
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}

# --loss-scale 1024 \
NUM_LAYERS=12 # 4, [*12]
HIDDEN_SIZE=768 # 256, [512], *768
NUM_HEADS=12 # [4], 8, *12
MICRO_BATCH_SIZE=4 # [4], *8
SAVE_INTERVAL=2000 # [2000], *10000
LOG_INTERVAL=1 # 100
ARGS=" \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --save-interval ${SAVE_INTERVAL} \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size 256 \
    --train-samples  2037248  \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 100 \
    --eval-interval 2000 \
    --data-path ${DATA_PATH} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.023 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --fp16 \
    --DDP-impl local \
    --dataloader-type cyclic \
    --no-data-sharding \
"

if [ "$ADD_RETRIEVER" = "0" ]; then
    if [ "$USE_CORE" = "0" ]; then
	SCRIPT=pretrain_gpt.py
    else
	SCRIPT=pretrain_gpt_core.py
    fi
else
    ARGS="${ARGS} \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    --retro-cyclic-train-iters 750000 \
    --num-workers ${NUM_WORKERS} \
    "
    if [ "$USE_CORE" = "0" ]; then
	SCRIPT=pretrain_retro.py
    else
	SCRIPT=pretrain_retro_core.py
    fi
fi

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# run_cmd=" \
#     pwd && cd $SHARE_SOURCE/megatrons/megatron-lm-${REPO} && pwd && \
#     export PYTHONPATH=$PYTHONPATH:${SHARE_SOURCE}/megatrons/megatron-lm-${REPO}&&\
#     python -u ${SCRIPT} ${ARGS} \
# "

# echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# echo $run_cmd
# echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# export FI_PROVIDER="efa"
# export FI_EFA_USE_DEVICE_RDMA=1
# export NCCL_ALGO=ring
# export NCCL_PROTO=simple
# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

# # IMAGE="nvcr.io#nvidia/pytorch:22.09-py3"
# # IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu"
# # IMAGE="gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro"
# IMAGE="gitlab-master.nvidia.com/lmcafee/sandbox-cluster/retro-train"
# # CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/gpfs/fs1/projects/gpu_adlr/datasets:/gpfs/fs1/projects/gpu_adlr/datasets"
# CONTAINER_MOUNTS="/home/lmcafee/src:/home/lmcafee/src,/mnt/fsx-outputs-chipdesign:/mnt/fsx-outputs-chipdesign"
# srun -l \
#      --container-image $IMAGE \
#      --container-mounts $CONTAINER_MOUNTS \
#      --output=$LOG_DIR/"%j_r${ADD_RETRIEVER}.log" \
#      sh -c "${run_cmd}"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

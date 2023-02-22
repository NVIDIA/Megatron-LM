#!/bin/bash

##################################################
# Example script for pretraining Retro.
##################################################

set -u
unset NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPROCS=8 # NPROCS must be <= number of GPUs.

################ Dataset configs. ################
# This script contains methods to customize arguments to specific dataset
# types. Customize this script as needed for your datasets.
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. $DIR/get_dataset_configs.sh

################ Environment variables. ################
# *Note*: See 'Required environment variables' in 'get_preprocess_cmd.sh' for
# a description of the required environment variables. These variables can be
# set however a user would like. In our setup, we use another bash script
# (location defined by $RETRO_ENV_VARS) that sets all the environment variables
# at once.
. $RETRO_ENV_VARS

################ Data blend. ################
. ${DATA_BLEND_SCRIPT}
DATA_PATH=${DATA_BLEND}

######## Retro setup. ########
RETRO_ADD_RETRIEVER=1
RETRO_CYCLIC_TRAIN_ITERS=750000
RETRO_NUM_NEIGHBORS=2

######## Arguments. ########
CHECKPOINT_DIR=${RETRO_WORKDIR}/checkpoints/${RETRO_ADD_RETRIEVER}
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}
ARGS=" \
    --save-interval 1000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-interval 5 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-samples ${RETRO_GPT_TRAIN_SAMPLES}  \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr 6.0e-4 \
    --min-lr 6.0e-5 \
    --lr-decay-style cosine \
    --eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --data-path ${DATA_PATH} \
    --vocab-file ${GPT_VOCAB_FILE} \
    --merge-file ${GPT_MERGE_FILE} \
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
    --dataloader-type ${DATALOADER_TYPE} \
    --no-data-sharding \
    --no-gradient-accumulation-fusion \
"

if [ "$RETRO_ADD_RETRIEVER" = "0" ]; then
    SCRIPT=pretrain_gpt.py
else
    ARGS="${ARGS} \
    --retro-add-retriever \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-cyclic-train-iters ${RETRO_CYCLIC_TRAIN_ITERS} \
    --retro-num-neighbors ${RETRO_NUM_NEIGHBORS} \
    "
    SCRIPT=pretrain_retro.py
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "ARGS = '$ARGS'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

python -m torch.distributed.launch \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    ${SCRIPT} \
    ${ARGS} \

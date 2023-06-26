#!/bin/bash

#SBATCH -p luna -A adlr -t 04:00:00 --dependency=singleton --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --job-name=adlr-nlp:foundation-model-medium_dp1_adaptve_routing-22.12-noflash-repeat

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1

BRANCH=${1}
COMMIT=${2}
CONTAINER=${3}
NUMBER=${4}

NAME="foundation-model-medium_dp1_adaptive_routing-22.12-noflash-${NUMBER}"

SOURCE="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/megatron-lm"
OUTPUT="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/pretraining.${BRANCH}.${COMMIT}.${CONTAINER}/${NAME}/"

SCRIPTS_DIR="/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/"

CHECKPOINTS_DIR="${OUTPUT}/checkpoints"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
LOGS_DIR="${OUTPUT}/logs"

mkdir -p ${CHECKPOINTS_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${LOGS_DIR}

# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/mshoeybi/checkpoints/foundation_model/speed/${NAME}"

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/gpt3_blend.sh

BPE_DIR="/lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/bpe"

#    --num-layers-per-virtual-pipeline-stage 3 \

options=" \
    --exit-duration-in-mins 230 \
    --exit-interval 100000 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --recompute-activations \
    --sequence-parallel \
    --num-layers 12 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 244141 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file ${BPE_DIR}/gpt2-vocab.json \
    --merge-file ${BPE_DIR}/gpt2-merges.txt \
    --save-interval 2000 \
    --save ${CHECKPOINTS_DIR} \
    --load ${CHECKPOINTS_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --DDP-impl local \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --timing-log-level 1 \
    --timing-log-option minmax \
"

run_cmd="${SCRIPTS_DIR}/bind.sh --cpu=${SCRIPTS_DIR}/dgxa100_ccx.sh --mem=${SCRIPTS_DIR}/dgxa100_ccx.sh python -u ${SOURCE}/pretrain_gpt.py ${options}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

#     --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch_flash_att:22.12-py3" \

srun -l \
     --container-image nvcr.io#nvidia/pytorch:22.09-py3 \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
     --output=${LOGS_DIR}/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x


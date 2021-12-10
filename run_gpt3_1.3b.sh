#!/bin/bash

#SBATCH -p luna -A adlr -t 4:00:00 --nodes=16 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=adlr-nlp-largelm:switch_1.3b_RUNVAR_expert 

NAME="gpt3-1.3b_switch_RUNVAR_expert"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/rprenger/switch/${NAME}"

TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"

mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/gpt3_blend.sh

BPE_DIR="/lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/bpe"

options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --rampup-batch-size 32 32 2929688 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 244141 \
    --lr 2.0e-4 \
    --min-lr 2.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file ${BPE_DIR}/gpt2-vocab.json \
    --merge-file ${BPE_DIR}/gpt2-merges.txt \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --fp16 \
    --DDP-impl torch \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --checkpoint-activations "

run_cmd="cd $DIR && python pretrain_gpt.py ${options}"

srun -l \
     --container-image "/lustre/fsw/adlr/adlr-nlp/images/pytorch+bf16_nccl_fusion.sqsh" \
     --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr,/home/rprenger/workspace:/home/rprenger/workspace" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x


#! /bin/bash

export CUBLAS_WORKSPACE_CONFIG=:16:8

NAME=nextllm_determinism_debug
BASE_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm
SCRIPTS=${BASE_DIR}/scripts
MEGATRON=${BASE_DIR}/source/megatron-lm
OUTPUT_DIR=${BASE_DIR}/output/debug
LOGDIR=${OUTPUT_DIR}/logs/${NAME}
CHECKPOINT_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/checkpoints/${NAME}
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard/${NAME}

WORLD_SIZE=8

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/gpt3_blend.sh

BPE_DIR="/lustre/fsw/adlr/adlr-nlp/data/pile-cc1-cc2-shuf/bpe"

TRAIN_COMMAND=(
    ${MEGATRON}/pretrain_gpt.py
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    #--num-layers-per-virtual-pipeline-stage 1 \
    --recompute-activations \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 768 \
    --num-attention-heads 24 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
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
    --save-interval 20000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --exit-interval 1 \
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
)

#    --num-layers-per-virtual-pipeline-stage 1

#    --use-flash-attn

#    --load ${CHECKPOINT_DIR}

CUDA_DEVICE_MAX_CONNECTIONS=1 \
CUBLAS_WORKSPACE_CONFIG=:16:8 \
torchrun --nproc_per_node ${WORLD_SIZE} ${TRAIN_COMMAND[*]}

#    --global-batch-size 256
#    --rampup-batch-size 32 32 1953125

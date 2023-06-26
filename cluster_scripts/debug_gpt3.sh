#! /bin/bash


NAME=gpt3_126m_2_2_debug
BASE_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source
SCRIPTS=${BASE_DIR}/scripts
MEGATRON=${BASE_DIR}/megatron-lm
OUTPUT_DIR=${BASE_DIR}/output/debug
LOGDIR=${OUTPUT_DIR}/logs/${NAME}
CHECKPOINT_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/checkpoints/${NAME}
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard/${NAME}

WORLD_SIZE=8

# Get the data blend
. /lustre/fsw/adlr/adlr-nlp-large/data/gpt3/gpt3_blend.sh

TRAIN_COMMAND=(
    ${MEGATRON}/pretrain_gpt.py
    --exit-duration-in-mins 230
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
    --num-layers 24
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 2048
    --max-position-embeddings 2048
    --micro-batch-size 1
    --global-batch-size 8
    --train-samples 192000000
    --lr-decay-samples 166400000
    --lr-warmup-samples 162761
    --lr 6.0e-4
    --min-lr 6.0e-5
    --lr-decay-style cosine
    --log-interval 10
    --exit-interval 1000
    --log-num-zeros-in-grad
    --eval-iters 200
    --eval-interval 2000
    --data-path ${DATA_BLEND}
    --vocab-file /lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-vocab.json
    --merge-file /lustre/fsw/adlr/adlr-nlp-large/data/bpe/gpt2-merges.txt
    --split 98,2,0
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.023
    --log-params-norm
    --log-num-zeros-in-grad
    --timing-log-level 0
    --bf16
    --DDP-impl local
    --save-interval 1000
    --save ${CHECKPOINT_DIR}
)

#    --num-layers-per-virtual-pipeline-stage 1

#    --use-flash-attn

#    --load ${CHECKPOINT_DIR}

CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node ${WORLD_SIZE} ${TRAIN_COMMAND[*]}

#    --global-batch-size 256
#    --rampup-batch-size 32 32 1953125

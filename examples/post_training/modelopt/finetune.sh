#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Extra arguments of this script
MLM_DEFAULT_ARGS=" \
    --distributed-timeout-minutes 30 \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --finetune \
"


if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_MODEL_CKPT}
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_CKPT})!\n"
fi

if [ -z ${MLM_DATA_ARGS} ]; then
    MLM_DATA_ARGS=" \
        --train-samples 128000 \
        --lr-decay-samples 128000 \
        --lr-warmup-samples 0 \
        --split 100,0,0 \
        --finetune-hf-dataset nvidia/Daring-Anteater \
    "
fi

if [ -z ${MLM_TRAIN_ARGS} ]; then
    MLM_TRAIN_ARGS=" \
        --recompute-activations \
        --no-gradient-accumulation-fusion \
        --reset-position-ids \
        --reset-attention-mask \
        --eod-mask-loss \
        --global-batch-size 128 \
        --micro-batch-size 1 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-check-for-nan-in-loss-and-grad \
    "
fi

if [ -z ${MLM_OPTIM_ARGS} ]; then
    MLM_OPTIM_ARGS=" \
        --lr 1.0e-5 \
        --min-lr 1.0e-7 \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.010 \
    "
fi

if [ -z ${MLM_EVAL_ARGS} ]; then
    MLM_EVAL_ARGS=" \
        --eval-iters 1 \
        --eval-interval 1000 \
        --save-interval 1000 \
        --log-interval 100 \
    "
fi

${LAUNCH_SCRIPT} ${SCRIPT_DIR}/finetune.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_CKPT} \
    --save ${MLM_MODEL_SAVE} \
    ${MLM_DATA_ARGS} \
    ${MLM_OPTIM_ARGS} \
    ${MLM_TRAIN_ARGS} \
    ${MLM_EVAL_ARGS} \
    ${MLM_RESUME_ARGS} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

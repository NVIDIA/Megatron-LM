#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"


# Set up cache dir for HF to avoid out of space error
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

# Extra arguments of this script
MLM_DEFAULT_ARGS=" \
    --modelopt-enabled \
    --distributed-timeout-minutes 60 \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
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
	--sft \
	--tokenizer-type SFTTokenizer \
	--per-split-data-args-path ${BLEND_PATH} \
    "
fi

if [ -z ${MLM_TRAIN_ARGS} ]; then
    MLM_TRAIN_ARGS=" \
        --no-gradient-accumulation-fusion \
        --micro-batch-size 1 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-check-for-nan-in-loss-and-grad \
    "
fi

if [ -z ${MLM_OPTIM_ARGS} ]; then
    MLM_OPTIM_ARGS=" \
        --lr 5.0e-5 \
        --min-lr 1.0e-7 \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.010 \
        --use-distributed-optimizer \
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

export HF_TOKEN=${HF_TOKEN}

if [[ ${MODEL_ARGS} == *"MambaModel"* ]]; then
    PRETRAIN_EXE=${SCRIPT_DIR}/../../../pretrain_mamba.py
else
    PRETRAIN_EXE=${SCRIPT_DIR}/../../../pretrain_gpt.py
fi

${LAUNCH_SCRIPT} ${PRETRAIN_EXE} \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_CKPT} \
    --save ${MLM_MODEL_SAVE} \
    ${MLM_DATA_ARGS} \
    ${MLM_OPTIM_ARGS} \
    ${MLM_TRAIN_ARGS} \
    ${MLM_EVAL_ARGS} \
    ${MLM_RESUME_ARGS} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

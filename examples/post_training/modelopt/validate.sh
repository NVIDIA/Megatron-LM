#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Extra arguments of this script
MLM_DEFAULT_ARGS="--finetune --auto-detect-ckpt-format --export-te-mcore-model"
MLM_EXTRA_ARGS="--sequence-parallel"


if [ -z ${MLM_MODEL_CKPT} ]; then
    printf "${MLM_ERROR} Variable ${PURPLE}MLM_MODEL_CKPT${WHITE} must be set!\n"
    exit 1
fi

if [ -z ${PROMPTS_PATH} ]; then
    printf "${MLM_ERROR} Variable ${PURPLE}PROMPTS_PATH${WHITE} must be set!\n"
    exit 1
fi

if [ -z ${STEPS} ]; then
    STEPS=1
fi

if [ -z ${SAVE_GT_PATH} ]; then
    SAVE_ARGS=""
else
    SAVE_ARGS="--save-ground-truth-path ${SAVE_GT_PATH}"
fi

if [ -z ${GT_PATH}]; then
    GT_ARGS=""
else
    GT_ARGS="--ground-truth-path ${GT_PATH}"
fi

if [ -z ${OSL} ]; then
    STEPS=64
fi


${LAUNCH_SCRIPT} ${SCRIPT_DIR}/validate.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_CKPT} \
    --prompts-path ${PROMPTS_PATH} \
    --steps ${STEPS} \
    --osl ${OSL} \
    ${GT_ARGS} \
    ${SAVE_ARGS} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}


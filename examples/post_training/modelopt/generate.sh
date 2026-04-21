#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Extra arguments of this script
MLM_DEFAULT_ARGS="--finetune --auto-detect-ckpt-format --export-te-mcore-model"


if [ -z ${MLM_MODEL_CKPT} ]; then
    printf "${MLM_ERROR} Variable ${PURPLE}MLM_MODEL_CKPT${WHITE} must be set!\n"
    exit 1
fi

if [ -z ${DRAFT_LEN} ]; then
    DRAFT_LEN=0
fi


if [ -z ${PROMPTS_PATH} ]; then
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/generate.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-model-parallel-size ${EP} \
        --pipeline-model-parallel-size ${PP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --load ${MLM_MODEL_CKPT} \
        --draft-length ${DRAFT_LEN} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

else
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/generate.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-model-parallel-size ${EP} \
        --pipeline-model-parallel-size ${PP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --load ${MLM_MODEL_CKPT} \
        --data ${PROMPTS_PATH} \
        --draft-length ${DRAFT_LEN} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
fi

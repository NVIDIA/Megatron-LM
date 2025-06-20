#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Default arguments of this script
MLM_DEFAULT_ARGS="--finetune --auto-detect-ckpt-format --export-te-mcore-model --use-cpu-initialization"


if [ -z ${HF_TOKEN} ]; then
    printf "${MLM_WARNING} Variable ${PURPLE}HF_TOKEN${WHITE} is not set! HF snapshot download may fail!\n"
fi

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_mlm
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi


if [ -z ${MLM_MODEL_CKPT} ]; then
    if [ -z ${HF_MODEL_CKPT} ]; then
        HF_MODEL_CKPT=${1}
    fi
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --pretrained-model-path ${HF_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
else
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --load ${MLM_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
fi

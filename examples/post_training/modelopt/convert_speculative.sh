#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Default arguments of this script
MLM_DEFAULT_ARGS="
    --distributed-timeout-minutes 60 \
    --finetune \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
"

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}-Eagle3
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi

if [ -z ${MLM_MODEL_CKPT} ]; then
    printf "${MLM_ERROR} Variable ${PURPLE}MLM_MODEL_CKPT${WHITE} is not set! Import the Hugging Face checkpoint with Megatron-Bridge first (see README.md).\n"
    exit 1
fi

${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_speculative.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_CKPT} \
    --save ${MLM_MODEL_SAVE} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

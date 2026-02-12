#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Extra arguments of this script
MLM_DEFAULT_ARGS="
    --distributed-timeout-minutes 30 \
    --finetune --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --sequence-parallel \
"

QUANT_CFG=$2

if [ -z ${QUANT_CFG} ]; then
    QUANT_CFG='FP8_DEFAULT_CFG'
    printf "${MLM_WARNING} Variable ${PURPLE}QUANT_CFG${WHITE} is not set (default: ${QUANT_CFG})!\n"
fi

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_quant
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi

if [ -z ${MLM_MODEL_CKPT} ]; then
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/quantize.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --pretrained-model-path ${HF_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        --export-quant-cfg ${QUANT_CFG} \
        --references "${MLM_REF_LABEL}" \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
else
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/quantize.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --load ${MLM_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        --export-quant-cfg ${QUANT_CFG} \
        --references "${MLM_REF_LABEL}" \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
fi

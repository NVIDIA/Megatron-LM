#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Extra arguments of this script
MLM_DEFAULT_ARGS="--finetune --auto-detect-ckpt-format --export-te-mcore-model --sequence-parallel"

${LAUNCH_SCRIPT} ${SCRIPT_DIR}/mmlu.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_CKPT} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

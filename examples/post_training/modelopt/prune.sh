#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

if [ ${TP} -ne 1 ]; then
    printf "${MLM_ERROR} TP must be 1. Only PP>=1 is supported for pruning.\n"
    exit 1
fi

# Extra arguments of this script
MLM_DEFAULT_ARGS="
    --distributed-timeout-minutes 30 \
    --finetune --auto-detect-ckpt-format \
    --no-gradient-accumulation-fusion \
    --export-te-mcore-model
"

# Pruning target arguments - set these environment variables to enable pruning
# Example: export TARGET_HIDDEN_SIZE=3072 TARGET_FFN_HIDDEN_SIZE=9216
# Example: export LAYERS_TO_DROP="1 5 10"

# Define pruning argument mappings: "env_var:cli_arg"
PRUNE_ARG_MAPPINGS=(
    "TARGET_FFN_HIDDEN_SIZE:--target-ffn-hidden-size"
    "TARGET_HIDDEN_SIZE:--target-hidden-size"
    "TARGET_NUM_ATTENTION_HEADS:--target-num-attention-heads"
    "TARGET_NUM_QUERY_GROUPS:--target-num-query-groups"
    "TARGET_MAMBA_NUM_HEADS:--target-mamba-num-heads"
    "TARGET_MAMBA_HEAD_DIM:--target-mamba-head-dim"
    "TARGET_NUM_LAYERS:--target-num-layers"
    "LAYERS_TO_DROP:--layers-to-drop"
)

# Build arguments from environment variables
PRUNE_ARGS=""
for mapping in "${PRUNE_ARG_MAPPINGS[@]}"; do
    env_var="${mapping%%:*}"
    cli_arg="${mapping##*:}"
    if [ ! -z "${!env_var}" ]; then
        PRUNE_ARGS="${PRUNE_ARGS} ${cli_arg} ${!env_var}"
    fi
done

if [ -z "${PRUNE_ARGS}" ]; then
    printf "${MLM_WARNING} No pruning arguments specified. Set TARGET_* or LAYERS_TO_DROP environment variables.\n"
fi

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_pruned
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi

if [ -z ${MLM_MODEL_CKPT} ]; then
    LOAD_ARGS="--pretrained-model-path ${HF_MODEL_CKPT}"
else
    LOAD_ARGS="--load ${MLM_MODEL_CKPT}"
fi

${LAUNCH_SCRIPT} ${SCRIPT_DIR}/prune.py \
    ${MODEL_ARGS} \
    ${LOAD_ARGS} \
    --pipeline-model-parallel-size ${PP} \
    --tensor-model-parallel-size ${TP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --save ${MLM_MODEL_SAVE} \
    --references "${MLM_REF_LABEL}" \
    --calib-size 1024 \
    ${PRUNE_ARGS} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

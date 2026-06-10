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
    --export-default-te-spec
"

# Pruning configuration is supplied via MLM_EXTRA_ARGS. At minimum, pass
# --prune-export-config '<json>' (required by prune.py). Example:
#   MLM_EXTRA_ARGS='--prune-export-config {"hidden_size":3072,"ffn_hidden_size":9216}' ./prune.sh ...
# Optionally add --prune-intermediate-ckpt <dir> to cache scores for re-runs.
# Supported hparams: hidden_size, ffn_hidden_size, num_attention_heads, num_query_groups,
#   mamba_num_heads, mamba_head_dim, num_moe_experts, moe_ffn_hidden_size,
#   moe_shared_expert_intermediate_size, num_layers.

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_pruned
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi

if [ -z ${MLM_MODEL_CKPT} ]; then
    LOAD_ARGS="--pretrained-model-path ${HF_MODEL_CKPT}"
else
    LOAD_ARGS="--load ${MLM_MODEL_CKPT}"
fi


set -ex

${LAUNCH_SCRIPT} ${SCRIPT_DIR}/prune.py \
    ${MODEL_ARGS} \
    ${LOAD_ARGS} \
    --pipeline-model-parallel-size ${PP} \
    --tensor-model-parallel-size ${TP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --save ${MLM_MODEL_SAVE} \
    --references "${MLM_REF_LABEL}" \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}

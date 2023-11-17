#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/commons.sh
setup;

export WORLD_SIZE_IN_GPUS=8
export GLOBAL_BATCH_SIZE=24
export PIPELINE_SIZE=8
export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
export ENABLE_ZERO_BUBBLE=1

export AIP_RUN_NAME=$(basename $0 | cut -d '.' -f 1)
export ENABLE_EXACTLY_NUMERIC_MATCH=1
launch

check_loss "$(loss_of 0_test_pp_1f1b_exact)"
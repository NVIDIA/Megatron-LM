#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/commons.sh
setup;

export WORLD_SIZE_IN_GPUS=8
export GLOBAL_BATCH_SIZE=24
export PIPELINE_SIZE=1
export LAYERS=4

export AIP_RUN_NAME=$(basename $0 | cut -d '.' -f 1)
launch

check_loss "7.381942E+00"
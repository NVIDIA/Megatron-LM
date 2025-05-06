#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
echo ""

export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

test=""
# test="::test_normal_attention"
# test="::test_mla_attention"
# test="::test_mla_attention ./test_te_attention.py::test_normal_attention"

pytest --log-cli-level=DEBUG -s ./test_te_attention.py${test}
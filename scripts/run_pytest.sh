#!/bin/bash

set -u

cd /lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore

pip install pytest-cov
pip install pytest_mock
pip install nltk

# SUBDIR=""
# SUBDIR=data
# SUBDIR=models
# SUBDIR=pipeline_parallel
# SUBDIR=tensor_parallel
# SUBDIR=test_basic.py
# SUBDIR=test_parallel_state.py
# SUBDIR=test_utilities.py
# SUBDIR=test_utils.py
# SUBDIR=transformer

# SUBDIR=transformer/test_attention.py
# SUBDIR=transformer/test_core_attention.py
# SUBDIR=transformer/test_mlp.py
# SUBDIR=transformer/test_module.py
SUBDIR=transformer/test_retro_attention.py
# SUBDIR=transformer/test_spec_customization.py # *
# SUBDIR=transformer/test_switch_mlp.py
# SUBDIR=transformer/test_transformer_block.py
# SUBDIR=transformer/test_transformer_layer.py # *

NPROCS=8
torchrun --nproc_per_node=${NPROCS} -m pytest --cov-report=term --cov-report=html --cov=megatron/core tests/unit_tests/${SUBDIR}

# eof

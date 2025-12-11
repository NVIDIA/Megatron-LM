# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Various loading and saving strategies """
from megatron.core.dist_checkpointing.strategies.common import register_default_common_strategies

# We load "common" strategies by default to be always available
register_default_common_strategies()

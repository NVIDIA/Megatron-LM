# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Various loading and saving strategies """

# We mock imports to populate the `default_strategies` objects.
# Since they are defined in base but populated in common, we have to mock
# import both modules.
from megatron.core.dist_checkpointing.strategies.base import _import_trigger
from megatron.core.dist_checkpointing.strategies.common import _import_trigger

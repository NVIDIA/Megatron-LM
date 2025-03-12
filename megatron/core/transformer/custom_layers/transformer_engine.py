# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings

warnings.warn(
    """The 'megatron.core.transformer.custom_layers.transformer_engine' 
    module is deprecated and will be removed in 0.10.0. Please use 
    'megatron.core.extensions.transformer_engine' instead.""",
    DeprecationWarning,
    stacklevel=2,
)
from megatron.core.extensions.transformer_engine import *

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .module import MegatronModule
from .spec_utils import ModuleSpec, build_module
from .transformer_config import MLATransformerConfig, TransformerConfig
from .transformer_layer import TransformerLayer, TransformerLayerSubmodules

# Keep common exports initialized before feature adapters import them.
# isort: split
from . import transformer_layer as _transformer_layer
from .hyper_connection_transformer_layer import HyperConnectionTransformerLayer
from .wide_residual_config import WideResidualConfig
from .wide_residual_transformer_layer import WideResidualTransformerLayer

# Tuple ModuleSpec imports use vars(module), bypassing a lazy attribute lookup.
_transformer_layer.HyperConnectionTransformerLayer = HyperConnectionTransformerLayer

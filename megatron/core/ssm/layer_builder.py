# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Shared layer-building utilities for Mamba hybrid architectures.

Used by:
- MambaStack: for main decoder layers
- MultiTokenPredictionLayer: for MTP inner layers

This module provides a single source of truth for converting layer pattern strings
(e.g., "M*M*" or "MM") into actual layer modules.
"""

from typing import TYPE_CHECKING, List

from torch import nn

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.transformer.spec_utils import build_module

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_block import MambaStackSubmodules
    from megatron.core.transformer import TransformerConfig


def build_layers_from_pattern(
    pattern: str,
    submodules: "MambaStackSubmodules",
    config: "TransformerConfig",
    pg_collection: "ProcessGroupCollection",
    layer_offset: int = 0,
    residual_in_fp32: bool = False,
    is_mtp_layer: bool = False,
) -> nn.ModuleList:
    """
    Build layers from a pattern string. Single source of truth for pattern → layers.

    This function is shared between:
    - MambaStack (for main decoder layers)
    - MultiTokenPredictionLayer (for MTP inner layers)

    Args:
        pattern: Layer pattern string, e.g., "M*M*" or "MM"
            - M = Mamba layer
            - * = Attention (TransformerLayer)
            - - = MLP layer
            - E = MoE layer
        submodules: MambaStackSubmodules containing specs for each layer type
        config: TransformerConfig for the model
        pg_collection: Process group collection for distributed training
        layer_offset: Starting layer number offset (for layer_number assignment)
        residual_in_fp32: Whether to use FP32 for residual connections (Mamba layers)
        is_mtp_layer: Whether these layers are for MTP (affects attention layer building)

    Returns:
        nn.ModuleList of layers matching the pattern

    Example:
        >>> submodules = mamba_stack_spec.submodules
        >>> layers = build_layers_from_pattern("M*M", submodules, config, pg_collection)
        >>> # Returns [MambaLayer, TransformerLayer, MambaLayer]
    """
    layers = nn.ModuleList()

    for i, layer_type in enumerate(pattern):
        layer_number = i + 1 + layer_offset
        fp8_init_context = get_fp8_context(config, i + layer_offset, is_init=True)

        with fp8_init_context:
            if layer_type == Symbols.MAMBA:
                layer = build_module(
                    submodules.mamba_layer,
                    config=config,
                    residual_in_fp32=residual_in_fp32,
                    layer_number=layer_number,
                    pp_layer_offset=layer_offset,
                    pg_collection=pg_collection,
                )
            elif layer_type == Symbols.ATTENTION:
                # Transformer layers apply their own pp_layer_offset
                layer = build_module(
                    submodules.attention_layer,
                    config=config,
                    layer_number=i + 1,
                    pg_collection=pg_collection,
                    is_mtp_layer=is_mtp_layer,
                )
            elif layer_type == Symbols.MLP:
                # MLP layers apply their own pp_layer_offset
                layer = build_module(
                    submodules.mlp_layer,
                    config=config,
                    layer_number=i + 1,
                    pg_collection=pg_collection,
                )
            elif layer_type == Symbols.MOE:
                # MoE layers apply their own pp_layer_offset
                layer = build_module(
                    submodules.moe_layer,
                    config=config,
                    layer_number=i + 1,
                    pg_collection=pg_collection,
                    is_mtp_layer=is_mtp_layer,
                )
            else:
                raise ValueError(
                    f"Unknown layer type: '{layer_type}'. "
                    f"Valid types are: {Symbols.VALID}"
                )

        layers.append(layer)

    return layers


def get_layer_types_from_pattern(pattern: str) -> List[str]:
    """
    Convert a pattern string to a list of layer type symbols.

    This is a simple utility that validates and returns the pattern as a list.

    Args:
        pattern: Layer pattern string, e.g., "M*M*"

    Returns:
        List of layer type symbols, e.g., ["M", "*", "M", "*"]

    Raises:
        ValueError: If pattern contains invalid symbols
    """
    layer_types = list(pattern)
    for char in layer_types:
        if char not in Symbols.VALID:
            raise ValueError(
                f"Invalid layer type: '{char}'. Valid types are: {Symbols.VALID}"
            )
    return layer_types

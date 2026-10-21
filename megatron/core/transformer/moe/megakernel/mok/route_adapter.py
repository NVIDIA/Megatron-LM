# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility names for the shared authoritative-routing adapter."""

from megatron.core.transformer.moe.megakernel.route_adapter import (
    RoutingMapToIndices,
)
from megatron.core.transformer.moe.megakernel.route_adapter import (
    routing_map_to_indices as routing_map_to_mok_inputs,
)

__all__ = ["RoutingMapToIndices", "routing_map_to_mok_inputs"]

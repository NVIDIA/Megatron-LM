# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hashed n-gram memory for HybridModel, including DeepSeek-V4.1."""

from .config import EngramConfig
from .engram import Engram
from .layer_specs import apply_engram_to_hybrid_stack_spec
from .variants import ENGRAM_VARIANTS, EngramVariant, resolve_variant

__all__ = [
    "ENGRAM_VARIANTS",
    "Engram",
    "EngramConfig",
    "EngramVariant",
    "apply_engram_to_hybrid_stack_spec",
    "resolve_variant",
]

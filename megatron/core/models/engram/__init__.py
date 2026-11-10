# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hashed n-gram memory modules for Megatron Core (DeepSeek Engram and Qwen PLE)."""

from .config import EngramConfig
from .engram import Engram
from .layer_specs import apply_engram_to_hybrid_stack_spec
from .table_sharding import initialize_engram_parallel_state
from .variants import ENGRAM_VARIANTS, EngramVariant, resolve_variant

__all__ = [
    "ENGRAM_VARIANTS",
    "Engram",
    "EngramConfig",
    "EngramVariant",
    "apply_engram_to_hybrid_stack_spec",
    "initialize_engram_parallel_state",
    "resolve_variant",
]

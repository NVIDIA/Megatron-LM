# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hashed n-gram memory modules for Megatron Core (DeepSeek Engram and Qwen PLE)."""

from .config import EngramConfig
from .engram import Engram
from .layer_specs import apply_engram_to_hybrid_stack_spec
from .symmetric_lookup import close_engram_transports, initialize_engram_transports
from .table_sharding import initialize_engram_parallel_state
from .variants import ENGRAM_VARIANTS, EngramVariant, resolve_variant

__all__ = [
    "ENGRAM_VARIANTS",
    "Engram",
    "EngramConfig",
    "EngramVariant",
    "apply_engram_to_hybrid_stack_spec",
    "close_engram_transports",
    "initialize_engram_parallel_state",
    "initialize_engram_transports",
    "resolve_variant",
]

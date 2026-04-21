# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings

from .base_context import BaseInferenceContext
from .dynamic_chunk_allocator import ChunkAllocator
from .static_context import StaticInferenceContext

warnings.warn(
    "The following imports from `dynamic_context.py` will be removed "
    "in this file in `megatron-core` 0.14. The imports here result in "
    "a cyclic import issue that causes rotary embeddings to import "
    "from Apex rather than Transformer Engine.",
    DeprecationWarning,
)
from .dynamic_context import (
    ChunkOverflowError,
    ContextOverflowError,
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)

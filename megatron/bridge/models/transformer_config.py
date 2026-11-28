# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Compatibility import paths; deferred transformer configs now belong to Core.

No Bridge subclasses remain. New recipes should import directly from Core.
"""

from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

__all__ = ["MLATransformerConfig", "TransformerConfig"]

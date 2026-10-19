# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path.

Use ``megatron.core.models.gpt.deepseek_v4_hybrid_attention_module_specs``.
"""

from megatron.core.ops._compat import deprecated_module

__getattr__, __dir__ = deprecated_module(
    __name__, "megatron.core.models.gpt.deepseek_v4_hybrid_attention_module_specs"
)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash text backbone on Megatron-Core's HybridModel.

Layer specs, the single-pass hyper-connection wrapper, the Engram n-gram memory and the
V4.1 hybrid stack live here; the attention variant (CSA2) lives under
``megatron.core.transformer.experimental_attention_variant.csa2``.
"""

from megatron.core.models.deepseek_v41.layer_specs import (
    build_dsv41_hybrid_layer_pattern,
    dsv41_config_kwargs_from_model_layers,
    hybrid_dsv41_stack_spec,
)

__all__ = [
    "build_dsv41_hybrid_layer_pattern",
    "dsv41_config_kwargs_from_model_layers",
    "hybrid_dsv41_stack_spec",
]

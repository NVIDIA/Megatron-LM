# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT <-> Hybrid checkpoint conversion.

Pure state-dict transformation logic — no on-disk I/O, no distributed setup.
The offline CLI at ``tools/checkpoint/gpt_hybrid_conversion.py`` and the
in-memory auto-conversion hook in ``megatron/training/checkpointing.py``
both import from here.
"""

from megatron.core.models.hybrid.conversion.compatibility import (
    GPT_COMPATIBLE_PATTERN_SYMBOLS,
    VALID_LAYER_SYMBOLS,
    build_layer_index_mapping,
    parse_hybrid_layer_pattern,
    validate_pattern_gpt_compatible,
    validate_source_args_gpt_compatible,
)
from megatron.core.models.hybrid.conversion.transforms import (
    convert_gpt_to_hybrid,
    convert_hybrid_to_gpt,
    get_layer_num_from_key,
    initialize_ssm_layer_params,
    is_attention_param,
    is_layer_norm_for_ssm,
    is_mlp_param,
    is_ssm_param,
    replace_layer_num,
)

__all__ = [
    "GPT_COMPATIBLE_PATTERN_SYMBOLS",
    "VALID_LAYER_SYMBOLS",
    "build_layer_index_mapping",
    "convert_gpt_to_hybrid",
    "convert_hybrid_to_gpt",
    "get_layer_num_from_key",
    "initialize_ssm_layer_params",
    "is_attention_param",
    "is_layer_norm_for_ssm",
    "is_mlp_param",
    "is_ssm_param",
    "parse_hybrid_layer_pattern",
    "replace_layer_num",
    "validate_pattern_gpt_compatible",
    "validate_source_args_gpt_compatible",
]

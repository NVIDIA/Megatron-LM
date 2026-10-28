# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path; use ``megatron.core.ops.ssm.gdp``.

The historical package-level names resolve from the modules that define them.
"""

from megatron.core.ops._compat import deprecated_module

__getattr__, __dir__ = deprecated_module(
    __name__,
    "megatron.core.ops.ssm.gdp",
    exports={
        "chunk_gated_delta_product_varlen": "megatron.core.ops.ssm.gdp.chunk",
        "fused_recurrent_gated_delta_rule_update": "megatron.core.ops.ssm.gdp.fused_recurrent",
        "gdp_decode_prepare": "megatron.core.ops.ssm.gdp.prepare",
        "gdp_prefill_prepare": "megatron.core.ops.ssm.gdp.prepare",
        "build_gdp_chunk_descriptors": "megatron.core.ops.ssm.gdp.metadata",
        "max_gdp_chunk_counts": "megatron.core.ops.ssm.gdp.metadata",
    },
)

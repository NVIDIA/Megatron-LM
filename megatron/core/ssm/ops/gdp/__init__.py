# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility exports for :mod:`megatron.core.ops.ssm.gdp`."""

from megatron.core.ops.ssm.gdp import (
    __all__,
    build_gdp_chunk_descriptors,
    chunk_gated_delta_product_varlen,
    fused_recurrent_gated_delta_rule_update,
    gdp_decode_prepare,
    max_gdp_chunk_counts,
)

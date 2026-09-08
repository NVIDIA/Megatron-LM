# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# The modules in this package are forked from the Gated Delta Product kernels
# in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention), licensed under the MIT
# license. See the LICENSE file in the repository root.

"""Inference-only fork of the Gated Delta Product kernels from
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(v0.5.1).

Training and the static-batching inference path call the pip
`flash-linear-attention` kernels, which own the backward pass. Only the
dynamic-batching decode and prefill steps route here:

* `fused_recurrent_gated_delta_rule_update` -- decode.
* `gdp_decode_prepare` -- the reshape/gating step feeding decode.
* `chunk_gated_delta_product_varlen` -- prefill.

All entry points are forward-only.
"""

from .chunk import chunk_gated_delta_product_varlen
from .decode_prepare import gdp_decode_prepare
from .fused_recurrent import fused_recurrent_gated_delta_rule_update
from .metadata import build_gdp_chunk_descriptors, max_gdp_chunk_counts

__all__ = [
    "chunk_gated_delta_product_varlen",
    "fused_recurrent_gated_delta_rule_update",
    "gdp_decode_prepare",
    "build_gdp_chunk_descriptors",
    "max_gdp_chunk_counts",
]

# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# The native inference modules in this package are forked from the Gated Delta
# Product kernels in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention), licensed under the MIT
# license. See the LICENSE file in the repository root.

"""Inference-only fork of the Gated Delta Product kernels from
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(v0.5.1).

With batch-invariant mode disabled, training and static-batching inference call
the pip `flash-linear-attention` kernels, which own the backward pass. The native
dynamic-batching decode and prefill steps use this package's fork:

* `fused_recurrent_gated_delta_rule_update` -- decode (shared with GDN, whose
  gate and beta sigmoid it can apply in-kernel).
* `gdp_decode_prepare` -- the reshape/gating step feeding decode.
* `chunk_gated_delta_product_varlen` -- prefill.
* `gdp_prefill_prepare` -- the reshape/gating step feeding prefill.

These native entry points are forward-only. The independently implemented
``batch_invariant*`` modules provide the opt-in shared training/inference
arithmetic, including its backward pass and complete prefix-cache snapshots.
"""

from .chunk import chunk_gated_delta_product_varlen
from .fused_recurrent import fused_recurrent_gated_delta_rule_update
from .metadata import build_gdp_chunk_descriptors, max_gdp_chunk_counts
from .prepare import gdp_decode_prepare, gdp_prefill_prepare

__all__ = [
    "chunk_gated_delta_product_varlen",
    "fused_recurrent_gated_delta_rule_update",
    "gdp_decode_prepare",
    "gdp_prefill_prepare",
    "build_gdp_chunk_descriptors",
    "max_gdp_chunk_counts",
]

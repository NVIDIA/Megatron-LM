# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# The inference kernel modules in this package are forked from the Gated Delta Product kernels
# in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention), licensed under the MIT
# license. See the LICENSE file in the repository root.

"""GDP operation implementation and an inference-only fork of kernels from
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(v0.5.1).

``mixer`` owns the operation's parameters, checkpoint mappings and execution;
``context_parallel`` implements its head-partitioned communication. Training
and the static-batching inference path call the pip
`flash-linear-attention` kernels, which own the backward pass. Only the
dynamic-batching decode and prefill steps use the following local kernels:

* `fused_recurrent_gated_delta_rule_update` -- decode.
* `gdp_decode_prepare` -- the reshape/gating step feeding decode.
* `chunk_gated_delta_product_varlen` -- prefill.

Those three kernel entry points are forward-only; the operation also supports
training through its external backend. Global cache lifecycle stays external.
"""

from importlib import import_module

from .kernel_metadata import KERNELS as KERNELS

_ENTRY_POINT_MODULES = {
    "chunk_gated_delta_product_varlen": ".chunk",
    "gdp_decode_prepare": ".decode_prepare",
    "fused_recurrent_gated_delta_rule_update": ".fused_recurrent",
    "build_gdp_chunk_descriptors": ".metadata",
    "max_gdp_chunk_counts": ".metadata",
}

__all__ = [
    "KERNELS",
    "chunk_gated_delta_product_varlen",
    "fused_recurrent_gated_delta_rule_update",
    "gdp_decode_prepare",
    "build_gdp_chunk_descriptors",
    "max_gdp_chunk_counts",
]


def __getattr__(name: str):
    """Load a GDP entry point only when requested, not on namespace import."""
    if name not in _ENTRY_POINT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_ENTRY_POINT_MODULES[name], __name__), name)

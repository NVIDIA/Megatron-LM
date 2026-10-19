# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path.

Use ``megatron.core.ops.attention.dsa.kernels.tilelang_sparse_mla_fwd``.
"""

from megatron.core.ops._compat import deprecated_module

__getattr__, __dir__ = deprecated_module(
    __name__, "megatron.core.ops.attention.dsa.kernels.tilelang_sparse_mla_fwd"
)

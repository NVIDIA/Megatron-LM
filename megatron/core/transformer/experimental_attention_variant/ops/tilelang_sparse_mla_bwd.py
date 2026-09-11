# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for the canonical TileLang sparse MLA backward kernels."""

import sys

from megatron.core.ops.attention.dsa.kernels import tilelang_sparse_mla_bwd as _impl

sys.modules[__name__] = _impl

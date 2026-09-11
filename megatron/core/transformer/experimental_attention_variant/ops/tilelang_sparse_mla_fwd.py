# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for the canonical TileLang sparse MLA forward kernels."""

import sys

from megatron.core.ops.attention.dsa.kernels import tilelang_sparse_mla_fwd as _impl

sys.modules[__name__] = _impl

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.attention.dsa.kernels.sparse_mla`."""

import sys

from megatron.core.ops.attention.dsa.kernels import sparse_mla as _impl

sys.modules[__name__] = _impl

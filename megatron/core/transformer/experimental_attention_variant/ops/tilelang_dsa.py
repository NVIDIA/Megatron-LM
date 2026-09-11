# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.attention.dsa.kernels.tilelang_dsa`."""

import sys

from megatron.core.ops.attention.dsa.kernels import tilelang_dsa as _impl

sys.modules[__name__] = _impl

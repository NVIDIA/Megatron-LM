# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.attention.dsa.kernels.indexer`."""

import sys

from megatron.core.ops.attention.dsa.kernels import indexer as _impl

sys.modules[__name__] = _impl

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.attention.dsa.dsa_indexer_loss`."""

import sys

from megatron.core.ops.attention.dsa import dsa_indexer_loss as _impl

sys.modules[__name__] = _impl

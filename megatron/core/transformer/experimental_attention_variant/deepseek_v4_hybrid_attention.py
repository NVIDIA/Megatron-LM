# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.attention.dsv4`."""

import sys

from megatron.core.ops.attention import dsv4 as _impl

sys.modules[__name__] = _impl

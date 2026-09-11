# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.triton_cache_manager`."""

import sys

from megatron.core.ops.ssm import triton_cache_manager as _impl

sys.modules[__name__] = _impl

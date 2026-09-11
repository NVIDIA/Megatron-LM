# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.mamba2.ssd_chunk_scan`."""

import sys

from megatron.core.ops.ssm.mamba2 import ssd_chunk_scan as _impl

sys.modules[__name__] = _impl

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.context_parallel.gdp`."""

import sys

from megatron.core.ops.ssm.context_parallel import gdp as _impl

sys.modules[__name__] = _impl

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gdp.wy_fast`."""

import sys

from megatron.core.ops.ssm.gdp import wy_fast as _impl

sys.modules[__name__] = _impl

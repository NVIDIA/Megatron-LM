# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gdp.decode_prepare`."""

import sys

from megatron.core.ops.ssm.gdp import decode_prepare as _impl

sys.modules[__name__] = _impl

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gdp.fused_recurrent`."""

import sys

from megatron.core.ops.ssm.gdp import fused_recurrent as _impl

sys.modules[__name__] = _impl

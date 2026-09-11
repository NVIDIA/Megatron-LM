# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gdp.scaled_dot_kkt`."""

import sys

from megatron.core.ops.ssm.gdp import scaled_dot_kkt as _impl

sys.modules[__name__] = _impl

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gdp.solve_tril`."""

import sys

from megatron.core.ops.ssm.gdp import solve_tril as _impl

sys.modules[__name__] = _impl

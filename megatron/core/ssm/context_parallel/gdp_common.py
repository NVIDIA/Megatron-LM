# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.context_parallel.gdp_common`."""

import sys

from megatron.core.ops.ssm.context_parallel import gdp_common as _impl

sys.modules[__name__] = _impl

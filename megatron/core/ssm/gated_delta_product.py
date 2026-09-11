# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gdp.mixer`."""

import sys

from megatron.core.ops.ssm.gdp import mixer as _impl

sys.modules[__name__] = _impl

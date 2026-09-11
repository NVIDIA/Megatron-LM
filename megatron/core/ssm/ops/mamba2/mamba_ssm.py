# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.mamba2.mamba_ssm`."""

import sys

from megatron.core.ops.ssm.mamba2 import mamba_ssm as _impl

sys.modules[__name__] = _impl

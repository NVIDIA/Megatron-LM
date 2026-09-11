# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.mamba2.ssd_bmm`."""

import sys

from megatron.core.ops.ssm.mamba2 import ssd_bmm as _impl

sys.modules[__name__] = _impl

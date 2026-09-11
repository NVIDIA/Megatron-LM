# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.mamba2.ssd_state_passing`."""

import sys

from megatron.core.ops.ssm.mamba2 import ssd_state_passing as _impl

sys.modules[__name__] = _impl

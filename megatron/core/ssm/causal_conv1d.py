# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.common.causal_conv1d_cp`."""

import sys

from megatron.core.ops.ssm.common import causal_conv1d_cp as _impl

sys.modules[__name__] = _impl

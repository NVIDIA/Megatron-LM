# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gated_delta.common`."""

import sys

from megatron.core.ops.ssm.gated_delta import common as _impl

sys.modules[__name__] = _impl

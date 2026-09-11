# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.gated_delta.gdn`."""

import sys

from megatron.core.ops.ssm.gated_delta import gdn as _impl

sys.modules[__name__] = _impl

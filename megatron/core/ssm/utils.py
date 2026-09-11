# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.common.checkpointing`."""

import sys

from megatron.core.ops.ssm.common import checkpointing as _impl

sys.modules[__name__] = _impl

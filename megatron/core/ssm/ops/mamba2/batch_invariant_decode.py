# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility alias for :mod:`megatron.core.ops.ssm.mamba2.batch_invariant_decode`."""

import sys

from megatron.core.ops.ssm.mamba2 import batch_invariant_decode as _impl

sys.modules[__name__] = _impl

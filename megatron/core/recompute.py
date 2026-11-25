# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Alias of the module moved to
``megatron.core.activation_policy.recompute.recompute``."""

import sys

from megatron.core.activation_policy.recompute import recompute as _module

sys.modules[__name__] = _module

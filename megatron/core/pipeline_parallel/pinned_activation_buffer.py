# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Alias of the module moved to
``megatron.core.activation_policy.offload.pinned_activation_buffer``."""

import sys

from megatron.core.activation_policy.offload import pinned_activation_buffer as _module

sys.modules[__name__] = _module

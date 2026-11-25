# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Alias of the module moved to
``megatron.core.activation_policy.offload.fine_grained_activation_offload``."""

import sys

from megatron.core.activation_policy.offload import fine_grained_activation_offload as _module

sys.modules[__name__] = _module

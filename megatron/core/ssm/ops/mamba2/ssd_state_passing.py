# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path; use ``megatron.core.ops.ssm.mamba2.ssd_state_passing``."""

from megatron.core.ops._compat import deprecated_module

__getattr__, __dir__ = deprecated_module(__name__, "megatron.core.ops.ssm.mamba2.ssd_state_passing")

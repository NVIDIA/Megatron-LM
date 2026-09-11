# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional causal-conv1d training and CUDA update entry points."""

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update_cuda = None

__all__ = ["causal_conv1d_fn", "causal_conv1d_update_cuda"]

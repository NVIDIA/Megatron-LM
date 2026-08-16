# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for the linear-attention / SSM mixers.

Organized by which recurrence a kernel belongs to:

* `common` -- shared by every variant (causal convolution, slot-indexed state
  gather/scatter, autotune determinism).
* `mamba2` -- the Mamba2 state-space duality kernels, adapted from vLLM and
  state-spaces/mamba.

The re-exports below are the historical top-level entry points and are kept so
existing callers do not have to spell out the subpackage.

This package is internal to Megatron Core. The kernels here are implementation
details of the SSM mixers, not a supported kernel library.
"""

try:
    from .mamba2.ssd_combined import mamba_chunk_scan_combined_varlen
except ImportError:
    mamba_chunk_scan_combined_varlen = None

try:
    from .common.causal_conv1d_varlen import causal_conv1d_varlen_fn
except ImportError:
    causal_conv1d_varlen_fn = None

__all__ = ["mamba_chunk_scan_combined_varlen", "causal_conv1d_varlen_fn"]

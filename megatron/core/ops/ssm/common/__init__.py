# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kernels and helpers shared by every linear-attention / SSM mixer.

Nothing here is specific to a recurrence. The short causal convolution that
front-runs the recurrence, the slot-indexed state gather/scatter used by prefix
caching, and the autotune-configuration filtering that makes the kernels
deterministic are all common to Mamba2, Gated Delta Product and their
relatives. Recurrence-specific kernels live in the sibling `mamba2` and `gdp`
packages.

The module-level helpers are shared the same way: `causal_conv1d_cp` (convolution
over context-parallel shards and its determinism guard), `packed_seq`,
`checkpointing` and `inference` (the dynamic-inference mixin).
"""

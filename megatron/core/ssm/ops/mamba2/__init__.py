# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels implementing the Mamba2 state-space duality (SSD) recurrence.

The `ssd_*` modules are the chunked scan decomposition -- block matmul, chunk
state, state passing, chunk scan -- plus the `ssd_combined` entry point that
drives them. `mamba_ssm` holds the decode-step kernels and
`batch_invariant_decode` the buffered-replay variant used when
`batch_invariant_mode` is on.

Kernels that are not specific to this recurrence (the causal convolution, state
gather/scatter, autotune determinism) live in `ops.common`.
"""

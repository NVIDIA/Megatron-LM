# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SSM operation implementations, their kernels and operation-local communication.

Organized by operation family:

* `common` -- shared by every variant: causal convolution (including its
  context-parallel form), slot-indexed state gather/scatter, autotune determinism,
  packed-sequence and checkpoint helpers, and the dynamic-inference mixin.
* `mamba2` -- the Mamba2 mixer, its context-parallel transforms and the state-space
  duality kernels, adapted from vLLM and state-spaces/mamba.
* `gdp` -- the Gated Delta Product mixer, its context-parallel transforms and the
  kernels forked from flash-linear-attention.
* `gated_delta` -- the GDN and GDN2 operations.
* `context_parallel` -- chunkwise context-parallel communication.

Stack assembly and global recurrent-cache management stay outside this package.
Packages here do not re-export; import each name from the module that defines it.

This package is internal to Megatron Core. The kernels here are implementation
details of the SSM mixers, not a supported kernel library.
"""

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Normalization kernels with explicit arithmetic and storage contracts.

The BF16 add/RMSNorm primitive accepts contiguous CUDA tensors shaped
``[..., hidden]`` (hidden divisible by eight, up to 8192) and a BF16 weight vector.
It returns the normalized output and
the BF16-rounded residual sum, both in BF16. Its optional in-place mode owns
and overwrites the branch-output buffer. It has no persistent state, masks,
packed-sequence metadata, or distributed inputs, and implements no backward.
Import the kernel module explicitly when selecting this inference primitive;
importing this package loads no optional dependencies.
"""

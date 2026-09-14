# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Low-level kernels for the internal chunked gated delta rule implementation.

Keep the public FLA-compatible contract in the parent chunk module while placing independent
optimized kernel stages in this package.
"""

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ordered collective implementations over explicitly supplied process groups.

Inputs are rank/shard partials; reduce-scatter returns owned rows and all-reduce
returns replicated results. Symmetric kernels borrow caller-owned buffers and handles.
This package imports no backends and re-exports no implementations."""

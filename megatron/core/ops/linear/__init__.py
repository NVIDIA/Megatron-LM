# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row-parallel operation implementations over local K shards.

Implementations own scratch and autograd state, accept explicit process groups,
and return independent activations. This package imports no optional libraries
and re-exports no implementations."""

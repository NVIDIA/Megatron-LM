# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row-parallel operation implementations over local K shards.

Implementations own scratch and autograd state, accept explicit process groups,
and return independent activations. MXFP8 with BF16 master weights uses TE's
quantized forward/backward and the ordered collective fallback, for one partial
per rank. Several fixed K shards per rank currently require unquantized BF16.
This package imports no optional libraries and re-exports no implementations."""

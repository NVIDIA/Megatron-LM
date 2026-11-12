# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Captured H100 backward work indexing, not an approved latency formula."""

def job_counts(lengths, query_heads, head_dim):
    lengths = tuple(lengths)
    if (not lengths or any(type(length) is not int or length < 1 for length in lengths)
            or type(query_heads) is not int or query_heads < 1
            or type(head_dim) is not int or head_dim not in (128, 256)):
        raise ValueError("positive real lengths/heads and captured D128/D256 kernels required")
    partitions = head_dim // 128
    return tuple(query_heads * partitions * ((length + 127) // 128) for length in lengths)

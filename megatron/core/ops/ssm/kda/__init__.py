# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kimi Delta Attention recurrence backends.

Backends consume BTHD query, key, value, log-decay and beta tensors. Packed
inputs use a singleton batch dimension and cumulative sequence lengths. The
cuDNN backend additionally composes boundary states over the explicit context
parallel process group supplied by FLA's chunkwise CP context.
"""

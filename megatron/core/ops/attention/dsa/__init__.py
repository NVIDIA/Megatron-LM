# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA operations: per-microbatch CLI state and optional target accumulation.

CLI targets use the owner's fixed, sorted Top-K slots. The carrier belongs to one
checkpointed forward and is not model state. Importing this package loads no optional backend.
"""

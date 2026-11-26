# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Domain checks shared by the frozen geometry kernels."""


def require(condition, message):
    """Reject unsupported source geometry without changing its numerical formulas."""
    if not condition:
        raise ValueError(message)

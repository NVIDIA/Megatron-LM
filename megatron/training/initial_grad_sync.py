# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Opt-in one-time allocator preparation before the first gradient finalizer."""

from functools import wraps
from typing import Callable


def release_initial_grad_sync_cache(finalize: Callable) -> Callable:
    """Release unused CUDA blocks once, preserving the selected finalizer."""
    if getattr(finalize, "_releases_initial_grad_sync_cache", False):
        return finalize
    released = False

    @wraps(finalize)
    def wrapper(*args, **kwargs):
        nonlocal released
        if not released:
            import torch

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            released = True
        return finalize(*args, **kwargs)

    wrapper._releases_initial_grad_sync_cache = True
    return wrapper

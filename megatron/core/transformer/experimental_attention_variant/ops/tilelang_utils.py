# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from collections import OrderedDict
from unittest.mock import MagicMock

import torch

try:
    import tilelang
    from tilelang import language as T  # pylint: disable=unused-import

    HAVE_TILELANG = True
except (ImportError, OSError):
    tilelang = MagicMock()
    T = MagicMock()
    HAVE_TILELANG = False


def _noop_jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


def tilelang_jit(*args, **kwargs):
    """Return TileLang's jit decorator when available, otherwise a no-op decorator."""
    if HAVE_TILELANG:
        return tilelang.jit(*args, **kwargs)
    return _noop_jit(*args, **kwargs)


def require_tilelang():
    """Raise a clear error when a fused TileLang kernel is used without TileLang installed."""
    if not HAVE_TILELANG:
        raise ImportError(
            "TileLang is required to use fused DSA TileLang kernels. "
            "Install tilelang or use the unfused fallback path."
        )


def _env_int(name: str, default: int) -> int:
    """Parse a positive integer environment variable, falling back to ``default``."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


_TILELANG_KERNEL_CACHE_MAX = _env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 512)


def _cache_put_lru(cache: OrderedDict, key, value):
    """Insert ``value`` as the most-recently-used entry, evicting oldest past the cap."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _TILELANG_KERNEL_CACHE_MAX:
        cache.popitem(last=False)


def _get_cached_kernel(cache: OrderedDict, lock, key, build_fn):
    """Return a cached compiled kernel for ``key``, building it via ``build_fn`` on miss."""
    with lock:
        kernel = cache.pop(key, None)
        if kernel is None:
            kernel = build_fn()
        _cache_put_lru(cache, key, kernel)
        return kernel


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 1:
        return x
    return _ceil_div(x, multiple) * multiple


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _normalize_sm_scale(sm_scale):
    """Coerce a softmax scale to a stable float so it can key the kernel cache."""
    if sm_scale is None:
        return None
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.detach().item())
    else:
        sm_scale = float(sm_scale)
    # Avoid tiny floating-point jitter creating cache-key churn.
    return round(sm_scale, 12)


def _as_2d_weights(weights: torch.Tensor) -> torch.Tensor:
    """Preserve the heads axis while accepting legacy trailing singleton weights."""
    if weights.ndim == 3 and weights.size(-1) == 1:
        return weights.squeeze(-1)
    return weights

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

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

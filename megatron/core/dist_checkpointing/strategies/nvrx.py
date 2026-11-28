# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for interacting with the experimental nvidia-resiliency-ext API."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Callable, Dict

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False

NVRX_MIN_VERSION = "0.7.1"


def has_nvrx_async_support() -> bool:
    """Checks whether an NVRx with the async checkpointing API Megatron uses is installed."""
    return is_nvrx_min_version()


def make_nvrx_async_request(
    async_request_cls: type,
    async_fn: Callable[..., Any],
    async_fn_args: Any,
    finalize_fns: list[Callable[..., Any]],
    async_fn_kwargs: Dict[str, Any] | None = None,
    preload_fn: Callable[..., Any] | None = None,
):
    """Builds an AsyncRequest using the expected NVRx API."""
    return async_request_cls(
        async_fn,
        async_fn_args,
        finalize_fns,
        async_fn_kwargs=async_fn_kwargs or {},
        preload_fn=preload_fn,
    )


def is_nvrx_min_version(version: str = NVRX_MIN_VERSION) -> bool:
    """Check if minimum version of `NVRx` is installed."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )

    try:
        import nvidia_resiliency_ext as nvrx

        HAVE_NVRX = True
    except (ImportError, ModuleNotFoundError):
        HAVE_NVRX = False

    if not HAVE_NVRX:
        return False

    nvrx_version = getattr(nvrx, "__version__", None)
    if nvrx_version is None:
        try:
            nvrx_version = package_version("nvidia-resiliency-ext")
        except PackageNotFoundError:
            return False

    return PkgVersion(str(nvrx_version)) >= PkgVersion(version)

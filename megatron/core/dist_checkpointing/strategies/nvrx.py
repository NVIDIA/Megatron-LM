# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for interacting with the experimental nvidia-resiliency-ext API."""

from importlib import import_module
from typing import Any, Callable, Dict

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False

NVRX_MIN_VERSION = "0.6.0.dev33+b2bb3d7"


def has_nvrx_async_support() -> bool:
    """Checks whether the NVRx async checkpointing symbols Megatron uses are importable."""
    try:
        core = import_module("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
        cached_metadata_reader = import_module(
            "nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader"
        )
        filesystem_async = import_module(
            "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async"
        )
        state_dict_saver = import_module(
            "nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver"
        )
    except (ImportError, ModuleNotFoundError):
        return False

    required_symbols = (
        getattr(core, "AsyncCallsQueue", None),
        getattr(core, "AsyncRequest", None),
        getattr(cached_metadata_reader, "CachedMetadataFileSystemReader", None),
        getattr(filesystem_async, "FileSystemWriterAsync", None),
        getattr(filesystem_async, "get_write_results_queue", None),
        getattr(state_dict_saver, "CheckpointMetadataCache", None),
        getattr(state_dict_saver, "save_state_dict_async_finalize", None),
        getattr(state_dict_saver, "save_state_dict_async_plan", None),
    )
    assert (
        is_nvrx_min_version()
    ), f"Minimum required nvidia-resiliency-ext package version is {NVRX_MIN_VERSION}."

    return all(symbol is not None for symbol in required_symbols) and hasattr(
        filesystem_async, "_results_queue"
    )


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

    nvrx_version = str(nvrx.__version__) if HAVE_NVRX else "0.0.0"

    return PkgVersion(nvrx_version) >= PkgVersion(version)

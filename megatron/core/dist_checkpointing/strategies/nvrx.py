"""Helpers for interacting with the experimental nvidia-resiliency-ext API."""

import inspect
from importlib import import_module
from typing import Any, Callable, Dict


def has_nvrx_async_support() -> bool:
    """Checks whether the NVRx async checkpointing symbols Megatron uses are importable."""
    try:
        core = import_module("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
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
        getattr(filesystem_async, "FileSystemWriterAsync", None),
        getattr(filesystem_async, "_results_queue", None),
        getattr(filesystem_async, "get_write_results_queue", None),
        getattr(state_dict_saver, "CheckpointMetadataCache", None),
        getattr(state_dict_saver, "save_state_dict_async_finalize", None),
        getattr(state_dict_saver, "save_state_dict_async_plan", None),
    )
    return all(symbol is not None for symbol in required_symbols)


def filter_supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drops kwargs that are not accepted by the provided callable."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return {}

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs

    supported_names = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {name: value for name, value in kwargs.items() if name in supported_names}


def make_nvrx_async_request(
    async_request_cls: type,
    async_fn: Callable[..., Any],
    async_fn_args: Any,
    finalize_fns: list[Callable[..., Any]],
    async_fn_kwargs: Dict[str, Any] | None = None,
    preload_fn: Callable[..., Any] | None = None,
):
    """Builds an AsyncRequest while tolerating API drift in optional fields."""
    kwargs = filter_supported_kwargs(
        async_request_cls,
        {
            "async_fn_kwargs": async_fn_kwargs or {},
            "preload_fn": preload_fn,
        },
    )
    return async_request_cls(async_fn, async_fn_args, finalize_fns, **kwargs)

import functools
from typing import Any, Callable

import torch


def _tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Single-entry cache for functions with tensor inputs."""
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result
        if (
            (last_args is not None and last_kwargs is not None)
            and (len(args) == len(last_args) and len(kwargs) == len(last_kwargs))
            and all(a is b for a, b in zip(args, last_args, strict=False))
            and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())
        ):
            return last_result
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@_tensor_cache
def _prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@_tensor_cache
def _prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in _prepare_lens(cu_seqlens).unbind()
        ]
    )


@_tensor_cache
def _prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return _prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@_tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Convert cumulative sequence lengths to per-token (batch_id, position) pairs."""
    position_ids = _prepare_position_ids(cu_seqlens)
    return torch.stack([_prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)

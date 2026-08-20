"""Strict public wrapper for the SM100 fused GDR backward CuTeDSL kernel.

Copyright (c) 2026 The Qwen team, Alibaba Group.
Licensed under the MIT License.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import torch

_BT = 64
_H = 64
_D = 128


@dataclass(frozen=True)
class _VarlenMetadata:
    _cu_seqlens_ref: weakref.ReferenceType
    chunk_offsets: torch.Tensor
    num_sequences: int
    num_chunks: int
    uniform_sequence_length: int

    @property
    def cu_seqlens(self) -> torch.Tensor:
        tensor = self._cu_seqlens_ref()
        if tensor is None:
            raise RuntimeError("cu_seqlens owner is no longer alive")
        return tensor


@dataclass
class _MetadataEntry:
    owner: weakref.ReferenceType
    version: int
    total_tokens: int
    chunk_size: int
    metadata: _VarlenMetadata


_METADATA_CACHE: dict[int, _MetadataEntry] = {}


def _clear_metadata_cache_for_test() -> None:
    _METADATA_CACHE.clear()


def _check_sm100(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError("fused_gdr_bwd requires CUDA tensors")
    capability = torch.cuda.get_device_capability(tensor.device)
    if capability != (10, 0):
        raise ValueError(f"fused_gdr_bwd requires SM100, got capability {capability}")


def _prepare_varlen_metadata(
    cu_seqlens: torch.Tensor, total_tokens: int, chunk_size: int
) -> _VarlenMetadata:
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise TypeError("cu_seqlens must be contiguous torch.int32")

    key = id(cu_seqlens)
    version = cu_seqlens._version
    cached = _METADATA_CACHE.get(key)
    if (
        cached is not None
        and cached.owner() is cu_seqlens
        and cached.version == version
        and cached.total_tokens == total_tokens
        and cached.chunk_size == chunk_size
    ):
        return cached.metadata

    if cu_seqlens.device.type == "cpu":
        values = cu_seqlens.detach().tolist()
        if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
            raise ValueError(
                "cu_seqlens must contain at least one sequence and span "
                f"[0, {total_tokens}], got {values}"
            )
        lengths = [end - start for start, end in zip(values, values[1:])]
        if any(length <= 0 or length % chunk_size for length in lengths):
            raise ValueError(
                "sequence lengths must be positive multiples of " f"{chunk_size}, got {lengths}"
            )
        chunk_offsets = torch.tensor(
            [offset // chunk_size for offset in values], dtype=torch.int32, device=cu_seqlens.device
        )
        num_sequences = len(lengths)
        num_chunks = sum(lengths) // chunk_size
        uniform_sequence_length = (
            lengths[0] if lengths and all(length == lengths[0] for length in lengths) else 0
        )
    else:
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences < 1:
            raise ValueError("cu_seqlens must contain at least one sequence")
        if total_tokens % chunk_size:
            raise ValueError(f"total_tokens must be divisible by {chunk_size}")
        chunk_offsets = (cu_seqlens // chunk_size).contiguous()
        num_chunks = total_tokens // chunk_size
        uniform_sequence_length = total_tokens if num_sequences == 1 else 0

    def remove_cached_owner(owner_ref):
        current = _METADATA_CACHE.get(key)
        if current is not None and current.owner is owner_ref:
            _METADATA_CACHE.pop(key, None)

    owner_ref = weakref.ref(cu_seqlens, remove_cached_owner)
    metadata = _VarlenMetadata(
        _cu_seqlens_ref=owner_ref,
        chunk_offsets=chunk_offsets,
        num_sequences=num_sequences,
        num_chunks=num_chunks,
        uniform_sequence_length=uniform_sequence_length,
    )
    _METADATA_CACHE[key] = _MetadataEntry(
        owner=owner_ref,
        version=version,
        total_tokens=total_tokens,
        chunk_size=chunk_size,
        metadata=metadata,
    )
    return metadata


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(
    *, q, k, v, a, g, beta, do, dht, h, scale, cu_seqlens, chunk_size, state_v_first
) -> None:
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q).__name__}")
    _check_sm100(q)
    if q.ndim != 4:
        raise ValueError(f"q must be rank 4, got rank {q.ndim}")
    if type(chunk_size) is not int or chunk_size != _BT:
        raise ValueError(f"chunk_size must be {_BT}, got {chunk_size}")
    if state_v_first is not False:
        raise NotImplementedError(f"state_v_first={state_v_first!r} is not supported")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {scale}")
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be 1D and contain at least two offsets")

    total_tokens = q.shape[1]
    num_sequences = cu_seqlens.numel() - 1
    device = q.device
    io_shape = (1, total_tokens, _H, _D)
    scalar_shape = (1, total_tokens, _H)
    _require_tensor("q", q, dtype=torch.bfloat16, shape=io_shape, device=device)
    _require_tensor("k", k, dtype=torch.bfloat16, shape=io_shape, device=device)
    _require_tensor("v", v, dtype=torch.bfloat16, shape=io_shape, device=device)
    _require_tensor("do", do, dtype=torch.bfloat16, shape=io_shape, device=device)
    _require_tensor("a", a, dtype=torch.bfloat16, shape=(1, total_tokens, _H, _BT), device=device)
    _require_tensor("g", g, dtype=torch.float32, shape=scalar_shape, device=device)
    _require_tensor("beta", beta, dtype=torch.float32, shape=scalar_shape, device=device)
    _require_tensor(
        "dht", dht, dtype=torch.float32, shape=(num_sequences, _H, _D, _D), device=device
    )
    _require_tensor(
        "h", h, dtype=torch.bfloat16, shape=(1, total_tokens // _BT, _H, _D, _D), device=device
    )
    _require_tensor(
        "cu_seqlens", cu_seqlens, dtype=torch.int32, shape=(num_sequences + 1,), device=device
    )


def _allocate_outputs(q, k, v, g, beta, dht):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        torch.empty_like(dht),
    )


def _launch_fused_gdr_bwd_out(**kwargs):
    from .launcher import launch_fused_gdr_bwd

    launch_fused_gdr_bwd(**kwargs)


def fused_gdr_bwd(
    q, k, v, a, g, beta, do, dht, h, scale=None, cu_seqlens=None, chunk_size=64, state_v_first=False
):
    if cu_seqlens is None:
        raise NotImplementedError("packed cu_seqlens are required")
    if dht is None:
        raise NotImplementedError("dht=None is not supported")
    normalized_scale = float(_D**-0.5 if scale is None else scale)
    _validate_inputs(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
        h=h,
        scale=normalized_scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
    )
    metadata = _prepare_varlen_metadata(cu_seqlens, total_tokens=q.shape[1], chunk_size=chunk_size)
    outputs = _allocate_outputs(q, k, v, g, beta, dht)
    _launch_fused_gdr_bwd_out(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
        h=h,
        scale=normalized_scale,
        metadata=metadata,
        outputs=outputs,
    )
    return outputs

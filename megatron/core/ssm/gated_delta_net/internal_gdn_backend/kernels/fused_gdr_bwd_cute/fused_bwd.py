"""Strict public wrapper for the SM100 fused GDR backward CuTeDSL kernel.

Copyright (c) 2026 The Qwen team, Alibaba Group.
Licensed under the MIT License.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_BT = 64
_H = 64
_D = 128


@dataclass(frozen=True)
class _VarlenMetadata:
    cu_seqlens: torch.Tensor
    chunk_offsets: torch.Tensor
    num_sequences: int
    num_chunks: int
    uniform_sequence_length: int
    has_partial_chunks: bool


def _check_sm100(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError("fused_gdr_bwd requires CUDA tensors")
    capability = torch.cuda.get_device_capability(tensor.device)
    if capability[0] != 10:
        raise ValueError(
            "fused_gdr_bwd requires a Blackwell data-center GPU (SM10x), "
            f"got capability {capability}"
        )


def _validate_chunk_offsets(chunk_offsets: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    if not isinstance(chunk_offsets, torch.Tensor):
        raise TypeError("chunk_offsets must be a torch.Tensor")
    if chunk_offsets.dtype != torch.int32 or not chunk_offsets.is_contiguous():
        raise TypeError("chunk_offsets must be contiguous torch.int32")
    if chunk_offsets.device != cu_seqlens.device:
        raise ValueError(
            f"chunk_offsets must be on {cu_seqlens.device}, got {chunk_offsets.device}"
        )
    if tuple(chunk_offsets.shape) != tuple(cu_seqlens.shape):
        raise ValueError(
            f"chunk_offsets must have shape {tuple(cu_seqlens.shape)}, "
            f"got {tuple(chunk_offsets.shape)}"
        )
    return chunk_offsets


def _prepare_varlen_metadata(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    chunk_size: int,
    chunk_offsets: torch.Tensor | None = None,
    num_chunks: int | None = None,
    trusted_chunk_offsets: bool = False,
) -> _VarlenMetadata:
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise TypeError("cu_seqlens must be contiguous torch.int32")
    if chunk_offsets is not None:
        chunk_offsets = _validate_chunk_offsets(chunk_offsets, cu_seqlens)
        if cu_seqlens.device.type != "cpu" and not trusted_chunk_offsets:
            raise ValueError(
                "non-CPU chunk_offsets require trusted_chunk_offsets=True after "
                "host-side metadata validation"
            )

    if cu_seqlens.device.type == "cpu":
        values = cu_seqlens.detach().tolist()
        if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
            raise ValueError(
                "cu_seqlens must contain at least one sequence and span "
                f"[0, {total_tokens}], got {values}"
            )
        lengths = [end - start for start, end in zip(values, values[1:])]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"sequence lengths must be positive, got {lengths}")
        chunks_per_sequence = [(length + chunk_size - 1) // chunk_size for length in lengths]
        expected_offsets = [0]
        for chunks in chunks_per_sequence:
            expected_offsets.append(expected_offsets[-1] + chunks)
        if chunk_offsets is None:
            chunk_offsets = torch.tensor(
                expected_offsets, dtype=torch.int32, device=cu_seqlens.device
            )
        else:
            expected = torch.tensor(expected_offsets, dtype=torch.int32, device=cu_seqlens.device)
            if not bool(torch.equal(chunk_offsets, expected)):
                raise ValueError(
                    "chunk_offsets must equal the per-sequence chunk prefix, "
                    f"got {chunk_offsets.tolist()} for cu_seqlens {values}"
                )
        num_sequences = len(lengths)
        expected_num_chunks = sum(chunks_per_sequence)
        if num_chunks is not None and num_chunks != expected_num_chunks:
            raise ValueError(f"num_chunks must be {expected_num_chunks}, got {num_chunks}")
        num_chunks = expected_num_chunks
        has_partial_chunks = any(length % chunk_size for length in lengths)
        uniform_sequence_length = (
            lengths[0]
            if lengths
            and not has_partial_chunks
            and all(length == lengths[0] for length in lengths)
            else 0
        )
    else:
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences < 1:
            raise ValueError("cu_seqlens must contain at least one sequence")
        if num_chunks is None:
            if total_tokens % chunk_size:
                raise ValueError(
                    "num_chunks is required for CUDA cu_seqlens when total_tokens "
                    f"is not divisible by {chunk_size}"
                )
            num_chunks = total_tokens // chunk_size
        elif num_chunks < 1:
            raise ValueError("num_chunks must be positive")
        if chunk_offsets is None:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            chunks_per_sequence = (lengths + chunk_size - 1) // chunk_size
            chunk_offsets = torch.empty_like(cu_seqlens)
            chunk_offsets[0] = 0
            torch.cumsum(chunks_per_sequence, dim=0, out=chunk_offsets[1:])
        has_partial_chunks = num_chunks * chunk_size != total_tokens
        uniform_sequence_length = (
            total_tokens if num_sequences == 1 and not has_partial_chunks else 0
        )

    return _VarlenMetadata(
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        num_sequences=num_sequences,
        num_chunks=num_chunks,
        uniform_sequence_length=uniform_sequence_length,
        has_partial_chunks=has_partial_chunks,
    )


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
    *, q, k, v, a, g, beta, do, dht, h, scale, cu_seqlens, chunk_size, state_v_first, num_chunks
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
    _require_tensor("h", h, dtype=torch.bfloat16, shape=(1, num_chunks, _H, _D, _D), device=device)
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


def fused_gdr_bwd(
    q,
    k,
    v,
    a,
    g,
    beta,
    do,
    dht,
    h,
    scale=None,
    cu_seqlens=None,
    chunk_offsets=None,
    chunk_size=64,
    state_v_first=False,
    trusted_chunk_offsets=False,
):
    if cu_seqlens is None:
        raise NotImplementedError("packed cu_seqlens are required")
    if dht is None:
        raise NotImplementedError("dht=None is not supported")
    normalized_scale = float(_D**-0.5 if scale is None else scale)
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q).__name__}")
    if q.ndim != 4:
        raise ValueError(f"q must be rank 4, got rank {q.ndim}")
    if not isinstance(h, torch.Tensor):
        raise TypeError(f"h must be a torch.Tensor, got {type(h).__name__}")
    if h.ndim != 5:
        raise ValueError(f"h must be rank 5, got rank {h.ndim}")
    metadata = _prepare_varlen_metadata(
        cu_seqlens,
        total_tokens=q.shape[1],
        chunk_size=chunk_size,
        chunk_offsets=chunk_offsets,
        num_chunks=h.shape[1],
        trusted_chunk_offsets=trusted_chunk_offsets,
    )
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
        num_chunks=metadata.num_chunks,
    )
    outputs = _allocate_outputs(q, k, v, g, beta, dht)
    # Imported lazily so the wrapper's validation stays importable without CuTe DSL.
    from .launcher import launch_fused_gdr_bwd

    launch_fused_gdr_bwd(
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

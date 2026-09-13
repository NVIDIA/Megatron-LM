# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed sequence helpers for variable-length (THD) attention."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.utils.packed_seq import PackedSeqParams


@dataclass(frozen=True)
class PackedTHDBatch:
    """Dense packed representation of a jagged no-padding batch."""

    input_ids: torch.Tensor
    labels: torch.Tensor | None
    loss_mask: torch.Tensor | None
    position_ids: torch.Tensor
    packed_seq_params: Any
    cu_seqlens_padded: torch.Tensor
    lengths: torch.Tensor
    padded_lengths: torch.Tensor
    cp_size: int = 1
    cp_rank: int = 0
    cp_group: Any | None = None


def _make_packed_seq_params(
    *,
    cu_seqlens_padded: torch.Tensor,
    max_seqlen: int,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_group: Any | None = None,
):
    extra_args = {}
    if cp_size > 1:
        extra_args["local_cp_size"] = cp_size
        if cp_group is not None:
            extra_args["cp_group"] = cp_group
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        cp_rank=cp_rank,
        **extra_args,
    )


def _slice_along_dim(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    index = [slice(None)] * tensor.dim()
    index[dim] = slice(start, end)
    return tensor[tuple(index)]


def _assign_along_dim(dst: torch.Tensor, dim: int, start: int, src: torch.Tensor) -> None:
    index = [slice(None)] * dst.dim()
    index[dim] = slice(start, start + src.size(dim))
    dst[tuple(index)] = src


def _split_full_to_cp_local(
    tensor: torch.Tensor, *, cu_seqlens_padded: torch.Tensor, cp_size: int, cp_rank: int, dim: int
) -> torch.Tensor:
    if cp_size <= 1:
        return tensor

    total_local = int(cu_seqlens_padded[-1].item()) // cp_size
    out_shape = list(tensor.shape)
    out_shape[dim] = total_local
    local = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)

    for idx in range(int(cu_seqlens_padded.numel()) - 1):
        full_start = int(cu_seqlens_padded[idx].item())
        full_end = int(cu_seqlens_padded[idx + 1].item())
        padded_len = full_end - full_start
        if padded_len <= 0:
            continue
        chunk = padded_len // (2 * cp_size)
        local_start = full_start // cp_size

        first = _slice_along_dim(
            tensor, dim, full_start + cp_rank * chunk, full_start + (cp_rank + 1) * chunk
        )
        second = _slice_along_dim(
            tensor, dim, full_end - (cp_rank + 1) * chunk, full_end - cp_rank * chunk
        )
        _assign_along_dim(local, dim, local_start, first)
        _assign_along_dim(local, dim, local_start + chunk, second)

    return local


def _reconstruct_full_from_cp_parts(
    parts: list[torch.Tensor], *, cu_seqlens_padded: torch.Tensor, cp_size: int, dim: int
) -> torch.Tensor:
    if cp_size <= 1:
        return parts[0]

    total_full = int(cu_seqlens_padded[-1].item())
    out_shape = list(parts[0].shape)
    out_shape[dim] = total_full
    full = torch.zeros(out_shape, dtype=parts[0].dtype, device=parts[0].device)

    for idx in range(int(cu_seqlens_padded.numel()) - 1):
        full_start = int(cu_seqlens_padded[idx].item())
        full_end = int(cu_seqlens_padded[idx + 1].item())
        padded_len = full_end - full_start
        if padded_len <= 0:
            continue
        chunk = padded_len // (2 * cp_size)
        local_start = full_start // cp_size

        for rank, part in enumerate(parts):
            first = _slice_along_dim(part, dim, local_start, local_start + chunk)
            second = _slice_along_dim(part, dim, local_start + chunk, local_start + 2 * chunk)
            _assign_along_dim(full, dim, full_start + rank * chunk, first)
            _assign_along_dim(full, dim, full_end - (rank + 1) * chunk, second)

    return full


def reconstruct_packed_from_cp_parts(
    parts: list[torch.Tensor], *, cu_seqlens_padded: torch.Tensor, cp_size: int, dim: int
) -> torch.Tensor:
    """Reconstruct a full packed THD tensor from CP-local zigzag parts."""
    return _reconstruct_full_from_cp_parts(
        parts, cu_seqlens_padded=cu_seqlens_padded, cp_size=cp_size, dim=dim
    )


def split_packed_to_cp_local(
    tensor: torch.Tensor, *, cu_seqlens_padded: torch.Tensor, cp_size: int, cp_rank: int, dim: int
) -> torch.Tensor:
    """Slice a full packed THD tensor back to one CP rank's zigzag shard."""
    return _split_full_to_cp_local(
        tensor, cu_seqlens_padded=cu_seqlens_padded, cp_size=cp_size, cp_rank=cp_rank, dim=dim
    )


def _packed_cu_seqlens(packed_seq_params: Any) -> torch.Tensor | None:
    if packed_seq_params is None:
        return None
    cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    if cu_seqlens is None:
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
    return cu_seqlens


def has_packed_thd_params(packed_seq_params: Any) -> bool:
    return _packed_cu_seqlens(packed_seq_params) is not None


def _sequence_dim(tensor: torch.Tensor) -> int:
    return tensor.dim() - 1 if tensor.dim() > 1 else 0


def _with_cp_metadata(packed_seq_params: Any, *, cp_size: int, cp_rank: int, cp_group: Any):
    updated = copy(packed_seq_params)
    updated.local_cp_size = cp_size
    updated.cp_rank = cp_rank
    updated.cp_group = cp_group
    return updated


def parallel_state_from_model(model: Any) -> Any:
    """Return the MLite parallel state from a raw model or wrapper."""

    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        ps = getattr(current, "ps", None)
        if ps is not None:
            return ps
        ps = getattr(current, "_parallel_state", None)
        if ps is not None:
            return ps
        current = getattr(current, "module", None)
    return None


def prepare_packed_thd_for_context_parallel(
    packed_seq_params: Any,
    tensors: tuple[torch.Tensor | None, ...],
    *,
    cp_size: int,
    cp_rank: int,
    cp_group: Any = None,
    dims: tuple[int | None, ...] | None = None,
) -> tuple[Any, tuple[torch.Tensor | None, ...]]:
    """Split plain packed THD tensors to one CP rank without knowing batch keys."""

    tensor_tuple = tuple(tensors)
    cu_seqlens = _packed_cu_seqlens(packed_seq_params)
    if cu_seqlens is None or cp_size <= 1:
        return packed_seq_params, tensor_tuple
    if int(getattr(packed_seq_params, "local_cp_size", None) or 1) > 1:
        return packed_seq_params, tensor_tuple
    if dims is not None and len(dims) != len(tensor_tuple):
        raise ValueError(
            f"dims length {len(dims)} does not match tensors length {len(tensor_tuple)}."
        )

    local_tensors: list[torch.Tensor | None] = []
    for idx, tensor in enumerate(tensor_tuple):
        if tensor is None:
            local_tensors.append(None)
            continue
        dim = _sequence_dim(tensor) if dims is None or dims[idx] is None else int(dims[idx])
        local_tensors.append(
            _split_full_to_cp_local(
                tensor,
                cu_seqlens_padded=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
                dim=dim,
            )
        )
    return (
        _with_cp_metadata(packed_seq_params, cp_size=cp_size, cp_rank=cp_rank, cp_group=cp_group),
        tuple(local_tensors),
    )


def prepare_packed_thd_kwargs_for_context_parallel(
    model: Any,
    kwargs: dict[str, Any],
    *,
    tensor_keys: tuple[str, ...] = ("input_ids", "labels", "loss_mask", "position_ids"),
) -> None:
    packed_seq_params = kwargs.get("packed_seq_params")
    if not has_packed_thd_params(packed_seq_params):
        return

    ps = parallel_state_from_model(model)
    packed_seq_params, tensors = prepare_packed_thd_for_context_parallel(
        packed_seq_params,
        tuple(kwargs.get(key) for key in tensor_keys),
        cp_size=int(getattr(ps, "cp_size", 1) or 1),
        cp_rank=int(getattr(ps, "cp_rank", 0) or 0),
        cp_group=getattr(ps, "cp_group", None),
    )
    if packed_seq_params is not None or "packed_seq_params" in kwargs:
        kwargs["packed_seq_params"] = packed_seq_params
    for key, tensor in zip(tensor_keys, tensors, strict=True):
        if tensor is not None or key in kwargs:
            kwargs[key] = tensor


@dataclass(frozen=True)
class ThdPackMeta:
    """Per-sequence THD padding layout, recomputable from true seq lengths.

    Shared by a model's pack/unpack pair so the connector never needs to hold
    ``PackedSeqParams`` or padded token tensors to reverse a model output.
    """

    lengths: torch.Tensor
    padded_lengths: torch.Tensor
    cu_seqlens_padded: torch.Tensor
    cp_size: int
    cp_group: Any | None


def thd_pack_meta(
    seq_lens: torch.Tensor,
    *,
    tp_size: int = 1,
    cp_size: int = 1,
    cp_group: Any | None = None,
    contiguous: bool = False,
) -> ThdPackMeta:
    """Compute the padded THD layout for ``seq_lens`` without copying tokens.

    ``contiguous=False`` aligns to Megatron/TE zigzag CP (``2*cp``); ``True``
    aligns to contiguous CP (``cp``). Mirrors :func:`pack_nested_thd` padding.
    """
    lengths = seq_lens.to(dtype=torch.int32)
    cp_align = (cp_size if contiguous else 2 * cp_size) if cp_size > 1 else 1
    align_size = max(int(tp_size), 1) * cp_align
    pad_size = (align_size - lengths % align_size) % align_size
    padded_lengths = lengths + pad_size
    cu_seqlens_padded = torch.zeros(
        lengths.numel() + 1, dtype=torch.int32, device=lengths.device
    )
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
    return ThdPackMeta(lengths, padded_lengths, cu_seqlens_padded, cp_size, cp_group)


def unpack_thd_to_nested(
    output: torch.Tensor, meta: ThdPackMeta, *, contiguous: bool = False
) -> torch.Tensor:
    """Reverse a model output back to jagged true-length form using ``meta``.

    Gathers CP-local shards (zigzag or contiguous reconstruct) then slices each
    sequence's true length out of the padded layout.
    """
    if output.dim() >= 2 and output.shape[0] == 1:
        flat = output[0]
    elif output.dim() >= 2 and output.shape[1] == 1:
        flat = output[:, 0]
    else:
        flat = output

    if meta.cp_size > 1:
        parts = _all_gather_cp_tensor(flat, cp_size=meta.cp_size, cp_group=meta.cp_group)
        if contiguous:
            flat = torch.cat(parts, dim=0)
        else:
            flat = _reconstruct_full_from_cp_parts(
                parts, cu_seqlens_padded=meta.cu_seqlens_padded, cp_size=meta.cp_size, dim=0
            )

    pieces = []
    for idx, length_t in enumerate(meta.lengths):
        length = int(length_t.item())
        start = int(meta.cu_seqlens_padded[idx].item())
        pieces.append(flat[start : start + length])
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _all_gather_cp_tensor(
    tensor: torch.Tensor, *, cp_size: int, cp_group: Any
) -> list[torch.Tensor]:
    if cp_size <= 1:
        return [tensor]
    if cp_group is None:
        raise ValueError("CP THD gather requires cp_group.")
    try:
        from torch.distributed.nn.functional import all_gather

        return list(all_gather(tensor, group=cp_group))
    except Exception:
        parts = [torch.empty_like(tensor) for _ in range(cp_size)]
        dist.all_gather(parts, tensor, group=cp_group)
        return parts


def _roll_packed_thd_left_local(
    tensor: torch.Tensor, *, cu_seqlens_padded: torch.Tensor, dims: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = dims if dims >= 0 else tensor.dim() + dims
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Invalid roll dim {dims} for tensor with {tensor.dim()} dims.")

    rolled = tensor.clone()
    for idx in range(int(cu_seqlens_padded.numel()) - 1):
        start = int(cu_seqlens_padded[idx].item())
        end = int(cu_seqlens_padded[idx + 1].item())
        if end <= start:
            continue

        index = [slice(None)] * tensor.dim()
        index[dim] = slice(start, end)
        seq = torch.roll(tensor[tuple(index)], shifts=-1, dims=dim)

        zero_index = [slice(None)] * seq.dim()
        zero_index[dim] = slice(-1, None)
        seq[tuple(zero_index)] = 0
        rolled[tuple(index)] = seq

    return rolled, rolled.sum()


def roll_packed_thd_left(
    tensor: torch.Tensor,
    *,
    cu_seqlens_padded: torch.Tensor | None = None,
    packed_seq_params: Any | None = None,
    dims: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll a THD packed tensor left without crossing sequence boundaries."""

    cp_size = 1
    cp_rank = 0
    cp_group = None
    if packed_seq_params is not None:
        cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q", None)
        cp_size = int(getattr(packed_seq_params, "local_cp_size", None) or 1)
        cp_rank = int(getattr(packed_seq_params, "cp_rank", 0) or 0)
        cp_group = getattr(packed_seq_params, "cp_group", None)
    if cu_seqlens_padded is None:
        raise ValueError("THD packed roll requires cu_seqlens.")

    dim = dims if dims >= 0 else tensor.dim() + dims
    if cp_size <= 1:
        return _roll_packed_thd_left_local(tensor, cu_seqlens_padded=cu_seqlens_padded, dims=dim)

    parts = _all_gather_cp_tensor(tensor, cp_size=cp_size, cp_group=cp_group)
    full = _reconstruct_full_from_cp_parts(
        parts, cu_seqlens_padded=cu_seqlens_padded, cp_size=cp_size, dim=dim
    )
    rolled_full, token_sum = _roll_packed_thd_left_local(
        full, cu_seqlens_padded=cu_seqlens_padded, dims=dim
    )
    local = _split_full_to_cp_local(
        rolled_full, cu_seqlens_padded=cu_seqlens_padded, cp_size=cp_size, cp_rank=cp_rank, dim=dim
    )
    return local, token_sum


def pack_nested_thd(
    input_ids: torch.Tensor,
    *,
    tp_size: int = 1,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_group: Any | None = None,
    split_cp: bool = True,
    labels: torch.Tensor | None = None,
    roll_labels: bool = False,
    loss_mask: torch.Tensor | None = None,
    roll_loss_mask: bool = False,
) -> PackedTHDBatch:
    """Pack a jagged no-padding batch into Megatron Lite's THD model input.

    Mirrors VERL/Megatron's THD engine convention: each sequence is
    padded to the tensor-parallel alignment, concatenated, then represented as
    a single ``[1, local_padded_tokens]`` token row plus ``PackedSeqParams``.
    For CP>1 the local row uses Megatron/TE zigzag chunking unless
    ``split_cp=False``.  The latter keeps full packed tokens and plain
    ``PackedSeqParams`` while still padding each sample to CP-compatible
    alignment, matching the external VERL runtime contract.
    """

    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if not getattr(input_ids, "is_nested", False):
        raise TypeError("pack_nested_thd expects a jagged NestedTensor input_ids.")
    if labels is not None and not getattr(labels, "is_nested", False):
        raise TypeError(
            "pack_nested_thd expects jagged NestedTensor labels when labels are provided."
        )
    if loss_mask is not None and not getattr(loss_mask, "is_nested", False):
        raise TypeError(
            "pack_nested_thd expects jagged NestedTensor loss_mask when loss_mask is provided."
        )

    align_size = max(int(tp_size), 1) * (2 * cp_size if cp_size > 1 else 1)
    device = input_ids.device
    offsets = input_ids.offsets().to(device=device)
    lengths = offsets.diff().to(dtype=torch.int32)
    pad_size = (align_size - lengths % align_size) % align_size
    padded_lengths = lengths + pad_size

    cu_seqlens_padded = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=device)
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
    total_padded = int(cu_seqlens_padded[-1].item())
    total_local = total_padded // cp_size if split_cp else total_padded
    max_seqlen = int(padded_lengths.max().item()) if padded_lengths.numel() else 0

    packed_input = torch.zeros(total_local, dtype=input_ids.dtype, device=device)
    packed_labels = (
        torch.zeros(total_local, dtype=labels.dtype, device=device) if labels is not None else None
    )
    packed_loss_mask = (
        torch.zeros(total_local, dtype=loss_mask.dtype, device=device)
        if loss_mask is not None
        else None
    )
    # Megatron's rotary embedding slices position embeddings on the CP rank.
    # Keep packed position ids full-length while CP-slicing tokens/labels/masks.
    position_ids = torch.zeros(total_padded, dtype=torch.long, device=device)

    for idx, length_t in enumerate(lengths):
        length = int(length_t.item())
        padded_length = int(padded_lengths[idx].item())
        full_start = int(cu_seqlens_padded[idx].item())
        local_start = full_start // cp_size if split_cp else full_start

        seq_input = torch.zeros(padded_length, dtype=input_ids.dtype, device=device)
        seq_input[:length] = input_ids[idx]
        seq_labels = None
        if labels is not None:
            assert packed_labels is not None
            seq_labels = torch.zeros(padded_length, dtype=labels.dtype, device=device)
            seq_labels[:length] = labels[idx]
            if roll_labels and length > 0:
                seq_labels[:length] = torch.roll(seq_labels[:length], shifts=-1, dims=0)
                seq_labels[length - 1] = 0
        seq_loss_mask = None
        if loss_mask is not None:
            assert packed_loss_mask is not None
            seq_loss_mask = torch.zeros(padded_length, dtype=loss_mask.dtype, device=device)
            seq_loss_mask[:length] = loss_mask[idx]
            if roll_loss_mask and length > 0:
                seq_loss_mask[:length] = torch.roll(seq_loss_mask[:length], shifts=-1, dims=0)
                seq_loss_mask[length - 1] = 0
        seq_positions = torch.zeros(padded_length, dtype=torch.long, device=device)
        seq_positions[:length] = torch.arange(length, dtype=torch.long, device=device)

        local_input = (
            _split_full_to_cp_local(
                seq_input,
                cu_seqlens_padded=torch.tensor(
                    [0, padded_length], dtype=torch.int32, device=device
                ),
                cp_size=cp_size,
                cp_rank=cp_rank,
                dim=0,
            )
            if split_cp
            else seq_input
        )
        packed_input[local_start : local_start + local_input.numel()] = local_input
        if seq_labels is not None:
            local_labels = (
                _split_full_to_cp_local(
                    seq_labels,
                    cu_seqlens_padded=torch.tensor(
                        [0, padded_length], dtype=torch.int32, device=device
                    ),
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    dim=0,
                )
                if split_cp
                else seq_labels
            )
            assert packed_labels is not None
            packed_labels[local_start : local_start + local_labels.numel()] = local_labels
        if seq_loss_mask is not None:
            local_loss_mask = (
                _split_full_to_cp_local(
                    seq_loss_mask,
                    cu_seqlens_padded=torch.tensor(
                        [0, padded_length], dtype=torch.int32, device=device
                    ),
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    dim=0,
                )
                if split_cp
                else seq_loss_mask
            )
            assert packed_loss_mask is not None
            packed_loss_mask[local_start : local_start + local_loss_mask.numel()] = local_loss_mask
        position_ids[full_start : full_start + padded_length] = seq_positions

    return PackedTHDBatch(
        input_ids=packed_input.unsqueeze(0),
        labels=packed_labels.unsqueeze(0) if packed_labels is not None else None,
        loss_mask=(packed_loss_mask.unsqueeze(0) if packed_loss_mask is not None else None),
        position_ids=position_ids.unsqueeze(0),
        packed_seq_params=_make_packed_seq_params(
            cu_seqlens_padded=cu_seqlens_padded,
            max_seqlen=max_seqlen,
            cp_size=cp_size if split_cp else 1,
            cp_rank=cp_rank if split_cp else 0,
            cp_group=cp_group if split_cp else None,
        ),
        cu_seqlens_padded=cu_seqlens_padded,
        lengths=lengths,
        padded_lengths=padded_lengths,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_group=cp_group,
    )


def unpack_packed_thd_to_nested(output: torch.Tensor, batch: PackedTHDBatch) -> torch.Tensor:
    """Unpack ``[1, total_padded, ...]`` THD model output back to jagged form."""

    if output.dim() >= 2 and output.shape[0] == 1:
        flat = output[0]
    elif output.dim() >= 2 and output.shape[1] == 1:
        flat = output[:, 0]
    else:
        flat = output

    if batch.cp_size > 1:
        dim = 0
        parts = _all_gather_cp_tensor(flat, cp_size=batch.cp_size, cp_group=batch.cp_group)
        flat = _reconstruct_full_from_cp_parts(
            parts, cu_seqlens_padded=batch.cu_seqlens_padded, cp_size=batch.cp_size, dim=dim
        )

    pieces = []
    for idx, length_t in enumerate(batch.lengths):
        length = int(length_t.item())
        start = int(batch.cu_seqlens_padded[idx].item())
        pieces.append(flat[start : start + length])
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


__all__ = [
    "PackedSeqParams",
    "PackedTHDBatch",
    "ThdPackMeta",
    "has_packed_thd_params",
    "pack_nested_thd",
    "parallel_state_from_model",
    "prepare_packed_thd_for_context_parallel",
    "prepare_packed_thd_kwargs_for_context_parallel",
    "reconstruct_packed_from_cp_parts",
    "roll_packed_thd_left",
    "split_packed_to_cp_local",
    "thd_pack_meta",
    "unpack_packed_thd_to_nested",
    "unpack_thd_to_nested",
]

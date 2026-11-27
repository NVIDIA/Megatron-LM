# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional, Tuple

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from ..fp8_utils import get_high_precision_storage_view, requantize_whole_params_from_high_precision


def all_gather_v(
    local: torch.Tensor,
    sizes: List[int],
    group: torch.distributed.ProcessGroup,
    *,
    out: Optional[torch.Tensor] = None,
    async_op: bool = False,
) -> Tuple[List[torch.Tensor], Optional[torch.distributed.Work]]:
    """Gather flat tensors of the given per-rank sizes on a group supporting uneven all-gather.

    With a flat receive buffer, return views of its used prefix. Otherwise retain the local
    input and allocate one receive tensor per peer. For an async gather, wait on the returned
    work before using or overwriting the received tensors.
    """
    rank = group.rank()
    assert local.ndim == 1 and len(sizes) == group.size() and local.numel() == sizes[rank]
    if out is None:
        slots = [
            local if owner == rank else local.new_empty((size,)) for owner, size in enumerate(sizes)
        ]
    else:
        total = sum(sizes)
        assert out.ndim == 1 and out.is_contiguous() and out.numel() >= total
        assert out.dtype == local.dtype and out.device == local.device
        slots = list(out[:total].split(sizes))
        if local.numel():
            slots[rank].copy_(local.detach())
    work = torch.distributed.all_gather(slots, slots[rank], group=group, async_op=async_op)
    return slots, work


def _compact_param_gather_uses_grad_buffer(config) -> bool:
    """Whether a compact LayerWise all-gather receives into ``bucket.grad_data``.

    Configuring overlap selects gradient storage for all supported dtypes, even when the
    call is forced synchronous. With overlap off, only the MXFP8 setting selects it.
    """
    return config.overlap_param_gather or config.reuse_grad_buf_for_mxfp8_param_ag


def _layerwise_gather_sources(
    params: List[torch.nn.Parameter], quantized: bool
) -> List[torch.Tensor]:
    """The tensors an owner contributes to a variable-size LayerWise all-gather.

    Quantized params can't be flattened (MXFP8 has no view(-1)), so they travel as BF16 cast
    from the owner's FP32 master; the copy-back requantizes them on every rank.
    """
    if not quantized:
        return [get_high_precision_storage_view(param) for param in params]
    for param in params:
        assert getattr(param, 'main_param', None) is not None, (
            "LayerWise quantized param gather needs the owner's FP32 main_param to stage the "
            "all-gather source."
        )
    return [param.main_param.to(torch.bfloat16) for param in params]


@torch.no_grad()
def _copy_back_layerwise_gather(
    params_list: List[List[torch.nn.Parameter]],
    gather_list: List[torch.Tensor],
    local_rank: int,
    quantized: bool,
) -> None:
    """Publish a variable-size LayerWise all-gather into the model params.

    Plain params of the other ranks are copied from their gather slots; the local rank's
    params already hold their update. Quantized params were gathered as BF16 staged from the
    owners' FP32 masters and are requantized on every rank, owner included, so that all
    replicas hold identical quantized values.
    """
    quantized_params, quantized_values = [], []
    for rank, params in enumerate(params_list):
        if not params:
            continue
        if quantized:
            templates = [torch.empty(p.shape, device="meta", dtype=torch.bfloat16) for p in params]
            quantized_params.extend(params)
            quantized_values.extend(_unflatten_dense_tensors(gather_list[rank], templates))
        elif rank != local_rank:
            destinations = [get_high_precision_storage_view(param) for param in params]
            values = _unflatten_dense_tensors(gather_list[rank], destinations)
            for destination, value in zip(destinations, values):
                destination.copy_(value)
    if quantized_params:
        requantize_whole_params_from_high_precision(quantized_params, quantized_values)


@torch.no_grad()
def start_layerwise_param_gather(
    params_by_rank: List[List[torch.nn.Parameter]],
    sizes: List[int],
    group: torch.distributed.ProcessGroup,
    quantized: bool,
    *,
    grad_buffer: Optional[torch.Tensor] = None,
    async_op: bool = False,
) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.distributed.Work]]:
    """Pack the local owner's parameters and start one gather per bucket or transport dtype."""
    if not any(sizes):
        return None, None
    rank = group.rank()
    if sizes[rank]:
        sources = _layerwise_gather_sources(params_by_rank[rank], quantized)
        local = _flatten_dense_tensors(sources)
    else:
        first = next(params[0] for params in params_by_rank if params)
        dtype = torch.bfloat16 if quantized else first.dtype
        local = torch.empty(0, device=first.device, dtype=dtype)
    out = grad_buffer.view(local.dtype) if grad_buffer is not None else None
    return all_gather_v(local, sizes, group, out=out, async_op=async_op)

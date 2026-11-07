# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paged K/V layouts for packed sequences, on one rank or zigzag context parallel."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions,
)

_PAGE = 128

_CACHE_MAX = 16
_cache: "OrderedDict[tuple, PagedLayout]" = OrderedDict()


@dataclass(frozen=True)
class PagedLayout:
    """Query segments that read a paged copy of K/V held in global packed order.

    On one rank each sequence is one segment. With context parallelism it is the zigzag front
    and back chunk of this rank; `kv_reorder_idx` sorts all-gathered rank-major tokens into
    packed order and `gathered_positions` is its inverse. `bwd_k_gather_idx` is None when every
    segment reads its whole sequence.
    """

    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    seqused_k: torch.Tensor
    max_seqlen_k: int
    page_table: torch.Tensor
    paged_scatter_idx: torch.Tensor
    num_pages: int
    bwd_cu_seqlens_k: torch.Tensor
    bwd_k_gather_idx: Optional[torch.Tensor]
    gathered_positions: Optional[torch.Tensor]
    kv_reorder_idx: Optional[torch.Tensor]


def _cumsum0(x):
    return torch.cat((x.new_zeros(1), x.cumsum(0)))


def _ranges(starts, lens):
    """Concatenation of arange(s, s + n) over (starts, lens)."""
    total = int(lens.sum().item())
    ids = torch.repeat_interleave(
        torch.arange(lens.numel(), device=lens.device), lens, output_size=total
    )
    within = torch.arange(total, device=lens.device) - _cumsum0(lens)[:-1].index_select(0, ids)
    return starts.index_select(0, ids) + within


def paged_layout(cu_seqlens, cp_size=1, cp_rank=0):
    """Build (and cache) the layout of this rank's query rows."""
    device = cu_seqlens.device
    key = (tuple(int(x) for x in cu_seqlens.tolist()), int(cp_size), int(cp_rank), str(device))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    cu = cu_seqlens.to(torch.int64)
    starts, ends = cu[:-1], cu[1:]
    lens = ends - starts
    if (lens <= 0).any():
        raise ValueError("every packed sequence must have length > 0")
    seq = torch.arange(lens.numel(), device=device)
    if cp_size == 1:
        q_lens, k_lens, seg_seq = lens, lens, seq
        gathered = reorder = k_gather = None
    else:
        if (lens % (2 * cp_size) != 0).any():
            raise ValueError(
                f"cp: every sequence length must be divisible by 2*cp_size "
                f"(2*{cp_size}); lengths {lens.tolist()}"
            )
        h = lens // (2 * cp_size)
        q_lens = h.repeat_interleave(2)
        k_lens = torch.stack(((cp_rank + 1) * h, lens - cp_rank * h), dim=1).reshape(-1)
        seg_seq = seq.repeat_interleave(2)
        gathered = torch.cat(
            [
                build_packed_allgather_cp_local_positions(cu, cp_size, rank, device)
                for rank in range(cp_size)
            ]
        )
        reorder = torch.argsort(gathered)
        k_gather = _ranges(starts.index_select(0, seg_seq), k_lens)

    pages = (lens + _PAGE - 1) // _PAGE
    page_off = _cumsum0(pages)[:-1]
    cols = torch.arange(int(pages.max().item()), device=device)
    page_table = torch.where(cols < pages[:, None], page_off[:, None] + cols, 0)
    tokens = torch.arange(int(cu[-1].item()), device=device)
    seq_of_token = torch.searchsorted(ends, tokens, right=True)

    layout = PagedLayout(
        cu_seqlens_q=_cumsum0(q_lens).to(torch.int32),
        max_seqlen_q=int(q_lens.max().item()),
        seqused_k=k_lens.to(torch.int32),
        max_seqlen_k=int(k_lens.max().item()),
        page_table=page_table.index_select(0, seg_seq).to(torch.int32),
        paged_scatter_idx=tokens + (page_off * _PAGE - starts).index_select(0, seq_of_token),
        num_pages=int(pages.sum().item()),
        bwd_cu_seqlens_k=_cumsum0(k_lens).to(torch.int32),
        bwd_k_gather_idx=k_gather,
        gathered_positions=gathered,
        kv_reorder_idx=reorder,
    )
    _cache[key] = layout
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return layout

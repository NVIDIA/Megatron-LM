# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Zigzag context-parallel layout plans for packed sequences."""

from collections import OrderedDict
from dataclasses import dataclass

import torch

_PAGE = 128

_PLAN_CACHE_MAX = 16
_plan_cache: "OrderedDict[tuple, CpPlan]" = OrderedDict()


@dataclass(frozen=True)
class CpPlan:
    """Host-built layout for one (cu_seqlens, cp_size, cp_rank) triple."""

    cp_size: int
    cp_rank: int
    n_seq: int
    local_positions: torch.Tensor
    kv_reorder_idx: torch.Tensor
    kv_unreorder_idx: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    seqused_k: torch.Tensor
    page_table: torch.Tensor
    paged_scatter_idx: torch.Tensor
    num_pages: int
    bwd_cu_seqlens_k: torch.Tensor
    bwd_k_gather_idx: torch.Tensor


def _rank_local_positions(starts, ends, h, cp_size, cp_rank, device):
    """Zigzag local positions for one rank: per seq, front chunk then back."""
    front_lo = starts + cp_rank * h
    back_lo = ends - (cp_rank + 1) * h
    seg_starts = torch.stack((front_lo, back_lo), dim=1).reshape(-1)
    seg_lens = torch.stack((h, h), dim=1).reshape(-1)
    t_local = int(seg_lens.sum().item())
    seg_ids = torch.repeat_interleave(
        torch.arange(seg_lens.numel(), device=device), seg_lens, output_size=t_local
    )
    within = torch.arange(t_local, device=device) - torch.repeat_interleave(
        seg_lens.cumsum(0) - seg_lens, seg_lens, output_size=t_local
    )
    return seg_starts.index_select(0, seg_ids) + within


def build_cp_plan(cu_seqlens, cp_size, cp_rank, h_kv, device):
    """Build (and cache) the CpPlan for this rank."""
    device = torch.device(device)
    key = (
        tuple(int(x) for x in cu_seqlens.tolist()),
        int(cp_size),
        int(cp_rank),
        int(h_kv),
        str(device),
    )
    if key in _plan_cache:
        _plan_cache.move_to_end(key)
        return _plan_cache[key]

    cu = cu_seqlens.to(device=device, dtype=torch.int64).contiguous()
    n_seq = cu.numel() - 1
    starts, ends = cu[:-1], cu[1:]
    L = ends - starts
    if (L <= 0).any():
        raise ValueError("cp: every packed sequence must have length > 0")
    if (L % (2 * cp_size) != 0).any():
        raise ValueError(
            f"cp: every sequence length must be divisible by 2*cp_size "
            f"(2*{cp_size}); lengths {L.tolist()}"
        )
    h = L // (2 * cp_size)
    r = cp_rank

    local_positions = _rank_local_positions(starts, ends, h, cp_size, r, device)
    t_local = local_positions.numel()
    t_global = int(cu[-1].item())

    gathered_positions = torch.cat(
        [_rank_local_positions(starts, ends, h, cp_size, rank, device) for rank in range(cp_size)]
    )
    kv_reorder_idx = torch.argsort(gathered_positions)
    kv_unreorder_idx = torch.empty_like(kv_reorder_idx)
    kv_unreorder_idx[kv_reorder_idx] = torch.arange(t_global, device=device)

    seg_qlens = h.repeat_interleave(2)
    cu_q = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), seg_qlens.cumsum(0))).to(
        torch.int32
    )
    seqused_k = torch.stack(((r + 1) * h, L - r * h), dim=1).reshape(-1).to(torch.int32)

    pages_i = (L + _PAGE - 1) // _PAGE
    page_off = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), pages_i.cumsum(0)[:-1]))
    num_pages = int(pages_i.sum().item())
    max_pages = int(pages_i.max().item())
    page_table = torch.zeros(2 * n_seq, max_pages, dtype=torch.int32, device=device)
    cols = torch.arange(max_pages, device=device)
    rows = (page_off[:, None] + cols[None, :]).to(torch.int32)
    page_table[0::2] = rows
    page_table[1::2] = rows

    seq_of_token = torch.searchsorted(cu[1:], torch.arange(t_global, device=device), right=True)
    paged_scatter_idx = (
        torch.arange(t_global, device=device) + (page_off * _PAGE - starts)[seq_of_token]
    )

    bwd_cu_k = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=device), seqused_k.to(torch.int64).cumsum(0))
    ).to(torch.int32)
    total_k = int(seqused_k.to(torch.int64).sum().item())
    k_seg_seq_start = starts.repeat_interleave(2)
    k_lens = seqused_k.to(torch.int64)
    k_ids = torch.repeat_interleave(
        torch.arange(2 * n_seq, device=device), k_lens, output_size=total_k
    )
    k_within = torch.arange(total_k, device=device) - torch.repeat_interleave(
        k_lens.cumsum(0) - k_lens, k_lens, output_size=total_k
    )
    bwd_k_gather_idx = k_seg_seq_start.index_select(0, k_ids) + k_within

    plan = CpPlan(
        cp_size=cp_size,
        cp_rank=cp_rank,
        n_seq=n_seq,
        local_positions=local_positions,
        kv_reorder_idx=kv_reorder_idx,
        kv_unreorder_idx=kv_unreorder_idx,
        cu_seqlens_q=cu_q,
        max_seqlen_q=int(h.max().item()),
        seqused_k=seqused_k,
        page_table=page_table,
        paged_scatter_idx=paged_scatter_idx,
        num_pages=num_pages,
        bwd_cu_seqlens_k=bwd_cu_k,
        bwd_k_gather_idx=bwd_k_gather_idx,
    )
    _plan_cache[key] = plan
    while len(_plan_cache) > _PLAN_CACHE_MAX:
        _plan_cache.popitem(last=False)
    return plan

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training entry points: exact forward, FlashAttention-4 backward on its out and lse."""

import torch
import torch.distributed as dist

from .cp_plan import _PAGE, _rank_local_positions, build_cp_plan

try:
    from flash_attn.cute.interface import _flash_attn_bwd
except ImportError:
    _flash_attn_bwd = None


def _bwd(attn, q, k, v, out, dout, lse, cu_q=None, cu_k=None, max_q=None, max_k=None):
    return _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout.contiguous(),
        lse,
        softmax_scale=attn.softmax_scale,
        causal=True,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        deterministic=attn.head_dim == 128,
    )[:3]


class _Dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn):
        out, lse = attn.prefill(q, k, v, return_lse=True)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.attn = attn
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        return (*_bwd(ctx.attn, q, k, v, out, dout, lse), None)


def dense_attention(q, k, v, attn):
    """Causal attention on [B, L, H, D] tensors."""
    return _Dense.apply(q.contiguous(), k.contiguous(), v.contiguous(), attn)


def thd_paging(cu_seqlens):
    """Page table, keys per sequence and flat page row per token, one segment per sequence."""
    cu = [int(x) for x in cu_seqlens.tolist()]
    lens = [b - a for a, b in zip(cu[:-1], cu[1:])]
    pages = [-(-n // _PAGE) for n in lens]
    base = [sum(pages[:i]) for i in range(len(lens))]
    device = cu_seqlens.device
    page_table = torch.zeros(len(lens), max(max(pages), 1), dtype=torch.int32)
    rows = []
    for i, n in enumerate(lens):
        page_table[i, : pages[i]] = torch.arange(base[i], base[i] + pages[i], dtype=torch.int32)
        t = torch.arange(n)
        rows.append((base[i] + t // _PAGE) * _PAGE + t % _PAGE)
    return (
        page_table.to(device),
        torch.tensor(lens, dtype=torch.int32, device=device),
        torch.cat(rows).to(device),
        sum(pages),
        max(lens),
    )


def _to_paged(x, rows, num_pages):
    paged = torch.zeros(num_pages * _PAGE, *x.shape[1:], dtype=x.dtype, device=x.device)
    paged.index_copy_(0, rows, x)
    return paged.view(num_pages, _PAGE, *x.shape[1:])


class _Varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn, cu_seqlens):
        page_table, seqused, rows, num_pages, max_len = thd_paging(cu_seqlens)
        cu = cu_seqlens.to(torch.int32).contiguous()
        out, lse = attn.prefill_paged(
            q,
            _to_paged(k, rows, num_pages),
            _to_paged(v, rows, num_pages),
            cu,
            seqused,
            page_table,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu)
        ctx.attn, ctx.max_len = attn, max_len
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu = ctx.saved_tensors
        grads = _bwd(ctx.attn, q, k, v, out, dout, lse, cu, cu, ctx.max_len, ctx.max_len)
        return (*grads, None, None)


def varlen_attention(q, k, v, attn, cu_seqlens):
    """Causal attention on packed [T, H, D] tensors, one sequence per cu_seqlens segment."""
    return _Varlen.apply(q.contiguous(), k.contiguous(), v.contiguous(), attn, cu_seqlens)


def cp_forward(q_l, k_glob, v_glob, attn, plan):
    """This rank's rows. k_glob/v_glob hold every token in global packed order."""
    out, lse = attn.prefill_paged(
        q_l,
        _to_paged(k_glob, plan.paged_scatter_idx, plan.num_pages),
        _to_paged(v_glob, plan.paged_scatter_idx, plan.num_pages),
        plan.cu_seqlens_q,
        plan.seqused_k,
        plan.page_table,
        return_lse=True,
    )
    return out, lse


def cp_backward(q_l, k_glob, v_glob, out, dout, lse, attn, plan):
    """dq for this rank; dk/dv FP32 partials in global packed order (to be summed over ranks)."""
    k_c = k_glob.index_select(0, plan.bwd_k_gather_idx)
    v_c = v_glob.index_select(0, plan.bwd_k_gather_idx)
    dq, dk_c, dv_c = _bwd(
        attn,
        q_l,
        k_c,
        v_c,
        out,
        dout,
        lse,
        plan.cu_seqlens_q,
        plan.bwd_cu_seqlens_k,
        plan.max_seqlen_q,
        int(plan.seqused_k.max().item()),
    )
    dk = torch.zeros(k_glob.shape, dtype=torch.float32, device=k_glob.device)
    dv = torch.zeros_like(dk)
    dk.index_add_(0, plan.bwd_k_gather_idx, dk_c.float())
    dv.index_add_(0, plan.bwd_k_gather_idx, dv_c.float())
    return dq, dk, dv


def _gather_global(x, group, cp_size, plan):
    gathered = torch.empty((cp_size * x.shape[0],) + x.shape[1:], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
    return gathered.index_select(0, plan.kv_reorder_idx)


def _scatter_local(x_glob, group, cp_size, plan):
    x = x_glob.index_select(0, plan.kv_unreorder_idx).contiguous()
    out = torch.empty((x.shape[0] // cp_size,) + x.shape[1:], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x, group=group)
    return out


class _CpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_l, k_l, v_l, attn, plan, group, cp_size):
        k_glob = _gather_global(k_l, group, cp_size, plan)
        v_glob = _gather_global(v_l, group, cp_size, plan)
        out, lse = cp_forward(q_l, k_glob, v_glob, attn, plan)
        ctx.save_for_backward(q_l, k_l, v_l, out, lse)
        ctx.attn, ctx.plan, ctx.group, ctx.cp_size = attn, plan, group, cp_size
        return out

    @staticmethod
    def backward(ctx, dout):
        q_l, k_l, v_l, out, lse = ctx.saved_tensors
        plan, group, cp_size = ctx.plan, ctx.group, ctx.cp_size
        k_glob = _gather_global(k_l, group, cp_size, plan)
        v_glob = _gather_global(v_l, group, cp_size, plan)
        dq, dk, dv = cp_backward(q_l, k_glob, v_glob, out, dout, lse, ctx.attn, plan)
        dk = _scatter_local(dk, group, cp_size, plan).to(k_l.dtype)
        dv = _scatter_local(dv, group, cp_size, plan).to(v_l.dtype)
        return dq, dk, dv, None, None, None, None


def cp_all_gather_attention(q_l, k_l, v_l, attn, cu_seqlens, group):
    """Context parallel over the zigzag packed layout, K/V all-gathered."""
    cp_size, cp_rank = dist.get_world_size(group), dist.get_rank(group)
    plan = build_cp_plan(cu_seqlens, cp_size, cp_rank, attn.h_kv, q_l.device)
    return _CpAllGather.apply(
        q_l.contiguous(), k_l.contiguous(), v_l.contiguous(), attn, plan, group, cp_size
    )


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, dout):
        grad = torch.empty_like(dout)
        dist.all_to_all_single(grad, dout.contiguous(), group=ctx.group)
        return grad, None


def zigzag_positions(cu_seqlens, cp_size):
    """Global packed position of each local token, rank-major over all ranks."""
    cu = cu_seqlens.to(torch.int64)
    starts, ends = cu[:-1], cu[1:]
    h = (ends - starts) // (2 * cp_size)
    return torch.cat(
        [_rank_local_positions(starts, ends, h, cp_size, r, cu.device) for r in range(cp_size)]
    )


def a2a_attention(q_l, k_l, v_l, attn, cu_seqlens, group):
    """Context parallel by head scatter: each rank runs full sequences for its share of heads.

    attn must be built for the per-rank head counts (Hq / cp, Hkv / cp).
    """
    cp_size = dist.get_world_size(group)
    pos = zigzag_positions(cu_seqlens, cp_size)
    inv = torch.argsort(pos)

    def to_heads(x):
        t, h, d = x.shape
        x = (
            x.view(t, cp_size, h // cp_size, d)
            .transpose(0, 1)
            .reshape(cp_size * t, h // cp_size, d)
        )
        return _AllToAll.apply(x, group).index_select(0, inv)

    def to_tokens(x):
        x = _AllToAll.apply(x.index_select(0, pos), group)
        t = x.shape[0] // cp_size
        return x.view(cp_size, t, x.shape[1], x.shape[2]).transpose(0, 1).reshape(t, -1, x.shape[2])

    out = varlen_attention(to_heads(q_l), to_heads(k_l), to_heads(v_l), attn, cu_seqlens)
    return to_tokens(out)

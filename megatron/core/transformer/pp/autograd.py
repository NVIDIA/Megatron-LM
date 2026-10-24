# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training entry points: PP forward, FlashAttention-4 backward on its out and lse."""

import torch
import torch.distributed as dist

from .cp_plan import _PAGE, paged_layout

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


def _to_paged(x, layout):
    paged = x.new_zeros((layout.num_pages * _PAGE,) + x.shape[1:])
    paged.index_copy_(0, layout.paged_scatter_idx, x)
    return paged.view(layout.num_pages, _PAGE, *x.shape[1:])


def paged_forward(q, k, v, attn, layout):
    """Out and lse of q's rows. k/v hold every token in global packed order."""
    return attn.prefill_paged(
        q,
        _to_paged(k, layout),
        _to_paged(v, layout),
        layout.cu_seqlens_q,
        layout.seqused_k,
        layout.page_table,
        return_lse=True,
    )


def paged_backward(q, k, v, out, dout, lse, attn, layout):
    """dq for q's rows. Under CP, dk/dv are FP32 partials in global packed order."""
    idx = layout.bwd_k_gather_idx
    k_c, v_c = (k, v) if idx is None else (k.index_select(0, idx), v.index_select(0, idx))
    dq, dk, dv = _bwd(
        attn,
        q,
        k_c,
        v_c,
        out,
        dout,
        lse,
        layout.cu_seqlens_q,
        layout.bwd_cu_seqlens_k,
        layout.max_seqlen_q,
        layout.max_seqlen_k,
    )
    if idx is None:
        return dq, dk, dv
    dk_g = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv_g = torch.zeros_like(dk_g)
    dk_g.index_add_(0, idx, dk.float())
    dv_g.index_add_(0, idx, dv.float())
    return dq, dk_g, dv_g


def _gather_global(x, group, layout):
    if group is None:
        return x
    cp_size = dist.get_world_size(group)
    gathered = x.new_empty((cp_size * x.shape[0],) + x.shape[1:])
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
    return gathered.index_select(0, layout.kv_reorder_idx)


def _scatter_local(x_glob, group, layout):
    x = x_glob.index_select(0, layout.gathered_positions).contiguous()
    out = x.new_empty((x.shape[0] // dist.get_world_size(group),) + x.shape[1:])
    dist.reduce_scatter_tensor(out, x, group=group)
    return out


class _Paged(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn, layout, group):
        k_g, v_g = _gather_global(k, group, layout), _gather_global(v, group, layout)
        out, lse = paged_forward(q, k_g, v_g, attn, layout)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.attn, ctx.layout, ctx.group = attn, layout, group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        layout, group = ctx.layout, ctx.group
        k_g, v_g = _gather_global(k, group, layout), _gather_global(v, group, layout)
        dq, dk, dv = paged_backward(q, k_g, v_g, out, dout, lse, ctx.attn, layout)
        if group is not None:
            dk = _scatter_local(dk, group, layout).to(k.dtype)
            dv = _scatter_local(dv, group, layout).to(v.dtype)
        return dq, dk, dv, None, None, None


def varlen_attention(q, k, v, attn, cu_seqlens):
    """Causal attention on packed [T, H, D] tensors, one sequence per cu_seqlens segment."""
    layout = paged_layout(cu_seqlens)
    return _Paged.apply(q.contiguous(), k.contiguous(), v.contiguous(), attn, layout, None)


def cp_all_gather_attention(q_l, k_l, v_l, attn, cu_seqlens, group):
    """Context parallel over the zigzag packed layout, K/V all-gathered."""
    layout = paged_layout(cu_seqlens, dist.get_world_size(group), dist.get_rank(group))
    return _Paged.apply(q_l.contiguous(), k_l.contiguous(), v_l.contiguous(), attn, layout, group)


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


def a2a_attention(q_l, k_l, v_l, attn, cu_seqlens, group):
    """Context parallel by head scatter: each rank runs full sequences for its share of heads.

    attn must be built for the per-rank head counts (Hq / cp, Hkv / cp).
    """
    cp_size = dist.get_world_size(group)
    order = paged_layout(cu_seqlens, cp_size, dist.get_rank(group))

    def to_heads(x):
        t, h, d = x.shape
        x = (
            x.view(t, cp_size, h // cp_size, d)
            .transpose(0, 1)
            .reshape(cp_size * t, h // cp_size, d)
        )
        return _AllToAll.apply(x, group).index_select(0, order.kv_reorder_idx)

    def to_tokens(x):
        x = _AllToAll.apply(x.index_select(0, order.gathered_positions), group)
        t = x.shape[0] // cp_size
        return x.view(cp_size, t, x.shape[1], x.shape[2]).transpose(0, 1).reshape(t, -1, x.shape[2])

    out = varlen_attention(to_heads(q_l), to_heads(k_l), to_heads(v_l), attn, cu_seqlens)
    return to_tokens(out)

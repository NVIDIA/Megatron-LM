# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Contiguous packed CSA row ownership, boundary exchange and indexer metadata."""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

from . import packed_layout
from .packed_sparse_attention import indexer_topk


def _thd_cp_position_ids(
    cu_seqlens_padded: torch.Tensor, global_start: int, local_rows: int
) -> torch.Tensor:
    """Map a consecutive CP row interval to positions within packed sequences."""
    global_rows = torch.arange(
        int(global_start),
        int(global_start) + int(local_rows),
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    sequence_ids = torch.bucketize(
        global_rows, cu_seqlens_padded[1:], out_int32=True, right=True
    ).clamp_max(cu_seqlens_padded.shape[0] - 2)
    sequence_starts = cu_seqlens_padded[sequence_ids]
    sequence_ends = cu_seqlens_padded[sequence_ids + 1]
    valid_rows = (global_rows >= sequence_starts) & (global_rows < sequence_ends)
    return torch.where(valid_rows, global_rows - sequence_starts, 0)


def apply_thd_cp_local_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows."""
    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    if inverse:
        # The fused kernel is in-place, but sparse-attention backward needs its original output.
        rope_input = rope_input.clone()
    output = fused_mla_rope_inplace(
        rope_input,
        cos,
        sin,
        nope_dim,
        pos_dim,
        cu_seqlens_q=cu_seqlens_padded,
        inverse=inverse,
        remove_interleaving=True,
        position_ids=_thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0]),
    )
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


def apply_thd_cp_local_rope_unfused(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    config,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply unfused RoPE to a consecutive interval of packed CP rows."""
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])
    freqs = torch.index_select(rotary_pos_emb, 0, position_ids.long())

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    content, rotary = torch.split(rope_input, [nope_dim, pos_dim], dim=-1)
    rotary = _apply_rotary_pos_emb_bshd(
        rotary,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        mscale=1.0,
        mla_rotary_interleaved=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    output = torch.cat((content, rotary), dim=-1)
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange fixed left-boundary windows and scatter gradients back to senders."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup):
        """Receive fixed left-boundary hidden rows needed by this CP rank."""
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if cp_size > 1 and tensor.shape[0] < d_window:
            raise RuntimeError(
                "DSv4 CP boundary exchange requires local rows >= D_window: "
                f"local_rows={tensor.shape[0]}, D_window={d_window}."
            )
        boundary = tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            send_tail = tensor[-d_window:].contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        for req in dist.batch_isend_irecv(ops) if ops else []:
            req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Send boundary gradients back to ranks that own those hidden rows."""
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        d_window = ctx.d_window
        grad_input = grad_boundary.new_zeros(ctx.input_shape)

        ops = []
        if cp_rank > 0:
            send_grad = grad_boundary.contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_grad, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            recv_grad = grad_boundary.new_empty(grad_boundary.shape)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_grad, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        for req in dist.batch_isend_irecv(ops) if ops else []:
            req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-d_window:] = recv_grad
        return grad_input, None, None


def exchange_cp_boundary_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
    csa_window_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Exchange hidden-state rows immediately left of this rank's token block."""
    d_comp = 8 if compress_ratio == 4 else compress_ratio if compress_ratio > 1 else 0
    d_window = max(int(csa_window_size), d_comp)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary_hidden = _LeftBoundaryExchange.apply(hidden_flat, d_window, cp_group)
    return boundary_hidden.reshape((d_window,) + tuple(hidden_states.shape[1:]))


def prepare_cp_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity compressor input for this rank's token block.

    Returns:
        ``hidden_compact``: rank-local compressor input, shape
            ``(compact_group_capacity * ratio, ...)``.
        ``compressed_group_ids``: original per-sequence compressed group id for each
            compact group, shape ``(compact_group_capacity,)``. For example,
            with ``ratio=4``, ``comp_id=3`` maps to RoPE position ``12``.
        ``compressed_position_ids``: precomputed RoPE positions, with padding
            groups mapped to position zero.
        ``cu_seqlens_compressed``: global sequence-major compressed prefixes.
        ``seq_to_rank_row``: map from global sequence-major compressed rows
            to their canonical rank-major all-gather rows.
            If rank 0 owns ``A0, A1`` and rank 1 owns ``B0, B1``, with four
            slots per rank, logical rows ``[A0, A1, B0, B1]`` are stored as
            ``[A0, A1, pad, pad | B0, B1, pad, pad]`` and map to ``[0, 1, 4, 5]``.
    """
    cp_size = int(cp_size)
    ratio = int(ratio)
    d_comp = 8 if ratio == 4 else ratio
    global_start = int(global_start)
    l_local = hidden_local.shape[0]
    group_alignment = 32 // math.gcd(32, ratio)
    c_cap = max(1, (l_local + d_comp) // ratio)
    c_cap = ((c_cap + group_alignment - 1) // group_alignment) * group_alignment
    return packed_layout.compact_compressor_input(
        hidden_local, boundary_hidden, cu_seqlens, global_start, ratio, d_comp, c_cap, cp_size
    )


@torch.compile
def build_cp_indexer_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
    total_k_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the indexer's packed local-Q/full-K metadata."""
    # Each real Q segment intersects its sequence with this rank's row interval,
    # while K keeps the sequence's full compressed segment. The final synthetic
    # segment owns the fixed-capacity K tail so the logical endpoint matches the
    # physical buffer. Causal offsets restore each non-empty local Q segment's
    # position in the original sequence.
    global_end = global_start + local_rows
    zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    local_starts = cu_seqlens_q[:-1].clamp_min(global_start)
    local_ends = cu_seqlens_q[1:].clamp_max(global_end)
    q_lens = (local_ends - local_starts).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    padding_q = (global_end - cu_seqlens_q[-1].clamp_min(global_start)).clamp_min(0)
    cu_q_topk = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
    # The gathered K buffer has fixed CP capacity, which can exceed the number
    # of valid compressed rows. Assign its tail to the synthetic padding
    # segment so cu_k[-1] still matches the physical K tensor length.
    k_capacity_end = cu_seqlens_compressed.new_full((1,), total_k_rows)
    cu_k_topk = torch.cat((cu_seqlens_compressed, k_capacity_end))
    q_causal_offsets = torch.cat(
        (torch.where(q_lens > 0, local_starts - cu_seqlens_q[:-1], 0), zero)
    )
    return cu_q_topk, cu_k_topk, q_causal_offsets


def compute_cp_indexer_topk(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Return local top-k and its local-Q/full-K packed layout."""
    topk_width = int(topk_width)
    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return None, None
    max_seqlen_kv = int(max_seqlen_q) // int(ratio)

    global_start = int(global_start)
    l_local = q_indexer_local.shape[0]
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    cu_q_topk, cu_k_topk, q_causal_offsets = build_cp_indexer_layout(
        cu_seqlens_q, cu_seqlens_compressed, global_start, l_local, k_indexer_seq_major.shape[0]
    )

    if max_seqlen_kv == 0 or k_indexer_seq_major.shape[0] == 0:
        topk = torch.full(
            (l_local, topk_width), -1, device=q_indexer_local.device, dtype=torch.int32
        )
        return topk, (cu_q_topk, cu_k_topk, q_causal_offsets)

    topk, _ = indexer_topk(
        q_indexer_local,
        k_indexer_seq_major,
        weights_indexer_local,
        topk=topk_width,
        ratio=ratio,
        indexer_softmax_scale=indexer_softmax_scale,
        cu_seqlens_q=cu_q_topk,
        cu_seqlens_kv=cu_k_topk,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_kv=int(max_seqlen_kv),
        q_causal_offsets=q_causal_offsets,
    )
    return topk, (cu_q_topk, cu_k_topk, q_causal_offsets)


def validate_packed_inputs(packed_seq_params, config, cp_group):
    """Validate the static packed contract without modifying any process group.

    main represents a runtime CP1 microbatch with cp_group=None. Dynamic CP
    transitions are a separate integration; reject them instead of accidentally
    using a stale build-time group during forward or recompute.
    """
    if config.dsa_kernel_backend != "cudnn":
        raise ValueError("Packed DSv4 attention requires dsa_kernel_backend='cudnn'.")
    if packed_seq_params.qkv_format != "thd":
        raise ValueError("DSv4 packed attention requires qkv_format='thd'.")
    if cp_group is None:
        raise ValueError("DSv4 requires an explicit build-time CP process group, including CP1.")
    runtime_size = packed_seq_params.local_cp_size
    runtime_group = packed_seq_params.cp_group
    if runtime_size is not None:
        if runtime_size == 1 and runtime_group is not None:
            raise ValueError("Runtime CP1 metadata must use cp_group=None.")
        if runtime_size > 1 and (runtime_group is None or runtime_group.size() != runtime_size):
            raise ValueError("Runtime CP metadata must provide the matching process group.")
        if runtime_size != cp_group.size() or (runtime_size > 1 and runtime_group is not cp_group):
            raise ValueError(
                "DSv4 currently supports static CP; runtime group transitions are unsupported."
            )
    if cp_group.size() > 1 and config.attention_cp_layout != "contiguous":
        raise ValueError("DSv4 packed CP requires attention_cp_layout='contiguous'.")
    if packed_seq_params.max_seqlen_q is None or packed_seq_params.max_seqlen_kv is None:
        raise ValueError("Packed DSv4 requires host-known maximum query and KV sequence lengths.")
    q = packed_seq_params.cu_seqlens_q
    kv = packed_seq_params.cu_seqlens_kv
    if q is None or kv is None or q.ndim != 1 or q.shape != kv.shape or q.numel() < 2:
        raise ValueError("Packed DSv4 requires query and KV cumulative lengths for self attention.")
    for cu in (
        q,
        kv,
        packed_seq_params.cu_seqlens_q_padded,
        packed_seq_params.cu_seqlens_kv_padded,
    ):
        if cu is not None and (
            cu.shape != q.shape or cu.dtype != torch.int32 or cu.device != q.device
        ):
            raise ValueError(
                "Packed DSv4 cumulative lengths must be matching int32 tensors on one device."
            )

    if q.data_ptr() != kv.data_ptr():
        torch._assert_async((q == kv).all(), "DSv4 packed CSA supports self attention only.")
    padded_q = packed_seq_params.cu_seqlens_q_padded
    padded_kv = packed_seq_params.cu_seqlens_kv_padded
    physical_q = q if padded_q is None else padded_q
    physical_kv = kv if padded_kv is None else padded_kv
    if physical_q.data_ptr() != physical_kv.data_ptr():
        torch._assert_async(
            (physical_q == physical_kv).all(),
            "Packed CSA requires matching physical Q/KV document boundaries.",
        )

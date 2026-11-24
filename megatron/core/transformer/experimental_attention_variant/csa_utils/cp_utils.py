# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MCore-facing utilities for the DSv4 THD context-parallel path.

This module owns CP row mapping, boundary exchange, compressor-input layout,
and indexer top-k metadata. It reuses MCore's fused MLA RoPE and calls the
retained compaction kernel; ``csa.py`` calls final-index lowering directly.
"""

import math
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

from . import thd_layout_kernels
from .fused_sparse_attention import (
    FUSED_INDEXER_MAX_SAFE_ROWS,
    THDCompactIndexerWorkspace,
    build_thd_compact_k_layout,
    indexer_topk,
    pack_thd_compact_k,
)
from .packed_zigzag import PackedZigzagLayout, build_packed_zigzag_layout

CPIndexerLayout = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# =============================================================================
# RoPE Wrappers
# =============================================================================


def thd_global_rows_position_ids(
    cu_seqlens_padded: torch.Tensor, global_rows: torch.Tensor
) -> torch.Tensor:
    """Map global packed rows to positions within their packed sequences.

    Rows outside every sequence (for example the zero-filled left halo of CP rank 0) map to
    position zero.
    """
    sequence_ids = torch.bucketize(
        global_rows, cu_seqlens_padded[1:], out_int32=True, right=True
    ).clamp_max(cu_seqlens_padded.shape[0] - 2)
    sequence_starts = cu_seqlens_padded[sequence_ids]
    sequence_ends = cu_seqlens_padded[sequence_ids + 1]
    valid_rows = (global_rows >= sequence_starts) & (global_rows < sequence_ends)
    return torch.where(valid_rows, global_rows - sequence_starts, 0)


def thd_cp_position_ids(
    cu_seqlens_padded: torch.Tensor, global_start: int, local_rows: int
) -> torch.Tensor:
    """Map a consecutive CP row interval to positions within packed sequences."""
    global_rows = torch.arange(
        int(global_start),
        int(global_start) + int(local_rows),
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    return thd_global_rows_position_ids(cu_seqlens_padded, global_rows)


# Retained for callers outside megatron.core that already use the private name.
_thd_cp_position_ids = thd_cp_position_ids


def _check_thd_cp_rope_rows(
    x: torch.Tensor, global_start: Optional[int], position_ids: Optional[torch.Tensor]
) -> None:
    """Require exactly one THD CP row-to-position source for the local RoPE helpers."""
    if (global_start is None) == (position_ids is None):
        raise ValueError(
            "THD CP local RoPE takes exactly one of global_start (a consecutive row interval) "
            "and position_ids (explicit per-row positions)."
        )
    if position_ids is not None and position_ids.shape != (x.shape[0],):
        raise ValueError(
            f"THD CP local RoPE position_ids must have shape ({x.shape[0]},), "
            f"got {tuple(position_ids.shape)}."
        )


def apply_thd_cp_local_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: Optional[int],
    inverse: bool = False,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows.

    ``global_start`` describes one consecutive interval of global rows; the kernel derives the
    positions from ``cu_seqlens_padded``. Pass ``global_start=None`` and ``position_ids``
    (one position per row) for a non-consecutive row layout such as the zigzag partition.
    """
    _check_thd_cp_rope_rows(x, global_start, position_ids)
    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    if inverse:
        # The fused kernel is in-place, but sparse-attention backward needs its original output.
        rope_input = rope_input.clone()
    row_kwargs = (
        {"thd_global_start": global_start}
        if position_ids is None
        else {"position_ids": position_ids}
    )
    output = fused_mla_rope_inplace(
        rope_input,
        cos,
        sin,
        nope_dim,
        pos_dim,
        cu_seqlens_q=cu_seqlens_padded,
        inverse=inverse,
        remove_interleaving=True,
        **row_kwargs,
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
    global_start: Optional[int],
    config,
    inverse: bool = False,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply unfused RoPE to local THD CP rows.

    ``global_start`` describes one consecutive interval of global rows. Pass
    ``global_start=None`` and ``position_ids`` (one position per row) for a non-consecutive
    row layout such as the zigzag partition.
    """
    _check_thd_cp_rope_rows(x, global_start, position_ids)
    if position_ids is None:
        position_ids = thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])
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


# =============================================================================
# Boundary Hidden Exchange
# =============================================================================


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
        if tensor.shape[0] < d_window:
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
        for req in dist.batch_isend_irecv(ops):
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
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-d_window:] = recv_grad
        return grad_input, None, None


class _ZigzagBoundaryExchange(torch.autograd.Function):
    """Exchange the two left-boundary windows of a zigzag CP rank.

    Under the zigzag partition, rank ``r`` holds chunk ``r`` (segment A) followed by chunk
    ``2 * CP - 1 - r`` (segment B), each ``segment_rows`` long. The window left of segment A is
    the tail of rank ``r - 1``'s segment A (zeros on rank 0, where A starts the sequence); the
    window left of segment B is the tail of rank ``r + 1``'s segment B, and on the last rank it
    is the tail of its own segment A. The result is ``[halo_A | halo_B]``. Backward returns each
    halo gradient to the rank that owns those rows; no row receives two gradients.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup):
        """Receive the ``(2 * d_window, ...)`` left windows of this rank's two segments."""
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if tensor.shape[0] % 2 != 0 or tensor.shape[0] // 2 < d_window:
            raise RuntimeError(
                "DSv4 zigzag CP boundary exchange requires two equal segments of at least "
                f"D_window rows: local_rows={tensor.shape[0]}, D_window={d_window}."
            )
        segment_rows = tensor.shape[0] // 2
        boundary = tensor.new_zeros((2 * d_window,) + tuple(tensor.shape[1:]))
        halo_a, halo_b = boundary[:d_window], boundary[d_window:]

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.irecv, halo_a, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
            send_b_tail = tensor[2 * segment_rows - d_window :].contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_b_tail, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            send_a_tail = tensor[segment_rows - d_window : segment_rows].contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_a_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
            ops.append(
                dist.P2POp(
                    dist.irecv, halo_b, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        else:
            # The last rank's segment B (chunk CP) directly follows its own segment A.
            halo_b.copy_(tensor[segment_rows - d_window : segment_rows])
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Send halo gradients back to the segment tails they were read from."""
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        d_window = ctx.d_window
        segment_rows = ctx.input_shape[0] // 2
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        grad_a, grad_b = grad_boundary[:d_window], grad_boundary[d_window:]

        ops = []
        recv_a_tail = recv_b_tail = None
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_a.contiguous(),
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
            recv_b_tail = grad_boundary.new_empty(grad_b.shape)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_b_tail, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            recv_a_tail = grad_boundary.new_empty(grad_a.shape)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_a_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_b.contiguous(),
                    dist.get_global_rank(cp_group, cp_rank + 1),
                    cp_group,
                )
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if recv_a_tail is not None:
            grad_input[segment_rows - d_window : segment_rows] = recv_a_tail
        else:
            # Last rank: its halo_B was a local copy of its own segment-A tail.
            grad_input[segment_rows - d_window : segment_rows] = grad_b
        if recv_b_tail is not None:
            grad_input[2 * segment_rows - d_window :] = recv_b_tail
        return grad_input, None, None


def dsv4_cp_window_rows(compress_ratio: int, csa_window_size: int) -> int:
    """Rows of left context a DSv4 CP segment needs: the sliding window or one compressor
    window (8 rows for the overlapping ratio-4 compressor), whichever is larger."""
    d_comp = 8 if compress_ratio == 4 else compress_ratio if compress_ratio > 1 else 0
    return max(int(csa_window_size), d_comp)


def exchange_cp_boundary_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
    csa_window_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Exchange hidden-state rows immediately left of this rank's token block."""
    d_window = dsv4_cp_window_rows(compress_ratio, csa_window_size)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary_hidden = _LeftBoundaryExchange.apply(hidden_flat, d_window, cp_group)
    return boundary_hidden.reshape((d_window,) + tuple(hidden_states.shape[1:]))


def exchange_cp_zigzag_boundary_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
    csa_window_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Exchange the hidden-state rows left of both segments of a zigzag CP rank.

    Returns ``[halo_A | halo_B]`` with ``2 * D_window`` rows; see ``_ZigzagBoundaryExchange``.
    """
    d_window = dsv4_cp_window_rows(compress_ratio, csa_window_size)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary_hidden = _ZigzagBoundaryExchange.apply(hidden_flat, d_window, cp_group)
    return boundary_hidden.reshape((2 * d_window,) + tuple(hidden_states.shape[1:]))


# =============================================================================
# Zigzag Partition Geometry
# =============================================================================


def thd_cp_zigzag_segments(
    local_rows: int, cp_rank: int, cp_size: int
) -> Tuple[int, Tuple[int, int]]:
    """Return ``(segment_rows, (start_A, start_B))`` of a zigzag CP rank's two row segments.

    The packed rows are cut into ``2 * cp_size`` chunks; rank ``r`` holds chunk ``r`` (segment
    A) followed by chunk ``2 * cp_size - 1 - r`` (segment B). This is MCore's per-sequence
    zigzag order for a pack holding a single sequence.
    """
    local_rows, cp_rank, cp_size = int(local_rows), int(cp_rank), int(cp_size)
    if local_rows % 2 != 0:
        raise ValueError(
            f"Zigzag CP needs an even number of local rows (two equal chunks), got {local_rows}."
        )
    segment_rows = local_rows // 2
    return segment_rows, (cp_rank * segment_rows, (2 * cp_size - 1 - cp_rank) * segment_rows)


class DSv4ZigzagCpLayout(NamedTuple):
    """Geometry of one zigzag CP rank for a DSv4 THD layer, shared by every layer of a microbatch.

    ``q_position_ids`` gives the in-sequence position of each local row ``[A | B]``;
    ``kv_position_ids`` those of ``[halo_A | halo_B | A | B]``, the rows that enter
    ``linear_kv_proj``. ``compressor_layout`` is the merged two-segment compressor layout, or
    None for window-only layers.
    """

    segment_rows: int
    segment_starts: Tuple[int, int]
    d_window: int
    q_position_ids: torch.Tensor
    kv_position_ids: torch.Tensor
    compressor_layout: Optional[thd_layout_kernels.CPCompressorLayout]


_ZIGZAG_LAYOUT_CACHE_ATTR = "_dsv4_zigzag_cp_layout_cache"


def check_dsv4_zigzag_cp_pack(
    cu_seqlens: torch.Tensor, local_rows: int, d_window: int, compress_ratio: int
) -> int:
    """Fail closed outside the supported zigzag envelope; return the segment rows.

    Supported: one sequence per pack (MCore zigzags each sequence separately, so a pack with
    several sequences has more than two segments per rank), two segments of at least
    ``d_window`` rows (each halo then comes from a single peer) and, for compressed layers,
    segments that are whole compression blocks. The ratio-4 compressor (overlapping windows,
    CSA indexer) is not supported. All checks use shapes and host integers only.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != 2:
        raise NotImplementedError(
            "The two-segment zigzag layout expects one sequence per pack; got cumulative "
            f"lengths of shape {tuple(cu_seqlens.shape)}. "
            "Packed inputs use the per-document layout."
        )
    if int(compress_ratio) == 4:
        raise NotImplementedError(
            "DSv4 zigzag CP does not support the ratio-4 compressor and CSA indexer yet."
        )
    segment_rows = int(local_rows) // 2
    if int(local_rows) % 2 != 0 or segment_rows < int(d_window):
        raise ValueError(
            "DSv4 zigzag CP needs two equal chunks of at least D_window rows per rank: "
            f"local_rows={local_rows}, D_window={d_window}."
        )
    if int(compress_ratio) > 1 and segment_rows % int(compress_ratio) != 0:
        raise ValueError(
            "DSv4 zigzag CP needs chunks that are whole compression blocks: "
            f"chunk_rows={segment_rows}, compress_ratio={compress_ratio}. Pad the sequence to "
            "a multiple of 2 * CP * compress_ratio."
        )
    return segment_rows


def build_dsv4_zigzag_cp_layout(
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    local_rows: int,
    compress_ratio: int,
    csa_window_size: int,
) -> DSv4ZigzagCpLayout:
    """Build a zigzag CP rank's row positions and compressor layout on device (no host sync)."""
    d_window = dsv4_cp_window_rows(compress_ratio, csa_window_size)
    check_dsv4_zigzag_cp_pack(cu_seqlens, local_rows, d_window, compress_ratio)
    segment_rows, segment_starts = thd_cp_zigzag_segments(local_rows, cp_rank, cp_size)
    # The zigzag rows assume the single sequence fills the whole pack.
    torch._assert_async(
        (cu_seqlens[0] == 0) & (cu_seqlens[-1] == int(local_rows) * int(cp_size)),
        "DSv4 zigzag CP requires padded cumulative lengths [0, local_rows * cp_size].",
    )

    def rows(start, count):
        return torch.arange(start, start + count, dtype=cu_seqlens.dtype, device=cu_seqlens.device)

    q_rows = torch.cat([rows(start, segment_rows) for start in segment_starts])
    halo_rows = torch.cat([rows(start - d_window, d_window) for start in segment_starts])
    q_position_ids = thd_global_rows_position_ids(cu_seqlens, q_rows)
    kv_position_ids = torch.cat(
        (thd_global_rows_position_ids(cu_seqlens, halo_rows), q_position_ids)
    )
    compressor_layout = None
    if int(compress_ratio) > 1:
        compressor_layout = thd_layout_kernels.build_cp_zigzag_compressor_layout(
            cu_seqlens, segment_starts, segment_rows, cp_size, int(compress_ratio)
        )
    return DSv4ZigzagCpLayout(
        segment_rows, segment_starts, d_window, q_position_ids, kv_position_ids, compressor_layout
    )


def get_dsv4_zigzag_cp_layout(
    packed_seq_params,
    cu_seqlens: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    local_rows: int,
    compress_ratio: int,
    csa_window_size: int,
    config,
) -> Union[DSv4ZigzagCpLayout, PackedZigzagLayout]:
    """Return this microbatch's zigzag CP layout, built once and shared by its layers.

    The layout is memoized on ``packed_seq_params``, which is built per microbatch. It is
    never memoized while a CUDA graph is captured or when CUDA graphs are configured: a graph
    must compute the layout from its own static ``cu_seqlens`` buffer.
    """
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    capturing = cu_seqlens.is_cuda and torch.cuda.is_current_stream_capturing()
    use_cache = (
        packed_seq_params is not None
        and not capturing
        and getattr(config, "cuda_graph_impl", "none") == "none"
    )
    key = (
        cu_seqlens.data_ptr(),
        cp_size,
        cp_rank,
        int(local_rows),
        int(compress_ratio),
        int(csa_window_size),
    )
    cache = getattr(packed_seq_params, _ZIGZAG_LAYOUT_CACHE_ATTR, None) if use_cache else None
    if cache is not None and key in cache:
        return cache[key]
    if cu_seqlens.ndim == 1 and cu_seqlens.numel() > 2:
        if capturing:
            raise ValueError(
                "Packed DSv4 zigzag needs eager mode to select its halo exchange splits."
            )
        bounds = getattr(packed_seq_params, "thd_cp_host_cu_seqlens_q", None)
        # A scheduler may pad the metadata with repeated prefix values. The prebuilt route
        # compacts those away; the compressor and attention must use the original sequence IDs.
        if bounds is None or len(bounds) != cu_seqlens.numel():
            bounds = cu_seqlens.detach().cpu().tolist()
        layout = build_packed_zigzag_layout(
            bounds,
            cp_rank,
            cp_size,
            local_rows,
            int(compress_ratio),
            dsv4_cp_window_rows(compress_ratio, csa_window_size),
            cu_seqlens.device,
        )
    else:
        layout = build_dsv4_zigzag_cp_layout(
            cu_seqlens, cp_rank, cp_size, local_rows, compress_ratio, csa_window_size
        )
    if use_cache:
        if cache is None:
            cache = {}
            setattr(packed_seq_params, _ZIGZAG_LAYOUT_CACHE_ATTR, cache)
        # Small index tensors shared by every layer: keep them out of activation offloading.
        for tensor in (layout.q_position_ids, layout.kv_position_ids):
            tensor._do_not_offload = True
        cache[key] = layout
    return layout


# =============================================================================
# Compressed Metadata And Compressor Inputs
# =============================================================================


def prepare_cp_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build fixed-capacity compressor input for this rank's token block.

    Returns:
        ``hidden_compact``: rank-local compressor input, shape
            ``(compact_group_capacity * ratio, ...)``.
        ``compressed_group_ids``: original per-sequence compressed group id for each
            compact group, shape ``(compact_group_capacity,)``. For example,
            with ``ratio=4``, ``comp_id=3`` maps to RoPE position ``12``.
        ``compressed_position_ids``: precomputed RoPE positions, with padding
            groups mapped to position zero.
        ``local_cu_seqlens`` and ``local_cu_seqlens_compressed``: prefixes that
            describe the physically compacted token/group segments. They let the
            regular THD fused compressor consume a pre-grouped CP buffer directly.
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
    return thd_layout_kernels.CompressorInputCompact.apply(
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


@torch.compile
def _build_cp_indexer_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the indexer's packed local-Q/full-K metadata."""
    # Each real Q segment intersects its sequence with this rank's row interval,
    # while K keeps the sequence's full compressed segment. The final synthetic
    # segment holds CP capacity padding and has zero K rows. Causal offsets
    # restore each non-empty local Q segment's position in the original sequence.
    global_end = global_start + local_rows
    zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    local_starts = cu_seqlens_q[:-1].clamp_min(global_start)
    local_ends = cu_seqlens_q[1:].clamp_max(global_end)
    q_lens = (local_ends - local_starts).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    padding_q = (global_end - cu_seqlens_q[-1].clamp_min(global_start)).clamp_min(0)
    cu_q_topk = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
    cu_k_topk = torch.cat((cu_seqlens_compressed, cu_seqlens_compressed[-1:]))
    q_causal_offsets = torch.cat(
        (torch.where(q_lens > 0, local_starts - cu_seqlens_q[:-1], 0), zero)
    )
    return cu_q_topk, cu_k_topk, q_causal_offsets


def build_cp_compact_indexer_layout(
    logical_layout: CPIndexerLayout,
    cu_seqlens_compressed: torch.Tensor,
    total_k_rows: int,
    ratio: int,
) -> Tuple[CPIndexerLayout, torch.Tensor]:
    """Pad each THD K segment with an invisible row for compact Top-K."""
    cu_q_topk, _, q_causal_offsets = logical_layout
    cu_k_topk, source_row_map = build_thd_compact_k_layout(
        cu_q_topk, cu_seqlens_compressed, total_k_rows, ratio
    )
    return (cu_q_topk, cu_k_topk, q_causal_offsets), source_row_map


def pack_cp_compact_indexer_k(
    k_indexer_seq_major: torch.Tensor, source_row_map: torch.Tensor
) -> torch.Tensor:
    """Insert zero-valued, causally invisible K rows for compact THD."""
    return pack_thd_compact_k(k_indexer_seq_major, source_row_map)


# Verified fused-kernel row-limit defect: see FUSED_INDEXER_MAX_SAFE_ROWS in
# dsa_fused_safety.py (single source of truth; re-exported here for the
# balanced-indexer prebuild and tests). Policy at THIS wrapper: the balanced
# synthetic-layout path fails closed above the limit; ordinary reference calls
# proceed unchanged — the once-per-process correctness warning fires inside
# the CSA _indexer_topk_core funnel. The neighboring DSv3.2 backend applies the
# same shared warning in dsa_cudnn_kernels before each direct cuDNN invocation.


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
    use_fused: bool,
    deterministic: bool = False,
    precision: str = "bf16",
    compact_workspace: Optional[THDCompactIndexerWorkspace] = None,
    return_softmax: bool = False,
    indexer_layout: Optional[CPIndexerLayout] = None,
    logical_indexer_layout: Optional[CPIndexerLayout] = None,
    max_seqlen_kv: Optional[int] = None,
    prebuilt_layout: Optional[CPIndexerLayout] = None,
    synthetic_layout: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[CPIndexerLayout], Optional[torch.Tensor]]:
    """Return local top-k, packed layout, and optional compact Top-K softmax.

    Ordinary and balanced production calls use the compact scorer and workspace
    contract. Legacy calls with ``prebuilt_layout`` use dense scoring and an
    unpadded K layout; they return None as the third (compact softmax) result.

    ``max_seqlen_kv`` optionally overrides the K capacity (default
    ``max_seqlen_q // ratio``). Compact scoring adds two guard rows per segment;
    its workspace must use the same bound. The legacy dense path uses this value
    as the width of its FP32 score matrix.

    ``prebuilt_layout`` optionally supplies a ``(cu_q, cu_k, q_causal_offsets)`` tuple from a
    previous ``_build_cp_indexer_layout(cu_seqlens_q, cu_seqlens_compressed, global_start,
    rows)`` call with identical arguments, skipping the rebuild (the layout is constant across
    layers within a microbatch). Only the fused path consumes the layout for masking; the
    unfused path recomputes its masking from ``(cu_seqlens_q, global_start)`` and returns the
    tuple as metadata only. Callers passing a synthetic layout that differs from that
    recomputation (the balanced zigzag path) must set ``synthetic_layout=True`` and use the fused
    path. Non-empty calls with ``synthetic_layout=True`` and ``use_fused=False`` raise
    ``ValueError`` rather than silently mis-mask. An ordinary cached
    ``_build_cp_indexer_layout`` result keeps the default.
    """
    topk_width = int(topk_width)
    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return None, None, None
    max_seqlen_kv = int(max_seqlen_q) // int(ratio) if max_seqlen_kv is None else int(max_seqlen_kv)
    if max_seqlen_kv == 0:
        return None, None, None

    if precision == "mxfp8" and not use_fused:
        raise ValueError(
            "MXFP8 CP indexer requires fused compact scoring; BF16 fallback is invalid"
        )

    if synthetic_layout and not use_fused:
        raise ValueError(
            "synthetic_layout=True requires use_fused=True because the unfused path "
            "recomputes masking and ignores the synthetic layout."
        )

    if use_fused and int(q_indexer_local.shape[0]) > FUSED_INDEXER_MAX_SAFE_ROWS:
        # See FUSED_INDEXER_MAX_SAFE_ROWS. Policy split (after the zero-work exits
        # above, which never launch the kernel):
        # - the balanced synthetic-layout path fails closed: it necessarily issues
        #   multiple fused calls per microbatch, so a later above-limit call is the
        #   verified-corrupt pattern, and a synthetic layout cannot take the unfused
        #   path (the unfused arm treats layouts as metadata only);
        # - pre-existing callers (reference CP path and non-CP paths) keep their
        #   behavior and proceed fused; the once-per-process high-severity
        #   correctness warning fires in the CSA _indexer_topk_core funnel;
        #   DSv3.2 direct callers are guarded separately in dsa_cudnn_kernels.
        if synthetic_layout:
            raise RuntimeError(
                f"fused indexer top-k with {int(q_indexer_local.shape[0])} query rows "
                f"exceeds the verified-safe limit of {FUSED_INDEXER_MAX_SAFE_ROWS} for "
                "the current fused kernel package (silent corruption of rows >= 32768 "
                "unless first-in-process; verified on GB200 with cudnn-frontend 1.26.0, "
                "no known-good version yet), and this synthetic layout cannot take the "
                "unfused path. Reduce per-call rows (higher CP degree or smaller pack "
                "capacity), or disable dsa_cp_balance_indexer and use the contiguous "
                "reference path with an unfused backend."
            )

    global_start = int(global_start)
    l_local = q_indexer_local.shape[0]
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    if prebuilt_layout is not None:
        logical_indexer_layout = indexer_layout = prebuilt_layout
    else:
        if logical_indexer_layout is None:
            logical_indexer_layout = build_cp_indexer_layout(
                cu_seqlens_q,
                cu_seqlens_compressed,
                global_start,
                l_local,
                k_indexer_seq_major.shape[0],
            )
        if not use_fused:
            indexer_layout = logical_indexer_layout
        elif indexer_layout is None:
            indexer_layout, source_row_map = build_cp_compact_indexer_layout(
                logical_indexer_layout, cu_seqlens_compressed, k_indexer_seq_major.shape[0], ratio
            )
            k_indexer_seq_major = pack_cp_compact_indexer_k(k_indexer_seq_major, source_row_map)

        if use_fused:
            # Each real segment has one or two invisible K padding rows.
            max_seqlen_kv += 2
    cu_q_topk, cu_k_topk, q_causal_offsets = indexer_layout

    if not use_fused:
        global_rows = torch.arange(
            global_start,
            global_start + l_local,
            dtype=cu_seqlens_q.dtype,
            device=cu_seqlens_q.device,
        )
        sequence_ids = torch.bucketize(
            global_rows, cu_seqlens_q[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_q.shape[0] - 2)
        positions = global_rows - cu_seqlens_q[sequence_ids]
        visible_k = torch.minimum(
            torch.div(positions + 1, int(ratio), rounding_mode="floor"),
            cu_seqlens_compressed[sequence_ids + 1] - cu_seqlens_compressed[sequence_ids],
        ).clamp_min(0)
        valid_q = (global_rows >= cu_seqlens_q[sequence_ids]) & (
            global_rows < cu_seqlens_q[sequence_ids + 1]
        )

        k_rows = torch.arange(
            k_indexer_seq_major.shape[0],
            dtype=cu_seqlens_compressed.dtype,
            device=cu_seqlens_compressed.device,
        )
        k_sequence_ids = torch.bucketize(
            k_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_compressed.shape[0] - 2)
        k_positions = k_rows - cu_seqlens_compressed[k_sequence_ids]
        output = torch.full(
            (l_local, topk_width), -1, dtype=torch.int32, device=q_indexer_local.device
        )
        selected_width = min(topk_width, k_indexer_seq_major.shape[0])
        for start in range(0, l_local, 128):
            end = min(start + 128, l_local)
            scores = torch.einsum(
                "rhd,kd->rhk", q_indexer_local[start:end].float(), k_indexer_seq_major.float()
            )
            scores = torch.relu(scores) * weights_indexer_local[start:end].float().unsqueeze(-1)
            scores = scores.sum(dim=1) * float(indexer_softmax_scale)
            valid_k = (
                (k_sequence_ids.unsqueeze(0) == sequence_ids[start:end].unsqueeze(1))
                & (k_positions.unsqueeze(0) < visible_k[start:end].unsqueeze(1))
                & valid_q[start:end].unsqueeze(1)
            )
            scores = scores.masked_fill(~valid_k, float("-inf"))
            values, rows = torch.topk(scores, selected_width, dim=-1)
            local_rows = k_positions[rows].to(torch.int32)
            output[start:end, :selected_width] = torch.where(torch.isfinite(values), local_rows, -1)
        return output, logical_indexer_layout, None

    topk_result = indexer_topk(
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
        compact_workspace=compact_workspace,
        use_compact=prebuilt_layout is None,
        precision=precision,
        deterministic=deterministic,
        return_softmax=return_softmax,
    )
    compact_predict = None
    if return_softmax:
        topk, _, compact_predict = topk_result
    else:
        topk, _ = topk_result
    return topk, logical_indexer_layout, compact_predict


def validate_packed_inputs(packed_seq_params, config, cp_group):
    """Validate the static packed contract without modifying any process group.

    CP1 metadata may use None or the explicit singleton group required by the
    main runtime-CP contract. Dynamic CP transitions remain unsupported; reject them
    instead of accidentally
    using a stale build-time group during forward or recompute.
    """
    if config.dsa_kernel_backend != "cudnn":
        raise ValueError("Packed DSv4 attention requires dsa_kernel_backend='cudnn'.")
    if config.deterministic_mode:
        raise NotImplementedError(
            "Packed DSv4 attention does not yet support deterministic_mode; "
            "deterministic THD/CP backward is tracked in #7921."
        )
    if packed_seq_params.qkv_format != "thd":
        raise ValueError("DSv4 packed attention requires qkv_format='thd'.")
    if cp_group is None:
        raise ValueError("DSv4 requires an explicit build-time CP process group, including CP1.")
    runtime_size = packed_seq_params.local_cp_size
    runtime_group = packed_seq_params.cp_group
    if runtime_size is not None:
        if runtime_size == 1 and runtime_group is not None and runtime_group.size() != 1:
            raise ValueError("Runtime CP1 metadata requires None or a singleton cp_group.")
        if runtime_size > 1 and (runtime_group is None or runtime_group.size() != runtime_size):
            raise ValueError("Runtime CP metadata must provide the matching process group.")
        if runtime_size != cp_group.size() or (runtime_size > 1 and runtime_group is not cp_group):
            raise ValueError(
                "DSv4 currently supports static CP; runtime group transitions are unsupported."
            )
    if (
        cp_group.size() > 1
        and config.attention_cp_layout != "contiguous"
        and not (
            config.attention_cp_layout == "zigzag" and getattr(config, "dsv4_cp_zigzag", False)
        )
    ):
        raise ValueError(
            "DSv4 packed CP requires attention_cp_layout='contiguous', or 'zigzag' with "
            "dsv4_cp_zigzag=True."
        )
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

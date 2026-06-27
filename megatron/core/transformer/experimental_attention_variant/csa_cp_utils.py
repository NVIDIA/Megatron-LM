# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MCore-facing utilities for the DSv4 THD context-parallel path.

This module owns CP partition semantics, boundary exchange, compressor-input
layout, and indexer top-k metadata. It calls the retained RoPE/compaction
kernels; ``csa.py`` calls the two final-index kernels directly.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant import csa_cp_layout_kernels
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import indexer_topk

# =============================================================================
# CP Partition Modes And Layout
# =============================================================================

DSV4_CP_PARTITION_CONTIGUOUS = "contiguous"
DSV4_CP_PARTITION_TWO_CHUNK = "two_chunk"


def local_q_cp_chunk_ranges(
    partition_mode: Optional[str], local_rows: int, cp_size: int, cp_rank: int
) -> Tuple[Tuple[int, int], ...]:
    """Return this rank's local rows as global packed-token ranges."""
    mode = DSV4_CP_PARTITION_CONTIGUOUS if partition_mode is None else partition_mode
    if mode not in (DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK):
        raise RuntimeError(
            "Unsupported CSA CP partition mode: " f"{mode!r}. Expected contiguous or two_chunk."
        )
    local_rows, cp_size, cp_rank = int(local_rows), int(cp_size), int(cp_rank)
    if local_rows <= 0:
        raise RuntimeError(f"local_rows must be positive, got {local_rows}.")
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise RuntimeError(f"Invalid CP rank/size: cp_rank={cp_rank}, cp_size={cp_size}.")

    if mode == DSV4_CP_PARTITION_CONTIGUOUS:
        start = cp_rank * local_rows
        return ((start, start + local_rows),)

    if cp_size == 1:
        return ((0, local_rows),)
    if local_rows % 2 != 0:
        raise RuntimeError(
            "DSv4 two-chunk CP partition expects even local_rows: "
            f"local_rows={local_rows}, cp_size={cp_size}."
        )
    chunk_len = local_rows // 2
    total_chunks = 2 * cp_size
    chunk_ids = (cp_rank, total_chunks - 1 - cp_rank)
    return tuple((chunk_id * chunk_len, (chunk_id + 1) * chunk_len) for chunk_id in chunk_ids)


def local_kv_cp_chunk_ranges(
    partition_mode: Optional[str], local_rows: int, boundary_rows: int, cp_size: int, cp_rank: int
) -> Tuple[Tuple[int, int], ...]:
    """Return boundary rows followed by local rows as global packed-token ranges."""
    local_ranges = local_q_cp_chunk_ranges(partition_mode, local_rows, cp_size, cp_rank)
    boundary_rows = int(boundary_rows)
    if boundary_rows <= 0:
        raise RuntimeError(f"boundary_rows must be positive, got {boundary_rows}.")
    if boundary_rows % len(local_ranges) != 0:
        raise RuntimeError(
            "CP boundary rows must be divisible by local chunk count: "
            f"boundary_rows={boundary_rows}, chunks={len(local_ranges)}."
        )
    d_window = boundary_rows // len(local_ranges)
    boundary_ranges = tuple((int(start) - d_window, int(start)) for start, _ in local_ranges)
    return boundary_ranges + local_ranges


def _compressed_group_capacity(rows: int, ratio: int, d_comp: int) -> int:
    """Return compressed group capacity with row alignment for TE FP8 linear inputs."""
    c_cap = max(1, (int(rows) + int(d_comp)) // int(ratio))
    group_alignment = 32 // math.gcd(32, int(ratio))
    return ((c_cap + group_alignment - 1) // group_alignment) * group_alignment


def thd_cp_local_row_indices(
    partition_mode: Optional[str],
    padded_total_rows: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Return global THD row indices selected by this CP rank.

    ``padded_total_rows`` is the final padded packed-token length. The output is
    an int64 tensor of shape ``(padded_total_rows / cp_size,)``. Its row order is
    exactly the order expected by ``local_q_cp_chunk_ranges`` for the selected
    CSA CP partition mode.
    """
    padded_total_rows = int(padded_total_rows)
    cp_size, cp_rank = int(cp_size), int(cp_rank)
    if padded_total_rows <= 0:
        raise RuntimeError(f"padded_total_rows must be positive, got {padded_total_rows}.")
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise RuntimeError(f"Invalid CP rank/size: cp_rank={cp_rank}, cp_size={cp_size}.")
    if padded_total_rows % cp_size != 0:
        raise RuntimeError(
            "DSv4 THD CP partition expects padded_total_rows divisible by cp_size: "
            f"padded_total_rows={padded_total_rows}, cp_size={cp_size}."
        )

    local_rows = padded_total_rows // cp_size
    return torch.cat(
        [
            torch.arange(start, end, dtype=torch.long, device=device)
            for start, end in local_q_cp_chunk_ranges(
                partition_mode, local_rows=local_rows, cp_size=cp_size, cp_rank=cp_rank
            )
        ],
        dim=0,
    )


def _normalize_row_ranges(
    chunk_ranges: Sequence[Tuple[int, int]], op_name: str
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...], int]:
    """Validate row ranges and return normalized ranges, lengths, and total rows."""
    if not chunk_ranges:
        raise RuntimeError(f"{op_name} expects at least one chunk range.")
    normalized = []
    lengths = []
    for start, end in chunk_ranges:
        start = int(start)
        end = int(end)
        length = end - start
        if length <= 0:
            raise RuntimeError(f"{op_name} expects positive chunk lengths, got {length}.")
        normalized.append((start, end))
        lengths.append(length)
    return tuple(normalized), tuple(lengths), sum(lengths)


# =============================================================================
# RoPE Wrappers
# =============================================================================


def apply_thd_cp_local_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    inverse: bool = False,
    clamp_to_valid_token: bool = False,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows."""
    chunk_ranges, lengths, l_local = _normalize_row_ranges(chunk_ranges, "DSv4 THD CP local RoPE")
    if x.shape[0] != l_local:
        raise RuntimeError(
            f"DSv4 THD CP local RoPE expects x rows to be {l_local}, got {x.shape[0]}."
        )
    if len(chunk_ranges) == 1 or (len(chunk_ranges) == 2 and lengths[0] == lengths[1]):
        return csa_cp_layout_kernels.ThdLocalRope.apply(
            x,
            cos,
            sin,
            cu_seqlens_padded,
            int(chunk_ranges[0][0]),
            int(lengths[0]),
            nope_dim,
            pos_dim,
            int(chunk_ranges[1][0]) if len(chunk_ranges) == 2 else 0,
            bool(inverse),
            bool(clamp_to_valid_token),
        )

    parts = []
    row_start = 0
    for chunk_range, length in zip(chunk_ranges, lengths):
        parts.append(
            csa_cp_layout_kernels.ThdLocalRope.apply(
                x.narrow(0, row_start, int(length)),
                cos,
                sin,
                cu_seqlens_padded,
                int(chunk_range[0]),
                int(length),
                nope_dim,
                pos_dim,
                0,
                bool(inverse),
                bool(clamp_to_valid_token),
            )
        )
        row_start += int(length)
    return torch.cat(parts, dim=0)


# =============================================================================
# Boundary Hidden Exchange
# =============================================================================


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange fixed left-boundary windows and scatter gradients back to senders."""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        d_window: int,
        cp_group: torch.distributed.ProcessGroup,
        two_chunk: bool,
    ):
        """Receive fixed left-boundary hidden rows needed by this CP rank."""
        cp_size = 1 if cp_group is None else cp_group.size()
        cp_rank = 0 if cp_group is None else cp_group.rank()
        if cp_size <= 1:
            total_chunks, local_chunk_ids = 1, (0,)
            owner_ranks = (0,)
        elif two_chunk:
            total_chunks = 2 * cp_size
            local_chunk_ids = (cp_rank, total_chunks - 1 - cp_rank)
            owner_ranks = tuple(
                min(chunk_id, total_chunks - 1 - chunk_id) for chunk_id in range(total_chunks)
            )
        else:
            total_chunks, local_chunk_ids = cp_size, (cp_rank,)
            owner_ranks = tuple(range(cp_size))
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        ctx.local_chunk_ids = local_chunk_ids
        ctx.owner_ranks = owner_ranks

        local_chunks = len(local_chunk_ids)
        if tensor.shape[0] % local_chunks != 0:
            raise RuntimeError(
                "DSv4 CP boundary exchange expects equal local chunks: "
                f"local={tensor.shape[0]}, chunks={local_chunks}."
            )
        chunk_len = tensor.shape[0] // local_chunks
        ctx.chunk_len = chunk_len
        if chunk_len < d_window:
            raise RuntimeError(
                "DSv4 CP boundary exchange requires chunk_len >= D_window: "
                f"chunk_len={chunk_len}, D_window={d_window}."
            )
        if cp_size <= 1:
            return tensor.new_zeros((local_chunks * d_window,) + tuple(tensor.shape[1:]))

        recv = tensor.new_zeros((local_chunks, d_window) + tuple(tensor.shape[1:]))
        ops = []
        send_tensors = []
        to_peer = dist.get_global_rank if hasattr(dist, "get_global_rank") else lambda _, rank: rank

        for local_idx, chunk_id in enumerate(local_chunk_ids):
            if chunk_id > 0:
                prev_chunk_id = chunk_id - 1
                source_rank = owner_ranks[prev_chunk_id]
                if source_rank == cp_rank:
                    source_idx = local_chunk_ids.index(prev_chunk_id)
                    source_start = source_idx * chunk_len
                    recv[local_idx].copy_(
                        tensor[source_start + chunk_len - d_window : source_start + chunk_len]
                    )
                else:
                    ops.append(
                        dist.P2POp(
                            dist.irecv, recv[local_idx], to_peer(cp_group, source_rank), cp_group
                        )
                    )

            next_chunk_id = chunk_id + 1
            if next_chunk_id < total_chunks:
                target_rank = owner_ranks[next_chunk_id]
                if target_rank != cp_rank:
                    start = local_idx * chunk_len
                    send_tensors.append(
                        tensor[start + chunk_len - d_window : start + chunk_len].contiguous()
                    )
                    ops.append(
                        dist.P2POp(
                            dist.isend, send_tensors[-1], to_peer(cp_group, target_rank), cp_group
                        )
                    )

        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv.reshape((local_chunks * d_window,) + tuple(tensor.shape[1:]))

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Send boundary gradients back to ranks that own those hidden rows."""
        cp_group = ctx.cp_group
        cp_size = 1 if cp_group is None else cp_group.size()
        cp_rank = 0 if cp_group is None else cp_group.rank()
        d_window = ctx.d_window
        chunk_len = ctx.chunk_len
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        if cp_size <= 1:
            return grad_input, None, None, None

        local_chunk_ids = ctx.local_chunk_ids
        owner_ranks = ctx.owner_ranks
        total_chunks = len(owner_ranks)
        local_chunks = len(local_chunk_ids)
        grad_chunks = grad_boundary.reshape(
            (local_chunks, d_window) + tuple(grad_boundary.shape[1:])
        )
        ops = []
        send_tensors = []
        recv_specs = []
        to_peer = dist.get_global_rank if hasattr(dist, "get_global_rank") else lambda _, rank: rank

        for local_idx, chunk_id in enumerate(local_chunk_ids):
            if chunk_id > 0:
                prev_chunk_id = chunk_id - 1
                source_rank = owner_ranks[prev_chunk_id]
                if source_rank == cp_rank:
                    source_idx = local_chunk_ids.index(prev_chunk_id)
                    source_start = source_idx * chunk_len
                    grad_input[
                        source_start + chunk_len - d_window : source_start + chunk_len
                    ] += grad_chunks[local_idx]
                else:
                    send_tensors.append(grad_chunks[local_idx].contiguous())
                    ops.append(
                        dist.P2POp(
                            dist.isend, send_tensors[-1], to_peer(cp_group, source_rank), cp_group
                        )
                    )

            next_chunk_id = chunk_id + 1
            if next_chunk_id < total_chunks:
                target_rank = owner_ranks[next_chunk_id]
                if target_rank != cp_rank:
                    recv = grad_chunks.new_zeros((d_window,) + tuple(grad_chunks.shape[2:]))
                    recv_specs.append((local_idx, recv))
                    ops.append(
                        dist.P2POp(dist.irecv, recv, to_peer(cp_group, target_rank), cp_group)
                    )

        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        for local_idx, recv in recv_specs:
            start = local_idx * chunk_len
            grad_input[start + chunk_len - d_window : start + chunk_len] += recv
        return grad_input, None, None, None


def exchange_cp_boundary_hidden(
    hidden_states: torch.Tensor,
    csa_compress_ratios: Sequence[int],
    csa_window_size: int,
    partition_mode: Optional[str],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Exchange hidden-state boundary rows for the selected CSA CP partition mode."""
    d_comp = max(
        (8 if ratio == 4 else ratio for ratio in csa_compress_ratios if ratio and ratio > 1),
        default=0,
    )
    d_window = max(int(csa_window_size), d_comp)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    cp_size = 1 if cp_group is None else cp_group.size()
    cp_rank = 0 if cp_group is None else cp_group.rank()
    local_chunks = len(
        local_q_cp_chunk_ranges(partition_mode, hidden_flat.shape[0], cp_size, cp_rank)
    )
    boundary_hidden = _LeftBoundaryExchange.apply(
        hidden_flat, d_window, cp_group, local_chunks == 2
    )
    return boundary_hidden.reshape((local_chunks * d_window,) + tuple(hidden_states.shape[1:]))


# =============================================================================
# Compressed Metadata And Compressor Inputs
# =============================================================================


def build_cp_compressor_prep_compact_fused(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    cp_size: int,
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity compressor input for the selected CP partition mode.

    Returns:
        ``hidden_compact``: rank-local compressor input, shape
            ``(compact_group_capacity * ratio, ...)``.
        ``comp_ids_local``: original per-sequence compressed group id for each
            compact group, shape ``(compact_group_capacity,)``. For example,
            with ``ratio=4``, ``comp_id=3`` maps to RoPE position ``12``.
        ``rank_row_for_seq_row``: map from global sequence-major compressed rows
            to their canonical rank-major all-gather rows.
    """
    d_window = int(d_window)

    two_chunk = len(chunk_ranges) == 2
    if two_chunk:
        chunk_ranges, chunk_lengths, l_local = _normalize_row_ranges(
            chunk_ranges, "DSv4 two-chunk compressor-prep"
        )
        if chunk_lengths[0] != chunk_lengths[1]:
            raise RuntimeError(
                "DSv4 two-chunk compressor-prep expects exactly two equal-length chunks."
            )
        chunk_starts = (chunk_ranges[0][0], chunk_ranges[1][0])
        chunk_len = chunk_lengths[0]
    else:
        start, end = chunk_ranges[0]
        chunk_ranges = ((int(start), int(end)),)
        l_local = chunk_ranges[0][1] - chunk_ranges[0][0]
    if hidden_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP compressor-prep expects hidden rows to be "
            f"{l_local}, got {hidden_local.shape[0]}."
        )
    expected_boundary_rows = len(chunk_ranges) * d_window
    if boundary_hidden.shape[0] != expected_boundary_rows:
        raise RuntimeError(
            "DSv4 CP compressor-prep expects one boundary window per chunk: "
            f"boundary={boundary_hidden.shape[0]}, expected={expected_boundary_rows}."
        )

    if two_chunk:
        chunk_specs = [
            (0, chunk_len, 0, chunk_starts[0]),
            (chunk_len, chunk_len, d_window, chunk_starts[1]),
        ]
    else:
        chunk_specs = [(0, l_local, 0, chunk_ranges[0][0])]

    parts = []
    for row_start, rows, boundary_start, global_start in chunk_specs:
        c_cap = (
            (int(rows) + int(d_comp)) // int(ratio)
            if two_chunk
            else _compressed_group_capacity(rows, ratio, d_comp)
        )
        parts.append(
            csa_cp_layout_kernels.CompressorInputCompact.apply(
                hidden_local.narrow(0, row_start, rows),
                boundary_hidden.narrow(0, boundary_start, d_window),
                cu_seqlens,
                global_start,
                rows,
                ratio,
                d_comp,
                d_window,
                c_cap,
            )
        )
    if two_chunk:
        hidden_parts, comp_parts = zip(*parts)
        target_c_cap = _compressed_group_capacity(l_local, ratio, len(chunk_ranges) * d_comp)
        current_c_cap = sum(part.shape[0] for part in comp_parts)
        if current_c_cap > target_c_cap:
            raise RuntimeError(
                "DSv4 two-chunk compressor-prep produced more groups than its aligned capacity: "
                f"current={current_c_cap}, capacity={target_c_cap}."
            )
        if current_c_cap < target_c_cap:
            pad_groups = target_c_cap - current_c_cap
            hidden_parts = hidden_parts + (
                hidden_local.new_zeros((pad_groups * int(ratio),) + tuple(hidden_local.shape[1:])),
            )
            comp_parts = comp_parts + (
                torch.full((pad_groups,), -1, dtype=torch.int32, device=hidden_local.device),
            )
        hidden_compact = torch.cat(hidden_parts, dim=0)
        comp_ids_local = torch.cat(comp_parts, dim=0)
    else:
        hidden_compact, comp_ids_local = parts[0]

    # A compressed group belongs canonically to the chunk containing its last
    # token. From that chunk's first visible compressed row, its fixed-capacity
    # rank-major slot follows directly; no (seq, comp, valid) tensors or repack
    # kernel are needed.
    cp_size = int(cp_size)
    ratio = int(ratio)
    total_chunks = cp_size * (2 if two_chunk else 1)
    seq_major_rows = (l_local * cp_size) // ratio
    logical_rows = torch.arange(seq_major_rows, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    n_seq = cu_seqlens.shape[0] - 1
    seq_ids = torch.bucketize(
        logical_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    comp_ids = logical_rows - cu_seqlens_compressed[seq_ids]
    group_last_rows = cu_seqlens[seq_ids] + (comp_ids + 1) * ratio - 1
    canonical_chunks = torch.div(
        group_last_rows, chunk_len if two_chunk else l_local, rounding_mode="floor"
    ).clamp_(0, total_chunks - 1)

    chunk_length = chunk_len if two_chunk else l_local
    chunk_starts = torch.arange(
        total_chunks, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    ) * int(chunk_length)
    first_seq_ids = torch.bucketize(
        chunk_starts, cu_seqlens[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    first_comp_ids = torch.div(
        (chunk_starts - int(d_comp) - cu_seqlens[first_seq_ids]).clamp_min_(0) + ratio - 1,
        ratio,
        rounding_mode="floor",
    )
    first_logical_rows = cu_seqlens_compressed[first_seq_ids] + first_comp_ids
    rank_slots = logical_rows - first_logical_rows[canonical_chunks]
    if two_chunk:
        late_chunks = canonical_chunks >= cp_size
        owner_ranks = torch.where(
            late_chunks, total_chunks - 1 - canonical_chunks, canonical_chunks
        )
        rank_slots = rank_slots + late_chunks * int(c_cap)
    else:
        owner_ranks = canonical_chunks
    rank_rows = owner_ranks * comp_ids_local.shape[0] + rank_slots
    rank_row_for_seq_row = torch.where(logical_rows < cu_seqlens_compressed[-1], rank_rows, -1).to(
        torch.int32
    )
    return hidden_compact, comp_ids_local, rank_row_for_seq_row


def compute_cp_indexer_topk_logical_fused(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> Optional[torch.Tensor]:
    """Run CP-aware indexer top-k for the selected CP partition mode.

    This is a workaround for the DSA fused indexer forward mask contract. A CP
    chunk may need a mask whose first local query already sees a compressed K
    prefix, for example:

    ```
    1 1 1 0 0 0 0 0
    1 1 1 1 0 0 0 0
    1 1 1 1 1 0 0 0
    ```

    The fused kernel assumes the last Q row can see all K rows, then derives
    earlier rows' causal masks from that endpoint. A CP chunk's last Q row may
    only see a prefix of the global K rows, so this helper copies that visible
    prefix before calling the kernel.
    """
    topk_width = int(topk_width)
    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return None

    chunk_ranges, lengths, l_local = _normalize_row_ranges(chunk_ranges, "DSv4 CP indexer top-k")
    if q_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects q_indexer_local rows to be "
            f"{l_local}, got {q_indexer_local.shape[0]}."
        )
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    outputs = []
    row_start = 0
    for (global_start, _), length in zip(chunk_ranges, lengths):
        global_start, length = int(global_start), int(length)
        global_end = global_start + length
        q_for_topk = q_indexer_local.narrow(0, row_start, int(length))
        weights_for_topk = weights_indexer_local.narrow(0, row_start, int(length))
        local_starts = cu_seqlens_q[:-1].clamp_min(global_start)
        local_ends = cu_seqlens_q[1:].clamp_max(global_end)
        q_lens = (local_ends - local_starts).clamp_min(0)
        k_lens = torch.minimum(
            torch.div(
                (local_ends - cu_seqlens_q[:-1]).clamp_min(0), int(ratio), rounding_mode="floor"
            ),
            cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1],
        )
        k_lens = torch.where(q_lens > 0, k_lens, 0)
        q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
        k_prefix = torch.cumsum(k_lens, dim=0, dtype=torch.int32)
        zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
        padding_q = (global_end - cu_seqlens_q[-1].clamp_min(global_start)).clamp_min(0)
        cu_q_topk = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
        cu_k_topk = torch.cat((zero, k_prefix, k_prefix[-1:]))

        k_rows = torch.arange(
            k_indexer_seq_major.shape[0],
            dtype=cu_seqlens_compressed.dtype,
            device=cu_seqlens_compressed.device,
        )
        k_seq_ids = torch.bucketize(k_rows, cu_k_topk[1:-1], out_int32=True, right=True).clamp_max(
            cu_seqlens_q.shape[0] - 2
        )
        valid_k = k_rows < k_prefix[-1]
        source_rows = cu_seqlens_compressed[k_seq_ids] + k_rows - cu_k_topk[k_seq_ids]
        k_for_topk = torch.index_select(
            k_indexer_seq_major, 0, torch.where(valid_k, source_rows, 0)
        )
        k_for_topk = k_for_topk * valid_k.view((-1,) + (1,) * (k_for_topk.ndim - 1))
        seq_lens_topk = torch.repeat_interleave(
            torch.cat((k_lens, zero)), torch.cat((q_lens, padding_q.view(1))), output_size=length
        )
        topk_chunk, _ = indexer_topk(
            q_for_topk,
            k_for_topk,
            weights_for_topk,
            topk=topk_width,
            ratio=ratio,
            indexer_softmax_scale=indexer_softmax_scale,
            cu_seqlens_q=cu_q_topk,
            cu_seqlens_kv=cu_k_topk,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_kv=int(max_seqlen_kv),
            visible_k_lengths=seq_lens_topk,
        )
        outputs.append(topk_chunk)
        row_start += length
    return torch.cat(outputs, dim=0)

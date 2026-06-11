# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MCore-facing utilities for the DSv4 THD context-parallel path.

This module is the boundary between attention code and the small CP layout
kernels in ``csa_cp_kernels.py``.  It owns CP partition metadata, boundary
communication, fixed-shape collectives, and the typed wrappers used by CSA/DSv4
attention modules.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant import csa_cp_kernels


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
            "Unsupported CSA CP partition mode: "
            f"{mode!r}. Expected contiguous or two_chunk."
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


def _two_chunk_layout(
    chunk_ranges: Sequence[Tuple[int, int]], op_name: str
) -> Tuple[Tuple[int, int], int, int]:
    ranges, lengths, l_local = _normalize_row_ranges(chunk_ranges, op_name)
    if len(ranges) != 2 or lengths[0] != lengths[1]:
        raise RuntimeError(f"{op_name} expects exactly two equal-length chunks.")
    return (int(ranges[0][0]), int(ranges[1][0])), int(lengths[0]), int(l_local)


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
        return csa_cp_kernels.ThdLocalRope.apply(
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
            csa_cp_kernels.ThdLocalRope.apply(
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


def apply_thd_cp_compressed_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    comp_ids_local: torch.Tensor,
    ratio: int,
    nope_dim: int,
    pos_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply fused RoPE to compressed rows using their original compression ids."""
    return csa_cp_kernels.ThdCompressedRope.apply(
        x, cos, sin, comp_ids_local, ratio, nope_dim, pos_dim, bool(inverse)
    )


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
                            dist.irecv,
                            recv[local_idx],
                            to_peer(cp_group, source_rank),
                            cp_group,
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
                            dist.isend,
                            send_tensors[-1],
                            to_peer(cp_group, target_rank),
                            cp_group,
                        )
                    )

        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv.reshape((local_chunks * d_window,) + tuple(tensor.shape[1:]))

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
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
                            dist.isend,
                            send_tensors[-1],
                            to_peer(cp_group, source_rank),
                            cp_group,
                        )
                    )

            next_chunk_id = chunk_id + 1
            if next_chunk_id < total_chunks:
                target_rank = owner_ranks[next_chunk_id]
                if target_rank != cp_rank:
                    recv = grad_chunks.new_zeros((d_window,) + tuple(grad_chunks.shape[2:]))
                    recv_specs.append((local_idx, recv))
                    ops.append(
                        dist.P2POp(
                            dist.irecv,
                            recv,
                            to_peer(cp_group, target_rank),
                            cp_group,
                        )
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
    ratios = (ratio for ratio in csa_compress_ratios if ratio and ratio > 1)
    d_comp = max((8 if ratio == 4 else ratio for ratio in ratios), default=0)
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


def build_global_compressed_cu_seqlens(
    cu_seqlens_padded: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Build fixed-shape compressed sequence prefix sums.

    Inputs:
        cu_seqlens_padded: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        ratio: positive compression ratio.

    Output:
        int32 CUDA tensor, shape ``(n_seq + 1,)`` where each prefix increments by
        ``floor(seq_len / ratio)``. This is intentionally a PyTorch utility, not
        a CuTe kernel, because it is fixed-shape and does not need a host sync.
    """
    if ratio <= 0:
        raise RuntimeError(f"DSv4 CP compressed cu_seqlens expects positive ratio, got {ratio}.")
    out = torch.empty_like(cu_seqlens_padded)
    out[:1].zero_()
    compressed_lens = torch.div(
        cu_seqlens_padded[1:] - cu_seqlens_padded[:-1], int(ratio), rounding_mode="floor"
    )
    out[1:].copy_(torch.cumsum(compressed_lens, dim=0, dtype=torch.int32))
    return out


def build_cp_rank_major_compressed_metadata_fused(
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    cp_size: int,
    ratio: int,
    d_comp: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rank-major compressed metadata for the selected CP partition mode."""
    if len(chunk_ranges) == 2:
        _, chunk_len, _ = _two_chunk_layout(chunk_ranges, "DSv4 two-chunk compressed metadata")
        c_cap_per_chunk = (chunk_len + int(d_comp)) // int(ratio)
        return csa_cp_kernels.build_compressed_row_metadata(
            cu_seqlens,
            int(cp_size),
            chunk_len,
            ratio,
            d_comp,
            c_cap_per_chunk,
            c_cap_per_rank=2 * int(c_cap_per_chunk),
            use_two_chunk=True,
        )
    _, _, l_local = _normalize_row_ranges(chunk_ranges, "DSv4 CP compressed metadata")
    c_cap = (l_local + int(d_comp)) // int(ratio)
    return csa_cp_kernels.build_compressed_row_metadata(
        cu_seqlens, int(cp_size), l_local, ratio, d_comp, c_cap
    )


def build_cp_compressor_prep_compact_fused(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity compressor input for the selected CP partition mode."""
    d_window = int(d_window)

    two_chunk = len(chunk_ranges) == 2
    if two_chunk:
        chunk_starts, chunk_len, l_local = _two_chunk_layout(
            chunk_ranges, "DSv4 two-chunk compressor-prep"
        )
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
        c_cap = (int(rows) + int(d_comp)) // int(ratio)
        parts.append(
            csa_cp_kernels.CompressorInputCompact.apply(
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
    if not two_chunk:
        return parts[0]
    hidden_parts, cu_parts, seq_parts, comp_parts, valid_parts = zip(*parts)
    cu_deltas = sum(cu[1:] - cu[:-1] for cu in cu_parts)
    cu_compact = torch.cat(
        (
            torch.zeros((1,), dtype=cu_seqlens.dtype, device=cu_seqlens.device),
            torch.cumsum(cu_deltas, dim=0),
        )
    )
    return (
        torch.cat(hidden_parts, dim=0),
        cu_compact,
        torch.cat(seq_parts, dim=0),
        torch.cat(comp_parts, dim=0),
        torch.cat(valid_parts, dim=0),
    )


# =============================================================================
# Fixed-Shape CP Collectives
# =============================================================================


class _AllGatherFixedCPTensor(torch.autograd.Function):
    """All-gather a fixed local tensor and reduce-scatter gradients in backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cp_group) -> torch.Tensor:
        cp_size = 1 if cp_group is None else cp_group.size()
        ctx.cp_group = cp_group
        ctx.input_shape = tuple(tensor.shape)
        ctx.cp_size = cp_size

        if cp_size == 1:
            return tensor

        local = tensor.contiguous()
        output_shape = (cp_size * local.shape[0],) + tuple(local.shape[1:])
        output = local.new_empty(output_shape)
        if hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(output, local, group=cp_group)
        elif hasattr(dist, "_all_gather_base"):
            dist._all_gather_base(output, local, group=cp_group)
        else:
            raise RuntimeError("DSv4 CP fixed all-gather requires all_gather_into_tensor support")
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cp_size = ctx.cp_size
        if cp_size == 1:
            return grad_output, None

        expected_shape = (cp_size * ctx.input_shape[0],) + tuple(ctx.input_shape[1:])
        if tuple(grad_output.shape) != expected_shape:
            raise RuntimeError(
                "DSv4 CP fixed all-gather backward received an unexpected gradient shape: "
                f"grad={tuple(grad_output.shape)}, expected={expected_shape}"
            )

        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_empty(ctx.input_shape)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(
                grad_input, grad_output, op=dist.ReduceOp.SUM, group=ctx.cp_group
            )
        elif hasattr(dist, "_reduce_scatter_base"):
            dist._reduce_scatter_base(
                grad_input, grad_output, op=dist.ReduceOp.SUM, group=ctx.cp_group
            )
        else:
            raise RuntimeError(
                "DSv4 CP fixed all-gather backward requires reduce_scatter_tensor support"
            )
        return grad_input, None


def all_gather_fixed_cp_tensor(
    tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """All-gather fixed-capacity CP tensors along dim 0 with autograd support."""
    return _AllGatherFixedCPTensor.apply(tensor, cp_group)


# =============================================================================
# KV Packing And Compressed KV Repacking
# =============================================================================


def pack_cp_kv_full_fused(
    kv_local: torch.Tensor,
    boundary_kv: torch.Tensor,
    compressed_kv_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    d_window: int,
    ratio: int,
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[int]]:
    """Pack full KV for the selected CP partition mode."""
    d_window = int(d_window)
    shared_compressed_base = None
    if len(chunk_ranges) == 2:
        chunk_starts, chunk_len, l_local = _two_chunk_layout(chunk_ranges, "DSv4 two-chunk KV pack")
        window_capacity = max(1, int(chunk_len) + d_window * (cu_seqlens.shape[0] - 1))
        shared_compressed_base = 2 * window_capacity
        total_capacity = max(1, shared_compressed_base + int(compressed_kv_rank_major.shape[0]))
        dummy_ids = torch.empty((1,), dtype=torch.int32, device=kv_local.device)
        dummy_valid = torch.empty((1,), dtype=torch.bool, device=kv_local.device)
        seq_ids_rank_major = comp_ids_rank_major = rank_major_by_seq_major = dummy_ids
        valid_rank_major = dummy_valid
        cu_seqlens_compressed = cu_seqlens
        global_start = local_rows = kernel_ratio = 0
        chunk0_start, chunk1_start, chunk_count = int(chunk_starts[0]), int(chunk_starts[1]), 2
    else:
        chunk_ranges, _, l_local = _normalize_row_ranges(chunk_ranges, "DSv4 CP KV pack")
        total_capacity = max(
            1,
            int(l_local)
            + d_window * (cu_seqlens.shape[0] - 1)
            + int(compressed_kv_rank_major.shape[0]),
        )
        if rank_major_by_seq_major is None:
            rank_major_by_seq_major = torch.empty((1,), dtype=torch.int32, device=kv_local.device)
        if cu_seqlens_compressed is None:
            cu_seqlens_compressed = cu_seqlens
        global_start = int(chunk_ranges[0][0])
        local_rows = int(l_local)
        kernel_ratio = int(ratio)
        chunk0_start = chunk1_start = chunk_count = chunk_len = window_capacity = 0
    if kv_local.shape[0] != l_local:
        raise RuntimeError(
            f"DSv4 CP KV pack expects local KV rows to be {l_local}, got {kv_local.shape[0]}."
        )
    expected_boundary_rows = len(chunk_ranges) * d_window
    if boundary_kv.shape[0] != expected_boundary_rows:
        raise RuntimeError(
            "DSv4 CP KV pack expects one boundary window per chunk: "
            f"boundary={boundary_kv.shape[0]}, expected={expected_boundary_rows}."
        )

    kv_full = csa_cp_kernels.ThdFullKvPack.apply(
        kv_local,
        boundary_kv,
        compressed_kv_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        rank_major_by_seq_major,
        cu_seqlens_compressed,
        global_start,
        local_rows,
        d_window,
        kernel_ratio,
        int(total_capacity),
        chunk0_start,
        chunk1_start,
        chunk_count,
        int(chunk_len),
        window_capacity,
    )
    return kv_full, shared_compressed_base


def repack_rank_major_compressed_to_seq_major_fused(
    rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    seq_major_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repack gathered compressed KV from rank-major rows to sequence-major rows."""
    return csa_cp_kernels.repack_compressed_kv_to_seq_major(
        rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens_compressed,
        seq_major_rows,
    )


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
    """Run CP-aware indexer top-k for the selected CP partition mode."""
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

    from megatron.core.transformer.experimental_attention_variant.dsa_kernels import indexer_topk

    outputs = []
    row_start = 0
    for (global_start, _), length in zip(chunk_ranges, lengths):
        q_for_topk = q_indexer_local.narrow(0, row_start, int(length))
        weights_for_topk = weights_indexer_local.narrow(0, row_start, int(length))
        k_for_topk, cu_q_topk, cu_k_topk, seq_lens_topk = (
            csa_cp_kernels.build_indexer_topk_metadata(
                k_indexer_seq_major,
                cu_seqlens_q,
                cu_seqlens_compressed,
                int(global_start),
                int(length),
                ratio,
            )
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
        row_start += int(length)
    return torch.cat(outputs, dim=0)


# =============================================================================
# Final Attention And Indexer-Loss Indices
# =============================================================================


def build_cp_attention_indices_fused(
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: Optional[torch.Tensor] = None,
    max_n_compressed: int = 0,
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    shared_compressed_base: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build final sparse-attention indices for the selected CP partition mode."""
    compressed_width = max_n_compressed if ratio > 1 else 0
    if indexer_topk_compressed_logical_ids is not None:
        compressed_width = indexer_topk_compressed_logical_ids.shape[-1]

    if len(chunk_ranges) == 2:
        chunk_starts, chunk_len, l_local = _two_chunk_layout(
            chunk_ranges, "DSv4 two-chunk final idx"
        )
        if shared_compressed_base is None:
            raise RuntimeError("DSv4 two-chunk final idx requires a shared compressed base.")
        if compressed_width > 0 and (
            rank_major_by_seq_major is None or cu_seqlens_compressed is None
        ):
            raise RuntimeError("DSv4 two-chunk final idx with compression requires metadata.")
        if compressed_width == 0:
            cu_seqlens_compressed = cu_seqlens
            rank_major_by_seq_major = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)
        global_start, window_capacity_per_chunk = 0, int(shared_compressed_base) // 2
    else:
        chunk_ranges, _, l_local = _normalize_row_ranges(chunk_ranges, "DSv4 CP final idx")
        global_start = chunk_ranges[0][0]
        chunk_starts, chunk_len, window_capacity_per_chunk, shared_compressed_base = None, 0, 0, 0
    if (
        indexer_topk_compressed_logical_ids is not None
        and indexer_topk_compressed_logical_ids.shape[0] != l_local
    ):
        raise RuntimeError(
            "DSv4 CP final idx expects indexer rows to be "
            f"{l_local}, got {indexer_topk_compressed_logical_ids.shape[0]}."
        )
    return csa_cp_kernels.build_attention_indices(
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        indexer_topk_compressed_logical_ids,
        cu_seqlens_compressed=cu_seqlens_compressed,
        rank_major_by_seq_major=rank_major_by_seq_major,
        chunk_starts=chunk_starts,
        chunk_len=chunk_len,
        window_capacity_per_chunk=window_capacity_per_chunk,
        shared_compressed_base=shared_compressed_base,
    )


def build_cp_indexer_loss_indices_fused(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: torch.Tensor,
    rank_major_by_seq_major: torch.Tensor,
    shared_compressed_base: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build indexer-loss indices for the selected CP partition mode."""
    if len(chunk_ranges) == 2:
        chunk_starts, chunk_len, l_local = _two_chunk_layout(
            chunk_ranges, "DSv4 two-chunk indexer-loss idx"
        )
        if shared_compressed_base is None:
            raise RuntimeError("DSv4 two-chunk indexer-loss idx requires a shared compressed base.")
        global_start, window_capacity_per_chunk = 0, int(shared_compressed_base) // 2
    else:
        chunk_ranges, _, l_local = _normalize_row_ranges(chunk_ranges, "DSv4 CP indexer-loss idx")
        global_start = chunk_ranges[0][0]
        chunk_starts, chunk_len, window_capacity_per_chunk, shared_compressed_base = None, 0, 0, 0
    if indexer_topk_compressed_logical_ids.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer-loss idx expects indexer rows to be "
            f"{l_local}, got {indexer_topk_compressed_logical_ids.shape[0]}."
        )
    return csa_cp_kernels.build_indexer_loss_indices(
        cu_seqlens,
        cu_seqlens_compressed,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        indexer_topk_compressed_logical_ids,
        rank_major_by_seq_major,
        chunk_starts=chunk_starts,
        chunk_len=chunk_len,
        window_capacity_per_chunk=window_capacity_per_chunk,
        shared_compressed_base=shared_compressed_base,
    )

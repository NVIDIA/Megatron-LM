# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MCore-facing utilities for the DSv4 THD context-parallel path.

This module is the boundary between attention code and the small CP layout
kernels in ``csa_cp_kernels.py``.  It owns CP partition metadata, boundary
communication, fixed-shape collectives, and the typed wrappers used by CSA/DSv4
attention modules.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant import csa_cp_kernels


# =============================================================================
# CP Partition Modes And Layout
# =============================================================================

DSV4_CP_PARTITION_CONTIGUOUS = "contiguous"
DSV4_CP_PARTITION_TWO_CHUNK = "two_chunk"
DSV4_CP_PARTITION_MODES = (DSV4_CP_PARTITION_CONTIGUOUS, DSV4_CP_PARTITION_TWO_CHUNK)


def normalize_dsv4_cp_partition_mode(mode: Optional[str]) -> str:
    """Return a supported DSv4 CP partition mode, defaulting to contiguous chunks."""
    if mode is None:
        return DSV4_CP_PARTITION_CONTIGUOUS
    if mode not in DSV4_CP_PARTITION_MODES:
        raise RuntimeError(
            "Unsupported DSv4 CP partition mode: "
            f"{mode!r}. Expected one of {DSV4_CP_PARTITION_MODES}."
        )
    return mode


def two_chunk_cp_partition(
    padded_total_tokens: int, cp_size: int, cp_rank: int
) -> Tuple[Tuple[int, int], ...]:
    """Return the two packed-token chunks owned by ``cp_rank``."""
    padded_total_tokens = int(padded_total_tokens)
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise RuntimeError(f"Invalid CP rank/size: cp_rank={cp_rank}, cp_size={cp_size}.")
    if cp_size == 1:
        return ((0, padded_total_tokens),)
    total_chunks = 2 * cp_size
    if padded_total_tokens % total_chunks != 0:
        raise RuntimeError(
            "DSv4 two-chunk CP partition expects padded_total_tokens % (2 * cp_size) == 0: "
            f"padded_total_tokens={padded_total_tokens}, cp_size={cp_size}."
        )
    chunk_len = padded_total_tokens // total_chunks
    chunk_ids = (cp_rank, total_chunks - 1 - cp_rank)
    return tuple((chunk_id * chunk_len, (chunk_id + 1) * chunk_len) for chunk_id in chunk_ids)


def _two_chunk_layout(
    chunk_ranges: Sequence[Tuple[int, int]], op_name: str
) -> Tuple[Tuple[int, int], int, int]:
    """Validate two equal chunks and return ``(starts, chunk_len, local_len)``."""
    if len(chunk_ranges) != 2:
        raise RuntimeError(
            f"{op_name} expects exactly two chunks, got {len(chunk_ranges)}."
        )
    chunk0_start, chunk0_end = (int(chunk_ranges[0][0]), int(chunk_ranges[0][1]))
    chunk1_start, chunk1_end = (int(chunk_ranges[1][0]), int(chunk_ranges[1][1]))
    chunk0_len = chunk0_end - chunk0_start
    chunk1_len = chunk1_end - chunk1_start
    if chunk0_len <= 0 or chunk1_len <= 0 or chunk0_len != chunk1_len:
        raise RuntimeError(
            f"{op_name} expects two equal positive chunks, got lengths "
            f"{(chunk0_len, chunk1_len)}."
        )
    return (chunk0_start, chunk1_start), chunk0_len, 2 * chunk0_len


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
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
    inverse: bool = False,
    remove_interleaving: bool = True,
    row_offset: int = 0,
    chunk_len: Optional[int] = None,
    clamp_to_valid_token: bool = False,
    global_row_base: Optional[int] = None,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows."""
    if rotary_interleaved:
        raise RuntimeError("DSv4 THD CP local RoPE does not support rotary_interleaved=True.")
    if not remove_interleaving:
        raise RuntimeError("DSv4 THD CP local RoPE requires remove_interleaving=True.")
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise RuntimeError(
            "DSv4 THD CP local RoPE got invalid CP rank/size: "
            f"cp_rank={cp_rank}, cp_size={cp_size}."
        )
    if chunk_len is None:
        chunk_len = x.shape[0]
    if global_row_base is None:
        global_row_base = int(cp_rank) * int(chunk_len)
    global_row_base = int(global_row_base) + int(row_offset)
    return csa_cp_kernels.ThdLocalRope.apply(
        x,
        cos,
        sin,
        cu_seqlens_padded,
        global_row_base,
        x.shape[0],
        nope_dim,
        pos_dim,
        0,
        bool(inverse),
        bool(clamp_to_valid_token),
    )


def apply_thd_cp_two_chunk_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    rotary_interleaved: bool = False,
    inverse: bool = False,
    remove_interleaving: bool = True,
    row_offset: int = 0,
    clamp_to_valid_token: bool = False,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to a two-chunk local THD CP tensor."""
    if rotary_interleaved:
        raise RuntimeError("DSv4 THD CP two-chunk RoPE does not support rotary_interleaved=True.")
    if not remove_interleaving:
        raise RuntimeError("DSv4 THD CP two-chunk RoPE requires remove_interleaving=True.")
    chunk_starts, chunk_len, l_local = _two_chunk_layout(
        chunk_ranges, "DSv4 THD CP two-chunk RoPE"
    )
    if x.shape[0] != l_local:
        raise RuntimeError(
            f"DSv4 THD CP two-chunk RoPE expects x rows to be {l_local}, got {x.shape[0]}."
        )
    return csa_cp_kernels.ThdLocalRope.apply(
        x,
        cos,
        sin,
        cu_seqlens_padded,
        chunk_starts[0] + int(row_offset),
        chunk_len,
        nope_dim,
        pos_dim,
        chunk_starts[1] + int(row_offset),
        bool(inverse),
        bool(clamp_to_valid_token),
    )


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


def _group_peer(cp_group: torch.distributed.ProcessGroup, group_rank: int) -> int:
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(cp_group, group_rank)
    return group_rank


def _cp_chunk_owner_rank(chunk_id: int, cp_size: int, two_chunk: bool) -> int:
    if not two_chunk:
        return chunk_id
    if chunk_id < cp_size:
        return chunk_id
    return 2 * cp_size - 1 - chunk_id


def _local_cp_chunks(cp_size: int, cp_rank: int, two_chunk: bool) -> Tuple[int, Tuple[int, ...]]:
    if cp_size <= 1:
        return 1, (0,)
    if not two_chunk:
        return cp_size, (cp_rank,)
    total_chunks = 2 * cp_size
    return total_chunks, (cp_rank, total_chunks - 1 - cp_rank)


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
        total_chunks, local_chunk_ids = _local_cp_chunks(cp_size, cp_rank, bool(two_chunk))
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        ctx.local_chunk_ids = local_chunk_ids
        ctx.total_chunks = total_chunks
        ctx.use_two_chunk = bool(two_chunk)

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

        local_index_by_chunk = {chunk_id: idx for idx, chunk_id in enumerate(local_chunk_ids)}
        recv = tensor.new_zeros((local_chunks, d_window) + tuple(tensor.shape[1:]))
        ops = []
        send_tensors = []

        for local_idx, chunk_id in enumerate(local_chunk_ids):
            if chunk_id > 0:
                prev_chunk_id = chunk_id - 1
                source_rank = _cp_chunk_owner_rank(prev_chunk_id, cp_size, bool(two_chunk))
                if source_rank == cp_rank:
                    source_idx = local_index_by_chunk[prev_chunk_id]
                    source_start = source_idx * chunk_len
                    recv[local_idx].copy_(
                        tensor[source_start + chunk_len - d_window : source_start + chunk_len]
                    )
                else:
                    ops.append(
                        dist.P2POp(
                            dist.irecv,
                            recv[local_idx],
                            _group_peer(cp_group, source_rank),
                            cp_group,
                        )
                    )

            next_chunk_id = chunk_id + 1
            if next_chunk_id < total_chunks:
                target_rank = _cp_chunk_owner_rank(next_chunk_id, cp_size, bool(two_chunk))
                if target_rank != cp_rank:
                    start = local_idx * chunk_len
                    send_tensors.append(
                        tensor[start + chunk_len - d_window : start + chunk_len].contiguous()
                    )
                    ops.append(
                        dist.P2POp(
                            dist.isend,
                            send_tensors[-1],
                            _group_peer(cp_group, target_rank),
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

        total_chunks = ctx.total_chunks
        local_chunk_ids = ctx.local_chunk_ids
        local_chunks = len(local_chunk_ids)
        local_index_by_chunk = {chunk_id: idx for idx, chunk_id in enumerate(local_chunk_ids)}
        grad_chunks = grad_boundary.reshape(
            (local_chunks, d_window) + tuple(grad_boundary.shape[1:])
        )
        ops = []
        send_tensors = []
        recv_specs = []

        for local_idx, chunk_id in enumerate(local_chunk_ids):
            if chunk_id > 0:
                prev_chunk_id = chunk_id - 1
                source_rank = _cp_chunk_owner_rank(prev_chunk_id, cp_size, ctx.use_two_chunk)
                if source_rank == cp_rank:
                    source_idx = local_index_by_chunk[prev_chunk_id]
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
                            _group_peer(cp_group, source_rank),
                            cp_group,
                        )
                    )

            next_chunk_id = chunk_id + 1
            if next_chunk_id < total_chunks:
                target_rank = _cp_chunk_owner_rank(next_chunk_id, cp_size, ctx.use_two_chunk)
                if target_rank != cp_rank:
                    recv = grad_chunks.new_zeros((d_window,) + tuple(grad_chunks.shape[2:]))
                    recv_specs.append((local_idx, recv))
                    ops.append(
                        dist.P2POp(dist.irecv, recv, _group_peer(cp_group, target_rank), cp_group)
                    )

        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        for local_idx, recv in recv_specs:
            start = local_idx * chunk_len
            grad_input[start + chunk_len - d_window : start + chunk_len] += recv
        return grad_input, None, None, None


def exchange_left_boundary_tensor(
    tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Return fixed left-boundary tokens for contiguous CP partitioning."""
    return _LeftBoundaryExchange.apply(tensor, d_window, cp_group, False)


def exchange_two_chunk_left_boundary_tensor(
    tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Return fixed left-boundary tokens for two-chunk partitioning."""
    return _LeftBoundaryExchange.apply(tensor, d_window, cp_group, True)


# =============================================================================
# Compressed Metadata And Compressor Inputs
# =============================================================================


def build_global_compressed_cu_seqlens_fused(
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


def build_rank_major_compressed_metadata_fused(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap_per_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build seq ids, compression ids, and valid mask for rank-major compressed rows."""
    return csa_cp_kernels.build_compressed_row_metadata(
        cu_seqlens, cp_size, l_local, ratio, d_comp, c_cap_per_rank
    )


def build_two_chunk_rank_major_compressed_metadata_fused(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    chunk_len: int,
    ratio: int,
    d_comp: int,
    c_cap_per_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compressed-row metadata for two-chunk local rows."""
    return csa_cp_kernels.build_compressed_row_metadata(
        cu_seqlens,
        cp_size,
        chunk_len,
        ratio,
        d_comp,
        c_cap_per_chunk,
        c_cap_per_rank=2 * int(c_cap_per_chunk),
        use_two_chunk=True,
    )


def build_compressor_prep_compact_fused(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack local hidden and boundary hidden into fixed-capacity compressed rows."""
    device = hidden_local.device
    c_cap = (l_local + d_comp) // ratio
    if c_cap == 0:
        empty_ids = torch.full((0,), -1, dtype=torch.int32, device=device)
        return (
            hidden_local.new_zeros((0,) + tuple(hidden_local.shape[1:])),
            torch.zeros_like(cu_seqlens),
            empty_ids,
            empty_ids,
            torch.zeros((0,), dtype=torch.bool, device=device),
            c_cap,
        )
    hidden_compact, cu_compact, seq_ids_t, comp_ids_t, valid_t = (
        csa_cp_kernels.CompressorInputCompact.apply(
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            global_start,
            l_local,
            ratio,
            d_comp,
            d_window,
            c_cap,
        )
    )
    return hidden_compact, cu_compact, seq_ids_t, comp_ids_t, valid_t, c_cap


def build_two_chunk_compressor_prep_compact_fused(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Pack both local chunks into one fixed-capacity compressor input tensor."""
    chunk_starts, chunk_len, l_local = _two_chunk_layout(
        chunk_ranges, "DSv4 two-chunk compressor-prep"
    )
    if hidden_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 two-chunk compressor-prep expects hidden rows to be "
            f"{l_local}, got {hidden_local.shape[0]}."
        )
    expected_boundary = 2 * int(d_window)
    if boundary_hidden.shape[0] != expected_boundary:
        raise RuntimeError(
            "DSv4 two-chunk compressor-prep expects one boundary window per chunk: "
            f"boundary={boundary_hidden.shape[0]}, expected={expected_boundary}."
        )

    parts = [
        build_compressor_prep_compact_fused(
            hidden_local.narrow(0, i * chunk_len, chunk_len),
            boundary_hidden.narrow(0, i * int(d_window), int(d_window)),
            cu_seqlens,
            chunk_starts[i],
            chunk_len,
            ratio,
            d_comp,
            d_window,
        )
        for i in range(2)
    ]
    hidden_parts, cu_parts, seq_parts, comp_parts, valid_parts, c_caps = zip(*parts)
    if c_caps[0] != c_caps[1]:
        raise RuntimeError(
            "DSv4 two-chunk compressor-prep expects equal fixed capacity per chunk: "
            f"c_caps={c_caps}."
        )
    cu_deltas = sum(cu[1:] - cu[:-1] for cu in cu_parts)
    cu_compact = torch.cat(
        (
            torch.zeros((1,), dtype=cu_seqlens.dtype, device=cu_seqlens.device),
            torch.cumsum(cu_deltas, dim=0),
        )
    )
    c_cap_per_chunk = c_caps[0]
    c_cap = sum(c_caps)
    return (
        torch.cat(hidden_parts, dim=0),
        cu_compact,
        torch.cat(seq_parts, dim=0),
        torch.cat(comp_parts, dim=0),
        torch.cat(valid_parts, dim=0),
        c_cap,
        c_cap_per_chunk,
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
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    ratio: int,
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack window KV and compressed KV into the fixed THD KV layout."""
    capacity = max(
        1,
        int(l_local)
        + int(d_window) * (cu_seqlens.shape[0] - 1)
        + int(compressed_rank_major.shape[0]),
    )
    if rank_major_by_seq_major is None:
        rank_major_by_seq_major = torch.empty((1,), dtype=torch.int32, device=kv_local.device)
    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    kv_full = csa_cp_kernels.ThdFullKvPack.apply(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        rank_major_by_seq_major,
        cu_seqlens_compressed,
        int(global_start),
        int(l_local),
        int(d_window),
        int(ratio),
        int(capacity),
        0,
        0,
        0,
        0,
        0,
    )
    return kv_full


def pack_two_chunk_cp_kv_full_fused(
    kv_local: torch.Tensor,
    boundary_kv: torch.Tensor,
    compressed_rank_major: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    d_window: int,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pack two local window chunks plus shared compressed KV into THD KV layout."""
    chunk_starts, chunk_len, l_local = _two_chunk_layout(chunk_ranges, "DSv4 two-chunk KV pack")
    if kv_local.shape[0] != l_local:
        raise RuntimeError(
            f"DSv4 two-chunk KV pack expects local KV rows to be {l_local}, got {kv_local.shape[0]}."
        )
    expected_boundary = 2 * int(d_window)
    if boundary_kv.shape[0] != expected_boundary:
        raise RuntimeError(
            "DSv4 two-chunk KV pack expects one boundary window per chunk: "
            f"boundary={boundary_kv.shape[0]}, expected={expected_boundary}."
        )
    chunk_count = len(chunk_starts)
    window_capacity = max(1, int(chunk_len) + int(d_window) * (cu_seqlens.shape[0] - 1))
    shared_compressed_base = chunk_count * window_capacity
    total_capacity = max(1, shared_compressed_base + int(compressed_rank_major.shape[0]))
    chunk1_start = int(chunk_starts[1]) if chunk_count > 1 else 0
    dummy_ids = torch.empty((1,), dtype=torch.int32, device=kv_local.device)
    dummy_valid = torch.empty((1,), dtype=torch.bool, device=kv_local.device)
    kv_full = csa_cp_kernels.ThdFullKvPack.apply(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        dummy_ids,
        dummy_ids,
        dummy_valid,
        cu_seqlens,
        dummy_ids,
        cu_seqlens,
        0,
        0,
        int(d_window),
        0,
        total_capacity,
        int(chunk_starts[0]),
        chunk1_start,
        chunk_count,
        int(chunk_len),
        window_capacity,
    )
    offsets = tuple(i * window_capacity for i in range(chunk_count))
    return kv_full, offsets


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


# =============================================================================
# Indexer Top-K Metadata
# =============================================================================


def build_cp_indexer_topk_inputs_fused(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor
]:
    """Build fixed-shape Q/K/weight and cu_seqlens inputs for indexer top-k."""
    if q_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects q_indexer_local rows to be "
            f"{l_local}, got {q_indexer_local.shape[0]}."
        )
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights_indexer_local rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    k_topk, cu_q_topk, cu_k_topk, seq_lens = csa_cp_kernels.build_indexer_topk_metadata(
        k_indexer_seq_major, cu_seqlens_q, cu_seqlens_compressed, global_start, l_local, ratio
    )
    max_q = int(max_seqlen_q) if max_seqlen_q is not None else l_local
    max_k = (
        int(max_seqlen_kv) if max_seqlen_kv is not None else max(1, k_indexer_seq_major.shape[0])
    )
    return (
        q_indexer_local,
        k_topk,
        weights_indexer_local,
        cu_q_topk,
        cu_k_topk,
        max_q,
        max_k,
        seq_lens,
    )


def compute_two_chunk_cp_indexer_topk_logical_fused(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
) -> torch.Tensor:
    """Run chunk-local indexer top-k and return logical compressed ids per local row."""
    topk_width = int(topk_width)
    if topk_width < 0:
        raise RuntimeError(f"DSv4 CP two-chunk indexer top-k got negative width: {topk_width}.")
    if ratio <= 1:
        raise RuntimeError(f"DSv4 CP two-chunk indexer top-k expects ratio > 1, got {ratio}.")
    chunk_starts, chunk_len, l_local = _two_chunk_layout(
        chunk_ranges, "DSv4 CP two-chunk indexer top-k"
    )
    if q_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP two-chunk indexer top-k expects q_indexer_local rows to be "
            f"{l_local}, got {q_indexer_local.shape[0]}."
        )
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP two-chunk indexer top-k expects weights rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return torch.empty((l_local, 0), dtype=torch.int32, device=q_indexer_local.device)

    chunk_outputs: List[torch.Tensor] = []
    for local_offset, global_start in ((0, chunk_starts[0]), (chunk_len, chunk_starts[1])):
        q_chunk = q_indexer_local.narrow(0, local_offset, chunk_len)
        weights_chunk = weights_indexer_local.narrow(0, local_offset, chunk_len)
        (
            q_for_topk,
            k_for_topk,
            weights_for_topk,
            cu_q_topk,
            cu_k_topk,
            max_q_topk,
            max_k_topk,
            seq_lens_topk,
        ) = build_cp_indexer_topk_inputs_fused(
            q_chunk,
            weights_chunk,
            k_indexer_seq_major,
            cu_seqlens_q,
            cu_seqlens_compressed,
            global_start,
            chunk_len,
            ratio,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        topk_compute_width = min(topk_width, max_k_topk)
        if topk_compute_width <= 0:
            topk_chunk = torch.empty(
                (chunk_len, 0), dtype=torch.int32, device=q_indexer_local.device
            )
        else:
            from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
                indexer_topk,
            )

            topk_chunk, _ = indexer_topk(
                q_for_topk,
                k_for_topk,
                weights_for_topk,
                topk=topk_compute_width,
                ratio=ratio,
                indexer_softmax_scale=indexer_softmax_scale,
                cu_seqlens_q=cu_q_topk,
                cu_seqlens_kv=cu_k_topk,
                max_seqlen_q=max_q_topk,
                max_seqlen_kv=max_k_topk,
                fixed_topk_width=topk_width,
                compute_topk_length=False,
                precomputed_seq_lens=seq_lens_topk,
            )
        if topk_chunk.shape[-1] < topk_width:
            pad = torch.full(
                (chunk_len, topk_width - topk_chunk.shape[-1]),
                -1,
                dtype=torch.int32,
                device=q_indexer_local.device,
            )
            topk_chunk = torch.cat([topk_chunk, pad], dim=-1)
        chunk_outputs.append(topk_chunk)

    return torch.cat(chunk_outputs, dim=0)


# =============================================================================
# Final Attention And Indexer-Loss Indices
# =============================================================================


def build_cp_flat_idxs_fused(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: Optional[torch.Tensor] = None,
    max_n_compressed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build final sparse-attention indices for contiguous CP partitioning."""
    if indexer_topk_compressed_logical_ids is not None:
        compressed_width = indexer_topk_compressed_logical_ids.shape[-1]
    elif ratio > 1:
        compressed_width = max_n_compressed
    else:
        compressed_width = 0
    return csa_cp_kernels.build_attention_indices(
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        indexer_topk_compressed_logical_ids,
    )


def build_two_chunk_cp_flat_idxs_fused(
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
    """Build final sparse-attention indices for two-chunk partitioning."""
    chunk_starts, chunk_len, l_local = _two_chunk_layout(chunk_ranges, "DSv4 two-chunk final idx")
    if indexer_topk_compressed_logical_ids is not None:
        if indexer_topk_compressed_logical_ids.shape[0] != l_local:
            raise RuntimeError(
                "DSv4 two-chunk final idx expects indexer rows to be "
                f"{l_local}, got {indexer_topk_compressed_logical_ids.shape[0]}."
            )
    if shared_compressed_base is None:
        raise RuntimeError("DSv4 two-chunk final idx requires a shared compressed base.")
    compressed_width = (
        int(indexer_topk_compressed_logical_ids.shape[-1])
        if indexer_topk_compressed_logical_ids is not None
        else int(max_n_compressed)
    )
    if ratio > 1 and compressed_width > 0:
        if rank_major_by_seq_major is None or cu_seqlens_compressed is None:
            raise RuntimeError("DSv4 two-chunk final idx with compression requires metadata.")
    else:
        cu_seqlens_compressed = cu_seqlens
        rank_major_by_seq_major = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)
    return csa_cp_kernels.build_attention_indices(
        cu_seqlens,
        0,
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
        window_capacity_per_chunk=int(shared_compressed_base) // 2,
        shared_compressed_base=int(shared_compressed_base),
    )


def build_cp_flat_idxs_for_indexer_loss_fused(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: torch.Tensor,
    indexer_rank_by_seq_major: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build indexer-loss indices for contiguous CP partitioning."""
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
        indexer_rank_by_seq_major,
    )


def build_two_chunk_cp_flat_idxs_for_indexer_loss_fused(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: torch.Tensor,
    indexer_rank_by_seq_major: torch.Tensor,
    shared_compressed_base: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build indexer-loss indices for two-chunk partitioning."""
    chunk_starts, chunk_len, l_local = _two_chunk_layout(
        chunk_ranges, "DSv4 two-chunk indexer-loss idx"
    )
    if indexer_topk_compressed_logical_ids.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 two-chunk indexer-loss idx expects indexer rows to be "
            f"{l_local}, got {indexer_topk_compressed_logical_ids.shape[0]}."
        )
    if shared_compressed_base is None:
        raise RuntimeError("DSv4 two-chunk indexer-loss idx requires a shared compressed base.")
    return csa_cp_kernels.build_indexer_loss_indices(
        cu_seqlens,
        cu_seqlens_compressed,
        0,
        l_local,
        d_window,
        window_size,
        ratio,
        indexer_topk_compressed_logical_ids,
        indexer_rank_by_seq_major,
        chunk_starts=chunk_starts,
        chunk_len=chunk_len,
        window_capacity_per_chunk=int(shared_compressed_base) // 2,
        shared_compressed_base=int(shared_compressed_base),
    )

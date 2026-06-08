# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant import csa_cp_kernels
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    batch_of_row,
    indexer_topk,
)


class _SingleRankCPGroup:
    """Small adapter for applying THD RoPE to already-global packed tensors."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


SINGLE_RANK_CP_GROUP = _SingleRankCPGroup()

DSV4_CP_PARTITION_CONTIGUOUS = "contiguous"
DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK = "packed_stream_two_chunk"
DSV4_CP_PARTITION_MODES = (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK,
)


def normalize_dsv4_cp_partition_mode(mode: Optional[str]) -> str:
    """Validate and normalize the DSv4 CP packed-token partition mode."""

    if mode is None:
        return DSV4_CP_PARTITION_CONTIGUOUS
    if mode not in DSV4_CP_PARTITION_MODES:
        raise RuntimeError(
            "Unsupported DSv4 CP partition mode: "
            f"{mode!r}. Expected one of {DSV4_CP_PARTITION_MODES}."
        )
    return mode


def cp_group_size(cp_group: Optional[torch.distributed.ProcessGroup]) -> int:
    if cp_group is None:
        return 1
    return cp_group.size()


def cp_group_rank(cp_group: Optional[torch.distributed.ProcessGroup]) -> int:
    if cp_group is None:
        return 0
    return cp_group.rank()


def can_use_csa_cp_fused_kernels(*tensors: Optional[torch.Tensor]) -> bool:
    """Return whether CSA CP fused memory kernels can run for these tensors."""

    return csa_cp_kernels.can_use_cute_kernels(*tensors)


class _THDOverlapTransformCute(torch.autograd.Function):
    """Autograd wrapper for CSA THD compressor overlap transform."""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        is_first_in_seg: torch.Tensor,
        head_dim: int,
        fill_value: float,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(tensor.shape)
        ctx.head_dim = head_dim
        ctx.save_for_backward(is_first_in_seg)
        return csa_cp_kernels.overlap_transform_thd(
            tensor,
            is_first_in_seg,
            head_dim,
            fill_value,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (is_first_in_seg,) = ctx.saved_tensors
        grad_input = csa_cp_kernels.overlap_transform_thd_backward(
            grad_output.contiguous(),
            is_first_in_seg,
            ctx.input_shape,
            ctx.head_dim,
        )
        return grad_input, None, None, None


def apply_thd_overlap_transform_fused(
    tensor: torch.Tensor,
    is_first_in_seg: torch.Tensor,
    head_dim: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Apply THD overlap transform with fused forward/backward kernels."""
    if not csa_cp_kernels.can_use_cute_kernels(tensor, is_first_in_seg):
        raise RuntimeError("DSv4 CP THD overlap transform requires CUDA tensors and CuTeDSL.")
    return _THDOverlapTransformCute.apply(tensor, is_first_in_seg, head_dim, float(fill_value))


class _THDChunkedCPRopeCute(torch.autograd.Function):
    """Autograd wrapper for THD CP RoPE on a known packed-token interval."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        global_row_base: int,
        nope_dim: int,
        pos_dim: int,
        inverse: bool,
        clamp_to_valid_token: bool,
    ) -> torch.Tensor:
        ctx.global_row_base = global_row_base
        ctx.nope_dim = nope_dim
        ctx.pos_dim = pos_dim
        ctx.inverse = inverse
        ctx.clamp_to_valid_token = clamp_to_valid_token
        ctx.save_for_backward(cos, sin, cu_seqlens_padded)
        return csa_cp_kernels.apply_thd_chunked_cp_rope(
            x,
            cos,
            sin,
            cu_seqlens_padded,
            global_row_base,
            nope_dim,
            pos_dim,
            inverse=inverse,
            clamp_to_valid_token=clamp_to_valid_token,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin, cu_seqlens_padded = ctx.saved_tensors
        grad_x = csa_cp_kernels.apply_thd_chunked_cp_rope(
            grad_output.contiguous(),
            cos,
            sin,
            cu_seqlens_padded,
            ctx.global_row_base,
            ctx.nope_dim,
            ctx.pos_dim,
            inverse=ctx.inverse,
            clamp_to_valid_token=ctx.clamp_to_valid_token,
            adjoint=True,
        )
        return grad_x, None, None, None, None, None, None, None, None


class _CPCompressedRopeCute(torch.autograd.Function):
    """Autograd wrapper for compressed-row RoPE using compressor metadata."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        comp_ids_local: torch.Tensor,
        ratio: int,
        nope_dim: int,
        pos_dim: int,
        inverse: bool,
    ) -> torch.Tensor:
        ctx.ratio = ratio
        ctx.nope_dim = nope_dim
        ctx.pos_dim = pos_dim
        ctx.inverse = inverse
        ctx.save_for_backward(cos, sin, comp_ids_local)
        return csa_cp_kernels.apply_compressed_rope(
            x,
            cos,
            sin,
            comp_ids_local,
            ratio,
            nope_dim,
            pos_dim,
            inverse=inverse,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin, comp_ids_local = ctx.saved_tensors
        grad_x = csa_cp_kernels.apply_compressed_rope(
            grad_output.contiguous(),
            cos,
            sin,
            comp_ids_local,
            ctx.ratio,
            ctx.nope_dim,
            ctx.pos_dim,
            inverse=ctx.inverse,
            adjoint=True,
        )
        return grad_x, None, None, None, None, None, None, None


class _THDCPChunksRopeCute(torch.autograd.Function):
    """Autograd wrapper for the two-chunk THD CP RoPE fast path."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        chunk0_global_start: int,
        chunk1_global_start: int,
        chunk_len: int,
        nope_dim: int,
        pos_dim: int,
        inverse: bool,
        clamp_to_valid_token: bool,
    ) -> torch.Tensor:
        ctx.chunk0_global_start = int(chunk0_global_start)
        ctx.chunk1_global_start = int(chunk1_global_start)
        ctx.chunk_len = int(chunk_len)
        ctx.nope_dim = int(nope_dim)
        ctx.pos_dim = int(pos_dim)
        ctx.inverse = inverse
        ctx.clamp_to_valid_token = clamp_to_valid_token
        ctx.save_for_backward(cos, sin, cu_seqlens_padded)
        return csa_cp_kernels.apply_thd_cp_two_chunks_rope(
            x,
            cos,
            sin,
            cu_seqlens_padded,
            ctx.chunk0_global_start,
            ctx.chunk1_global_start,
            ctx.chunk_len,
            ctx.nope_dim,
            ctx.pos_dim,
            inverse=inverse,
            clamp_to_valid_token=clamp_to_valid_token,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin, cu_seqlens_padded = ctx.saved_tensors
        grad_x = csa_cp_kernels.apply_thd_cp_two_chunks_rope(
            grad_output.contiguous(),
            cos,
            sin,
            cu_seqlens_padded,
            ctx.chunk0_global_start,
            ctx.chunk1_global_start,
            ctx.chunk_len,
            ctx.nope_dim,
            ctx.pos_dim,
            inverse=ctx.inverse,
            clamp_to_valid_token=ctx.clamp_to_valid_token,
            adjoint=True,
        )
        return grad_x, None, None, None, None, None, None, None, None, None, None


def apply_thd_chunked_cp_rope_fused(
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
    """Apply THD CP RoPE for rows in one packed-token interval.

    ``global_row_base`` can override the contiguous formula
    ``cp_rank * chunk_len``. This is used when local rows are split across
    explicit global chunks and each slice has its own global start.
    """
    if not csa_cp_kernels.can_use_cute_kernels(x, cos, sin, cu_seqlens_padded):
        raise RuntimeError("DSv4 THD CP RoPE requires CUDA tensors and CuTeDSL kernels.")
    if rotary_interleaved:
        raise RuntimeError("DSv4 THD CP RoPE does not support rotary_interleaved=True.")
    if not remove_interleaving:
        raise RuntimeError("DSv4 THD CP RoPE requires remove_interleaving=True.")
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise RuntimeError(
            f"DSv4 THD CP RoPE got invalid CP rank/size: cp_rank={cp_rank}, "
            f"cp_size={cp_size}."
        )
    if chunk_len is None:
        chunk_len = x.shape[0]
    if global_row_base is None:
        global_row_base = int(cp_rank) * int(chunk_len)
    global_row_base = int(global_row_base) + int(row_offset)
    return _THDChunkedCPRopeCute.apply(
        x,
        cos,
        sin,
        cu_seqlens_padded,
        global_row_base,
        nope_dim,
        pos_dim,
        inverse,
        clamp_to_valid_token,
    )


def apply_thd_cp_chunks_rope_fused(
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
    """Apply THD RoPE to local rows already ordered as explicit global chunks.

    ``chunk_ranges`` describes the global padded-token interval for each local
    slice in first-dimension order. This is the path for packed-stream
    two-chunk CP partition; it avoids building a per-token positions tensor.
    """

    if not chunk_ranges:
        raise RuntimeError("DSv4 THD CP chunk RoPE requires at least one chunk range.")
    if not csa_cp_kernels.can_use_cute_kernels(x, cos, sin, cu_seqlens_padded):
        raise RuntimeError("DSv4 THD CP chunk RoPE requires CUDA tensors and CuTeDSL.")
    if rotary_interleaved:
        raise RuntimeError("DSv4 THD CP chunk RoPE does not support rotary_interleaved=True.")
    if not remove_interleaving:
        raise RuntimeError("DSv4 THD CP chunk RoPE requires remove_interleaving=True.")
    chunk_lengths = [int(end) - int(start) for start, end in chunk_ranges]
    if any(length <= 0 for length in chunk_lengths):
        raise RuntimeError(f"DSv4 THD CP chunk RoPE got invalid chunks: {chunk_ranges}.")
    l_local = sum(chunk_lengths)
    if x.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 THD CP chunk RoPE expects x rows to match chunk ranges: "
            f"x={x.shape[0]}, chunks={l_local}."
        )

    if len(chunk_ranges) == 2 and chunk_lengths[0] == chunk_lengths[1]:
        chunk0_start = int(chunk_ranges[0][0]) + int(row_offset)
        chunk1_start = int(chunk_ranges[1][0]) + int(row_offset)
        return _THDCPChunksRopeCute.apply(
            x,
            cos,
            sin,
            cu_seqlens_padded,
            chunk0_start,
            chunk1_start,
            chunk_lengths[0],
            nope_dim,
            pos_dim,
            inverse,
            clamp_to_valid_token,
        )

    outputs: List[torch.Tensor] = []
    local_offset = 0
    for (global_start, _global_end), l_chunk in zip(chunk_ranges, chunk_lengths):
        x_chunk = x.narrow(0, local_offset, l_chunk)
        outputs.append(
            apply_thd_chunked_cp_rope_fused(
                x_chunk,
                cos,
                sin,
                nope_dim,
                pos_dim,
                cu_seqlens_padded,
                cp_rank=0,
                cp_size=1,
                rotary_interleaved=rotary_interleaved,
                inverse=inverse,
                remove_interleaving=remove_interleaving,
                row_offset=row_offset,
                chunk_len=l_chunk,
                clamp_to_valid_token=clamp_to_valid_token,
                global_row_base=int(global_start),
            )
        )
        local_offset += l_chunk

    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def apply_cp_compressed_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    comp_ids_local: torch.Tensor,
    ratio: int,
    nope_dim: int,
    pos_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply production compressed-row RoPE without sequence-prefix reconstruction."""
    if not csa_cp_kernels.can_use_cute_kernels(x, cos, sin, comp_ids_local):
        raise RuntimeError("DSv4 CP compressed RoPE requires CUDA tensors and CuTeDSL kernels.")
    return _CPCompressedRopeCute.apply(
        x, cos, sin, comp_ids_local, ratio, nope_dim, pos_dim, inverse
    )


def _group_peer(cp_group: torch.distributed.ProcessGroup, group_rank: int) -> int:
    """Translate a CP group-local rank to the global rank expected by P2P APIs."""
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(cp_group, group_rank)
    return group_rank


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange a fixed left boundary with reverse scatter/add in backward."""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        d_window: int,
        cp_group: torch.distributed.ProcessGroup,
    ):
        cp_size = cp_group_size(cp_group)
        cp_rank = cp_group_rank(cp_group)
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape

        if cp_size <= 1:
            return tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))
        if tensor.shape[0] < d_window:
            raise RuntimeError(
                "DSv4 THD CP boundary exchange requires local token capacity >= D_window: "
                f"local={tensor.shape[0]}, D_window={d_window}"
            )

        send = tensor[-d_window:].contiguous()
        recv = tensor.new_zeros(send.shape)
        ops = []
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, recv, _group_peer(cp_group, cp_rank - 1), cp_group))
        if cp_rank + 1 < cp_size:
            ops.append(dist.P2POp(dist.isend, send, _group_peer(cp_group, cp_rank + 1), cp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        cp_group = ctx.cp_group
        cp_size = cp_group_size(cp_group)
        cp_rank = cp_group_rank(cp_group)
        d_window = ctx.d_window

        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        if cp_size <= 1:
            return grad_input, None, None

        send = grad_boundary.contiguous()
        recv = grad_boundary.new_zeros(send.shape)
        ops = []
        if cp_rank + 1 < cp_size:
            ops.append(dist.P2POp(dist.irecv, recv, _group_peer(cp_group, cp_rank + 1), cp_group))
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.isend, send, _group_peer(cp_group, cp_rank - 1), cp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-d_window:] += recv
        return grad_input, None, None


def exchange_left_boundary_tensor(
    tensor: torch.Tensor,
    d_window: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Return the fixed left boundary tensor for this CP rank."""
    return _LeftBoundaryExchange.apply(tensor, d_window, cp_group)


def _chunked_cp_owner_rank(chunk_id: int, cp_size: int) -> int:
    if chunk_id < cp_size:
        return chunk_id
    return 2 * cp_size - 1 - chunk_id


class _ChunkedLeftBoundaryExchange(torch.autograd.Function):
    """Exchange left boundaries for packed-stream two-chunk CP local rows."""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        d_window: int,
        cp_group: torch.distributed.ProcessGroup,
    ):
        cp_size = cp_group_size(cp_group)
        cp_rank = cp_group_rank(cp_group)
        local_chunks = 1 if cp_size <= 1 else 2
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        ctx.local_chunks = local_chunks

        if tensor.shape[0] % local_chunks != 0:
            raise RuntimeError(
                "DSv4 chunked CP boundary exchange expects equal local chunks: "
                f"local={tensor.shape[0]}, chunks={local_chunks}."
            )
        chunk_len = tensor.shape[0] // local_chunks
        ctx.chunk_len = chunk_len
        if chunk_len < d_window:
            raise RuntimeError(
                "DSv4 chunked CP boundary exchange requires chunk_len >= D_window: "
                f"chunk_len={chunk_len}, D_window={d_window}."
            )
        if cp_size <= 1:
            return tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        total_chunks = 2 * cp_size
        local_chunk_ids = (cp_rank, total_chunks - 1 - cp_rank)
        ctx.local_chunk_ids = local_chunk_ids
        local_index_by_chunk = {chunk_id: idx for idx, chunk_id in enumerate(local_chunk_ids)}
        recv = tensor.new_zeros((local_chunks, d_window) + tuple(tensor.shape[1:]))
        ops = []
        send_tensors = []

        for local_idx, chunk_id in enumerate(local_chunk_ids):
            if chunk_id > 0:
                prev_chunk_id = chunk_id - 1
                source_rank = _chunked_cp_owner_rank(prev_chunk_id, cp_size)
                if source_rank == cp_rank:
                    source_idx = local_index_by_chunk[prev_chunk_id]
                    source_start = source_idx * chunk_len
                    recv[local_idx].copy_(tensor[source_start + chunk_len - d_window : source_start + chunk_len])
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
                target_rank = _chunked_cp_owner_rank(next_chunk_id, cp_size)
                if target_rank != cp_rank:
                    start = local_idx * chunk_len
                    send_tensors.append(tensor[start + chunk_len - d_window : start + chunk_len].contiguous())
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
        cp_size = cp_group_size(cp_group)
        cp_rank = cp_group_rank(cp_group)
        d_window = ctx.d_window
        local_chunks = ctx.local_chunks
        chunk_len = ctx.chunk_len
        grad_input = grad_boundary.new_zeros(ctx.input_shape)
        if cp_size <= 1:
            return grad_input, None, None

        total_chunks = 2 * cp_size
        local_chunk_ids = ctx.local_chunk_ids
        local_index_by_chunk = {chunk_id: idx for idx, chunk_id in enumerate(local_chunk_ids)}
        grad_chunks = grad_boundary.reshape((local_chunks, d_window) + tuple(grad_boundary.shape[1:]))
        ops = []
        send_tensors = []
        recv_specs = []

        for local_idx, chunk_id in enumerate(local_chunk_ids):
            if chunk_id > 0:
                prev_chunk_id = chunk_id - 1
                source_rank = _chunked_cp_owner_rank(prev_chunk_id, cp_size)
                if source_rank == cp_rank:
                    source_idx = local_index_by_chunk[prev_chunk_id]
                    source_start = source_idx * chunk_len
                    grad_input[source_start + chunk_len - d_window : source_start + chunk_len] += grad_chunks[
                        local_idx
                    ]
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
                target_rank = _chunked_cp_owner_rank(next_chunk_id, cp_size)
                if target_rank != cp_rank:
                    recv = grad_chunks.new_zeros((d_window,) + tuple(grad_chunks.shape[2:]))
                    recv_specs.append((local_idx, recv))
                    ops.append(
                        dist.P2POp(
                            dist.irecv,
                            recv,
                            _group_peer(cp_group, target_rank),
                            cp_group,
                        )
                    )

        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        for local_idx, recv in recv_specs:
            start = local_idx * chunk_len
            grad_input[start + chunk_len - d_window : start + chunk_len] += recv
        return grad_input, None, None


def exchange_chunked_left_boundary_tensor(
    tensor: torch.Tensor,
    d_window: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Return fixed left boundaries for packed-stream two-chunk CP local rows.

    For ``cp_size > 1``, ``tensor`` is interpreted as two equal local chunks in
    packed-stream order ``[rank, 2 * cp_size - 1 - rank]``. The output has
    ``2 * d_window`` rows, one boundary window per local chunk in the same
    order.  Backward scatters/adds each boundary gradient to the tail of the
    global previous chunk, including the self-boundary case such as rank 3's
    chunks ``[3, 4]`` when ``cp_size == 4``.
    """

    return _ChunkedLeftBoundaryExchange.apply(tensor, d_window, cp_group)


def contiguous_cp_partition(
    cu_seqlens_padded: torch.Tensor, cp_size: int, cp_rank: int
) -> Tuple[int, int]:
    """Return the global padded-token partition assigned to a contiguous CP rank.

    Args:
        cu_seqlens_padded: Global padded THD cumulative sequence lengths. The
            last entry is the padded total token count that CP partitions
            evenly across CP ranks.
        cp_size: Number of context-parallel ranks.
        cp_rank: Context-parallel rank whose contiguous partition is requested.

    Returns:
        A ``(global_start, l_local)`` tuple. ``global_start`` is the first
        global padded token owned by ``cp_rank`` and ``l_local`` is the fixed
        local padded token count per CP rank.
    """
    padded_total = int(cu_seqlens_padded[-1].item())
    if padded_total % cp_size != 0:
        raise RuntimeError(
            "DSv4 THD CP path requires padded_total_tokens % cp_size == 0: "
            f"total={padded_total}, cp_size={cp_size}"
        )
    l_local = padded_total // cp_size
    return cp_rank * l_local, l_local


def chunked_cp_partition(
    padded_total_tokens: int,
    cp_size: int,
    cp_rank: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return the packed-stream two-chunk CP token ranges for this rank.

    Args:
        padded_total_tokens: Host-known padded global token capacity. Production
            callers should pass a shape-derived value, not read this from a GPU
            cumulative-length tensor.
        cp_size: Number of context-parallel ranks.
        cp_rank: Context-parallel rank whose chunked partition is requested.

    Returns:
        Global padded-token intervals ``(start, end)`` in packed-stream two-chunk CP
        order: rank ``r`` owns chunk ``r`` followed by chunk
        ``2 * cp_size - 1 - r``. For ``cp_size == 1``, returns one full chunk.
    """

    padded_total_tokens = int(padded_total_tokens)
    if cp_size < 1 or cp_rank < 0 or cp_rank >= cp_size:
        raise RuntimeError(f"Invalid CP rank/size: cp_rank={cp_rank}, cp_size={cp_size}.")
    if cp_size == 1:
        return ((0, padded_total_tokens),)
    total_chunks = 2 * cp_size
    if padded_total_tokens % total_chunks != 0:
        raise RuntimeError(
            "DSv4 chunked CP partition expects padded_total_tokens % (2 * cp_size) == 0: "
            f"padded_total_tokens={padded_total_tokens}, cp_size={cp_size}."
        )
    chunk_len = padded_total_tokens // total_chunks
    chunk_ids = (cp_rank, total_chunks - 1 - cp_rank)
    return tuple((chunk_id * chunk_len, (chunk_id + 1) * chunk_len) for chunk_id in chunk_ids)


def build_cp_local_seq_positions(
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    l_local: int,
    device: torch.device,
) -> torch.Tensor:
    """Return sequence-local positions for this rank's contiguous CP rows.

    Args:
        cu_seqlens_padded: Global padded THD cumulative sequence lengths.
        global_start: First global padded token owned by this CP rank.
        l_local: Number of padded tokens owned by this CP rank.
        device: Device for the returned tensor.

    Returns:
        ``(l_local,)`` int64 tensor where entry ``i`` is the local position
        inside its packed sequence for global token ``global_start + i``.
    """
    global_ids = torch.arange(global_start, global_start + l_local, device=device, dtype=torch.long)
    cu_long = cu_seqlens_padded.to(device=device, dtype=torch.long)
    batch_ids = torch.searchsorted(cu_long, global_ids, right=True) - 1
    batch_ids = batch_ids.clamp(min=0, max=cu_long.numel() - 2)
    return global_ids - cu_long[batch_ids]


def build_cp_local_rope_positions(
    packed_seq_params,
    local_tokens: int,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Return sequence-local RoPE positions for this rank's contiguous CP chunk.

    THD CP partitions the global padded packed-token space contiguously. The
    local tensor rows are numbered from zero, but RoPE needs each row's position
    inside its original packed sequence, including the global CP offset when a
    sequence is split across ranks.

    Args:
        packed_seq_params: THD packed sequence metadata. This uses
            ``cu_seqlens_q_padded`` when present because CP partitions the
            padded THD capacity across CP ranks.
        local_tokens: Number of local tensor rows that will receive RoPE.
        cp_size: Number of context-parallel ranks.
        cp_rank: Context-parallel rank that owns these local rows.

    Returns:
        ``(local_tokens,)`` int64 tensor of sequence-local positions.
    """
    cu_seqlens_padded = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    global_start, l_local = contiguous_cp_partition(cu_seqlens_padded, cp_size, cp_rank)
    if local_tokens != l_local:
        raise RuntimeError(
            "DSv4 THD CP local RoPE expects contiguous equal local token chunks: "
            f"local={local_tokens}, expected={l_local}"
        )

    # Map this rank's local rows back to global padded THD row ids, then convert
    # them into per-sequence positions for RoPE.
    return build_cp_local_seq_positions(
        cu_seqlens_padded, global_start, l_local, cu_seqlens_padded.device
    )


def build_global_compressed_cu_seqlens(
    cu_seqlens_padded: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Return reference global seq-major compressed cumulative lengths."""
    seq_lens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    compressed_lens = seq_lens // ratio
    return torch.cat(
        [
            torch.zeros(1, dtype=cu_seqlens_padded.dtype, device=cu_seqlens_padded.device),
            compressed_lens.cumsum(0).to(cu_seqlens_padded.dtype),
        ]
    )


def build_global_compressed_cu_seqlens_fused(
    cu_seqlens_padded: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Return production global compressed prefix lengths with CuTeDSL."""
    if not csa_cp_kernels.can_use_cute_kernels(cu_seqlens_padded):
        raise RuntimeError("DSv4 CP compressed cu_seqlens requires CUDA tensors and CuTeDSL.")
    return csa_cp_kernels.build_global_compressed_cu_seqlens(cu_seqlens_padded, ratio)


def _ceil_div_nonnegative(numerator: int, denominator: int) -> int:
    if numerator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def _build_compressor_metadata_for_range(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_ids: List[int] = []
    comp_ids: List[int] = []
    valid: List[bool] = []
    global_end = global_start + l_local

    for b in range(int(cu_seqlens.shape[0]) - 1):
        seq_start = int(cu_seqlens[b].item())
        seq_end = int(cu_seqlens[b + 1].item())
        local_seq_start = max(seq_start, global_start)
        local_seq_end = min(seq_end, global_end)
        if local_seq_start >= local_seq_end:
            continue
        n_full_groups = (seq_end - seq_start) // ratio
        first_group = _ceil_div_nonnegative(
            max(seq_start, global_start - d_comp) - seq_start, ratio
        )
        stop_group = min(n_full_groups, (local_seq_end - seq_start) // ratio)
        for comp_id in range(first_group, stop_group):
            group_end = seq_start + (comp_id + 1) * ratio
            seq_ids.append(b)
            comp_ids.append(comp_id)
            valid.append(global_start <= group_end - 1 < global_end)

    if len(seq_ids) > c_cap:
        raise RuntimeError(
            "DSv4 compressor metadata produced more entries than fixed capacity: "
            f"produced={len(seq_ids)}, capacity={c_cap}"
        )

    seq_ids_t = torch.full((c_cap,), -1, dtype=torch.int32, device=device)
    comp_ids_t = torch.full((c_cap,), -1, dtype=torch.int32, device=device)
    valid_t = torch.zeros((c_cap,), dtype=torch.bool, device=device)
    if seq_ids:
        n = len(seq_ids)
        seq_ids_t[:n] = torch.tensor(seq_ids, dtype=torch.int32, device=device)
        comp_ids_t[:n] = torch.tensor(comp_ids, dtype=torch.int32, device=device)
        valid_t[:n] = torch.tensor(valid, dtype=torch.bool, device=device)
    return seq_ids_t, comp_ids_t, valid_t


def build_rank_major_compressed_metadata(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if csa_cp_kernels.can_use_cute_kernels(cu_seqlens):
        return build_rank_major_compressed_metadata_fused(
            cu_seqlens, cp_size, l_local, ratio, d_comp, c_cap
        )

    seq_parts = []
    comp_parts = []
    valid_parts = []
    for rank in range(cp_size):
        seq_ids, comp_ids, valid = _build_compressor_metadata_for_range(
            cu_seqlens,
            rank * l_local,
            l_local,
            ratio,
            d_comp,
            c_cap,
            device,
        )
        seq_parts.append(seq_ids)
        comp_parts.append(comp_ids)
        valid_parts.append(valid)
    return torch.cat(seq_parts), torch.cat(comp_parts), torch.cat(valid_parts)


def build_rank_major_compressed_metadata_fused(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rank-major compressed metadata with the production CuTeDSL kernel."""

    del device
    if not csa_cp_kernels.can_use_cute_kernels(cu_seqlens):
        raise RuntimeError("DSv4 CP rank-major metadata requires CUDA tensors and CuTeDSL kernels.")
    return csa_cp_kernels.build_rank_major_compressed_metadata(
        cu_seqlens, cp_size, l_local, ratio, d_comp, c_cap
    )


def build_chunked_rank_major_compressed_metadata_fused(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    chunk_len: int,
    ratio: int,
    d_comp: int,
    c_cap_per_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rank-major metadata for packed-stream two-chunk CP rows.

    Rank ``r`` owns global chunks ``r`` and ``2 * cp_size - 1 - r``. The output
    row order is still rank-major and matches all-gathered local compressed rows:
    all rows for rank 0's two chunks, then rank 1, and so on.
    """

    if cp_size <= 1:
        return build_rank_major_compressed_metadata_fused(
            cu_seqlens,
            cp_size,
            chunk_len,
            ratio,
            d_comp,
            c_cap_per_chunk,
        )
    if not csa_cp_kernels.can_use_cute_kernels(cu_seqlens):
        raise RuntimeError(
            "DSv4 chunked CP rank-major metadata requires CUDA tensors and CuTeDSL kernels."
        )
    return csa_cp_kernels.build_chunked_rank_major_compressed_metadata(
        cu_seqlens,
        cp_size,
        chunk_len,
        ratio,
        d_comp,
        c_cap_per_chunk,
    )


class _CompressorPrepCompact(torch.autograd.Function):
    """Gather local+left-boundary hidden rows into fixed compact compressor input."""

    @staticmethod
    def forward(
        ctx,
        hidden_local: torch.Tensor,
        boundary_hidden: torch.Tensor,
        local_src: torch.Tensor,
        boundary_src: torch.Tensor,
        source_is_boundary: torch.Tensor,
        compact_len: int,
    ) -> torch.Tensor:
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.save_for_backward(local_src, boundary_src, source_is_boundary)

        hidden_compact = hidden_local.new_zeros((compact_len,) + tuple(hidden_local.shape[1:]))
        if local_src.numel() == 0:
            return hidden_compact

        gathered = hidden_compact.new_zeros((local_src.numel(),) + tuple(hidden_local.shape[1:]))
        local_mask = ~source_is_boundary
        if bool(local_mask.any().item()):
            gathered[local_mask] = hidden_local.index_select(0, local_src[local_mask])
        if bool(source_is_boundary.any().item()):
            gathered[source_is_boundary] = boundary_hidden.index_select(
                0, boundary_src[source_is_boundary]
            )
        hidden_compact[: local_src.numel()] = gathered
        return hidden_compact

    @staticmethod
    def backward(ctx, grad_hidden_compact: torch.Tensor):
        local_src, boundary_src, source_is_boundary = ctx.saved_tensors
        grad_hidden = grad_hidden_compact.new_zeros(ctx.hidden_shape)
        grad_boundary = grad_hidden_compact.new_zeros(ctx.boundary_shape)
        if local_src.numel() == 0:
            return grad_hidden, grad_boundary, None, None, None, None

        grad_gathered = grad_hidden_compact[: local_src.numel()]
        local_mask = ~source_is_boundary
        if bool(local_mask.any().item()):
            grad_hidden.index_add_(0, local_src[local_mask], grad_gathered[local_mask])
        if bool(source_is_boundary.any().item()):
            grad_boundary.index_add_(
                0, boundary_src[source_is_boundary], grad_gathered[source_is_boundary]
            )
        return grad_hidden, grad_boundary, None, None, None, None


class _CompressorPrepCompactCute(torch.autograd.Function):
    """CuTe implementation of compressor-prep compact with explicit backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_local: torch.Tensor,
        boundary_hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        global_start: int,
        l_local: int,
        ratio: int,
        d_comp: int,
        d_window: int,
        c_cap: int,
    ):
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.global_start = global_start
        ctx.l_local = l_local
        ctx.ratio = ratio
        ctx.d_comp = d_comp
        ctx.d_window = d_window
        ctx.save_for_backward(cu_seqlens)
        return csa_cp_kernels.compressor_prep_compact(
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

    @staticmethod
    def backward(
        ctx,
        grad_hidden_compact: torch.Tensor,
        _grad_cu_compact: torch.Tensor,
        _grad_seq_ids: torch.Tensor,
        _grad_comp_ids: torch.Tensor,
        _grad_valid: torch.Tensor,
    ):
        (cu_seqlens,) = ctx.saved_tensors
        grad_hidden, grad_boundary = csa_cp_kernels.compressor_prep_compact_backward(
            grad_hidden_compact.contiguous(),
            ctx.hidden_shape,
            ctx.boundary_shape,
            cu_seqlens,
            ctx.global_start,
            ctx.l_local,
            ctx.ratio,
            ctx.d_comp,
            ctx.d_window,
        )
        return (
            grad_hidden,
            grad_boundary,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def build_compressor_prep_compact(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build compact compressor input for one CP rank.

    Args:
        hidden_local: Local hidden states, ``(L_local, 1, hidden_size)``.
        boundary_hidden: Fixed left boundary hidden states received by P2P,
            ``(D_window, 1, hidden_size)``.
        cu_seqlens: Global padded THD cumulative sequence lengths.
        global_start: First global padded token owned by this CP rank.
        l_local: Local padded token capacity.
        ratio: Compression ratio for the current CSA layer.
        d_comp: Left overlap needed by the compressor. DSv4 ratio 4 uses 8;
            other compressed layers use ``ratio``.
        d_window: Size of ``boundary_hidden``.

    Returns:
        ``(hidden_compact, cu_compact, seq_ids, comp_ids, valid, c_cap)``:

        * ``hidden_compact`` has fixed length ``c_cap * ratio`` and gathers
          only rows from ``boundary_hidden`` or ``hidden_local``. It has a
          custom backward that scatters grads to those two inputs.
        * ``cu_compact`` describes the real compact rows consumed by the
          compressor; tail capacity is padding.
        * ``seq_ids`` / ``comp_ids`` / ``valid`` describe each fixed
          compressed output row before rank-major all-gather.
    """
    device = hidden_local.device
    c_cap = (l_local + d_comp) // ratio
    compact_len = c_cap * ratio
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
    if csa_cp_kernels.can_use_cute_kernels(hidden_local, boundary_hidden, cu_seqlens):
        return build_compressor_prep_compact_fused(
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            global_start,
            l_local,
            ratio,
            d_comp,
            d_window,
        )

    ext_start = global_start - d_window
    src_positions: List[int] = []
    compact_cu = [0]
    global_end = global_start + l_local

    for b in range(int(cu_seqlens.shape[0]) - 1):
        seq_start = int(cu_seqlens[b].item())
        seq_end = int(cu_seqlens[b + 1].item())
        local_seq_start = max(seq_start, global_start)
        local_seq_end = min(seq_end, global_end)
        seq_group_count = 0
        if local_seq_start < local_seq_end:
            n_full_groups = (seq_end - seq_start) // ratio
            first_group = _ceil_div_nonnegative(
                max(seq_start, global_start - d_comp) - seq_start, ratio
            )
            stop_group = min(n_full_groups, (local_seq_end - seq_start) // ratio)
            for comp_id in range(first_group, stop_group):
                group_start = seq_start + comp_id * ratio
                group_end = group_start + ratio
                for pos in range(group_start, group_end):
                    src_positions.append(pos)
                seq_group_count += 1
        compact_cu.append(compact_cu[-1] + seq_group_count * ratio)

    if len(src_positions) > compact_len:
        raise RuntimeError(
            "DSv4 compressor-prep compact produced more tokens than fixed capacity: "
            f"produced={len(src_positions)}, capacity={compact_len}"
        )

    local_src = torch.empty((0,), dtype=torch.long, device=device)
    boundary_src = torch.empty((0,), dtype=torch.long, device=device)
    source_is_boundary = torch.empty((0,), dtype=torch.bool, device=device)
    if src_positions:
        src_global = torch.tensor(src_positions, dtype=torch.long, device=device)
        boundary_src = src_global - ext_start
        local_src = src_global - global_start
        source_is_boundary = (boundary_src >= 0) & (boundary_src < d_window)
        in_local = (local_src >= 0) & (local_src < l_local)
        if not bool((source_is_boundary | in_local).all().item()):
            raise RuntimeError("DSv4 compressor-prep compact source index out of boundary range.")
        if bool(source_is_boundary.any().item()) and d_window < d_comp:
            raise RuntimeError(
                "DSv4 compressor-prep compact requires D_window >= D_comp when "
                "a compressed group reads left-boundary tokens."
            )
        local_src = local_src.clamp(min=0).to(torch.long)
        boundary_src = boundary_src.clamp(min=0).to(torch.long)
    hidden_compact = _CompressorPrepCompact.apply(
        hidden_local,
        boundary_hidden,
        local_src,
        boundary_src,
        source_is_boundary,
        compact_len,
    )

    cu_compact = torch.tensor(compact_cu, dtype=cu_seqlens.dtype, device=device)
    seq_ids_t, comp_ids_t, valid_t = _build_compressor_metadata_for_range(
        cu_seqlens, global_start, l_local, ratio, d_comp, c_cap, device
    )
    return hidden_compact, cu_compact, seq_ids_t, comp_ids_t, valid_t, c_cap


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
    """Build compressor-prep compact rows with the production CuTeDSL kernel."""

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
    if not csa_cp_kernels.can_use_cute_kernels(hidden_local, boundary_hidden, cu_seqlens):
        raise RuntimeError("DSv4 CP compressor-prep compact requires CUDA tensors and CuTeDSL.")
    hidden_compact, cu_compact, seq_ids_t, comp_ids_t, valid_t = _CompressorPrepCompactCute.apply(
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
    return hidden_compact, cu_compact, seq_ids_t, comp_ids_t, valid_t, c_cap


def build_chunked_compressor_prep_compact_fused(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build compressor compact rows for explicit packed-stream CP chunks.

    Args:
        hidden_local: Local hidden rows ordered as ``chunk_ranges``.
        boundary_hidden: One left-boundary window per chunk, concatenated in the
            same order as ``chunk_ranges``.
        cu_seqlens: Global padded THD cumulative sequence lengths.
        chunk_ranges: Global padded-token intervals owned by this rank.
        ratio / d_comp / d_window: DSv4 compressor layout parameters.

    Returns:
        ``(hidden_compact, cu_compact, seq_ids, comp_ids, valid, c_cap, c_cap_per_chunk)``.
        ``c_cap`` is the fixed local compressed capacity for all chunks on this
        rank, and ``c_cap_per_chunk`` is the per-chunk capacity used by
        rank-major metadata reconstruction.
    """

    if not chunk_ranges:
        raise RuntimeError("DSv4 chunked compressor-prep requires at least one chunk range.")
    chunk_lengths = [int(end) - int(start) for start, end in chunk_ranges]
    if any(length <= 0 for length in chunk_lengths):
        raise RuntimeError(f"DSv4 chunked compressor-prep got invalid chunks: {chunk_ranges}.")
    if len(set(chunk_lengths)) != 1:
        raise RuntimeError(
            "DSv4 chunked compressor-prep expects equal-size chunks: "
            f"chunk_lengths={chunk_lengths}."
        )
    l_local = sum(chunk_lengths)
    if hidden_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 chunked compressor-prep expects hidden rows to match chunks: "
            f"hidden={hidden_local.shape[0]}, chunks={l_local}."
        )
    expected_boundary = len(chunk_ranges) * int(d_window)
    if boundary_hidden.shape[0] != expected_boundary:
        raise RuntimeError(
            "DSv4 chunked compressor-prep expects one boundary window per chunk: "
            f"boundary={boundary_hidden.shape[0]}, expected={expected_boundary}."
        )

    compact_parts: List[torch.Tensor] = []
    cu_parts: List[torch.Tensor] = []
    seq_parts: List[torch.Tensor] = []
    comp_parts: List[torch.Tensor] = []
    valid_parts: List[torch.Tensor] = []
    c_caps: List[int] = []
    local_offset = 0
    for chunk_id, ((global_start, _global_end), l_chunk) in enumerate(
        zip(chunk_ranges, chunk_lengths)
    ):
        (
            hidden_compact,
            cu_compact,
            seq_ids,
            comp_ids,
            valid,
            c_cap,
        ) = build_compressor_prep_compact_fused(
            hidden_local.narrow(0, local_offset, l_chunk),
            boundary_hidden.narrow(0, chunk_id * int(d_window), int(d_window)),
            cu_seqlens,
            int(global_start),
            l_chunk,
            ratio,
            d_comp,
            d_window,
        )
        compact_parts.append(hidden_compact)
        cu_parts.append(cu_compact)
        seq_parts.append(seq_ids)
        comp_parts.append(comp_ids)
        valid_parts.append(valid)
        c_caps.append(c_cap)
        local_offset += l_chunk

    if len(set(c_caps)) != 1:
        raise RuntimeError(
            "DSv4 chunked compressor-prep expects equal fixed capacity per chunk: "
            f"c_caps={c_caps}."
        )
    cu_deltas = torch.stack([cu[1:] - cu[:-1] for cu in cu_parts], dim=0).sum(dim=0)
    cu_compact = torch.cat(
        (
            torch.zeros((1,), dtype=cu_seqlens.dtype, device=cu_seqlens.device),
            torch.cumsum(cu_deltas, dim=0),
        )
    )
    c_cap_per_chunk = c_caps[0]
    c_cap = sum(c_caps)
    return (
        torch.cat(compact_parts, dim=0),
        cu_compact,
        torch.cat(seq_parts, dim=0),
        torch.cat(comp_parts, dim=0),
        torch.cat(valid_parts, dim=0),
        c_cap,
        c_cap_per_chunk,
    )


def _all_gather_into_tensor(output: torch.Tensor, tensor: torch.Tensor, cp_group) -> None:
    if hasattr(dist, "all_gather_into_tensor"):
        dist.all_gather_into_tensor(output, tensor, group=cp_group)
        return
    if hasattr(dist, "_all_gather_base"):
        dist._all_gather_base(output, tensor, group=cp_group)
        return
    raise RuntimeError("DSv4 CP fixed all-gather requires all_gather_into_tensor support")


def _reduce_scatter_tensor(output: torch.Tensor, tensor: torch.Tensor, cp_group) -> None:
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(output, tensor, op=dist.ReduceOp.SUM, group=cp_group)
        return
    if hasattr(dist, "_reduce_scatter_base"):
        dist._reduce_scatter_base(output, tensor, op=dist.ReduceOp.SUM, group=cp_group)
        return
    raise RuntimeError("DSv4 CP fixed all-gather backward requires reduce_scatter_tensor support")


class _AllGatherFixedCPTensor(torch.autograd.Function):
    """Fixed-capacity CP all-gather with reduce-scatter backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cp_group) -> torch.Tensor:
        cp_size = cp_group_size(cp_group)
        ctx.cp_group = cp_group
        ctx.input_shape = tuple(tensor.shape)
        ctx.cp_size = cp_size

        if cp_size == 1:
            return tensor

        local = tensor.contiguous()
        output_shape = (cp_size * local.shape[0],) + tuple(local.shape[1:])
        output = local.new_empty(output_shape)
        _all_gather_into_tensor(output, local, cp_group)
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
        _reduce_scatter_tensor(grad_input, grad_output, ctx.cp_group)
        return grad_input, None


def all_gather_fixed_cp_tensor(
    tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Gather fixed-capacity local CP rows into rank-major order.

    The first dimension is static and identical on every CP rank. Forward uses a
    single flat all-gather output buffer; backward is the matching reduce-scatter.
    """

    return _AllGatherFixedCPTensor.apply(tensor, cp_group)


class _KVFullPack(torch.autograd.Function):
    """Pack window and compressed KV rows with explicit backward scatter."""

    @staticmethod
    def forward(
        ctx,
        kv_local: torch.Tensor,
        boundary_kv: torch.Tensor,
        compressed_rank_major: torch.Tensor,
        source_kind: torch.Tensor,
        source_index: torch.Tensor,
        kv_full_capacity: int,
    ) -> torch.Tensor:
        ctx.local_shape = tuple(kv_local.shape)
        ctx.boundary_shape = tuple(boundary_kv.shape)
        ctx.compressed_shape = tuple(compressed_rank_major.shape)
        ctx.save_for_backward(source_kind, source_index)

        kv_full = kv_local.new_zeros((kv_full_capacity,) + tuple(kv_local.shape[1:]))
        if source_kind.numel() == 0:
            return kv_full

        out_pos = torch.arange(source_kind.numel(), device=kv_local.device, dtype=torch.long)
        local_mask = source_kind == 0
        boundary_mask = source_kind == 1
        compressed_mask = source_kind == 2
        if bool(local_mask.any().item()):
            kv_full[out_pos[local_mask]] = kv_local.index_select(0, source_index[local_mask])
        if bool(boundary_mask.any().item()):
            kv_full[out_pos[boundary_mask]] = boundary_kv.index_select(
                0, source_index[boundary_mask]
            )
        if bool(compressed_mask.any().item()):
            kv_full[out_pos[compressed_mask]] = compressed_rank_major.index_select(
                0, source_index[compressed_mask]
            )
        return kv_full

    @staticmethod
    def backward(ctx, grad_kv_full: torch.Tensor):
        source_kind, source_index = ctx.saved_tensors
        grad_local = grad_kv_full.new_zeros(ctx.local_shape)
        grad_boundary = grad_kv_full.new_zeros(ctx.boundary_shape)
        grad_compressed = grad_kv_full.new_zeros(ctx.compressed_shape)
        if source_kind.numel() == 0:
            return grad_local, grad_boundary, grad_compressed, None, None, None

        grad_rows = grad_kv_full[: source_kind.numel()]
        local_mask = source_kind == 0
        boundary_mask = source_kind == 1
        compressed_mask = source_kind == 2
        if bool(local_mask.any().item()):
            grad_local.index_add_(0, source_index[local_mask], grad_rows[local_mask])
        if bool(boundary_mask.any().item()):
            grad_boundary.index_add_(0, source_index[boundary_mask], grad_rows[boundary_mask])
        if bool(compressed_mask.any().item()):
            grad_compressed.index_add_(
                0, source_index[compressed_mask], grad_rows[compressed_mask]
            )
        return grad_local, grad_boundary, grad_compressed, None, None, None


class _KVFullIndexedPack(torch.autograd.Function):
    """Pack KV rows at explicit output positions with explicit backward scatter."""

    @staticmethod
    def forward(
        ctx,
        kv_local: torch.Tensor,
        boundary_kv: torch.Tensor,
        source_kind: torch.Tensor,
        source_index: torch.Tensor,
        out_pos: torch.Tensor,
        kv_full_capacity: int,
    ) -> torch.Tensor:
        ctx.local_shape = tuple(kv_local.shape)
        ctx.boundary_shape = tuple(boundary_kv.shape)
        ctx.save_for_backward(source_kind, source_index, out_pos)

        kv_full = kv_local.new_zeros((kv_full_capacity,) + tuple(kv_local.shape[1:]))
        if source_kind.numel() == 0:
            return kv_full

        local_mask = source_kind == 0
        boundary_mask = source_kind == 1
        if bool(local_mask.any().item()):
            kv_full[out_pos[local_mask]] = kv_local.index_select(0, source_index[local_mask])
        if bool(boundary_mask.any().item()):
            kv_full[out_pos[boundary_mask]] = boundary_kv.index_select(
                0, source_index[boundary_mask]
            )
        return kv_full

    @staticmethod
    def backward(ctx, grad_kv_full: torch.Tensor):
        source_kind, source_index, out_pos = ctx.saved_tensors
        grad_local = grad_kv_full.new_zeros(ctx.local_shape)
        grad_boundary = grad_kv_full.new_zeros(ctx.boundary_shape)
        if source_kind.numel() == 0:
            return grad_local, grad_boundary, None, None, None, None

        grad_rows = grad_kv_full.index_select(0, out_pos)
        local_mask = source_kind == 0
        boundary_mask = source_kind == 1
        if bool(local_mask.any().item()):
            grad_local.index_add_(0, source_index[local_mask], grad_rows[local_mask])
        if bool(boundary_mask.any().item()):
            grad_boundary.index_add_(0, source_index[boundary_mask], grad_rows[boundary_mask])
        return grad_local, grad_boundary, None, None, None, None


def pack_cp_kv_full(
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
) -> Tuple[torch.Tensor, Dict[int, int], Dict[Tuple[int, int], int]]:
    """PyTorch stand-in for the post-all-gather KV full pack kernel.

    Args:
        kv_local: Local uncompressed KV rows, ``(L_local, hidden)``.
        boundary_kv: Left boundary KV rows, ``(D_window, hidden)``.
        compressed_rank_major: Rank-major all-gathered compressed KV rows.
        seq_ids_rank_major / comp_ids_rank_major / valid_rank_major:
            Metadata for each rank-major compressed row.
        cu_seqlens: Global padded THD cumulative sequence lengths.
        global_start: First global padded token owned by this CP rank.
        l_local: Number of padded tokens owned by this CP rank.
        d_window: Left boundary window width.

    Returns:
        ``(kv_full, window_map, compressed_map)``. ``kv_full`` has static
        capacity with tail padding. Its custom backward scatters gradients to
        ``kv_local``, ``boundary_kv``, and ``compressed_rank_major``.
    """
    window_map: Dict[int, int] = {}
    compressed_map: Dict[Tuple[int, int], int] = {}
    global_end = global_start + l_local
    boundary_start = global_start - d_window
    n_sequences = int(cu_seqlens.shape[0]) - 1

    window_capacity = l_local + d_window * n_sequences
    compressed_capacity = int(compressed_rank_major.shape[0])
    kv_full_capacity = max(1, window_capacity + compressed_capacity)

    write_pos = 0
    source_kind: List[int] = []
    source_index: List[int] = []
    for b in range(n_sequences):
        seq_start = int(cu_seqlens[b].item())
        seq_end = int(cu_seqlens[b + 1].item())
        local_seq_start = max(seq_start, global_start)
        local_seq_end = min(seq_end, global_end)
        if local_seq_start >= local_seq_end:
            continue

        window_start = max(seq_start, local_seq_start - d_window)
        window_end = local_seq_end
        for pos in range(window_start, window_end):
            window_map[pos] = write_pos
            if pos < global_start:
                src = pos - boundary_start
                source_kind.append(1)
                source_index.append(src)
            else:
                src = pos - global_start
                source_kind.append(0)
                source_index.append(src)
            write_pos += 1

        valid_indices = [
            i
            for i in range(int(valid_rank_major.shape[0]))
            if bool(valid_rank_major[i].item()) and int(seq_ids_rank_major[i].item()) == b
        ]
        valid_indices.sort(key=lambda i: int(comp_ids_rank_major[i].item()))
        for i in valid_indices:
            comp_id = int(comp_ids_rank_major[i].item())
            compressed_map[(b, comp_id)] = write_pos
            source_kind.append(2)
            source_index.append(i)
            write_pos += 1

    if write_pos > kv_full_capacity:
        raise RuntimeError(
            "DSv4 KV full pack exceeded fixed output capacity: "
            f"written={write_pos}, capacity={kv_full_capacity}"
        )
    source_kind_t = torch.tensor(source_kind, dtype=torch.int8, device=kv_local.device)
    source_index_t = torch.tensor(source_index, dtype=torch.long, device=kv_local.device)
    kv_full = _KVFullPack.apply(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        source_kind_t,
        source_index_t,
        kv_full_capacity,
    )
    return kv_full, window_map, compressed_map


def pack_cp_window_kv_global(
    kv_local: torch.Tensor,
    boundary_kv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Pack window KV at global THD row positions for the window-only CP path."""
    window_map: Dict[int, int] = {}
    global_end = global_start + l_local
    boundary_start = global_start - d_window
    n_sequences = int(cu_seqlens.shape[0]) - 1

    out_pos: List[int] = []
    source_kind: List[int] = []
    source_index: List[int] = []
    for b in range(n_sequences):
        seq_start = int(cu_seqlens[b].item())
        seq_end = int(cu_seqlens[b + 1].item())
        local_seq_start = max(seq_start, global_start)
        local_seq_end = min(seq_end, global_end)
        if local_seq_start >= local_seq_end:
            continue

        window_start = max(seq_start, local_seq_start - d_window)
        window_end = local_seq_end
        for pos in range(window_start, window_end):
            window_map[pos] = pos
            out_pos.append(pos)
            if pos < global_start:
                src = pos - boundary_start
                source_kind.append(1)
                source_index.append(src)
            else:
                src = pos - global_start
                source_kind.append(0)
                source_index.append(src)

    source_kind_t = torch.tensor(source_kind, dtype=torch.int8, device=kv_local.device)
    source_index_t = torch.tensor(source_index, dtype=torch.long, device=kv_local.device)
    out_pos_t = torch.tensor(out_pos, dtype=torch.long, device=kv_local.device)
    kv_full = _KVFullIndexedPack.apply(
        kv_local,
        boundary_kv,
        source_kind_t,
        source_index_t,
        out_pos_t,
        max(1, int(cu_seqlens[-1].item())),
    )
    return kv_full, window_map


def cp_kv_full_capacity(
    cu_seqlens: torch.Tensor, l_local: int, d_window: int, compressed_rows: int
) -> int:
    """Return the fixed upper-bound KV full capacity for fused CP packing."""

    return csa_cp_kernels.kv_full_capacity(cu_seqlens, l_local, d_window, compressed_rows)


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
    """Pack local window rows and rank-major compressed rows without Python maps.

    Args:
        kv_local: Local KV rows, ``(L_local, ...)``.
        boundary_kv: Fixed left boundary KV rows, ``(D_window, ...)``.
        compressed_rank_major: All-gathered compressed rows in rank-major order.
        seq_ids_rank_major / comp_ids_rank_major / valid_rank_major:
            Rank-major compressed metadata.
        cu_seqlens: Global padded THD cumulative sequence lengths.
        global_start: First global padded token owned by this rank.
        l_local: Local padded token capacity.
        d_window: Left boundary window width.
        ratio: Compression ratio. Use ``0`` or ``1`` for window-only layouts.

    Returns:
        Static-capacity ``kv_full`` laid out per active sequence as
        ``[window rows, compressed rows]`` with tail padding.
    """

    if not csa_cp_kernels.can_use_cute_kernels(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        rank_major_by_seq_major,
        cu_seqlens_compressed,
    ):
        raise RuntimeError("Fused CSA CP KV pack requires CUDA tensors and CuTe DSL kernels.")
    capacity = cp_kv_full_capacity(
        cu_seqlens, l_local, d_window, int(compressed_rank_major.shape[0])
    )
    return csa_cp_kernels.pack_kv_full(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        ratio,
        capacity,
        rank_major_by_seq_major,
        cu_seqlens_compressed,
    )


def pack_chunked_cp_kv_full_fused(
    kv_local: torch.Tensor,
    boundary_kv: torch.Tensor,
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    d_window: int,
    ratio: int,
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pack chunked local windows into one sparse-attention row space.

    When rank-major compressed metadata is available, the packed layout is
    ``[chunk0 window, chunk1 window, shared compressed]`` so that compressed KV
    rows are stored once and all chunk rows can still use one top-k index tensor.
    Without that metadata this falls back to concatenating one full ``kv_full``
    layout per chunk.
    """

    if not chunk_ranges:
        raise RuntimeError("DSv4 chunked KV pack requires at least one chunk range.")
    chunk_lengths = [int(end) - int(start) for start, end in chunk_ranges]
    if any(length <= 0 for length in chunk_lengths):
        raise RuntimeError(f"DSv4 chunked KV pack got invalid chunks: {chunk_ranges}.")
    l_local = sum(chunk_lengths)
    if kv_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 chunked KV pack expects local KV rows to match chunks: "
            f"kv_local={kv_local.shape[0]}, chunks={l_local}."
        )
    expected_boundary = len(chunk_ranges) * int(d_window)
    if boundary_kv.shape[0] != expected_boundary:
        raise RuntimeError(
            "DSv4 chunked KV pack expects one boundary window per chunk: "
            f"boundary={boundary_kv.shape[0]}, expected={expected_boundary}."
        )
    if (
        ratio > 1
        and compressed_rank_major.shape[0] > 0
        and rank_major_by_seq_major is not None
        and cu_seqlens_compressed is not None
    ):
        return csa_cp_kernels.pack_chunked_shared_kv_full(
            kv_local,
            boundary_kv,
            compressed_rank_major,
            cu_seqlens,
            tuple(int(start) for start, _end in chunk_ranges),
            chunk_lengths[0],
            d_window,
        )[:2]

    outputs: List[torch.Tensor] = []
    offsets: List[int] = []
    local_offset = 0
    kv_offset = 0
    for chunk_id, ((global_start, _global_end), l_chunk) in enumerate(
        zip(chunk_ranges, chunk_lengths)
    ):
        offsets.append(kv_offset)
        kv_chunk = pack_cp_kv_full_fused(
            kv_local.narrow(0, local_offset, l_chunk),
            boundary_kv.narrow(0, chunk_id * int(d_window), int(d_window)),
            compressed_rank_major,
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens,
            int(global_start),
            l_chunk,
            d_window,
            ratio,
            rank_major_by_seq_major=rank_major_by_seq_major,
            cu_seqlens_compressed=cu_seqlens_compressed,
        )
        outputs.append(kv_chunk)
        local_offset += l_chunk
        kv_offset += kv_chunk.shape[0]

    if len(outputs) == 1:
        return outputs[0], tuple(offsets)
    return torch.cat(outputs, dim=0), tuple(offsets)


def repack_rank_major_compressed_to_seq_major(
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    output_capacity: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rank-major compressed rows into global seq-major order.

    This is a local PyTorch lowering helper, not a communication op. It exists
    so the current THD indexer top-k kernel can consume seq-major compressed K
    while the CP communication contract remains rank-major.

    Returns:
        ``(compressed_seq_major, rank_major_by_seq_major)`` where the second
        tensor maps a seq-major compressed row id back to its rank-major row id.
    """
    total_comp = (
        output_capacity if output_capacity is not None else int(cu_seqlens_compressed[-1].item())
    )
    if csa_cp_kernels.can_use_cute_kernels(
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens_compressed,
    ):
        return repack_rank_major_compressed_to_seq_major_fused(
            compressed_rank_major,
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens_compressed,
            output_capacity=total_comp,
        )

    compressed_seq_major = compressed_rank_major.new_zeros(
        (total_comp,) + tuple(compressed_rank_major.shape[1:])
    )
    rank_major_by_seq_major = torch.full(
        (total_comp,), -1, dtype=torch.int32, device=compressed_rank_major.device
    )
    if total_comp == 0:
        return compressed_seq_major, rank_major_by_seq_major

    for rank_major_id in range(int(valid_rank_major.shape[0])):
        if not bool(valid_rank_major[rank_major_id].item()):
            continue
        seq_id = int(seq_ids_rank_major[rank_major_id].item())
        comp_id = int(comp_ids_rank_major[rank_major_id].item())
        if seq_id < 0 or comp_id < 0:
            continue
        seq_major_id = int(cu_seqlens_compressed[seq_id].item()) + comp_id
        if 0 <= seq_major_id < total_comp:
            compressed_seq_major[seq_major_id] = compressed_rank_major[rank_major_id]
            rank_major_by_seq_major[seq_major_id] = rank_major_id

    check_rows = (
        int(cu_seqlens_compressed[-1].item()) if output_capacity is not None else total_comp
    )
    if bool((rank_major_by_seq_major[:check_rows] < 0).any().item()):
        raise RuntimeError("DSv4 rank-major compressed metadata did not cover all valid rows.")
    return compressed_seq_major, rank_major_by_seq_major


def repack_rank_major_compressed_to_seq_major_fused(
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    output_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rank-major compressed rows with the production CuTeDSL kernel."""

    if not csa_cp_kernels.can_use_cute_kernels(
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens_compressed,
    ):
        raise RuntimeError("DSv4 CP rank-major repack requires CUDA tensors and CuTeDSL kernels.")
    return csa_cp_kernels.repack_rank_major_compressed_to_seq_major(
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens_compressed,
        output_capacity,
    )


def build_cp_indexer_topk_inputs(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    """Build CP-local packed inputs for the THD indexer top-k kernel.

    Args:
        q_indexer_local: Local indexer Q rows, ``(L_local, index_n_heads, index_head_dim)``.
        weights_indexer_local: Local indexer weights, ``(L_local, index_n_heads)``.
        k_indexer_seq_major: Global seq-major compressed indexer K rows after all-gather.
        cu_seqlens_q: Global padded THD cumulative Q lengths.
        cu_seqlens_compressed: Global seq-major compressed cumulative K lengths.
        global_start: First global padded token owned by this CP rank.
        l_local: Local padded token capacity.
        ratio: Compression ratio used by the indexer causal mask.
        max_seqlen_q / max_seqlen_kv: Static upper bounds passed to the THD
            indexer kernel. Production callers pass these from packed metadata
            instead of deriving dynamic maxima from GPU tensors.

    Returns:
        ``(q_topk, k_topk, weights_topk, cu_q_topk, cu_k_topk, max_q, max_k, local_row_ids)``.
        ``q_topk`` and ``weights_topk`` keep fixed ``L_local`` row capacity.
        ``k_topk`` keeps the fixed global compressed capacity and packs only
        the compressed prefixes visible to this CP rank. ``cu_q_topk`` and
        ``cu_k_topk`` have matching fixed segment counts but may describe
        different per-segment lengths, which represents the trapezoid causal
        mask for a sequence piece cut by CP. ``local_row_ids`` is the identity
        map because rows are no longer dynamically dropped.
    """
    if q_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects q_indexer_local length to match L_local: "
            f"q={q_indexer_local.shape[0]}, L_local={l_local}"
        )
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights_indexer_local length to match L_local: "
            f"weights={weights_indexer_local.shape[0]}, L_local={l_local}"
        )

    device = q_indexer_local.device
    if csa_cp_kernels.can_use_cute_kernels(
        q_indexer_local,
        weights_indexer_local,
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
    ):
        k_topk, cu_q_topk, cu_k_topk, _seq_lens = csa_cp_kernels.build_indexer_topk_inputs(
            k_indexer_seq_major,
            cu_seqlens_q,
            cu_seqlens_compressed,
            global_start,
            l_local,
            ratio,
        )
        local_row_ids = torch.arange(l_local, dtype=torch.long, device=device)
        max_q = int(max_seqlen_q) if max_seqlen_q is not None else l_local
        max_k = (
            int(max_seqlen_kv)
            if max_seqlen_kv is not None
            else max(1, k_indexer_seq_major.shape[0])
        )
        return (
            q_indexer_local,
            k_topk,
            weights_indexer_local,
            cu_q_topk,
            cu_k_topk,
            max_q,
            max_k,
            local_row_ids,
        )

    global_end = global_start + l_local
    cu_q = [0]
    cu_k = [0]
    max_q = 0
    max_k = 0
    k_topk = torch.zeros_like(k_indexer_seq_major)
    k_write = 0

    for seq_id in range(int(cu_seqlens_q.shape[0]) - 1):
        seq_start = int(cu_seqlens_q[seq_id].item())
        seq_end = int(cu_seqlens_q[seq_id + 1].item())
        local_seq_start = max(seq_start, global_start)
        local_seq_end = min(seq_end, global_end)
        q_len = max(0, local_seq_end - local_seq_start)
        seq_comp_start = int(cu_seqlens_compressed[seq_id].item())
        seq_comp_end = int(cu_seqlens_compressed[seq_id + 1].item())
        k_len = 0
        if q_len > 0:
            k_len = min((local_seq_end - seq_start) // ratio, seq_comp_end - seq_comp_start)
        if k_len > 0:
            k_topk[k_write : k_write + k_len] = k_indexer_seq_major[
                seq_comp_start : seq_comp_start + k_len
            ]
            k_write += k_len
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)
        max_q = max(max_q, q_len)
        max_k = max(max_k, k_len)

    actual_total = int(cu_seqlens_q[-1].item())
    padding_start = max(actual_total, global_start)
    padding_q = max(0, global_end - padding_start)
    cu_q.append(cu_q[-1] + padding_q)
    cu_k.append(cu_k[-1])
    max_q = max(max_q, padding_q)

    cu_q_topk = torch.tensor(cu_q, dtype=cu_seqlens_q.dtype, device=device)
    cu_k_topk = torch.tensor(cu_k, dtype=cu_seqlens_compressed.dtype, device=device)
    local_row_ids = torch.arange(l_local, dtype=torch.long, device=device)
    return (
        q_indexer_local,
        k_topk,
        weights_indexer_local,
        cu_q_topk,
        cu_k_topk,
        int(max_seqlen_q) if max_seqlen_q is not None else max_q,
        int(max_seqlen_kv) if max_seqlen_kv is not None else max_k,
        local_row_ids,
    )


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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
]:
    """Build fixed CP-local indexer top-k inputs with production CuTeDSL kernels."""

    if q_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects q_indexer_local length to match L_local: "
            f"q={q_indexer_local.shape[0]}, L_local={l_local}"
        )
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights_indexer_local length to match L_local: "
            f"weights={weights_indexer_local.shape[0]}, L_local={l_local}"
        )
    if not csa_cp_kernels.can_use_cute_kernels(
        q_indexer_local,
        weights_indexer_local,
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
    ):
        raise RuntimeError("DSv4 CP indexer top-k input pack requires CUDA tensors and CuTeDSL.")

    k_topk, cu_q_topk, cu_k_topk, seq_lens = csa_cp_kernels.build_indexer_topk_inputs(
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
        global_start,
        l_local,
        ratio,
    )
    max_q = int(max_seqlen_q) if max_seqlen_q is not None else l_local
    max_k = int(max_seqlen_kv) if max_seqlen_kv is not None else max(1, k_indexer_seq_major.shape[0])
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


def compute_chunked_cp_indexer_topk_logical_fused(
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
    """Run THD CP indexer top-k on each contiguous local chunk.

    Packed-stream two-chunk CP gives one rank rows from two global token intervals. The
    current cuDNN indexer forward/top-k kernel builds its causal/trapezoid mask
    from ``cu_seqlens_q`` and ``cu_seqlens_kv`` and therefore must see one
    contiguous global interval per call.  Sparse attention does not have this
    restriction because it consumes explicit ``topk_idxs``; this helper only
    splits indexer top-k and returns rows concatenated in local chunk order.
    """

    topk_width = int(topk_width)
    if topk_width < 0:
        raise RuntimeError(f"DSv4 CP chunked indexer top-k got negative width: {topk_width}.")
    if ratio <= 1:
        raise RuntimeError(f"DSv4 CP chunked indexer top-k expects ratio > 1, got {ratio}.")
    if not chunk_ranges:
        raise RuntimeError("DSv4 CP chunked indexer top-k requires at least one chunk range.")

    chunk_lengths = [int(end) - int(start) for start, end in chunk_ranges]
    if any(length <= 0 for length in chunk_lengths):
        raise RuntimeError(f"DSv4 CP chunked indexer top-k got invalid chunks: {chunk_ranges}.")
    l_local = sum(chunk_lengths)
    if q_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP chunked indexer top-k expects q_indexer_local rows to match chunks: "
            f"q={q_indexer_local.shape[0]}, chunks={l_local}."
        )
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP chunked indexer top-k expects weights rows to match chunks: "
            f"weights={weights_indexer_local.shape[0]}, chunks={l_local}."
        )
    if not csa_cp_kernels.can_use_cute_kernels(
        q_indexer_local,
        weights_indexer_local,
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
    ):
        raise RuntimeError("DSv4 CP chunked indexer top-k requires CUDA tensors and CuTeDSL.")

    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return torch.empty((l_local, 0), dtype=torch.int32, device=q_indexer_local.device)

    chunk_outputs: List[torch.Tensor] = []
    local_offset = 0
    for (global_start, _global_end), l_chunk in zip(chunk_ranges, chunk_lengths):
        q_chunk = q_indexer_local.narrow(0, local_offset, l_chunk)
        weights_chunk = weights_indexer_local.narrow(0, local_offset, l_chunk)
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
            int(global_start),
            l_chunk,
            ratio,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        topk_compute_width = min(topk_width, max_k_topk)
        if topk_compute_width <= 0:
            topk_chunk = torch.empty((l_chunk, 0), dtype=torch.int32, device=q_indexer_local.device)
        else:
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
            topk_chunk = pad_indexer_topk_to_fixed_width_fused(topk_chunk, topk_width)
        chunk_outputs.append(topk_chunk)
        local_offset += l_chunk

    if len(chunk_outputs) == 1:
        return chunk_outputs[0]
    return torch.cat(chunk_outputs, dim=0)


def pad_indexer_topk_to_fixed_width_fused(
    topk_indices: torch.Tensor,
    fixed_width: int,
) -> torch.Tensor:
    """Pad CP-local indexer top-k ids to the static production width."""
    if not csa_cp_kernels.can_use_cute_kernels(topk_indices):
        raise RuntimeError("DSv4 CP top-k index padding requires CUDA tensors and CuTeDSL.")
    return csa_cp_kernels.pad_topk_indices(topk_indices, int(fixed_width))


def map_cp_topk_logical_to_rank_major(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    indexer_topk_logical: torch.Tensor,
    rank_major_by_seq_major: torch.Tensor,
) -> torch.Tensor:
    """Map per-sequence logical compressed top-k ids to rank-major row ids."""
    out = torch.full_like(indexer_topk_logical, -1)
    if indexer_topk_logical.numel() == 0:
        return out
    batch_all = batch_of_row(cu_seqlens_q, total_q=int(cu_seqlens_q[-1].item()))
    for row in range(l_local):
        global_q = global_start + row
        if global_q >= int(cu_seqlens_q[-1].item()):
            continue
        seq_id = int(batch_all[global_q].item())
        seq_comp_start = int(cu_seqlens_compressed[seq_id].item())
        seq_comp_end = int(cu_seqlens_compressed[seq_id + 1].item())
        for col, comp_id_t in enumerate(indexer_topk_logical[row]):
            comp_id = int(comp_id_t.item())
            if comp_id < 0:
                continue
            seq_major_id = seq_comp_start + comp_id
            if seq_major_id < seq_comp_end:
                out[row, col] = rank_major_by_seq_major[seq_major_id]
    return out


def build_cp_flat_idxs(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    window_size: int,
    ratio: int,
    device: torch.device,
    window_map: Dict[int, int],
    compressed_map: Dict[Tuple[int, int], int],
    indexer_topk_compressed_logical_ids: Optional[torch.Tensor] = None,
    max_n_compressed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch stand-in for final CP+THD idx generation/lowering."""
    compressed_width = 0
    if ratio > 1:
        if indexer_topk_compressed_logical_ids is not None:
            compressed_width = indexer_topk_compressed_logical_ids.shape[-1]
        else:
            compressed_width = max_n_compressed
    out = torch.full(
        (l_local, window_size + compressed_width), -1, dtype=torch.int32, device=device
    )
    topk_length = torch.zeros((l_local,), dtype=torch.int32, device=device)
    batch_all = batch_of_row(cu_seqlens, total_q=int(cu_seqlens[-1].item()))

    for row in range(l_local):
        global_q = global_start + row
        if global_q >= int(cu_seqlens[-1].item()):
            continue
        seq_id = int(batch_all[global_q].item())
        seq_start = int(cu_seqlens[seq_id].item())
        pos_in_seq = global_q - seq_start
        w_start = max(seq_start, global_q - window_size + 1)
        w_positions = list(range(w_start, global_q + 1))
        w_positions = w_positions[-window_size:]
        write_col = 0
        for pos in w_positions:
            flat = window_map.get(pos)
            if flat is not None:
                out[row, write_col] = flat
                write_col += 1

        if compressed_width == 0:
            topk_length[row] = write_col
            continue
        if indexer_topk_compressed_logical_ids is not None:
            comp_ids = indexer_topk_compressed_logical_ids[row].tolist()
        else:
            n_visible = min((pos_in_seq + 1) // ratio, max_n_compressed)
            comp_ids = list(range(n_visible)) + [-1] * (compressed_width - n_visible)
        for comp_id in comp_ids[:compressed_width]:
            if comp_id < 0:
                continue
            flat = compressed_map.get((seq_id, int(comp_id)))
            if flat is not None:
                out[row, write_col] = flat
                write_col += 1
        topk_length[row] = write_col
    return out, topk_length


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
    """Generate/lower CP final sparse-attention ids without Python maps.

    Returns:
        ``(topk_idxs, topk_length)`` where ``topk_idxs`` is already in the
        fused ``kv_full`` flat row space.
    """

    if indexer_topk_compressed_logical_ids is not None:
        compressed_width = indexer_topk_compressed_logical_ids.shape[-1]
    elif ratio > 1:
        compressed_width = max_n_compressed
    else:
        compressed_width = 0
    if not csa_cp_kernels.can_use_cute_kernels(
        cu_seqlens, indexer_topk_compressed_logical_ids
    ):
        raise RuntimeError("Fused CSA CP final idx generation requires CUDA tensors and CuTe DSL.")
    return csa_cp_kernels.build_final_idxs(
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        indexer_topk_compressed_logical_ids,
    )


def _offset_valid_topk_idxs(topk_idxs: torch.Tensor, kv_offset: int) -> torch.Tensor:
    if kv_offset == 0:
        return topk_idxs
    return torch.where(topk_idxs >= 0, topk_idxs + int(kv_offset), topk_idxs)


def build_chunked_cp_flat_idxs_fused(
    cu_seqlens: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    kv_full_offsets: Sequence[int],
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: Optional[torch.Tensor] = None,
    max_n_compressed: int = 0,
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    shared_compressed_base: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sparse-attention ids for a concatenated chunked ``kv_full`` row space."""

    if len(chunk_ranges) != len(kv_full_offsets):
        raise RuntimeError(
            "DSv4 chunked final idx expects one KV offset per chunk: "
            f"chunks={len(chunk_ranges)}, offsets={len(kv_full_offsets)}."
        )
    chunk_lengths = [int(end) - int(start) for start, end in chunk_ranges]
    if any(length <= 0 for length in chunk_lengths):
        raise RuntimeError(f"DSv4 chunked final idx got invalid chunks: {chunk_ranges}.")
    l_local = sum(chunk_lengths)
    if (
        indexer_topk_compressed_logical_ids is not None
        and indexer_topk_compressed_logical_ids.shape[0] != l_local
    ):
        raise RuntimeError(
            "DSv4 chunked final idx expects indexer rows to match chunks: "
            f"indexer={indexer_topk_compressed_logical_ids.shape[0]}, chunks={l_local}."
        )
    if (
        ratio > 1
        and shared_compressed_base is not None
        and rank_major_by_seq_major is not None
        and cu_seqlens_compressed is not None
    ):
        if len(set(chunk_lengths)) != 1:
            raise RuntimeError(
                "DSv4 shared-compressed chunked final idx expects equal-size chunks: "
                f"chunk_lengths={chunk_lengths}."
            )
        window_capacity = int(shared_compressed_base) // len(chunk_ranges)
        compressed_width = (
            int(indexer_topk_compressed_logical_ids.shape[-1])
            if indexer_topk_compressed_logical_ids is not None
            else int(max_n_compressed)
        )
        return csa_cp_kernels.build_chunked_shared_final_idxs(
            cu_seqlens,
            cu_seqlens_compressed,
            rank_major_by_seq_major,
            tuple(int(start) for start, _end in chunk_ranges),
            chunk_lengths[0],
            window_capacity,
            int(shared_compressed_base),
            d_window,
            window_size,
            ratio,
            compressed_width,
            indexer_topk_compressed_logical_ids,
        )

    topk_chunks: List[torch.Tensor] = []
    length_chunks: List[torch.Tensor] = []
    local_offset = 0
    for (global_start, _global_end), l_chunk, kv_offset in zip(
        chunk_ranges, chunk_lengths, kv_full_offsets
    ):
        indexer_chunk = (
            None
            if indexer_topk_compressed_logical_ids is None
            else indexer_topk_compressed_logical_ids.narrow(0, local_offset, l_chunk)
        )
        topk_chunk, length_chunk = build_cp_flat_idxs_fused(
            cu_seqlens,
            int(global_start),
            l_chunk,
            d_window,
            window_size,
            ratio,
            indexer_chunk,
            max_n_compressed=max_n_compressed,
        )
        topk_chunks.append(_offset_valid_topk_idxs(topk_chunk, int(kv_offset)))
        length_chunks.append(length_chunk)
        local_offset += l_chunk

    if len(topk_chunks) == 1:
        return topk_chunks[0], length_chunks[0]
    return torch.cat(topk_chunks, dim=0), torch.cat(length_chunks, dim=0)


def build_cp_flat_idxs_for_indexer_loss(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    window_size: int,
    device: torch.device,
    window_map: Dict[int, int],
    compressed_map: Dict[Tuple[int, int], int],
    indexer_topk_compressed_logical_ids: torch.Tensor,
    indexer_topk_rank_major_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build non-compact Path-B ids: compressed top-k first, then window ids."""
    compressed_width = indexer_topk_compressed_logical_ids.shape[-1]
    out = torch.full(
        (l_local, compressed_width + window_size), -1, dtype=torch.int32, device=device
    )
    indexer_rank_major = torch.full(
        (l_local, compressed_width), -1, dtype=torch.int32, device=device
    )
    batch_all = batch_of_row(cu_seqlens, total_q=int(cu_seqlens[-1].item()))

    for row in range(l_local):
        global_q = global_start + row
        if global_q >= int(cu_seqlens[-1].item()):
            continue
        seq_id = int(batch_all[global_q].item())
        seq_start = int(cu_seqlens[seq_id].item())

        comp_ids = indexer_topk_compressed_logical_ids[row].tolist()
        comp_rank_major_ids = indexer_topk_rank_major_ids[row].tolist()
        for j, comp_id in enumerate(comp_ids[:compressed_width]):
            if comp_id < 0 or int(comp_rank_major_ids[j]) < 0:
                continue
            flat = compressed_map.get((seq_id, int(comp_id)))
            if flat is not None:
                out[row, j] = flat
                indexer_rank_major[row, j] = int(comp_rank_major_ids[j])

        w_start = max(seq_start, global_q - window_size + 1)
        w_positions = list(range(w_start, global_q + 1))[-window_size:]
        for j, pos in enumerate(w_positions):
            flat = window_map.get(pos)
            if flat is not None:
                out[row, compressed_width + j] = flat
    return out, indexer_rank_major


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
    """Generate Path-B ids and rank-major compressed ids in one fused kernel.

    The first returned tensor lowers sparse attention indices into fused
    ``kv_full`` layout. The second tensor maps the compressed top-k columns to
    rank-major indexer-K rows for the sparse indexer-loss kernel.
    """

    if not csa_cp_kernels.can_use_cute_kernels(
        cu_seqlens,
        cu_seqlens_compressed,
        indexer_topk_compressed_logical_ids,
        indexer_rank_by_seq_major,
    ):
        raise RuntimeError(
            "Fused CSA CP indexer-loss idx generation requires CUDA tensors and CuTe DSL."
        )
    return csa_cp_kernels.build_indexer_loss_final_idxs(
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


def build_chunked_cp_flat_idxs_for_indexer_loss_fused(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    chunk_ranges: Sequence[Tuple[int, int]],
    kv_full_offsets: Sequence[int],
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: torch.Tensor,
    indexer_rank_by_seq_major: torch.Tensor,
    shared_compressed_base: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build Path-B ids for one sparse-attention call over chunked local rows."""

    if len(chunk_ranges) != len(kv_full_offsets):
        raise RuntimeError(
            "DSv4 chunked indexer-loss idx expects one KV offset per chunk: "
            f"chunks={len(chunk_ranges)}, offsets={len(kv_full_offsets)}."
        )
    chunk_lengths = [int(end) - int(start) for start, end in chunk_ranges]
    if any(length <= 0 for length in chunk_lengths):
        raise RuntimeError(f"DSv4 chunked indexer-loss idx got invalid chunks: {chunk_ranges}.")
    l_local = sum(chunk_lengths)
    if indexer_topk_compressed_logical_ids.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 chunked indexer-loss idx expects indexer rows to match chunks: "
            f"indexer={indexer_topk_compressed_logical_ids.shape[0]}, chunks={l_local}."
        )
    if shared_compressed_base is not None:
        if len(set(chunk_lengths)) != 1:
            raise RuntimeError(
                "DSv4 shared-compressed chunked indexer-loss idx expects equal-size chunks: "
                f"chunk_lengths={chunk_lengths}."
            )
        window_capacity = int(shared_compressed_base) // len(chunk_ranges)
        return csa_cp_kernels.build_chunked_shared_indexer_loss_final_idxs(
            cu_seqlens,
            cu_seqlens_compressed,
            indexer_topk_compressed_logical_ids,
            indexer_rank_by_seq_major,
            tuple(int(start) for start, _end in chunk_ranges),
            chunk_lengths[0],
            window_capacity,
            int(shared_compressed_base),
            d_window,
            window_size,
            ratio,
        )

    topk_chunks: List[torch.Tensor] = []
    rank_major_chunks: List[torch.Tensor] = []
    local_offset = 0
    for (global_start, _global_end), l_chunk, kv_offset in zip(
        chunk_ranges, chunk_lengths, kv_full_offsets
    ):
        topk_chunk, rank_major_chunk = build_cp_flat_idxs_for_indexer_loss_fused(
            cu_seqlens,
            cu_seqlens_compressed,
            int(global_start),
            l_chunk,
            d_window,
            window_size,
            ratio,
            indexer_topk_compressed_logical_ids.narrow(0, local_offset, l_chunk),
            indexer_rank_by_seq_major,
        )
        topk_chunks.append(_offset_valid_topk_idxs(topk_chunk, int(kv_offset)))
        rank_major_chunks.append(rank_major_chunk)
        local_offset += l_chunk

    if len(topk_chunks) == 1:
        return topk_chunks[0], rank_major_chunks[0]
    return torch.cat(topk_chunks, dim=0), torch.cat(rank_major_chunks, dim=0)

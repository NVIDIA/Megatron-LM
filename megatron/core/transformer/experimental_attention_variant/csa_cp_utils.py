# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from megatron.core.transformer.experimental_attention_variant.dsa_kernels import batch_of_row


class _SingleRankCPGroup:
    """Small adapter for applying THD RoPE to already-global packed tensors."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


_SINGLE_RANK_CP_GROUP = _SingleRankCPGroup()


def cp_debug_trace(message: str) -> None:
    if not os.environ.get("DSV4_CP_DEBUG_TRACE"):
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "?"
    print(f"[dsv4-cp rank={rank}] {message}", flush=True)


def cp_group_size(cp_group: Optional[torch.distributed.ProcessGroup]) -> int:
    if cp_group is None:
        return 1
    return cp_group.size()


def cp_group_rank(cp_group: Optional[torch.distributed.ProcessGroup]) -> int:
    if cp_group is None:
        return 0
    return cp_group.rank()


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
        cp_debug_trace(f"boundary forward start D={d_window} local={tensor.shape[0]}")
        ops = []
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, recv, _group_peer(cp_group, cp_rank - 1), cp_group))
        if cp_rank + 1 < cp_size:
            ops.append(dist.P2POp(dist.isend, send, _group_peer(cp_group, cp_rank + 1), cp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        cp_debug_trace("boundary forward done")
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
        cp_debug_trace(f"boundary backward start D={d_window}")
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
        cp_debug_trace("boundary backward done")
        return grad_input, None, None


def exchange_left_boundary_tensor(
    tensor: torch.Tensor,
    d_window: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Return the fixed left boundary tensor for this CP rank."""
    return _LeftBoundaryExchange.apply(tensor, d_window, cp_group)


class _RouteBoundaryKVGradToLocal(torch.autograd.Function):
    """Route boundary-KV gradients back to the rank that owns the local tail."""

    @staticmethod
    def forward(
        ctx,
        kv_local: torch.Tensor,
        boundary_kv: torch.Tensor,
        d_window: int,
        cp_group: torch.distributed.ProcessGroup,
    ):
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.local_shape = tuple(kv_local.shape)
        ctx.boundary_shape = tuple(boundary_kv.shape)
        return boundary_kv

    @staticmethod
    def backward(ctx, grad_boundary_kv: torch.Tensor):
        cp_group = ctx.cp_group
        cp_size = cp_group_size(cp_group)
        cp_rank = cp_group_rank(cp_group)
        d_window = ctx.d_window

        grad_local = grad_boundary_kv.new_zeros(ctx.local_shape)
        grad_boundary = grad_boundary_kv.new_zeros(ctx.boundary_shape)
        if cp_size <= 1:
            return grad_local, grad_boundary, None, None

        send = grad_boundary_kv.contiguous()
        recv = grad_boundary_kv.new_zeros(send.shape)
        cp_debug_trace(f"boundary kv-grad route start D={d_window}")
        ops = []
        if cp_rank + 1 < cp_size:
            ops.append(dist.P2POp(dist.irecv, recv, _group_peer(cp_group, cp_rank + 1), cp_group))
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.isend, send, _group_peer(cp_group, cp_rank - 1), cp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if cp_rank + 1 < cp_size:
            grad_local[-d_window:] += recv
        cp_debug_trace("boundary kv-grad route done")
        return grad_local, grad_boundary, None, None


def route_boundary_kv_grad_to_local(
    kv_local: torch.Tensor,
    boundary_kv: torch.Tensor,
    d_window: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Use boundary KV in forward and route its backward grad to local tail KV."""
    return _RouteBoundaryKVGradToLocal.apply(kv_local, boundary_kv, d_window, cp_group)


def contiguous_cp_partition(
    cu_seqlens_padded: torch.Tensor, cp_size: int, cp_rank: int
) -> Tuple[int, int]:
    """Return the global padded-token partition assigned to a contiguous CP rank.

    Args:
        cu_seqlens_padded: Global padded THD cumulative sequence lengths. The
            last entry is the padded total token count that MCore partitions
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


def build_global_compressed_cu_seqlens(
    cu_seqlens_padded: torch.Tensor, ratio: int
) -> torch.Tensor:
    """Return global seq-major compressed cumulative lengths for ``ratio``."""
    seq_lens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    compressed_lens = seq_lens // ratio
    return torch.cat(
        [
            torch.zeros(1, dtype=cu_seqlens_padded.dtype, device=cu_seqlens_padded.device),
            compressed_lens.cumsum(0).to(cu_seqlens_padded.dtype),
        ]
    )


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


def pad_compressed_to_capacity(
    compressed: Optional[torch.Tensor],
    c_cap: int,
    head_dim: int,
    like: torch.Tensor,
    module: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Pad compressor output to the fixed local compressed capacity.

    Args:
        compressed: Compressor output ``(n_valid, 1, head_dim)`` or ``None``.
        c_cap: Fixed local compressed row capacity.
        head_dim: Last-dimension size of the compressed rows.
        like: Tensor providing device/dtype.
        module: Optional module whose parameters receive a zero dependency
            when ``compressed`` is ``None``.

    Returns:
        ``(c_cap, 1, head_dim)`` with valid rows at the front and tail zeros.
    """
    fixed = like.new_zeros((c_cap, 1, head_dim))
    if compressed is not None and compressed.shape[0] > 0:
        if compressed.shape[0] > c_cap:
            raise RuntimeError(
                "DSv4 local compressed output exceeded fixed capacity: "
                f"compressed={compressed.shape[0]}, capacity={c_cap}"
            )
        fixed[: compressed.shape[0]] = compressed
    elif module is not None:
        fixed = fixed + zero_module_parameter_dependency(module, fixed)
    return fixed


def all_gather_fixed_cp_tensor(
    tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    return torch.cat(differentiable_all_gather(tensor.contiguous(), group=cp_group), dim=0)


def zero_module_parameter_dependency(module: nn.Module, like: torch.Tensor) -> torch.Tensor:
    token = like.new_tensor(0.0)
    for param in module.parameters():
        token = token + param.sum().to(dtype=like.dtype) * like.new_tensor(0.0)
    return token


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


def repack_rank_major_compressed_to_seq_major(
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rank-major compressed rows into global seq-major order.

    This is a local PyTorch lowering helper, not a communication op. It exists
    so the current THD indexer top-k kernel can consume seq-major compressed K
    while the CP communication contract remains rank-major.

    Returns:
        ``(compressed_seq_major, rank_major_by_seq_major)`` where the second
        tensor maps a seq-major compressed row id back to its rank-major row id.
    """
    total_comp = int(cu_seqlens_compressed[-1].item())
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

    if bool((rank_major_by_seq_major < 0).any().item()):
        raise RuntimeError("DSv4 rank-major compressed metadata did not cover all valid rows.")
    return compressed_seq_major, rank_major_by_seq_major


def build_cp_indexer_topk_inputs(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
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

    Returns:
        ``(q_topk, k_topk, weights_topk, cu_q_topk, cu_k_topk, max_q, max_k, local_row_ids)``.
        ``cu_q_topk`` and ``cu_k_topk`` have matching segment counts but may
        describe different per-segment lengths.  That lets the THD indexer
        kernel represent the trapezoid causal mask for a sequence piece cut by
        CP.  ``local_row_ids`` maps rows returned by the kernel back to
        ``[0, L_local)``. Rows with no visible compressed KV are omitted and
        should be filled with ``-1`` by the caller.
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
    global_end = global_start + l_local
    q_parts: List[torch.Tensor] = []
    k_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    local_row_parts: List[torch.Tensor] = []
    cu_q = [0]
    cu_k = [0]
    max_q = 0
    max_k = 0

    for seq_id in range(int(cu_seqlens_q.shape[0]) - 1):
        seq_start = int(cu_seqlens_q[seq_id].item())
        seq_end = int(cu_seqlens_q[seq_id + 1].item())
        local_seq_start = max(seq_start, global_start)
        local_seq_end = min(seq_end, global_end)
        if local_seq_start >= local_seq_end:
            continue

        q_len = local_seq_end - local_seq_start
        seq_comp_start = int(cu_seqlens_compressed[seq_id].item())
        seq_comp_end = int(cu_seqlens_compressed[seq_id + 1].item())
        k_len = min((local_seq_end - seq_start) // ratio, seq_comp_end - seq_comp_start)
        if k_len <= 0:
            continue

        q_local_start = local_seq_start - global_start
        q_local_end = local_seq_end - global_start
        q_parts.append(q_indexer_local[q_local_start:q_local_end])
        weight_parts.append(weights_indexer_local[q_local_start:q_local_end])
        k_parts.append(k_indexer_seq_major[seq_comp_start : seq_comp_start + k_len])
        local_row_parts.append(
            torch.arange(q_local_start, q_local_end, dtype=torch.long, device=device)
        )
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)
        max_q = max(max_q, q_len)
        max_k = max(max_k, k_len)

    if not q_parts:
        q_empty = q_indexer_local.new_zeros((0,) + tuple(q_indexer_local.shape[1:]))
        k_empty = k_indexer_seq_major.new_zeros((0,) + tuple(k_indexer_seq_major.shape[1:]))
        weights_empty = weights_indexer_local.new_zeros(
            (0,) + tuple(weights_indexer_local.shape[1:])
        )
        cu_empty = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=device)
        local_rows_empty = torch.empty((0,), dtype=torch.long, device=device)
        return q_empty, k_empty, weights_empty, cu_empty, cu_empty, 0, 0, local_rows_empty

    q_topk = torch.cat(q_parts, dim=0).contiguous()
    k_topk = torch.cat(k_parts, dim=0).contiguous()
    weights_topk = torch.cat(weight_parts, dim=0).contiguous()
    cu_q_topk = torch.tensor(cu_q, dtype=cu_seqlens_q.dtype, device=device)
    cu_k_topk = torch.tensor(cu_k, dtype=cu_seqlens_compressed.dtype, device=device)
    local_row_ids = torch.cat(local_row_parts, dim=0).contiguous()
    return q_topk, k_topk, weights_topk, cu_q_topk, cu_k_topk, max_q, max_k, local_row_ids


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

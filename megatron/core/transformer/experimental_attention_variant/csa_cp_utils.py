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
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup):
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
    tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Return the fixed left boundary tensor for this CP rank."""
    return _LeftBoundaryExchange.apply(tensor, d_window, cp_group)


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
    """PyTorch stand-in for the compressor-prep compact kernel."""
    device = hidden_local.device
    c_cap = (l_local + d_comp) // ratio
    compact_len = c_cap * ratio
    hidden_compact = hidden_local.new_zeros((compact_len,) + tuple(hidden_local.shape[1:]))
    if c_cap == 0:
        empty_ids = torch.full((0,), -1, dtype=torch.int32, device=device)
        return (
            hidden_compact,
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

    if src_positions:
        src_global = torch.tensor(src_positions, dtype=torch.long, device=device)
        boundary_src = src_global - ext_start
        local_src = src_global - global_start
        in_boundary = (boundary_src >= 0) & (boundary_src < d_window)
        in_local = (local_src >= 0) & (local_src < l_local)
        if not bool((in_boundary | in_local).all().item()):
            raise RuntimeError("DSv4 compressor-prep compact source index out of boundary range.")
        compact_src = hidden_compact.new_zeros(
            (len(src_positions),) + tuple(hidden_local.shape[1:])
        )
        if bool(in_boundary.any().item()):
            compact_src[in_boundary] = boundary_hidden.index_select(
                0, boundary_src[in_boundary].to(torch.long)
            )
        if bool(in_local.any().item()):
            compact_src[in_local] = hidden_local.index_select(0, local_src[in_local].to(torch.long))
        hidden_compact[: len(src_positions)] = compact_src

    cu_compact = torch.tensor(compact_cu, dtype=cu_seqlens.dtype, device=device)
    seq_ids_t, comp_ids_t, valid_t = _build_compressor_metadata_for_range(
        cu_seqlens, global_start, l_local, ratio, d_comp, c_cap, device
    )
    return hidden_compact, cu_compact, seq_ids_t, comp_ids_t, valid_t, c_cap


def all_gather_fixed_cp_tensor(
    tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    return torch.cat(differentiable_all_gather(tensor.contiguous(), group=cp_group), dim=0)


def zero_module_parameter_dependency(module: nn.Module, like: torch.Tensor) -> torch.Tensor:
    token = like.new_tensor(0.0)
    for param in module.parameters():
        token = token + param.sum().to(dtype=like.dtype) * like.new_tensor(0.0)
    return token


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

    This CP fallback has a static output contract: valid per-sequence window
    and compressed entries are packed at the front, and unused capacity is kept
    as final tail padding. This stand-in mirrors that flat-id contract while
    still using Python metadata construction.
    """
    window_map: Dict[int, int] = {}
    compressed_map: Dict[Tuple[int, int], int] = {}
    global_end = global_start + l_local
    boundary_start = global_start - d_window
    n_sequences = int(cu_seqlens.shape[0]) - 1

    window_capacity = l_local + d_window * n_sequences
    compressed_capacity = int(compressed_rank_major.shape[0])
    kv_full_capacity = max(1, window_capacity + compressed_capacity)
    kv_full = kv_local.new_zeros((kv_full_capacity,) + tuple(kv_local.shape[1:]))

    write_pos = 0
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
                kv_full[write_pos] = boundary_kv[src]
            else:
                src = pos - global_start
                kv_full[write_pos] = kv_local[src]
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
            kv_full[write_pos] = compressed_rank_major[i]
            write_pos += 1

    if write_pos > kv_full_capacity:
        raise RuntimeError(
            "DSv4 KV full pack exceeded fixed output capacity: "
            f"written={write_pos}, capacity={kv_full_capacity}"
        )
    return kv_full, window_map, compressed_map


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

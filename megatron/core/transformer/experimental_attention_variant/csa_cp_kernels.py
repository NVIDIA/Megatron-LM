# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CuTe DSL kernels for DSv4 CSA context-parallel tensor layout work.

The functions in this module are intentionally narrow: they replace the CP
path's memory-layout helpers, not the communication ops or attention kernels.
The PyTorch reference implementations live in ``csa_cp_utils.py`` and are kept
for CPU tests and parity checks.
"""

import math
import os
from typing import Optional, Tuple

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    _CUTE_AVAILABLE = True
except ImportError:
    cutlass = None
    cute = None
    from_dlpack = None
    _CUTE_AVAILABLE = False


def can_use_cute_kernels(*tensors: Optional[torch.Tensor]) -> bool:
    """Return whether this process can run the fused CSA CP kernels."""

    if not _CUTE_AVAILABLE or os.environ.get("DSV4_CP_DISABLE_CUTE_KERNELS"):
        return False
    return all(tensor is None or tensor.is_cuda for tensor in tensors)


def _to_cute(tensor: torch.Tensor):
    return from_dlpack(tensor.detach())


def _row_width(shape: Tuple[int, ...]) -> int:
    if len(shape) == 1:
        return 1
    return math.prod(shape[1:])


def _flatten_rows(tensor: torch.Tensor, rows: int, row_width: Optional[int] = None) -> torch.Tensor:
    if row_width is None:
        row_width = _row_width(tuple(tensor.shape))
    return tensor.reshape(rows, row_width)


def _zero_rows(tensor: torch.Tensor, rows: int, row_width: int) -> None:
    total_work = rows * row_width
    if total_work > 0:
        _zero_2d_launch(_to_cute(_flatten_rows(tensor, rows, row_width)), row_width, total_work)


if _CUTE_AVAILABLE:

    @cute.kernel
    def _zero_2d_kernel(
        tensor: cute.Tensor,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            row = linear // row_width
            col = linear - row * row_width
            tensor[row, col] = cutlass.Float32(0.0).to(tensor.element_type)


    @cute.jit
    def _zero_2d_launch(
        tensor: cute.Tensor,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        _zero_2d_kernel.set_name_prefix("dsv4_cp_zero_2d")
        _zero_2d_kernel(tensor, row_width, total_work).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1)
        )


    @cute.kernel
    def _rank_major_metadata_kernel(
        cu_seqlens: cute.Tensor,
        seq_ids: cute.Tensor,
        comp_ids: cute.Tensor,
        valid: cute.Tensor,
        n_seq: cutlass.Int32,
        cp_size: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        c_cap: cutlass.Int32,
        total_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row < total_rows:
            rank = row // c_cap
            slot = row - rank * c_cap
            rank_start = rank * l_local
            rank_end = rank_start + l_local

            seq_ids[row] = -1
            comp_ids[row] = -1
            valid[row] = False

            running = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                local_seq_start = seq_start
                if local_seq_start < rank_start:
                    local_seq_start = rank_start
                local_seq_end = seq_end
                if local_seq_end > rank_end:
                    local_seq_end = rank_end

                if local_seq_start < local_seq_end:
                    n_full_groups = (seq_end - seq_start) // ratio
                    first_numer = rank_start - d_comp - seq_start
                    if first_numer < 0:
                        first_numer = 0
                    first_group = 0
                    if first_numer > 0:
                        first_group = (first_numer + ratio - 1) // ratio
                    stop_group = (local_seq_end - seq_start) // ratio
                    if stop_group > n_full_groups:
                        stop_group = n_full_groups
                    group_count = stop_group - first_group
                    if group_count < 0:
                        group_count = 0

                    if slot >= running and slot < running + group_count:
                        comp_id = first_group + slot - running
                        group_end = seq_start + (comp_id + 1) * ratio
                        seq_ids[row] = seq
                        comp_ids[row] = comp_id
                        valid[row] = group_end - 1 >= rank_start and group_end - 1 < rank_end
                    running = running + group_count


    @cute.jit
    def _rank_major_metadata_launch(
        cu_seqlens: cute.Tensor,
        seq_ids: cute.Tensor,
        comp_ids: cute.Tensor,
        valid: cute.Tensor,
        n_seq: cutlass.Int32,
        cp_size: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        c_cap: cutlass.Int32,
        total_rows: cutlass.Int32,
    ):
        _rank_major_metadata_kernel.set_name_prefix("dsv4_cp_rank_major_metadata")
        _rank_major_metadata_kernel(
            cu_seqlens,
            seq_ids,
            comp_ids,
            valid,
            n_seq,
            cp_size,
            l_local,
            ratio,
            d_comp,
            c_cap,
            total_rows,
        ).launch(grid=(cute.ceil_div(total_rows, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _compressor_prep_kernel(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_compact: cute.Tensor,
        seq_ids: cute.Tensor,
        comp_ids: cute.Tensor,
        valid: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        d_window: cutlass.Int32,
        c_cap: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        if linear < compact_len * row_width:
            row = linear // row_width
            col = linear - row * row_width
            hidden_compact[row, col] = cutlass.Float32(0.0).to(hidden_compact.element_type)

            running_tokens = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                global_end = global_start + l_local
                local_seq_start = seq_start
                if local_seq_start < global_start:
                    local_seq_start = global_start
                local_seq_end = seq_end
                if local_seq_end > global_end:
                    local_seq_end = global_end

                if local_seq_start < local_seq_end:
                    n_full_groups = (seq_end - seq_start) // ratio
                    first_numer = global_start - d_comp - seq_start
                    if first_numer < 0:
                        first_numer = 0
                    first_group = 0
                    if first_numer > 0:
                        first_group = (first_numer + ratio - 1) // ratio
                    stop_group = (local_seq_end - seq_start) // ratio
                    if stop_group > n_full_groups:
                        stop_group = n_full_groups
                    group_count = stop_group - first_group
                    if group_count < 0:
                        group_count = 0
                    seq_tokens = group_count * ratio

                    if row >= running_tokens and row < running_tokens + seq_tokens:
                        local_group_token = row - running_tokens
                        comp_id = first_group + local_group_token // ratio
                        token_in_group = local_group_token - (local_group_token // ratio) * ratio
                        src_global = seq_start + comp_id * ratio + token_in_group
                        boundary_start = global_start - d_window
                        if src_global < global_start:
                            boundary_row = src_global - boundary_start
                            hidden_compact[row, col] = boundary_hidden[boundary_row, col]
                        else:
                            local_row = src_global - global_start
                            hidden_compact[row, col] = hidden_local[local_row, col]
                    running_tokens = running_tokens + seq_tokens

        if linear < c_cap:
            slot = linear
            seq_ids[slot] = -1
            comp_ids[slot] = -1
            valid[slot] = False

            running = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                global_end = global_start + l_local
                local_seq_start = seq_start
                if local_seq_start < global_start:
                    local_seq_start = global_start
                local_seq_end = seq_end
                if local_seq_end > global_end:
                    local_seq_end = global_end

                if local_seq_start < local_seq_end:
                    n_full_groups = (seq_end - seq_start) // ratio
                    first_numer = global_start - d_comp - seq_start
                    if first_numer < 0:
                        first_numer = 0
                    first_group = 0
                    if first_numer > 0:
                        first_group = (first_numer + ratio - 1) // ratio
                    stop_group = (local_seq_end - seq_start) // ratio
                    if stop_group > n_full_groups:
                        stop_group = n_full_groups
                    group_count = stop_group - first_group
                    if group_count < 0:
                        group_count = 0

                    if slot >= running and slot < running + group_count:
                        comp_id = first_group + slot - running
                        group_end = seq_start + (comp_id + 1) * ratio
                        seq_ids[slot] = seq
                        comp_ids[slot] = comp_id
                        valid[slot] = group_end - 1 >= global_start and group_end - 1 < global_end
                    running = running + group_count

        if linear < n_seq + 1:
            upto = linear
            running = 0
            for seq in range(n_seq):
                if seq < upto:
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    global_end = global_start + l_local
                    local_seq_start = seq_start
                    if local_seq_start < global_start:
                        local_seq_start = global_start
                    local_seq_end = seq_end
                    if local_seq_end > global_end:
                        local_seq_end = global_end

                    if local_seq_start < local_seq_end:
                        n_full_groups = (seq_end - seq_start) // ratio
                        first_numer = global_start - d_comp - seq_start
                        if first_numer < 0:
                            first_numer = 0
                        first_group = 0
                        if first_numer > 0:
                            first_group = (first_numer + ratio - 1) // ratio
                        stop_group = (local_seq_end - seq_start) // ratio
                        if stop_group > n_full_groups:
                            stop_group = n_full_groups
                        group_count = stop_group - first_group
                        if group_count < 0:
                            group_count = 0
                        running = running + group_count * ratio
            cu_compact[linear] = running


    @cute.jit
    def _compressor_prep_launch(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_compact: cute.Tensor,
        seq_ids: cute.Tensor,
        comp_ids: cute.Tensor,
        valid: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        d_window: cutlass.Int32,
        c_cap: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        _compressor_prep_kernel.set_name_prefix("dsv4_cp_compressor_prep")
        _compressor_prep_kernel(
            hidden_local,
            boundary_hidden,
            hidden_compact,
            cu_seqlens,
            cu_compact,
            seq_ids,
            comp_ids,
            valid,
            n_seq,
            global_start,
            l_local,
            ratio,
            d_comp,
            d_window,
            c_cap,
            compact_len,
            row_width,
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _compressor_prep_backward_kernel(
        grad_hidden_compact: cute.Tensor,
        grad_hidden_local: cute.Tensor,
        grad_boundary_hidden: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        d_window: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        if linear < compact_len * row_width:
            row = linear // row_width
            col = linear - row * row_width

            running_tokens = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                global_end = global_start + l_local
                local_seq_start = seq_start
                if local_seq_start < global_start:
                    local_seq_start = global_start
                local_seq_end = seq_end
                if local_seq_end > global_end:
                    local_seq_end = global_end

                if local_seq_start < local_seq_end:
                    n_full_groups = (seq_end - seq_start) // ratio
                    first_numer = global_start - d_comp - seq_start
                    if first_numer < 0:
                        first_numer = 0
                    first_group = 0
                    if first_numer > 0:
                        first_group = (first_numer + ratio - 1) // ratio
                    stop_group = (local_seq_end - seq_start) // ratio
                    if stop_group > n_full_groups:
                        stop_group = n_full_groups
                    group_count = stop_group - first_group
                    if group_count < 0:
                        group_count = 0
                    seq_tokens = group_count * ratio

                    if row >= running_tokens and row < running_tokens + seq_tokens:
                        local_group_token = row - running_tokens
                        comp_id = first_group + local_group_token // ratio
                        token_in_group = local_group_token - (local_group_token // ratio) * ratio
                        src_global = seq_start + comp_id * ratio + token_in_group
                        boundary_start = global_start - d_window
                        if src_global < global_start:
                            boundary_row = src_global - boundary_start
                            grad_boundary_hidden[boundary_row, col] = grad_hidden_compact[row, col]
                        else:
                            local_row = src_global - global_start
                            grad_hidden_local[local_row, col] = grad_hidden_compact[row, col]
                    running_tokens = running_tokens + seq_tokens


    @cute.jit
    def _compressor_prep_backward_launch(
        grad_hidden_compact: cute.Tensor,
        grad_hidden_local: cute.Tensor,
        grad_boundary_hidden: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        d_window: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        _compressor_prep_backward_kernel.set_name_prefix("dsv4_cp_compressor_prep_backward")
        _compressor_prep_backward_kernel(
            grad_hidden_compact,
            grad_hidden_local,
            grad_boundary_hidden,
            cu_seqlens,
            n_seq,
            global_start,
            l_local,
            ratio,
            d_comp,
            d_window,
            compact_len,
            row_width,
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _kv_pack_kernel(
        kv_local: cute.Tensor,
        boundary_kv: cute.Tensor,
        compressed_rank_major: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        kv_full: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            out_row = linear // row_width
            col = linear - out_row * row_width
            kv_full[out_row, col] = cutlass.Float32(0.0).to(kv_full.element_type)

            offset = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                global_end = global_start + l_local
                local_seq_start = seq_start
                if local_seq_start < global_start:
                    local_seq_start = global_start
                local_seq_end = seq_end
                if local_seq_end > global_end:
                    local_seq_end = global_end

                if local_seq_start < local_seq_end:
                    window_start = local_seq_start - d_window
                    if window_start < seq_start:
                        window_start = seq_start
                    window_len = local_seq_end - window_start
                    comp_len = 0
                    if ratio > 1:
                        comp_len = (seq_end - seq_start) // ratio
                    seq_total = window_len + comp_len

                    if out_row >= offset and out_row < offset + seq_total:
                        inner = out_row - offset
                        if inner < window_len:
                            src_global = window_start + inner
                            if src_global < global_start:
                                boundary_row = src_global - (global_start - d_window)
                                kv_full[out_row, col] = boundary_kv[boundary_row, col]
                            else:
                                local_row = src_global - global_start
                                kv_full[out_row, col] = kv_local[local_row, col]
                        else:
                            comp_id = inner - window_len
                            for comp_row in range(compressed_rows):
                                if (
                                    valid_rank_major[comp_row]
                                    and seq_ids_rank_major[comp_row] == seq
                                    and comp_ids_rank_major[comp_row] == comp_id
                                ):
                                    kv_full[out_row, col] = compressed_rank_major[comp_row, col]
                    offset = offset + seq_total


    @cute.jit
    def _kv_pack_launch(
        kv_local: cute.Tensor,
        boundary_kv: cute.Tensor,
        compressed_rank_major: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        kv_full: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        _kv_pack_kernel.set_name_prefix("dsv4_cp_kv_full_pack")
        _kv_pack_kernel(
            kv_local,
            boundary_kv,
            compressed_rank_major,
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens,
            kv_full,
            n_seq,
            global_start,
            l_local,
            d_window,
            ratio,
            compressed_rows,
            kv_full_capacity,
            row_width,
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _kv_pack_backward_kernel(
        grad_kv_full: cute.Tensor,
        grad_kv_local: cute.Tensor,
        grad_boundary_kv: cute.Tensor,
        grad_compressed: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        if linear < kv_full_capacity * row_width:
            out_row = linear // row_width
            col = linear - out_row * row_width

            offset = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                global_end = global_start + l_local
                local_seq_start = seq_start
                if local_seq_start < global_start:
                    local_seq_start = global_start
                local_seq_end = seq_end
                if local_seq_end > global_end:
                    local_seq_end = global_end

                if local_seq_start < local_seq_end:
                    window_start = local_seq_start - d_window
                    if window_start < seq_start:
                        window_start = seq_start
                    window_len = local_seq_end - window_start
                    comp_len = 0
                    if ratio > 1:
                        comp_len = (seq_end - seq_start) // ratio
                    seq_total = window_len + comp_len

                    if out_row >= offset and out_row < offset + seq_total:
                        inner = out_row - offset
                        if inner < window_len:
                            src_global = window_start + inner
                            if src_global < global_start:
                                boundary_row = src_global - (global_start - d_window)
                                grad_boundary_kv[boundary_row, col] = grad_kv_full[out_row, col]
                            else:
                                local_row = src_global - global_start
                                grad_kv_local[local_row, col] = grad_kv_full[out_row, col]
                        else:
                            comp_id = inner - window_len
                            for comp_row in range(compressed_rows):
                                if (
                                    valid_rank_major[comp_row]
                                    and seq_ids_rank_major[comp_row] == seq
                                    and comp_ids_rank_major[comp_row] == comp_id
                                ):
                                    grad_compressed[comp_row, col] = grad_kv_full[out_row, col]
                    offset = offset + seq_total


    @cute.jit
    def _kv_pack_backward_launch(
        grad_kv_full: cute.Tensor,
        grad_kv_local: cute.Tensor,
        grad_boundary_kv: cute.Tensor,
        grad_compressed: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        _kv_pack_backward_kernel.set_name_prefix("dsv4_cp_kv_full_pack_backward")
        _kv_pack_backward_kernel(
            grad_kv_full,
            grad_kv_local,
            grad_boundary_kv,
            grad_compressed,
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens,
            n_seq,
            global_start,
            l_local,
            d_window,
            ratio,
            compressed_rows,
            kv_full_capacity,
            row_width,
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _repack_rank_major_kernel(
        compressed_rank_major: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        compressed_seq_major: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        rank_major_rows: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        if linear < seq_major_rows * row_width:
            row = linear // row_width
            col = linear - row * row_width
            compressed_seq_major[row, col] = cutlass.Float32(0.0).to(
                compressed_seq_major.element_type
            )
        if linear < seq_major_rows:
            rank_major_by_seq_major[linear] = -1

        if linear < rank_major_rows * row_width:
            rank_row = linear // row_width
            col = linear - rank_row * row_width
            if valid_rank_major[rank_row]:
                seq = seq_ids_rank_major[rank_row]
                comp = comp_ids_rank_major[rank_row]
                seq_major = cu_seqlens_compressed[seq] + comp
                if seq_major >= 0 and seq_major < seq_major_rows:
                    compressed_seq_major[seq_major, col] = compressed_rank_major[rank_row, col]
                    if col == 0:
                        rank_major_by_seq_major[seq_major] = rank_row


    @cute.jit
    def _repack_rank_major_launch(
        compressed_rank_major: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        compressed_seq_major: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        rank_major_rows: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        _repack_rank_major_kernel.set_name_prefix("dsv4_cp_repack_rank_major")
        _repack_rank_major_kernel(
            compressed_rank_major,
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens_compressed,
            compressed_seq_major,
            rank_major_by_seq_major,
            rank_major_rows,
            seq_major_rows,
            row_width,
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _final_idx_kernel(
        cu_seqlens: cute.Tensor,
        indexer_topk: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        has_indexer_topk: cutlass.Int32,
        has_compressed: cutlass.Int32,
        total_width: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row < l_local:
            for col in range(total_width):
                topk_idxs[row, col] = -1
            topk_length[row] = 0

            global_q = global_start + row
            seq_id = -1
            seq_start_found = 0
            seq_end_found = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                if global_q >= seq_start and global_q < seq_end:
                    seq_id = seq
                    seq_start_found = seq_start
                    seq_end_found = seq_end

            if seq_id >= 0:
                seq_offset = 0
                seq_window_start = 0
                seq_window_len = 0
                seq_comp_len = 0
                global_end = global_start + l_local
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_start = seq_start
                    if local_seq_start < global_start:
                        local_seq_start = global_start
                    local_seq_end = seq_end
                    if local_seq_end > global_end:
                        local_seq_end = global_end
                    if local_seq_start < local_seq_end:
                        window_start = local_seq_start - d_window
                        if window_start < seq_start:
                            window_start = seq_start
                        window_len = local_seq_end - window_start
                        comp_len = 0
                        if has_compressed != 0 and ratio > 1:
                            comp_len = (seq_end - seq_start) // ratio
                        if seq == seq_id:
                            seq_window_start = window_start
                            seq_window_len = window_len
                            seq_comp_len = comp_len
                        if seq < seq_id:
                            seq_offset = seq_offset + window_len + comp_len

                write_col = 0
                window_start_for_q = global_q - window_size + 1
                if window_start_for_q < seq_start_found:
                    window_start_for_q = seq_start_found
                window_count = global_q - window_start_for_q + 1
                for w in range(window_size):
                    pos = window_start_for_q + w
                    if w < window_count and pos >= seq_window_start and pos < seq_window_start + seq_window_len:
                        topk_idxs[row, write_col] = seq_offset + pos - seq_window_start
                        write_col = write_col + 1

                if compressed_width > 0:
                    pos_in_seq = global_q - seq_start_found
                    for j in range(compressed_width):
                        comp_id = -1
                        if has_indexer_topk != 0:
                            comp_id = indexer_topk[row, j]
                        else:
                            n_visible = (pos_in_seq + 1) // ratio
                            if n_visible > compressed_width:
                                n_visible = compressed_width
                            if j < n_visible:
                                comp_id = j
                        if comp_id >= 0 and comp_id < seq_comp_len:
                            topk_idxs[row, write_col] = seq_offset + seq_window_len + comp_id
                            write_col = write_col + 1
                topk_length[row] = write_col


    @cute.jit
    def _final_idx_launch(
        cu_seqlens: cute.Tensor,
        indexer_topk: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        has_indexer_topk: cutlass.Int32,
        has_compressed: cutlass.Int32,
        total_width: cutlass.Int32,
    ):
        _final_idx_kernel.set_name_prefix("dsv4_cp_final_idx")
        _final_idx_kernel(
            cu_seqlens,
            indexer_topk,
            topk_idxs,
            topk_length,
            n_seq,
            global_start,
            l_local,
            d_window,
            window_size,
            ratio,
            compressed_width,
            has_indexer_topk,
            has_compressed,
            total_width,
        ).launch(grid=(cute.ceil_div(l_local, 128), 1, 1), block=(128, 1, 1))


    @cute.kernel
    def _indexer_loss_final_idx_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk_logical: cute.Tensor,
        indexer_rank_by_seq_major: cute.Tensor,
        topk_idxs: cute.Tensor,
        indexer_rank_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        total_width: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row < l_local:
            for col in range(total_width):
                topk_idxs[row, col] = -1
            for col in range(compressed_width):
                indexer_rank_major[row, col] = -1

            global_q = global_start + row
            seq_id = -1
            seq_start_found = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                if global_q >= seq_start and global_q < seq_end:
                    seq_id = seq
                    seq_start_found = seq_start

            if seq_id >= 0:
                seq_offset = 0
                seq_window_start = 0
                seq_window_len = 0
                seq_comp_len = 0
                global_end = global_start + l_local
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_start = seq_start
                    if local_seq_start < global_start:
                        local_seq_start = global_start
                    local_seq_end = seq_end
                    if local_seq_end > global_end:
                        local_seq_end = global_end
                    if local_seq_start < local_seq_end:
                        window_start = local_seq_start - d_window
                        if window_start < seq_start:
                            window_start = seq_start
                        window_len = local_seq_end - window_start
                        comp_len = 0
                        if ratio > 1:
                            comp_len = (seq_end - seq_start) // ratio
                        if seq == seq_id:
                            seq_window_start = window_start
                            seq_window_len = window_len
                            seq_comp_len = comp_len
                        if seq < seq_id:
                            seq_offset = seq_offset + window_len + comp_len

                for j in range(compressed_width):
                    comp_id = indexer_topk_logical[row, j]
                    if comp_id >= 0 and comp_id < seq_comp_len:
                        topk_idxs[row, j] = seq_offset + seq_window_len + comp_id
                        seq_major_id = cu_seqlens_compressed[seq_id] + comp_id
                        if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                            rank_major_id = indexer_rank_by_seq_major[seq_major_id]
                            if rank_major_id >= 0:
                                indexer_rank_major[row, j] = rank_major_id

                window_start_for_q = global_q - window_size + 1
                if window_start_for_q < seq_start_found:
                    window_start_for_q = seq_start_found
                window_count = global_q - window_start_for_q + 1
                for w in range(window_size):
                    pos = window_start_for_q + w
                    if w < window_count and pos >= seq_window_start and pos < seq_window_start + seq_window_len:
                        topk_idxs[row, compressed_width + w] = (
                            seq_offset + pos - seq_window_start
                        )


    @cute.jit
    def _indexer_loss_final_idx_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk_logical: cute.Tensor,
        indexer_rank_by_seq_major: cute.Tensor,
        topk_idxs: cute.Tensor,
        indexer_rank_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        total_width: cutlass.Int32,
    ):
        _indexer_loss_final_idx_kernel.set_name_prefix("dsv4_cp_indexer_loss_final_idx")
        _indexer_loss_final_idx_kernel(
            cu_seqlens,
            cu_seqlens_compressed,
            indexer_topk_logical,
            indexer_rank_by_seq_major,
            topk_idxs,
            indexer_rank_major,
            n_seq,
            global_start,
            l_local,
            d_window,
            window_size,
            ratio,
            compressed_width,
            seq_major_rows,
            total_width,
        ).launch(grid=(cute.ceil_div(l_local, 128), 1, 1), block=(128, 1, 1))


def build_rank_major_compressed_metadata(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = cp_size * c_cap
    seq_ids = torch.empty((total_rows,), dtype=torch.int32, device=cu_seqlens.device)
    comp_ids = torch.empty((total_rows,), dtype=torch.int32, device=cu_seqlens.device)
    valid = torch.empty((total_rows,), dtype=torch.bool, device=cu_seqlens.device)
    _rank_major_metadata_launch(
        _to_cute(cu_seqlens),
        _to_cute(seq_ids),
        _to_cute(comp_ids),
        _to_cute(valid),
        cu_seqlens.shape[0] - 1,
        cp_size,
        l_local,
        ratio,
        d_comp,
        c_cap,
        total_rows,
    )
    return seq_ids, comp_ids, valid


def compressor_prep_compact(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    d_window: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compact_len = c_cap * ratio
    output_shape = (compact_len,) + tuple(hidden_local.shape[1:])
    hidden_compact = hidden_local.new_empty(output_shape)
    cu_compact = torch.empty_like(cu_seqlens)
    seq_ids = torch.empty((c_cap,), dtype=torch.int32, device=hidden_local.device)
    comp_ids = torch.empty((c_cap,), dtype=torch.int32, device=hidden_local.device)
    valid = torch.empty((c_cap,), dtype=torch.bool, device=hidden_local.device)

    row_width = _row_width(tuple(hidden_local.shape))
    hidden_local_flat = _flatten_rows(hidden_local, l_local, row_width)
    boundary_flat = _flatten_rows(boundary_hidden, d_window, row_width)
    compact_flat = _flatten_rows(hidden_compact, compact_len, row_width)
    total_work = max(compact_len * row_width, c_cap, cu_seqlens.shape[0])
    _compressor_prep_launch(
        _to_cute(hidden_local_flat),
        _to_cute(boundary_flat),
        _to_cute(compact_flat),
        _to_cute(cu_seqlens),
        _to_cute(cu_compact),
        _to_cute(seq_ids),
        _to_cute(comp_ids),
        _to_cute(valid),
        cu_seqlens.shape[0] - 1,
        global_start,
        l_local,
        ratio,
        d_comp,
        d_window,
        c_cap,
        compact_len,
        row_width,
        total_work,
    )
    return hidden_compact, cu_compact, seq_ids, comp_ids, valid


def compressor_prep_compact_backward(
    grad_hidden_compact: torch.Tensor,
    hidden_shape: Tuple[int, ...],
    boundary_shape: Tuple[int, ...],
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    d_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_hidden = grad_hidden_compact.new_empty(hidden_shape)
    grad_boundary = grad_hidden_compact.new_empty(boundary_shape)
    compact_len = grad_hidden_compact.shape[0]
    row_width = _row_width(tuple(hidden_shape))
    _zero_rows(grad_hidden, l_local, row_width)
    _zero_rows(grad_boundary, d_window, row_width)

    total_work = compact_len * row_width
    if total_work > 0:
        _compressor_prep_backward_launch(
            _to_cute(_flatten_rows(grad_hidden_compact, compact_len, row_width)),
            _to_cute(_flatten_rows(grad_hidden, l_local, row_width)),
            _to_cute(_flatten_rows(grad_boundary, d_window, row_width)),
            _to_cute(cu_seqlens),
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            ratio,
            d_comp,
            d_window,
            compact_len,
            row_width,
            total_work,
        )
    return grad_hidden, grad_boundary


class _KVFullPackCute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        kv_full_capacity: int,
    ) -> torch.Tensor:
        ctx.local_shape = tuple(kv_local.shape)
        ctx.boundary_shape = tuple(boundary_kv.shape)
        ctx.compressed_shape = tuple(compressed_rank_major.shape)
        ctx.global_start = global_start
        ctx.l_local = l_local
        ctx.d_window = d_window
        ctx.ratio = ratio
        ctx.kv_full_capacity = kv_full_capacity
        ctx.save_for_backward(
            seq_ids_rank_major, comp_ids_rank_major, valid_rank_major, cu_seqlens
        )

        output_shape = (kv_full_capacity,) + tuple(kv_local.shape[1:])
        kv_full = kv_local.new_empty(output_shape)
        row_width = _row_width(tuple(kv_local.shape))
        kv_local_flat = _flatten_rows(kv_local, l_local, row_width)
        boundary_flat = _flatten_rows(boundary_kv, d_window, row_width)
        compressed_rows = compressed_rank_major.shape[0]
        compressed_flat = _flatten_rows(compressed_rank_major, compressed_rows, row_width)
        kv_full_flat = _flatten_rows(kv_full, kv_full_capacity, row_width)
        total_work = kv_full_capacity * row_width
        _kv_pack_launch(
            _to_cute(kv_local_flat),
            _to_cute(boundary_flat),
            _to_cute(compressed_flat),
            _to_cute(seq_ids_rank_major),
            _to_cute(comp_ids_rank_major),
            _to_cute(valid_rank_major),
            _to_cute(cu_seqlens),
            _to_cute(kv_full_flat),
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            d_window,
            ratio,
            compressed_rows,
            kv_full_capacity,
            row_width,
            total_work,
        )
        return kv_full

    @staticmethod
    def backward(ctx, grad_kv_full: torch.Tensor):
        seq_ids_rank_major, comp_ids_rank_major, valid_rank_major, cu_seqlens = ctx.saved_tensors
        grad_kv_local = grad_kv_full.new_empty(ctx.local_shape)
        grad_boundary_kv = grad_kv_full.new_empty(ctx.boundary_shape)
        grad_compressed = grad_kv_full.new_empty(ctx.compressed_shape)

        l_local = ctx.l_local
        d_window = ctx.d_window
        compressed_rows = ctx.compressed_shape[0]
        kv_full_capacity = ctx.kv_full_capacity
        row_width = _row_width(ctx.local_shape)
        _zero_rows(grad_kv_local, l_local, row_width)
        _zero_rows(grad_boundary_kv, d_window, row_width)
        _zero_rows(grad_compressed, compressed_rows, row_width)

        total_work = kv_full_capacity * row_width
        if total_work > 0:
            _kv_pack_backward_launch(
                _to_cute(_flatten_rows(grad_kv_full, kv_full_capacity, row_width)),
                _to_cute(_flatten_rows(grad_kv_local, l_local, row_width)),
                _to_cute(_flatten_rows(grad_boundary_kv, d_window, row_width)),
                _to_cute(_flatten_rows(grad_compressed, compressed_rows, row_width)),
                _to_cute(seq_ids_rank_major),
                _to_cute(comp_ids_rank_major),
                _to_cute(valid_rank_major),
                _to_cute(cu_seqlens),
                cu_seqlens.shape[0] - 1,
                ctx.global_start,
                l_local,
                d_window,
                ctx.ratio,
                compressed_rows,
                kv_full_capacity,
                row_width,
                total_work,
            )
        return (
            grad_kv_local,
            grad_boundary_kv,
            grad_compressed,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def kv_full_capacity(cu_seqlens: torch.Tensor, l_local: int, d_window: int, ratio: int) -> int:
    # Static upper bound matching the HTML layout: per active sequence window rows
    # plus compressed rows, followed by tail padding.
    n_seq = cu_seqlens.shape[0] - 1
    compressed_capacity = 0
    if ratio > 1:
        compressed_capacity = (l_local * n_seq + d_window * n_seq) // ratio
    return max(1, l_local + d_window * n_seq + compressed_capacity)


def pack_kv_full(
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
    kv_full_capacity_value: int,
) -> torch.Tensor:
    return _KVFullPackCute.apply(
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
        kv_full_capacity_value,
    )


def repack_rank_major_compressed_to_seq_major(
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    seq_major_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output_shape = (seq_major_rows,) + tuple(compressed_rank_major.shape[1:])
    compressed_seq_major = compressed_rank_major.new_empty(output_shape)
    rank_major_by_seq_major = torch.empty(
        (seq_major_rows,), dtype=torch.int32, device=compressed_rank_major.device
    )
    rank_major_rows = compressed_rank_major.shape[0]
    row_width = _row_width(tuple(compressed_rank_major.shape))
    total_work = max(seq_major_rows * row_width, rank_major_rows * row_width, seq_major_rows)
    _repack_rank_major_launch(
        _to_cute(_flatten_rows(compressed_rank_major, rank_major_rows, row_width)),
        _to_cute(seq_ids_rank_major),
        _to_cute(comp_ids_rank_major),
        _to_cute(valid_rank_major),
        _to_cute(cu_seqlens_compressed),
        _to_cute(_flatten_rows(compressed_seq_major, seq_major_rows, row_width)),
        _to_cute(rank_major_by_seq_major),
        rank_major_rows,
        seq_major_rows,
        row_width,
        total_work,
    )
    return compressed_seq_major, rank_major_by_seq_major


def build_final_idxs(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    indexer_topk_compressed_logical_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_width = window_size + compressed_width
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    topk_length = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens.device)
    if indexer_topk_compressed_logical_ids is None:
        dummy = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
        has_indexer = 0
    else:
        dummy = indexer_topk_compressed_logical_ids
        has_indexer = 1
    _final_idx_launch(
        _to_cute(cu_seqlens),
        _to_cute(dummy),
        _to_cute(topk_idxs),
        _to_cute(topk_length),
        cu_seqlens.shape[0] - 1,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        has_indexer,
        1 if ratio > 1 and compressed_width > 0 else 0,
        total_width,
    )
    return topk_idxs, topk_length


def build_indexer_loss_final_idxs(
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
    compressed_width = indexer_topk_compressed_logical_ids.shape[-1]
    total_width = compressed_width + window_size
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    indexer_rank_major = torch.empty(
        (l_local, compressed_width), dtype=torch.int32, device=cu_seqlens.device
    )
    _indexer_loss_final_idx_launch(
        _to_cute(cu_seqlens),
        _to_cute(cu_seqlens_compressed),
        _to_cute(indexer_topk_compressed_logical_ids),
        _to_cute(indexer_rank_by_seq_major),
        _to_cute(topk_idxs),
        _to_cute(indexer_rank_major),
        cu_seqlens.shape[0] - 1,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        indexer_rank_by_seq_major.shape[0],
        total_width,
    )
    return topk_idxs, indexer_rank_major

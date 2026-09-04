# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTeDSL kernels for DSv4 packed-THD layout work.

This module contains the context-parallel compressor compaction/metadata kernels
and the final-index kernel shared by CP and generic packed THD paths.
``cp_utils.py`` dispatches CP layout work; ``csa.py`` calls final-index lowering
directly. Local THD RoPE reuses MCore's fused MLA implementation.
"""

import math
from typing import Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import utils
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    _CUTE_AVAILABLE = True
except ImportError:
    cuda = None
    cutlass = None
    cute = None
    from_dlpack = None
    make_fake_stream = None
    _CUTE_AVAILABLE = False

# =============================================================================
# CuTeDSL Kernel Definitions
# =============================================================================
# Device kernels and CuTe launch functions. These blocks describe the GPU work;
# tensor allocation, validation, and autograd integration stay in the wrappers.
# =============================================================================

if _CUTE_AVAILABLE:

    def _launch_named(
        kernel, name: str, args: Tuple, grid: Tuple, block: Tuple, stream: cuda.CUstream
    ):
        """Launch one CSA CP CuTe kernel and keep its timeline prefix local."""
        kernel.set_name_prefix(name)
        launcher = kernel(*args)
        try:
            launcher.launch(grid=grid, block=block, stream=stream)
        finally:
            if hasattr(kernel, "_name_prefix"):
                kernel._name_prefix = None
            dsl = getattr(launcher, "dsl", None)
            if dsl is not None and hasattr(dsl, "_name_prefix"):
                dsl._name_prefix = None

    # Kernel contract:
    #   hidden_local: local hidden rows, shape (l_local, row_width), bf16/fp16/fp32.
    #   boundary_hidden: left boundary rows, shape (d_window, row_width).
    #   hidden_compact: output compact rows, shape (compact_len, row_width).
    #   comp_ids: original per-sequence compressed group ids, shape (c_cap,).
    #   position_ids: RoPE positions for the compact groups, shape (c_cap,).
    #   local_cu_seqlens/local_cu_seqlens_comp: prefixes describing the physically
    #   compacted token/group segments, shape (n_seq + 1,).
    #   cu_seqlens_comp: global per-sequence compressed prefixes, shape (n_seq + 1,).
    #   Enumerates visible full compression groups in [global_start-d_comp,
    #   global_start+l_local), copies their ratio tokens from boundary/local
    #   hidden into hidden_compact, and emits compressed group ids.
    @cute.kernel
    def _compressor_input_compact_fwd_kernel(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        comp_ids: cute.Tensor,
        position_ids: cute.Tensor,
        local_cu_seqlens: cute.Tensor,
        local_cu_seqlens_comp: cute.Tensor,
        cu_seqlens_comp: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx

        if cutlass.const_expr(hidden_compact.element_type.width == 16 and row_width % 4 == 0):
            vec_elems = cutlass.Int32(4)
            vec_cols = row_width // 4
        else:
            vec_elems = cutlass.Int32(1)
            vec_cols = row_width

        range_start = global_start
        range_end = global_start + l_local
        first_range_group_start = range_start - d_comp

        # Metadata is consumed by a later launch on the same stream. Keeping the
        # short prefix scans in this launch avoids the eager diff/div/cumsum/cat
        # chains without requiring grid-wide synchronization.
        if bidx == 0 and tidx == 0:
            local_running_groups = 0
            global_running_groups = 0
            local_cu_seqlens[0] = 0
            local_cu_seqlens_comp[0] = 0
            cu_seqlens_comp[0] = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                global_running_groups = global_running_groups + (seq_end - seq_start) // ratio
                cu_seqlens_comp[seq + 1] = global_running_groups

                local_seq_end = seq_end
                if local_seq_end > range_end:
                    local_seq_end = range_end
                visible_group_count = 0
                if seq_start < local_seq_end and range_start < local_seq_end:
                    first_visible_numer = first_range_group_start - seq_start
                    if first_visible_numer < 0:
                        first_visible_numer = 0
                    first_visible_group = (first_visible_numer + ratio - 1) // ratio
                    stop_visible_group = (local_seq_end - seq_start) // ratio
                    visible_group_count = stop_visible_group - first_visible_group
                    if visible_group_count < 0:
                        visible_group_count = 0
                local_running_groups = local_running_groups + visible_group_count
                local_cu_seqlens_comp[seq + 1] = local_running_groups
                local_cu_seqlens[seq + 1] = local_running_groups * ratio

        src_global = cutlass.Int32(-1)
        visible_comp_id = cutlass.Int32(-1)
        if row < compact_len:
            if tidx == 0 or tidx == 32:
                running_tokens = 0
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_end = seq_end
                    if local_seq_end > range_end:
                        local_seq_end = range_end

                    if seq_start < local_seq_end and range_start < local_seq_end:
                        first_visible_numer = first_range_group_start - seq_start
                        if first_visible_numer < 0:
                            first_visible_numer = 0
                        first_visible_group = (first_visible_numer + ratio - 1) // ratio
                        stop_visible_group = (local_seq_end - seq_start) // ratio
                        visible_group_count = stop_visible_group - first_visible_group
                        if visible_group_count < 0:
                            visible_group_count = 0
                        visible_token_count = visible_group_count * ratio

                        if row >= running_tokens and row < running_tokens + visible_token_count:
                            local_visible_token = row - running_tokens
                            comp_id = first_visible_group + local_visible_token // ratio
                            token_in_group = (
                                local_visible_token - (local_visible_token // ratio) * ratio
                            )
                            src_global = seq_start + comp_id * ratio + token_in_group
                            visible_comp_id = comp_id
                        running_tokens = running_tokens + visible_token_count

            src_global = cute.arch.shuffle_sync(src_global, 0, mask=-1, mask_and_clamp=31)
            if row % ratio == 0 and tidx == 0:
                comp_ids[row // ratio] = visible_comp_id
                if visible_comp_id < 0:
                    position_ids[row // ratio] = 0
                else:
                    position_ids[row // ratio] = visible_comp_id * ratio

            vec_col = tidx
            while vec_col < vec_cols:
                dst_offset = row * row_width + vec_col * vec_elems
                if cutlass.const_expr(
                    hidden_compact.element_type.width == 16 and row_width % 4 == 0
                ):
                    dst_ptr = cute.recast_ptr(
                        hidden_compact.iterator + dst_offset, dtype=cutlass.Int64
                    )
                    value = cutlass.Int64(0)
                    if src_global >= 0:
                        if src_global < range_start:
                            src_row = src_global - (range_start - d_window)
                            src_offset = src_row * row_width + vec_col * vec_elems
                            src_ptr = cute.recast_ptr(
                                boundary_hidden.iterator + src_offset, dtype=cutlass.Int64
                            )
                            value = cute.arch.load(
                                src_ptr.llvm_ptr,
                                cutlass.Int64,
                                level1_eviction_priority="evict_no_allocate",
                            )
                        else:
                            src_row = src_global - range_start
                            src_offset = src_row * row_width + vec_col * vec_elems
                            src_ptr = cute.recast_ptr(
                                hidden_local.iterator + src_offset, dtype=cutlass.Int64
                            )
                            value = cute.arch.load(
                                src_ptr.llvm_ptr,
                                cutlass.Int64,
                                level1_eviction_priority="evict_no_allocate",
                            )
                    cute.arch.store(dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first")
                else:
                    value = cutlass.Float32(0.0).to(hidden_compact.element_type)
                    if src_global >= 0:
                        if src_global < range_start:
                            src_row = src_global - (range_start - d_window)
                            value = boundary_hidden[src_row, vec_col]
                        else:
                            src_row = src_global - range_start
                            value = hidden_local[src_row, vec_col]
                    hidden_compact[row, vec_col] = value
                vec_col = vec_col + 64

    # Launch contract for _compressor_input_compact_fwd_kernel.
    #   One block copies each compact row; group-leading blocks also emit comp_ids.
    @cute.jit
    def _compressor_input_compact_fwd_launch(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        comp_ids: cute.Tensor,
        position_ids: cute.Tensor,
        local_cu_seqlens: cute.Tensor,
        local_cu_seqlens_comp: cute.Tensor,
        cu_seqlens_comp: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _compressor_input_compact_fwd_kernel,
            "dsv4_cp_compressor_input_compact_fwd",
            (
                hidden_local,
                boundary_hidden,
                hidden_compact,
                cu_seqlens,
                comp_ids,
                position_ids,
                local_cu_seqlens,
                local_cu_seqlens_comp,
                cu_seqlens_comp,
                n_seq,
                global_start,
                l_local,
                ratio,
                d_comp,
                d_window,
                compact_len,
                row_width,
            ),
            grid=(compact_len, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    # Build the global sequence-major -> rank-major compact-row map after the
    # compaction launch has produced ``cu_seqlens_comp``. One thread owns one
    # logical compressed row; n_seq is tiny for the intended packed workloads,
    # so two short linear scans are cheaper than materializing bucketize inputs.
    @cute.kernel
    def _compressor_rank_row_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_comp: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        n_seq: cutlass.Int32,
        l_local: cutlass.Int32,
        cp_size: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        c_cap: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        logical_row = bidx * 128 + tidx

        if logical_row < seq_major_rows:
            physical_row = cutlass.Int32(-1)
            if logical_row < cu_seqlens_comp[n_seq]:
                seq_id = cutlass.Int32(0)
                for seq in range(n_seq):
                    if logical_row >= cu_seqlens_comp[seq + 1]:
                        seq_id = seq + 1

                comp_id = logical_row - cu_seqlens_comp[seq_id]
                group_last_row = cu_seqlens[seq_id] + (comp_id + 1) * ratio - 1
                owner_rank = group_last_row // l_local
                if owner_rank < 0:
                    owner_rank = 0
                if owner_rank >= cp_size:
                    owner_rank = cp_size - 1

                rank_start = owner_rank * l_local
                first_seq_id = cutlass.Int32(0)
                for seq in range(n_seq):
                    if rank_start >= cu_seqlens[seq + 1]:
                        first_seq_id = seq + 1
                if first_seq_id >= n_seq:
                    first_seq_id = n_seq - 1

                first_visible_numer = rank_start - d_comp - cu_seqlens[first_seq_id]
                if first_visible_numer < 0:
                    first_visible_numer = 0
                first_comp_id = (first_visible_numer + ratio - 1) // ratio
                first_logical_row = cu_seqlens_comp[first_seq_id] + first_comp_id
                physical_row = owner_rank * c_cap + logical_row - first_logical_row
            seq_to_rank_row[logical_row] = physical_row

    @cute.jit
    def _compressor_rank_row_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_comp: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        n_seq: cutlass.Int32,
        l_local: cutlass.Int32,
        cp_size: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        c_cap: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _compressor_rank_row_kernel,
            "dsv4_cp_compressor_rank_row",
            (
                cu_seqlens,
                cu_seqlens_comp,
                seq_to_rank_row,
                n_seq,
                l_local,
                cp_size,
                ratio,
                d_comp,
                c_cap,
                seq_major_rows,
            ),
            grid=(cute.ceil_div(seq_major_rows, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    # Kernel contract:
    #   grad_hidden_compact: gradient of compact rows, shape (compact_len, row_width).
    #   grad_hidden_local: output gradient, shape (l_local, row_width).
    #   grad_boundary_hidden: output gradient, shape (d_window, row_width).
    #   Reconstructs the same compact source mapping as forward and scatters each
    #   compact-row gradient back to either boundary or local hidden.
    @cute.kernel
    def _compressor_input_compact_bwd_kernel(
        grad_hidden_compact: cute.Tensor,
        grad_hidden_local: cute.Tensor,
        grad_boundary_hidden: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        out_row = bidx

        if cutlass.const_expr(grad_hidden_compact.element_type.width == 16 and row_width % 4 == 0):
            vec_elems = cutlass.Int32(4)
            vec_cols = row_width // 4
        else:
            vec_elems = cutlass.Int32(1)
            vec_cols = row_width

        compact_row = cutlass.Int32(-1)
        range_start = global_start
        range_end = global_start + l_local
        first_range_group_start = range_start - d_comp
        src_global = range_start + (out_row - d_window)
        dst_row = out_row - d_window
        if out_row < d_window:
            dst_row = out_row

        if tidx == 0 or tidx == 32:
            running_tokens = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                local_seq_end = seq_end
                if local_seq_end > range_end:
                    local_seq_end = range_end

                if seq_start < local_seq_end and range_start < local_seq_end:
                    first_visible_numer = first_range_group_start - seq_start
                    if first_visible_numer < 0:
                        first_visible_numer = 0
                    first_visible_group = (first_visible_numer + ratio - 1) // ratio
                    stop_visible_group = (local_seq_end - seq_start) // ratio
                    visible_group_count = stop_visible_group - first_visible_group
                    if visible_group_count < 0:
                        visible_group_count = 0
                    visible_token_count = visible_group_count * ratio

                    if src_global >= seq_start and src_global < seq_end:
                        seq_offset = src_global - seq_start
                        comp_id = seq_offset // ratio
                        token_in_group = seq_offset - comp_id * ratio
                        if comp_id >= first_visible_group and comp_id < stop_visible_group:
                            compact_row = (
                                running_tokens
                                + (comp_id - first_visible_group) * ratio
                                + token_in_group
                            )
                    running_tokens = running_tokens + visible_token_count

        compact_row = cute.arch.shuffle_sync(compact_row, 0, mask=-1, mask_and_clamp=31)

        vec_col = tidx
        while vec_col < vec_cols:
            dst_offset = dst_row * row_width + vec_col * vec_elems
            if cutlass.const_expr(
                grad_hidden_compact.element_type.width == 16 and row_width % 4 == 0
            ):
                value = cutlass.Int64(0)
                if compact_row >= 0 and compact_row < compact_len:
                    src_offset = compact_row * row_width + vec_col * vec_elems
                    src_ptr = cute.recast_ptr(
                        grad_hidden_compact.iterator + src_offset, dtype=cutlass.Int64
                    )
                    value = cute.arch.load(
                        src_ptr.llvm_ptr,
                        cutlass.Int64,
                        level1_eviction_priority="evict_no_allocate",
                    )
                if out_row < d_window:
                    dst_ptr = cute.recast_ptr(
                        grad_boundary_hidden.iterator + dst_offset, dtype=cutlass.Int64
                    )
                    cute.arch.store(dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first")
                else:
                    dst_ptr = cute.recast_ptr(
                        grad_hidden_local.iterator + dst_offset, dtype=cutlass.Int64
                    )
                    cute.arch.store(dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first")
            else:
                value = cutlass.Float32(0.0).to(grad_hidden_compact.element_type)
                if compact_row >= 0 and compact_row < compact_len:
                    value = grad_hidden_compact[compact_row, vec_col]
                if out_row < d_window:
                    grad_boundary_hidden[dst_row, vec_col] = value
                else:
                    grad_hidden_local[dst_row, vec_col] = value
            vec_col = vec_col + 64

    # Launch contract for _compressor_input_compact_bwd_kernel.
    #   total_work is the number of boundary + local gradient rows.
    @cute.jit
    def _compressor_input_compact_bwd_launch(
        grad_hidden_compact: cute.Tensor,
        grad_hidden_local: cute.Tensor,
        grad_boundary_hidden: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _compressor_input_compact_bwd_kernel,
            "dsv4_cp_compressor_input_compact_bwd",
            (
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
            ),
            grid=(total_work, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _build_attention_indices_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_unpadded: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        indexer_physical_rows: cute.Tensor,
        q_padding_mask: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
        total_width: cutlass.Int32,
        index_mode: cutlass.Constexpr,
        compressed_is_sequence_major: cutlass.Constexpr,
        has_unpadded_seqlens: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        # 0: selected top-k, 1: all visible compressed rows, 2: indexer loss.
        if cutlass.const_expr(index_mode == 0):
            row = linear
        else:
            row = bidx

        if row < l_local:
            global_q = global_start + row
            seq_start_found = cutlass.Int32(-1)
            seq_comp_start = cutlass.Int32(0)
            seq_comp_len = cutlass.Int32(0)
            seq_real_len = cutlass.Int32(0)
            if cutlass.const_expr(index_mode == 0):
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    if global_q >= seq_start and global_q < seq_end:
                        seq_start_found = seq_start
                        if cutlass.const_expr(has_unpadded_seqlens):
                            seq_real_len = cu_seqlens_unpadded[seq + 1] - cu_seqlens_unpadded[seq]
                        if ratio > 1 and compressed_width > 0:
                            seq_comp_start = cu_seqlens_compressed[seq]
                            seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start
            else:
                shared = utils.SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(4))
                if tidx == 0:
                    for seq in range(n_seq):
                        seq_start = cu_seqlens[seq]
                        seq_end = cu_seqlens[seq + 1]
                        if global_q >= seq_start and global_q < seq_end:
                            seq_start_found = seq_start
                            if cutlass.const_expr(has_unpadded_seqlens):
                                seq_real_len = (
                                    cu_seqlens_unpadded[seq + 1] - cu_seqlens_unpadded[seq]
                                )
                            if ratio > 1 and compressed_width > 0:
                                seq_comp_start = cu_seqlens_compressed[seq]
                                seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start
                    shared[0] = seq_start_found
                    shared[1] = seq_comp_start
                    shared[2] = seq_comp_len
                    shared[3] = seq_real_len
                cute.arch.sync_threads()
                seq_start_found = shared[0]
                seq_comp_start = shared[1]
                seq_comp_len = shared[2]
                seq_real_len = shared[3]

            is_padding = cutlass.Int32(0)
            if cutlass.const_expr(has_unpadded_seqlens):
                if cutlass.const_expr(index_mode == 0):
                    if seq_start_found < 0:
                        is_padding = 1
                    elif global_q - seq_start_found >= seq_real_len:
                        is_padding = 1
                    q_padding_mask[row] = is_padding.to(q_padding_mask.element_type)
                elif tidx == 0:
                    if seq_start_found < 0:
                        is_padding = 1
                    elif global_q - seq_start_found >= seq_real_len:
                        is_padding = 1
                    q_padding_mask[row] = is_padding.to(q_padding_mask.element_type)

            if cutlass.const_expr(index_mode != 0):
                col = tidx
                while col < total_width:
                    topk_value = cutlass.Int32(-1)
                    physical_value = cutlass.Int32(-1)
                    length = cutlass.Int32(0)
                    if seq_start_found >= 0:
                        window_start = global_q - window_size + 1
                        if window_start < seq_start_found:
                            window_start = seq_start_found
                        window_count = global_q - window_start + 1
                        if cutlass.const_expr(index_mode == 2):
                            if col < compressed_width:
                                comp_id = indexer_topk[row, col]
                                if comp_id >= 0 and comp_id < seq_comp_len:
                                    seq_major_id = seq_comp_start + comp_id
                                    if seq_major_id < seq_major_rows:
                                        if cutlass.const_expr(compressed_is_sequence_major):
                                            physical_value = seq_major_id
                                        else:
                                            physical_value = seq_to_rank_row[seq_major_id]
                                        if physical_value >= 0 and physical_value < compressed_rows:
                                            topk_value = compressed_base + physical_value
                                        else:
                                            physical_value = cutlass.Int32(-1)
                            else:
                                window_col = col - compressed_width
                                if window_col < window_count:
                                    pos = window_start + window_col
                                    if pos < global_start:
                                        topk_value = pos - (global_start - d_window)
                                    else:
                                        topk_value = d_window + pos - global_start
                        else:
                            comp_count = cutlass.Int32(0)
                            if ratio > 1 and compressed_width > 0:
                                comp_count = (global_q - seq_start_found + 1) // ratio
                                if comp_count > compressed_width:
                                    comp_count = compressed_width
                                if comp_count > seq_comp_len:
                                    comp_count = seq_comp_len
                            length = window_count + comp_count
                            if col < window_count:
                                pos = window_start + col
                                if pos < global_start:
                                    topk_value = pos - (global_start - d_window)
                                else:
                                    topk_value = d_window + pos - global_start
                            elif col < length:
                                seq_major_id = seq_comp_start + col - window_count
                                if seq_major_id < seq_major_rows:
                                    if cutlass.const_expr(compressed_is_sequence_major):
                                        physical_id = seq_major_id
                                    else:
                                        physical_id = seq_to_rank_row[seq_major_id]
                                    if physical_id >= 0 and physical_id < compressed_rows:
                                        topk_value = compressed_base + physical_id
                    elif total_width > 0 and cutlass.const_expr(index_mode != 2):
                        length = 1
                        if col == 0:
                            topk_value = 0
                    topk_idxs[row, col] = topk_value
                    if cutlass.const_expr(index_mode == 2):
                        if col < compressed_width:
                            indexer_physical_rows[row, col] = physical_value
                    elif col == 0:
                        topk_length[row] = length
                    col = col + 128
            else:
                for out_col in range(total_width):
                    topk_idxs[row, out_col] = -1
                topk_length[row] = 0

                if seq_start_found >= 0:
                    write_col = cutlass.Int32(0)
                    window_start = global_q - window_size + 1
                    if window_start < seq_start_found:
                        window_start = seq_start_found
                    window_count = global_q - window_start + 1
                    for window_col in range(window_size):
                        if window_col < window_count:
                            pos = window_start + window_col
                            if pos < global_start:
                                topk_idxs[row, write_col] = pos - (global_start - d_window)
                            else:
                                topk_idxs[row, write_col] = d_window + pos - global_start
                            write_col = write_col + 1

                    if ratio > 1 and compressed_width > 0:
                        for compressed_col in range(compressed_width):
                            comp_id = indexer_topk[row, compressed_col]
                            if comp_id >= 0 and comp_id < seq_comp_len:
                                seq_major_id = seq_comp_start + comp_id
                                if seq_major_id < seq_major_rows:
                                    if cutlass.const_expr(compressed_is_sequence_major):
                                        physical_id = seq_major_id
                                    else:
                                        physical_id = seq_to_rank_row[seq_major_id]
                                    if physical_id >= 0 and physical_id < compressed_rows:
                                        topk_idxs[row, write_col] = compressed_base + physical_id
                                        write_col = write_col + 1
                    topk_length[row] = write_col
                elif total_width > 0:
                    topk_idxs[row, 0] = 0
                    topk_length[row] = 1

    @cute.jit
    def _build_attention_indices_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_unpadded: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        indexer_physical_rows: cute.Tensor,
        q_padding_mask: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
        total_width: cutlass.Int32,
        index_mode: cutlass.Constexpr,
        compressed_is_sequence_major: cutlass.Constexpr,
        has_unpadded_seqlens: cutlass.Constexpr,
        launch_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_named(
            _build_attention_indices_kernel,
            "dsv4_thd_build_attention_indices",
            (
                cu_seqlens,
                cu_seqlens_unpadded,
                cu_seqlens_compressed,
                indexer_topk,
                seq_to_rank_row,
                topk_idxs,
                topk_length,
                indexer_physical_rows,
                q_padding_mask,
                n_seq,
                global_start,
                l_local,
                d_window,
                window_size,
                ratio,
                compressed_width,
                seq_major_rows,
                compressed_rows,
                compressed_base,
                total_width,
                index_mode,
                compressed_is_sequence_major,
                has_unpadded_seqlens,
            ),
            grid=(cute.ceil_div(launch_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


# =============================================================================
# Torch Wrapper Functions
# =============================================================================
# Torch-facing entry points called by csa.py and cp_utils.py. They validate
# inputs, allocate outputs, and dispatch the CuTeDSL kernels defined above.
# =============================================================================


def _require_cute(message: str, *tensors: Optional[torch.Tensor]) -> None:
    """Raise ``RuntimeError`` when a wrapper cannot use CuTeDSL kernels."""
    if not _CUTE_AVAILABLE or not all(tensor is None or tensor.is_cuda for tensor in tensors):
        raise RuntimeError(message)


_COMPILED_LAUNCH_CACHE = {}


def _run_compiled_launch(
    launch_fn,
    tensor_args: Tuple[torch.Tensor, ...],
    scalar_args: Tuple,
    static_arg_indices: Tuple[int, ...] = (),
) -> None:
    """Compile/cache a CuTe launch function and invoke it on the current stream."""
    static_arg_set = set(static_arg_indices)
    key = (
        launch_fn.__name__,
        tuple(
            (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())) for tensor in tensor_args
        ),
        tuple((i, scalar_args[i]) for i in static_arg_indices),
    )
    compiled = _COMPILED_LAUNCH_CACHE.get(key)
    if compiled is None:
        cap = torch.cuda.get_device_capability()
        arch = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}.get(cap)
        if arch is None:
            raise RuntimeError(
                f"Unsupported GPU compute capability {cap} for CSA CP CuTe kernels; "
                "supported architectures: sm_90a, sm_100a, sm_103a."
            )
        cute_tensor_args = []
        for tensor in tensor_args:
            cute_tensor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True)
            if tensor.ndim != 0:
                cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=tensor.ndim - 1)
            cute_tensor_args.append(cute_tensor)
        compiled = cute.compile(
            launch_fn,
            *cute_tensor_args,
            *(
                arg if i in static_arg_set else cutlass.Int32(arg)
                for i, arg in enumerate(scalar_args)
            ),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=f"--enable-tvm-ffi --gpu-arch {arch}",
        )
        _COMPILED_LAUNCH_CACHE[key] = compiled
    compiled(
        *tensor_args,
        *(cutlass.Int32(arg) for i, arg in enumerate(scalar_args) if i not in static_arg_set),
        cuda.CUstream(torch.cuda.current_stream(tensor_args[0].device).cuda_stream),
    )


class CompressorInputCompact(torch.autograd.Function):
    """Compact local and boundary rows and build the CP compressor metadata.

    Inputs:
        hidden_local: CUDA tensor, shape ``(l_local, ...)``.
        boundary_hidden: CUDA tensor, shape ``(d_window, ...)``.
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start: first global row in ``hidden_local``.
        ratio/d_comp/c_cap: compressor window and fixed group capacity.
        cp_size: number of context-parallel ranks.

    Outputs:
        ``hidden_compact`` shape ``(c_cap * ratio, ...)``, same dtype as hidden,
        ``comp_ids`` and ``position_ids`` int32 tensors of shape ``(c_cap,)``,
        local token/group prefixes and global group prefixes of shape
        ``(n_seq + 1,)``, and the global sequence-major to rank-major row map.
        The kernel copies the ratio source tokens for each visible compressed
        group while emitting all metadata needed by the compressor and gather.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_local: torch.Tensor,
        boundary_hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        global_start: int,
        ratio: int,
        d_comp: int,
        c_cap: int,
        cp_size: int,
    ):
        """Compact hidden rows and produce graph-safe compressor metadata."""
        _require_cute(
            "DSv4 CP compressor compaction requires CUDA tensors and CuTeDSL.",
            hidden_local,
            boundary_hidden,
            cu_seqlens,
        )
        l_local = hidden_local.shape[0]
        d_window = boundary_hidden.shape[0]
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.compact_args = (int(global_start), int(l_local), int(ratio), int(d_comp), int(d_window))
        ctx.save_for_backward(cu_seqlens)

        compact_len = int(c_cap) * int(ratio)
        hidden_compact = hidden_local.new_empty((compact_len,) + tuple(hidden_local.shape[1:]))
        comp_ids = torch.empty((int(c_cap),), dtype=torch.int32, device=hidden_local.device)
        position_ids = torch.empty_like(comp_ids)
        metadata_shape = (cu_seqlens.shape[0],)
        local_cu_seqlens = torch.empty(
            metadata_shape, dtype=torch.int32, device=hidden_local.device
        )
        local_cu_seqlens_comp = torch.empty_like(local_cu_seqlens)
        cu_seqlens_comp = torch.empty_like(local_cu_seqlens)
        row_width = math.prod(hidden_local.shape[1:])
        _run_compiled_launch(
            _compressor_input_compact_fwd_launch,
            (
                hidden_local.reshape(int(l_local), row_width),
                boundary_hidden.reshape(int(d_window), row_width),
                hidden_compact.reshape(compact_len, row_width),
                cu_seqlens,
                comp_ids,
                position_ids,
                local_cu_seqlens,
                local_cu_seqlens_comp,
                cu_seqlens_comp,
            ),
            (
                cu_seqlens.shape[0] - 1,
                int(global_start),
                int(l_local),
                int(ratio),
                int(d_comp),
                int(d_window),
                compact_len,
                row_width,
            ),
            static_arg_indices=(3, 4, 5, 7),
        )

        seq_major_rows = (int(l_local) * int(cp_size)) // int(ratio)
        seq_to_rank_row = torch.empty(
            (seq_major_rows,), dtype=torch.int32, device=hidden_local.device
        )
        if seq_major_rows > 0:
            _run_compiled_launch(
                _compressor_rank_row_launch,
                (cu_seqlens, cu_seqlens_comp, seq_to_rank_row),
                (
                    cu_seqlens.shape[0] - 1,
                    int(l_local),
                    int(cp_size),
                    int(ratio),
                    int(d_comp),
                    int(c_cap),
                    seq_major_rows,
                ),
                static_arg_indices=(3, 4),
            )
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            comp_ids,
            position_ids,
            local_cu_seqlens,
            local_cu_seqlens_comp,
            cu_seqlens_comp,
            seq_to_rank_row,
        )
        return (
            hidden_compact,
            comp_ids,
            position_ids,
            local_cu_seqlens,
            local_cu_seqlens_comp,
            cu_seqlens_comp,
            seq_to_rank_row,
        )

    @staticmethod
    def backward(
        ctx,
        grad_hidden_compact: torch.Tensor,
        _grad_comp_ids: Optional[torch.Tensor],
        _grad_position_ids: Optional[torch.Tensor],
        _grad_local_cu_seqlens: Optional[torch.Tensor],
        _grad_local_cu_seqlens_comp: Optional[torch.Tensor],
        _grad_cu_seqlens_comp: Optional[torch.Tensor],
        _grad_seq_to_rank_row: Optional[torch.Tensor],
    ):
        """Scatter compacted compressor gradients to local and boundary rows."""
        (cu_seqlens,) = ctx.saved_tensors
        global_start, l_local, ratio, d_comp, d_window = ctx.compact_args
        grad_hidden_compact = grad_hidden_compact.contiguous()
        grad_hidden = grad_hidden_compact.new_empty(ctx.hidden_shape)
        grad_boundary = grad_hidden_compact.new_empty(ctx.boundary_shape)
        compact_len = grad_hidden_compact.shape[0]
        row_width = math.prod(ctx.hidden_shape[1:])
        _run_compiled_launch(
            _compressor_input_compact_bwd_launch,
            (
                grad_hidden_compact.reshape(compact_len, row_width),
                grad_hidden.reshape(l_local, row_width),
                grad_boundary.reshape(d_window, row_width),
                cu_seqlens,
            ),
            (
                cu_seqlens.shape[0] - 1,
                global_start,
                l_local,
                ratio,
                d_comp,
                d_window,
                compact_len,
                row_width,
                l_local + d_window,
            ),
            static_arg_indices=(3, 4, 5, 7),
        )
        return (grad_hidden, grad_boundary, *([None] * 6))


def build_attention_indices(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    seq_to_rank_row: Optional[torch.Tensor] = None,
    for_indexer_loss: bool = False,
    compressed_base: int | None = None,
    compressed_rows: int | None = None,
    compressed_is_sequence_major: bool = False,
    cu_seqlens_unpadded: Optional[torch.Tensor] = None,
    output_alignment: int = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build final sparse-attention and optional indexer-loss indices.

    Inputs:
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start/l_local: this rank's global query start and row count.
        d_window/window_size: physical left-window capacity and per-query window width.
        ratio/compressed_width: compression mode and number of compressed columns.
        compressed_topk: optional per-sequence compressed ids, shape
            ``(l_local, compressed_width)``.
        cu_seqlens_compressed/seq_to_rank_row: compressed row metadata and the
            optional sequence-major to rank-major physical-row mapping.
        for_indexer_loss: put compressed ids first and return their physical rows.
        compressed_base: first compressed row in the final KV buffer. Defaults
            to ``d_window + l_local``, matching the context-parallel raw layout.
        compressed_rows: physical compressed-buffer capacity. Required whenever
            ``compressed_width`` is nonzero.
        compressed_is_sequence_major: compressed rows already use sequence-major
            order, so logical rows address them directly without an identity map.
        cu_seqlens_unpadded: optional true sequence lengths when ``cu_seqlens``
            describes CUDA-graph-padded rows.
        output_alignment: pad the final attention-index width to this multiple.
            The logical window/compressed widths and ``topk_length`` are unchanged.

    Outputs:
        ``topk_idxs`` int32, shape ``(l_local, padded_width)`` where
        ``padded_width`` rounds ``window_size + compressed_width`` up to
        ``output_alignment``;
        optional ``topk_length`` int32, shape ``(l_local,)``; and optional
        ``indexer_physical_rows`` int32, shape ``(l_local, compressed_width)``.
        The fourth result is an optional bool padding-row mask, shape ``(l_local,)``.
        Normal attention puts window ids first. Indexer-loss mode puts selected
        compressed ids first and returns their physical compressed-buffer rows.
    """
    if for_indexer_loss and compressed_topk is None:
        raise RuntimeError("DSv4 THD indexer-loss indices require compressed_topk.")
    _require_cute(
        "DSv4 THD final indices require CUDA tensors and CuTeDSL.",
        cu_seqlens,
        compressed_topk,
        cu_seqlens_compressed,
        seq_to_rank_row,
        cu_seqlens_unpadded,
    )
    if cu_seqlens_unpadded is not None and cu_seqlens_unpadded.shape != cu_seqlens.shape:
        raise ValueError("padded and unpadded cumulative lengths must have the same shape")
    global_start, l_local = int(global_start), int(l_local)
    compressed_width = int(compressed_width)
    output_alignment = int(output_alignment)
    if output_alignment <= 0:
        raise ValueError(f"output_alignment must be positive, got {output_alignment}")
    compressed_is_sequence_major = bool(compressed_is_sequence_major)
    if compressed_width > 0 and compressed_rows is None:
        raise RuntimeError("DSv4 THD final indices require the compressed-buffer capacity.")
    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if compressed_is_sequence_major:
        seq_major_rows = int(compressed_rows or 0)
    else:
        if compressed_width > 0 and seq_to_rank_row is None:
            raise RuntimeError(
                "Rank-major DSv4 final indices require a sequence-to-physical-row map."
            )
        seq_major_rows = 0 if seq_to_rank_row is None else seq_to_rank_row.shape[0]
    if seq_to_rank_row is None:
        seq_to_rank_row = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)

    if compressed_rows is None:
        compressed_rows = 0
    compressed_rows = int(compressed_rows)
    if compressed_base is None:
        compressed_base = int(d_window) + int(l_local)
    compressed_base = int(compressed_base)

    logical_width = int(window_size) + compressed_width
    total_width = (
        (logical_width + output_alignment - 1) // output_alignment * output_alignment
        if logical_width > 0
        else 0
    )
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    index_mode = 2 if for_indexer_loss else int(compressed_topk is None)
    if compressed_topk is None:
        compressed_topk = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
    if for_indexer_loss:
        topk_length_kernel = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)
        indexer_physical_rows = torch.empty_like(compressed_topk)
    else:
        topk_length_kernel = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens.device)
        indexer_physical_rows = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
    has_unpadded_seqlens = cu_seqlens_unpadded is not None
    # Store byte flags in CuTe and reinterpret them as torch.bool without a
    # conversion launch; both dtypes have one-byte PyTorch storage.
    if cu_seqlens_unpadded is None:
        cu_seqlens_unpadded = cu_seqlens
        q_padding_mask_kernel = torch.empty((1,), dtype=torch.uint8, device=cu_seqlens.device)
    else:
        q_padding_mask_kernel = torch.empty((l_local,), dtype=torch.uint8, device=cu_seqlens.device)
    launch_work = l_local * 128 if index_mode else l_local
    _run_compiled_launch(
        _build_attention_indices_launch,
        (
            cu_seqlens,
            cu_seqlens_unpadded,
            cu_seqlens_compressed,
            compressed_topk,
            seq_to_rank_row,
            topk_idxs,
            topk_length_kernel,
            indexer_physical_rows,
            q_padding_mask_kernel,
        ),
        (
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            d_window,
            window_size,
            ratio,
            compressed_width,
            seq_major_rows,
            compressed_rows,
            compressed_base,
            total_width,
            index_mode,
            compressed_is_sequence_major,
            has_unpadded_seqlens,
            launch_work,
        ),
        static_arg_indices=(11, 12, 13),
    )
    q_padding_mask = q_padding_mask_kernel.view(torch.bool) if has_unpadded_seqlens else None
    if for_indexer_loss:
        return topk_idxs, None, indexer_physical_rows, q_padding_mask
    return topk_idxs, topk_length_kernel, None, q_padding_mask

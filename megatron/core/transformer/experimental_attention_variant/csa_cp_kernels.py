# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTeDSL kernels for DSv4 THD context parallel layout work.

This module is intentionally below the MCore-facing CP utility layer.  Callers
outside ``csa_cp_utils.py`` should not import this file directly.  The functions
here allocate output tensors, compile/cache CuTeDSL launches, and execute small
layout kernels for the DSv4 CP path.
"""

import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
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

    def _launch_1d(kernel, name: str, args: Tuple, work: cutlass.Int32, stream: cuda.CUstream):
        """Launch a 1-D CuTe kernel with 128 threads per block."""
        kernel.set_name_prefix(name)
        kernel(*args).launch(
            grid=(cute.ceil_div(work, 128), 1, 1), block=(128, 1, 1), stream=stream
        )

    # Kernel contract:
    #   x/out: row-major tensor views, shape (local_rows, row_width), bf16/fp16/fp32.
    #   cos/sin: rotary tables, shape (max_seq_len, pos_dim), floating dtype.
    #   cu_seqlens: int32 global sequence prefix sums, shape (n_seq + 1,).
    #   chunk0/1_global_start and chunk_len map local rows back to two global
    #   packed-token chunks. Applies non-interleaved RoPE to each head's position
    #   dims; if adjoint is set, applies the backward/adjoint transform.
    @cute.kernel
    def _thd_local_rope_kernel(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        chunk0_global_start: cutlass.Int32,
        chunk1_global_start: cutlass.Int32,
        chunk_len: cutlass.Int32,
        clamp_to_valid_token: cutlass.Int32,
        row_width: cutlass.Constexpr,
        head_dim: cutlass.Int32,
        nope_dim: cutlass.Int32,
        pos_dim: cutlass.Int32,
        inverse: cutlass.Int32,
        adjoint: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            pair = linear % (pos_dim // 2)
            tmp = linear // (pos_dim // 2)
            head = tmp % (row_width // head_dim)
            row = tmp // (row_width // head_dim)
            col = head * head_dim + nope_dim + pair * 2

            local_row = row
            global_token = chunk0_global_start + local_row
            if row >= chunk_len:
                local_row = row - chunk_len
                global_token = chunk1_global_start + local_row

            if clamp_to_valid_token != 0:
                if global_token < 0:
                    global_token = 0
                last_token = cu_seqlens[n_seq] - 1
                if global_token > last_token:
                    global_token = last_token

            rope_pos = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                if global_token >= seq_start and global_token < seq_end:
                    rope_pos = global_token - seq_start

            x1 = x[row, col].to(cutlass.Float32)
            x2 = x[row, col + 1].to(cutlass.Float32)
            cos_left = cos[rope_pos, pair].to(cutlass.Float32)
            sin_left = sin[rope_pos, pair].to(cutlass.Float32)
            cos_right = cos[rope_pos, pos_dim // 2 + pair].to(cutlass.Float32)
            sin_right = sin[rope_pos, pos_dim // 2 + pair].to(cutlass.Float32)
            if inverse != 0:
                sin_left = cutlass.Float32(0.0) - sin_left
                sin_right = cutlass.Float32(0.0) - sin_right

            x1_cos_left = (x1 * cos_left).to(out.element_type).to(cutlass.Float32)
            x2_sin_left = (x2 * sin_left).to(out.element_type).to(cutlass.Float32)
            x2_cos_right = (x2 * cos_right).to(out.element_type).to(cutlass.Float32)
            x1_sin_right = (x1 * sin_right).to(out.element_type).to(cutlass.Float32)
            value_left = x1_cos_left - x2_sin_left
            value_right = x2_cos_right + x1_sin_right
            if adjoint != 0:
                x2_sin_right = (x2 * sin_right).to(out.element_type).to(cutlass.Float32)
                x1_sin_left = (x1 * sin_left).to(out.element_type).to(cutlass.Float32)
                value_left = x1_cos_left + x2_sin_right
                value_right = cutlass.Float32(0.0) - x1_sin_left + x2_cos_right
            out[row, col] = value_left.to(out.element_type)
            out[row, col + 1] = value_right.to(out.element_type)

    # Launch contract for _thd_local_rope_kernel.
    #   row_width is static; total_work is local_rows * n_heads * (pos_dim / 2).
    @cute.jit
    def _thd_local_rope_launch(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        chunk0_global_start: cutlass.Int32,
        chunk1_global_start: cutlass.Int32,
        chunk_len: cutlass.Int32,
        clamp_to_valid_token: cutlass.Int32,
        row_width: cutlass.Constexpr,
        head_dim: cutlass.Int32,
        nope_dim: cutlass.Int32,
        pos_dim: cutlass.Int32,
        inverse: cutlass.Int32,
        adjoint: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _thd_local_rope_kernel,
            "dsv4_cp_thd_local_rope",
            (
                x,
                out,
                cos,
                sin,
                cu_seqlens,
                n_seq,
                chunk0_global_start,
                chunk1_global_start,
                chunk_len,
                clamp_to_valid_token,
                row_width,
                head_dim,
                nope_dim,
                pos_dim,
                inverse,
                adjoint,
                total_work,
            ),
            total_work,
            stream,
        )

    # Kernel contract:
    #   x/out: row-major compressed tensor views, shape (rows, row_width).
    #   cos/sin: rotary tables, shape (max_seq_len, pos_dim).
    #   comp_ids: int32 compressed group id per row, shape (rows,); -1 is treated as 0.
    #   ratio maps comp_id to original token position comp_id * ratio.
    #   Copies no-RoPE dims and rotates each non-interleaved pair in the position dims.
    @cute.kernel
    def _thd_compressed_rope_kernel(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        comp_ids: cute.Tensor,
        ratio: cutlass.Int32,
        row_width: cutlass.Constexpr,
        head_dim: cutlass.Int32,
        nope_dim: cutlass.Int32,
        pos_dim: cutlass.Int32,
        inverse: cutlass.Int32,
        adjoint: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            row = linear // row_width
            col = linear - row * row_width
            head_col = col - (col // head_dim) * head_dim

            if head_col < nope_dim:
                out[row, col] = x[row, col]
            else:
                comp_id = comp_ids[row]
                if comp_id < 0:
                    comp_id = 0
                rope_pos = comp_id * ratio

                pos_col = head_col - nope_dim
                pair = pos_col // 2
                pair_base = col - head_col + nope_dim + pair * 2
                x1 = x[row, pair_base].to(cutlass.Float32)
                x2 = x[row, pair_base + 1].to(cutlass.Float32)

                cos_left = cos[rope_pos, pair].to(cutlass.Float32)
                sin_left = sin[rope_pos, pair].to(cutlass.Float32)
                cos_right = cos[rope_pos, pos_dim // 2 + pair].to(cutlass.Float32)
                sin_right = sin[rope_pos, pos_dim // 2 + pair].to(cutlass.Float32)
                if inverse != 0:
                    sin_left = cutlass.Float32(0.0) - sin_left
                    sin_right = cutlass.Float32(0.0) - sin_right

                x1_cos_left = (x1 * cos_left).to(out.element_type).to(cutlass.Float32)
                x2_sin_left = (x2 * sin_left).to(out.element_type).to(cutlass.Float32)
                x2_cos_right = (x2 * cos_right).to(out.element_type).to(cutlass.Float32)
                x1_sin_right = (x1 * sin_right).to(out.element_type).to(cutlass.Float32)
                value = x1_cos_left - x2_sin_left
                if pos_col - pair * 2 != 0:
                    value = x2_cos_right + x1_sin_right
                if adjoint != 0:
                    x2_sin_right = (x2 * sin_right).to(out.element_type).to(cutlass.Float32)
                    x1_sin_left = (x1 * sin_left).to(out.element_type).to(cutlass.Float32)
                    value = x1_cos_left + x2_sin_right
                    if pos_col - pair * 2 != 0:
                        value = cutlass.Float32(0.0) - x1_sin_left + x2_cos_right
                out[row, col] = value.to(out.element_type)

    # Launch contract for _thd_compressed_rope_kernel.
    #   row_width is static; total_work is rows * row_width.
    @cute.jit
    def _thd_compressed_rope_launch(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        comp_ids: cute.Tensor,
        ratio: cutlass.Int32,
        row_width: cutlass.Constexpr,
        head_dim: cutlass.Int32,
        nope_dim: cutlass.Int32,
        pos_dim: cutlass.Int32,
        inverse: cutlass.Int32,
        adjoint: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _thd_compressed_rope_kernel,
            "dsv4_cp_thd_compressed_rope",
            (
                x,
                out,
                cos,
                sin,
                comp_ids,
                ratio,
                row_width,
                head_dim,
                nope_dim,
                pos_dim,
                inverse,
                adjoint,
                total_work,
            ),
            total_work,
            stream,
        )

    # Kernel contract:
    #   hidden_local: local hidden rows, shape (l_local, row_width), bf16/fp16/fp32.
    #   boundary_hidden: left boundary rows, shape (d_window, row_width).
    #   hidden_compact: output compact rows, shape (compact_len, row_width).
    #   cu_compact: int32 output prefix, shape (n_seq + 1,).
    #   seq_ids/comp_ids/valid: compressed metadata outputs, shape (c_cap,).
    #   Enumerates visible full compression groups in [global_start-d_comp,
    #   global_start+l_local), copies their ratio tokens from boundary/local
    #   hidden into hidden_compact, and emits compact sequence metadata.
    @cute.kernel
    def _compressor_input_compact_fwd_kernel(
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
        row_width: cutlass.Constexpr,
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

    # Launch contract for _compressor_input_compact_fwd_kernel.
    #   total_work covers max(compact_len * row_width, c_cap, n_seq + 1).
    @cute.jit
    def _compressor_input_compact_fwd_launch(
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
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _compressor_input_compact_fwd_kernel,
            "dsv4_cp_compressor_input_compact_fwd",
            (
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
            ),
            total_work,
            stream,
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
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        d_window: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
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

    # Launch contract for _compressor_input_compact_bwd_kernel.
    #   total_work is compact_len * row_width.
    @cute.jit
    def _compressor_input_compact_bwd_launch(
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
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
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
                total_work,
            ),
            total_work,
            stream,
        )

    # Kernel contract:
    #   kv_local/boundary_kv/compressed_rank_major: source tensors for forward,
    #       or output gradient tensors for backward, each flattened to
    #       ``(rows, row_width)``.
    #   kv_full: output tensor for forward, or grad_kv_full input for backward,
    #       shape ``(kv_full_capacity, row_width)``.
    #   rank-major metadata inputs describe compressed rows after all-gather.
    #   direction: 0 packs source tensors into kv_full; 1 scatters kv_full
    #       gradients back to the source tensors.
    #   Computes the physical KV-full row mapping inline. No source-map tensor is
    #   materialized or saved for backward; backward recomputes the same mapping.
    @cute.kernel
    def _thd_full_kv_pack_kernel(
        kv_local: cute.Tensor,
        boundary_kv: cute.Tensor,
        compressed_rank_major: cute.Tensor,
        kv_full: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        window_capacity_per_chunk: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        use_seq_major_map: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Constexpr,
        direction: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        if linear < total_work:
            out_row = linear // row_width
            col = linear - out_row * row_width
            source_kind = -1
            source_index = 0

            if chunk_count != 0:
                shared_compressed_base = chunk_count * window_capacity_per_chunk
                if out_row >= shared_compressed_base:
                    compressed_idx = out_row - shared_compressed_base
                    if compressed_idx >= 0 and compressed_idx < compressed_rows:
                        source_kind = 2
                        source_index = compressed_idx
                else:
                    chunk_id = out_row // window_capacity_per_chunk
                    chunk_row = out_row - chunk_id * window_capacity_per_chunk
                    chunk_start = chunk0_start
                    if chunk_id == 1:
                        chunk_start = chunk1_start
                    chunk_end = chunk_start + chunk_len

                    offset = 0
                    for seq in range(n_seq):
                        seq_start = cu_seqlens[seq]
                        seq_end = cu_seqlens[seq + 1]
                        local_seq_start = seq_start
                        if local_seq_start < chunk_start:
                            local_seq_start = chunk_start
                        local_seq_end = seq_end
                        if local_seq_end > chunk_end:
                            local_seq_end = chunk_end
                        if local_seq_start < local_seq_end:
                            window_start = local_seq_start - d_window
                            if window_start < seq_start:
                                window_start = seq_start
                            window_len = local_seq_end - window_start
                            if chunk_row >= offset and chunk_row < offset + window_len:
                                src_global = window_start + chunk_row - offset
                                if src_global < chunk_start:
                                    source_kind = 1
                                    source_index = (
                                        chunk_id * d_window + src_global - (chunk_start - d_window)
                                    )
                                else:
                                    source_kind = 0
                                    source_index = chunk_id * chunk_len + src_global - chunk_start
                            offset = offset + window_len
            else:
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
                        comp_start = 0
                        comp_end = 0
                        if ratio > 1 and use_seq_major_map != 0:
                            comp_start = cu_seqlens_compressed[seq]
                            comp_end = cu_seqlens_compressed[seq + 1]
                            comp_len = comp_end - comp_start
                        elif ratio > 1:
                            comp_len = (seq_end - seq_start) // ratio
                        seq_total = window_len + comp_len

                        if out_row >= offset and out_row < offset + seq_total:
                            inner = out_row - offset
                            if inner < window_len:
                                src_global = window_start + inner
                                if src_global < global_start:
                                    source_kind = 1
                                    source_index = src_global - (global_start - d_window)
                                else:
                                    source_kind = 0
                                    source_index = src_global - global_start
                            else:
                                comp_id = inner - window_len
                                if use_seq_major_map != 0:
                                    seq_major = comp_start + comp_id
                                    if seq_major >= comp_start and seq_major < comp_end:
                                        if seq_major >= 0 and seq_major < seq_major_rows:
                                            rank_major = rank_major_by_seq_major[seq_major]
                                            if rank_major >= 0:
                                                source_kind = 2
                                                source_index = rank_major
                                else:
                                    for comp_row in range(compressed_rows):
                                        if (
                                            valid_rank_major[comp_row]
                                            and seq_ids_rank_major[comp_row] == seq
                                            and comp_ids_rank_major[comp_row] == comp_id
                                        ):
                                            source_kind = 2
                                            source_index = comp_row
                        offset = offset + seq_total

            if direction == 0:
                value = cutlass.Float32(0.0).to(kv_full.element_type)
                if source_kind == 0:
                    value = kv_local[source_index, col]
                elif source_kind == 1:
                    value = boundary_kv[source_index, col]
                elif source_kind == 2:
                    value = compressed_rank_major[source_index, col]
                kv_full[out_row, col] = value
            else:
                if source_kind == 0:
                    kv_local[source_index, col] = kv_full[out_row, col]
                elif source_kind == 1:
                    boundary_kv[source_index, col] = kv_full[out_row, col]
                elif source_kind == 2:
                    compressed_rank_major[source_index, col] = kv_full[out_row, col]

    # Launch contract for _thd_full_kv_pack_kernel.
    #   Supports contiguous CP and two-chunk CP. ``use_seq_major_map`` selects
    #   whether compressed ids must be translated through rank_major_by_seq_major.
    #   ``row_width`` is constexpr to preserve the existing compiled layout key.
    @cute.jit
    def _thd_full_kv_pack_launch(
        kv_local: cute.Tensor,
        boundary_kv: cute.Tensor,
        compressed_rank_major: cute.Tensor,
        kv_full: cute.Tensor,
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        window_capacity_per_chunk: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        use_seq_major_map: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Constexpr,
        direction: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _thd_full_kv_pack_kernel,
            "dsv4_cp_thd_full_kv_pack",
            (
                kv_local,
                boundary_kv,
                compressed_rank_major,
                kv_full,
                seq_ids_rank_major,
                comp_ids_rank_major,
                valid_rank_major,
                cu_seqlens,
                cu_seqlens_compressed,
                rank_major_by_seq_major,
                n_seq,
                global_start,
                l_local,
                chunk0_start,
                chunk1_start,
                chunk_count,
                chunk_len,
                window_capacity_per_chunk,
                d_window,
                ratio,
                compressed_rows,
                use_seq_major_map,
                seq_major_rows,
                kv_full_capacity,
                row_width,
                direction,
                total_work,
            ),
            total_work,
            stream,
        )

    # Kernel contract:
    #   rank_by_seq_major: int32 output reverse map, shape (total_seq_major_rows,).
    #   Initializes the reverse map before the deterministic rank-map build.
    @cute.kernel
    def _repack_compressed_kv_init_rank_map_kernel(
        rank_major_by_seq_major: cute.Tensor,
        seq_major_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < seq_major_rows:
            rank_major_by_seq_major[linear] = -1

    # Kernel contract:
    #   seq_ids/comp_ids/valid: rank-major metadata, shape (rows,).
    #   cu_compressed: int32 compressed sequence prefixes, shape (n_seq + 1,).
    #   rank_by_seq_major: int32 output reverse map, shape (total_seq_major_rows,).
    #   Builds a deterministic seq-major -> rank-major map. Multiple rank-major
    #   candidates can describe the same compressed group near CP chunk
    #   boundaries, so the kernel selects the greatest rank-major row with
    #   atomic max instead of racing concurrent value writes.
    @cute.kernel
    def _repack_compressed_kv_rank_map_kernel(
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        rank_major_rows: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        rank_row = bidx * 128 + tidx
        if rank_row < rank_major_rows and valid_rank_major[rank_row] == 1:
            seq = seq_ids_rank_major[rank_row]
            comp = comp_ids_rank_major[rank_row]
            seq_major = cu_seqlens_compressed[seq] + comp
            if seq_major >= 0 and seq_major < seq_major_rows:
                rank_map_ptr = rank_major_by_seq_major.iterator + seq_major
                cute.arch.atomic_max(
                    rank_map_ptr.llvm_ptr,
                    rank_row,
                    sem="relaxed",
                    scope="gpu",
                )

    # Kernel contract:
    #   rank_major: compressed rows in rank-major order, shape (rows, row_width).
    #   rank_by_seq_major: deterministic reverse map built by
    #   _repack_compressed_kv_rank_map_kernel.
    #   out: seq-major compressed rows, shape (total_seq_major_rows, row_width).
    #   Gathers selected rank-major rows into seq-major order and zero-fills
    #   rows with no valid compressed group.
    @cute.kernel
    def _repack_compressed_kv_to_seq_major_kernel(
        compressed_rank_major: cute.Tensor,
        compressed_seq_major: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        rank_major_rows: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        if linear < seq_major_rows * row_width:
            row = linear // row_width
            col = linear - row * row_width
            rank_row = rank_major_by_seq_major[row]
            value = cutlass.Float32(0.0).to(compressed_seq_major.element_type)
            if rank_row >= 0 and rank_row < rank_major_rows:
                value = compressed_rank_major[rank_row, col]
            compressed_seq_major[row, col] = value

    # Launch contract for _repack_compressed_kv_init_rank_map_kernel.
    #   launch_work is seq-major rows.
    @cute.jit
    def _repack_compressed_kv_init_rank_map_launch(
        rank_major_by_seq_major: cute.Tensor,
        seq_major_rows: cutlass.Int32,
        launch_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _repack_compressed_kv_init_rank_map_kernel,
            "dsv4_cp_repack_compressed_kv_init_rank_map",
            (rank_major_by_seq_major, seq_major_rows),
            launch_work,
            stream,
        )

    # Launch contract for _repack_compressed_kv_rank_map_kernel.
    #   launch_work is rank-major rows.
    @cute.jit
    def _repack_compressed_kv_rank_map_launch(
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        rank_major_rows: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        launch_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _repack_compressed_kv_rank_map_kernel,
            "dsv4_cp_repack_compressed_kv_rank_map",
            (
                seq_ids_rank_major,
                comp_ids_rank_major,
                valid_rank_major,
                cu_seqlens_compressed,
                rank_major_by_seq_major,
                rank_major_rows,
                seq_major_rows,
            ),
            launch_work,
            stream,
        )

    # Launch contract for _repack_compressed_kv_to_seq_major_kernel.
    #   total_work is seq-major rows * row_width.
    @cute.jit
    def _repack_compressed_kv_to_seq_major_launch(
        compressed_rank_major: cute.Tensor,
        compressed_seq_major: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        rank_major_rows: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _repack_compressed_kv_to_seq_major_kernel,
            "dsv4_cp_repack_compressed_kv_to_seq_major",
            (
                compressed_rank_major,
                compressed_seq_major,
                rank_major_by_seq_major,
                rank_major_rows,
                seq_major_rows,
                row_width,
                total_work,
            ),
            total_work,
            stream,
        )

    # Kernel contract:
    #   cu_seqlens: int32, shape (n_seq + 1,), global packed sequence prefixes.
    #   seq_ids/comp_ids: int32 outputs, shape (total_rows,).
    #   valid: bool output, shape (total_rows,).
    #   total_rows is cp_size * c_cap_per_rank. For each rank-major compressed
    #   row, computes (seq_id, compressed group id) and marks whether that group
    #   ends inside the owning rank/chunk.
    @cute.kernel
    def _build_compressed_row_metadata_kernel(
        cu_seqlens: cute.Tensor,
        seq_ids: cute.Tensor,
        comp_ids: cute.Tensor,
        valid: cute.Tensor,
        n_seq: cutlass.Int32,
        cp_size: cutlass.Int32,
        chunk_len: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        c_cap_per_chunk: cutlass.Int32,
        c_cap_per_rank: cutlass.Int32,
        two_chunk: cutlass.Int32,
        total_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row < total_rows:
            rank = row // c_cap_per_rank
            rank_slot = row - rank * c_cap_per_rank
            slot = rank_slot
            rank_start = rank * chunk_len
            row_is_padding = False
            if two_chunk != 0:
                local_chunk = rank_slot // c_cap_per_chunk
                if local_chunk >= 2:
                    row_is_padding = True
                else:
                    slot = rank_slot - local_chunk * c_cap_per_chunk
                    chunk_id = rank
                    if local_chunk != 0:
                        chunk_id = cp_size * 2 - 1 - rank
                    rank_start = chunk_id * chunk_len
            rank_end = rank_start + chunk_len

            seq_ids[row] = -1
            comp_ids[row] = -1
            valid[row] = False

            if not row_is_padding:
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

    # Launch contract for _build_compressed_row_metadata_kernel.
    #   two_chunk selects one contiguous chunk per rank or two chunks
    #   per rank; c_cap_per_rank is the fixed local compressed capacity.
    @cute.jit
    def _build_compressed_row_metadata_launch(
        cu_seqlens: cute.Tensor,
        seq_ids: cute.Tensor,
        comp_ids: cute.Tensor,
        valid: cute.Tensor,
        n_seq: cutlass.Int32,
        cp_size: cutlass.Int32,
        chunk_len: cutlass.Int32,
        ratio: cutlass.Int32,
        d_comp: cutlass.Int32,
        c_cap_per_chunk: cutlass.Int32,
        c_cap_per_rank: cutlass.Int32,
        two_chunk: cutlass.Int32,
        total_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _build_compressed_row_metadata_kernel,
            "dsv4_cp_build_compressed_row_metadata",
            (
                cu_seqlens,
                seq_ids,
                comp_ids,
                valid,
                n_seq,
                cp_size,
                chunk_len,
                ratio,
                d_comp,
                c_cap_per_chunk,
                c_cap_per_rank,
                two_chunk,
                total_rows,
            ),
            total_rows,
            stream,
        )

    # Kernel contract:
    #   k_indexer_seq_major: compressed indexer K, shape (seq_major_rows, row_width).
    #   cu_q: int32 query prefixes, shape (n_seq + 1,).
    #   cu_compressed: int32 compressed K prefixes, shape (n_seq + 1,).
    #   k_topk: output packed K for local trapezoid top-k, shape (k_rows, row_width).
    #   cu_q_topk/cu_k_topk/seq_lens: int32 outputs, shape (n_seq + 1)/(n_seq + 1)/(n_seq).
    #   Copies the compressed K rows visible to local query rows and emits local
    #   q/k prefix sums for the indexer kernel.
    @cute.kernel
    def _build_indexer_topk_metadata_kernel(
        k_seq_major: cute.Tensor,
        k_topk: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        cu_q_topk: cute.Tensor,
        cu_k_topk: cute.Tensor,
        seq_lens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        row_width: cutlass.Constexpr,
        seq_major_rows: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if bidx == 0 and tidx == 0:
            global_end = global_start + l_local
            q_prefix = 0
            k_prefix = 0
            cu_q_topk[0] = cutlass.Int32(0)
            cu_k_topk[0] = cutlass.Int32(0)
            for seq in range(n_seq):
                seq_start = cu_seqlens_q[seq]
                seq_end = cu_seqlens_q[seq + 1]
                local_start = seq_start
                if local_start < global_start:
                    local_start = global_start
                local_end = seq_end
                if local_end > global_end:
                    local_end = global_end

                q_len = 0
                k_len = 0
                if local_start < local_end:
                    q_len = local_end - local_start
                    seq_comp_start = cu_seqlens_compressed[seq]
                    seq_comp_end = cu_seqlens_compressed[seq + 1]
                    seq_comp_len = seq_comp_end - seq_comp_start
                    k_len = (local_end - seq_start) // ratio
                    if k_len > seq_comp_len:
                        k_len = seq_comp_len

                q_prefix = q_prefix + q_len
                k_prefix = k_prefix + k_len
                cu_q_topk[seq + 1] = q_prefix
                cu_k_topk[seq + 1] = k_prefix
                for row in range(q_len):
                    seq_lens[q_prefix - q_len + row] = k_len

            actual_total = cu_seqlens_q[n_seq]
            padding_q = 0
            if global_end > actual_total:
                padding_start = actual_total
                if padding_start < global_start:
                    padding_start = global_start
                if padding_start < global_end:
                    padding_q = global_end - padding_start
            cu_q_topk[n_seq + 1] = q_prefix + padding_q
            cu_k_topk[n_seq + 1] = k_prefix
            for row in range(padding_q):
                seq_lens[q_prefix + row] = cutlass.Int32(0)

        linear = bidx * 128 + tidx
        if linear < total_work:
            dst_row = linear // row_width
            col = linear - dst_row * row_width
            k_topk[dst_row, col] = cutlass.Float32(0.0).to(k_topk.element_type)

            global_end = global_start + l_local
            k_prefix = 0
            src_row = -1
            for seq in range(n_seq):
                seq_start = cu_seqlens_q[seq]
                seq_end = cu_seqlens_q[seq + 1]
                local_start = seq_start
                if local_start < global_start:
                    local_start = global_start
                local_end = seq_end
                if local_end > global_end:
                    local_end = global_end

                k_len = 0
                if local_start < local_end:
                    seq_comp_start = cu_seqlens_compressed[seq]
                    seq_comp_end = cu_seqlens_compressed[seq + 1]
                    seq_comp_len = seq_comp_end - seq_comp_start
                    k_len = (local_end - seq_start) // ratio
                    if k_len > seq_comp_len:
                        k_len = seq_comp_len
                    if dst_row >= k_prefix and dst_row < k_prefix + k_len:
                        src_row = seq_comp_start + dst_row - k_prefix
                k_prefix = k_prefix + k_len

            if src_row >= 0 and src_row < seq_major_rows:
                k_topk[dst_row, col] = k_seq_major[src_row, col]

    # Launch contract for _build_indexer_topk_metadata_kernel.
    #   total_work covers max(k_rows * row_width, n_seq + 1).
    @cute.jit
    def _build_indexer_topk_metadata_launch(
        k_seq_major: cute.Tensor,
        k_topk: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        cu_q_topk: cute.Tensor,
        cu_k_topk: cute.Tensor,
        seq_lens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        row_width: cutlass.Constexpr,
        seq_major_rows: cutlass.Int32,
        total_work: cutlass.Int32,
        launch_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _build_indexer_topk_metadata_kernel,
            "dsv4_cp_build_indexer_topk_metadata",
            (
                k_seq_major,
                k_topk,
                cu_seqlens_q,
                cu_seqlens_compressed,
                cu_q_topk,
                cu_k_topk,
                seq_lens,
                n_seq,
                global_start,
                l_local,
                ratio,
                row_width,
                seq_major_rows,
                total_work,
            ),
            launch_work,
            stream,
        )

    # Kernel contract:
    #   cu_seqlens: int32 global query prefixes, shape (n_seq + 1,).
    #   optional indexer_topk: int32 compressed logical ids, shape (l_local, compressed_width).
    #   rank_major_by_seq_major: int32 reverse compressed map when compression is active.
    #   topk_idxs: int32 output, shape (l_local, window_size + compressed_width).
    #   topk_length: int32 output valid count, shape (l_local,).
    #   Lowers each local query row to physical KV-full ids: window ids first,
    #   then compressed ids from either dense causal range or indexer top-k.
    @cute.kernel
    def _build_attention_indices_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        window_capacity_per_chunk: cutlass.Int32,
        shared_compressed_base: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        has_indexer_topk: cutlass.Int32,
        has_compressed: cutlass.Int32,
        total_width: cutlass.Int32,
        parallel_dense: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        row_count = l_local
        if chunk_count != 0:
            row_count = chunk_count * chunk_len

        row = 0
        if parallel_dense != 0:
            if linear < row_count * total_width:
                row = linear // total_width
                col = linear - row * total_width
                topk_value = -1
                length = 0

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
                    global_end = global_start + row_count
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

                    window_start_for_q = global_q - window_size + 1
                    if window_start_for_q < seq_start_found:
                        window_start_for_q = seq_start_found
                    if window_start_for_q < seq_window_start:
                        window_start_for_q = seq_window_start
                    window_count = global_q - window_start_for_q + 1
                    if window_count < 0:
                        window_count = 0
                    if window_count > window_size:
                        window_count = window_size

                    comp_count = 0
                    if has_compressed != 0 and compressed_width > 0 and ratio > 1:
                        comp_count = (global_q - seq_start_found + 1) // ratio
                        if comp_count > compressed_width:
                            comp_count = compressed_width
                        if comp_count > seq_comp_len:
                            comp_count = seq_comp_len
                    length = window_count + comp_count
                    if col < window_count:
                        topk_value = seq_offset + window_start_for_q - seq_window_start + col
                    elif col < length:
                        topk_value = seq_offset + seq_window_len + col - window_count
                elif total_width > 0:
                    # Fused DSA sparse attention expects at least one KV id per
                    # query row. Padded THD rows carry zero loss gradient, so a
                    # dummy in-range id is enough to keep the kernel contract.
                    length = 1
                    if col == 0:
                        topk_value = 0

                topk_idxs[row, col] = topk_value
                if col == 0:
                    topk_length[row] = length
        else:
            row = linear
            if row >= row_count:
                row = row_count
        if parallel_dense == 0 and row < row_count:
            for col in range(total_width):
                topk_idxs[row, col] = -1
            topk_length[row] = 0

            local_start = global_start
            row_in_local = row
            chunk_id = 0
            if chunk_count != 0:
                chunk_id = row // chunk_len
                row_in_local = row - chunk_id * chunk_len
                local_start = chunk0_start
                if chunk_id == 1:
                    local_start = chunk1_start
            global_q = local_start + row_in_local
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
                seq_comp_start = 0
                seq_comp_end = 0
                global_end = local_start + row_count
                if chunk_count != 0:
                    global_end = local_start + chunk_len
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_start = seq_start
                    if local_seq_start < local_start:
                        local_seq_start = local_start
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
                            if chunk_count != 0:
                                comp_len = (
                                    cu_seqlens_compressed[seq + 1] - cu_seqlens_compressed[seq]
                                )
                            else:
                                comp_len = (seq_end - seq_start) // ratio
                        if seq == seq_id:
                            seq_window_start = window_start
                            seq_window_len = window_len
                            seq_comp_len = comp_len
                            if chunk_count != 0 and has_compressed != 0 and ratio > 1:
                                seq_comp_start = cu_seqlens_compressed[seq]
                                seq_comp_end = cu_seqlens_compressed[seq + 1]
                        if seq < seq_id:
                            seq_offset = seq_offset + window_len
                            if chunk_count == 0:
                                seq_offset = seq_offset + comp_len

                write_col = 0
                window_start_for_q = global_q - window_size + 1
                if window_start_for_q < seq_start_found:
                    window_start_for_q = seq_start_found
                window_count = global_q - window_start_for_q + 1
                for w in range(window_size):
                    pos = window_start_for_q + w
                    if (
                        w < window_count
                        and pos >= seq_window_start
                        and pos < seq_window_start + seq_window_len
                    ):
                        window_value = seq_offset + pos - seq_window_start
                        if chunk_count != 0:
                            window_value = chunk_id * window_capacity_per_chunk + window_value
                        topk_idxs[row, write_col] = window_value
                        write_col = write_col + 1

                if has_compressed != 0 and compressed_width > 0 and ratio > 1:
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
                            if chunk_count != 0:
                                seq_major_id = seq_comp_start + comp_id
                                if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                                    rank_major_id = rank_major_by_seq_major[seq_major_id]
                                    if rank_major_id >= 0:
                                        topk_idxs[row, write_col] = (
                                            shared_compressed_base + rank_major_id
                                        )
                                        write_col = write_col + 1
                            else:
                                topk_idxs[row, write_col] = seq_offset + seq_window_len + comp_id
                                write_col = write_col + 1
                topk_length[row] = write_col
            elif total_width > 0:
                # Same padding-row contract as the parallel dense path above.
                topk_idxs[row, 0] = 0
                topk_length[row] = 1

    # Launch contract for _build_attention_indices_kernel.
    #   Supports contiguous and two-chunk CP. total_work is
    #   l_local * (window_size + compressed_width).
    @cute.jit
    def _build_attention_indices_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        window_capacity_per_chunk: cutlass.Int32,
        shared_compressed_base: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        has_indexer_topk: cutlass.Int32,
        has_compressed: cutlass.Int32,
        total_width: cutlass.Int32,
        parallel_dense: cutlass.Int32,
        launch_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _build_attention_indices_kernel,
            "dsv4_cp_build_attention_indices",
            (
                cu_seqlens,
                cu_seqlens_compressed,
                indexer_topk,
                rank_major_by_seq_major,
                topk_idxs,
                topk_length,
                n_seq,
                global_start,
                l_local,
                chunk0_start,
                chunk1_start,
                chunk_count,
                chunk_len,
                window_capacity_per_chunk,
                shared_compressed_base,
                d_window,
                window_size,
                ratio,
                compressed_width,
                seq_major_rows,
                has_indexer_topk,
                has_compressed,
                total_width,
                parallel_dense,
            ),
            launch_work,
            stream,
        )

    # Kernel contract:
    #   Same inputs as _build_attention_indices_kernel, but emits compressed-first indexer-loss order.
    #   topk_idxs: int32 output, shape (l_local, compressed_width + window_size),
    #       compressed ids first then window ids.
    #   indexer_rank_major: int32 output, shape (l_local, compressed_width), mapping
    #       each compressed logical top-k id to rank-major indexer-K row.
    @cute.kernel
    def _build_indexer_loss_indices_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk_logical: cute.Tensor,
        indexer_rank_by_seq_major: cute.Tensor,
        topk_idxs: cute.Tensor,
        indexer_rank_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        window_capacity_per_chunk: cutlass.Int32,
        shared_compressed_base: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        total_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            row = linear // total_width
            col = linear - row * total_width
            topk_value = -1
            rank_major_value = -1

            local_start = global_start
            row_in_local = row
            chunk_id = 0
            if chunk_count != 0:
                chunk_id = row // chunk_len
                row_in_local = row - chunk_id * chunk_len
                local_start = chunk0_start
                if chunk_id == 1:
                    local_start = chunk1_start
            global_q = local_start + row_in_local
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
                global_end = local_start + l_local
                if chunk_count != 0:
                    global_end = local_start + chunk_len
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_start = seq_start
                    if local_seq_start < local_start:
                        local_seq_start = local_start
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
                            comp_len = cu_seqlens_compressed[seq + 1] - cu_seqlens_compressed[seq]
                        if seq == seq_id:
                            seq_window_start = window_start
                            seq_window_len = window_len
                            seq_comp_len = comp_len
                        if seq < seq_id:
                            seq_offset = seq_offset + window_len
                            if chunk_count == 0:
                                seq_offset = seq_offset + comp_len

                if col < compressed_width:
                    comp_id = indexer_topk_logical[row, col]
                    if comp_id >= 0 and comp_id < seq_comp_len:
                        seq_major_id = cu_seqlens_compressed[seq_id] + comp_id
                        if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                            rank_major_id = indexer_rank_by_seq_major[seq_major_id]
                            if rank_major_id >= 0:
                                if chunk_count != 0:
                                    topk_value = shared_compressed_base + rank_major_id
                                else:
                                    topk_value = seq_offset + seq_window_len + comp_id
                                rank_major_value = rank_major_id
                else:
                    window_col = col - compressed_width
                    window_start_for_q = global_q - window_size + 1
                    if window_start_for_q < seq_start_found:
                        window_start_for_q = seq_start_found
                    window_count = global_q - window_start_for_q + 1
                    pos = window_start_for_q + window_col
                    if (
                        window_col < window_count
                        and pos >= seq_window_start
                        and pos < seq_window_start + seq_window_len
                    ):
                        topk_value = seq_offset + pos - seq_window_start
                        if chunk_count != 0:
                            topk_value = chunk_id * window_capacity_per_chunk + topk_value

            topk_idxs[row, col] = topk_value
            if col < compressed_width:
                indexer_rank_major[row, col] = rank_major_value

    # Launch contract for _build_indexer_loss_indices_kernel.
    #   Supports contiguous and two-chunk CP. total_work is
    #   l_local * (compressed_width + window_size).
    @cute.jit
    def _build_indexer_loss_indices_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk_logical: cute.Tensor,
        indexer_rank_by_seq_major: cute.Tensor,
        topk_idxs: cute.Tensor,
        indexer_rank_major: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        window_capacity_per_chunk: cutlass.Int32,
        shared_compressed_base: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        total_width: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_1d(
            _build_indexer_loss_indices_kernel,
            "dsv4_cp_build_indexer_loss_indices",
            (
                cu_seqlens,
                cu_seqlens_compressed,
                indexer_topk_logical,
                indexer_rank_by_seq_major,
                topk_idxs,
                indexer_rank_major,
                n_seq,
                global_start,
                l_local,
                chunk0_start,
                chunk1_start,
                chunk_count,
                chunk_len,
                window_capacity_per_chunk,
                shared_compressed_base,
                d_window,
                window_size,
                ratio,
                compressed_width,
                seq_major_rows,
                total_width,
                total_work,
            ),
            total_work,
            stream,
        )


# =============================================================================
# Torch Wrapper Functions
# =============================================================================
# Torch-facing entry points called by csa_cp_utils.py. They validate inputs,
# allocate outputs, and dispatch the CuTeDSL kernels defined above.
# =============================================================================


def _require_cute(message: str, *tensors: Optional[torch.Tensor]) -> None:
    """Raise ``RuntimeError`` when a wrapper cannot use CuTeDSL kernels."""
    if (
        not _CUTE_AVAILABLE
        or os.environ.get("DSV4_CP_DISABLE_CUTE_KERNELS")
        or not all(tensor is None or tensor.is_cuda for tensor in tensors)
    ):
        raise RuntimeError(message)


def _make_compiled_launch_runner():
    """Build the CuTe launch helper with private helpers and persistent caches."""
    compiled_launch_cache = {}

    def _cute_scalar(value):
        if isinstance(value, bool):
            return cutlass.Int32(1 if value else 0)
        if isinstance(value, int):
            return cutlass.Int32(value)
        if isinstance(value, float):
            return cutlass.Float32(value)
        return value

    @lru_cache(maxsize=None)
    def _gpu_arch_flag() -> str:
        cap = torch.cuda.get_device_capability()
        arch_map = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}
        arch = arch_map.get(cap)
        if arch is None:
            raise RuntimeError(f"Unsupported GPU compute capability {cap} for CSA CP CuTe kernels.")
        return arch

    def _run_compiled_launch(
        launch_fn,
        tensor_args: Tuple[torch.Tensor, ...],
        scalar_args: Tuple,
        key_arg_indices: Optional[Tuple[int, ...]] = None,
        static_arg_indices: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Compile/cache a CuTe launch function and invoke it on the current stream.

        Inputs:
            launch_fn: ``@cute.jit`` launcher.
            tensor_args: Torch CUDA tensors passed through DLPack with dynamic leading
                layouts.
            scalar_args: Python scalar launch parameters.
            key_arg_indices: Scalar indices that participate in the compile cache key.
            static_arg_indices: Scalar indices passed as compile-time constexpr values.

        Output:
            None. The launch writes into output tensors passed in ``tensor_args``.
        """
        key_arg_indices = (
            tuple(range(len(scalar_args))) if key_arg_indices is None else tuple(key_arg_indices)
        )
        static_arg_indices = () if static_arg_indices is None else tuple(static_arg_indices)
        static_arg_set = set(static_arg_indices)
        key = (
            launch_fn.__name__,
            tuple(
                (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
                for tensor in tensor_args
            ),
            key_arg_indices,
            tuple(scalar_args[i] for i in key_arg_indices),
            static_arg_indices,
        )
        compiled = compiled_launch_cache.get(key)
        if compiled is None:
            fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
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
                    arg if i in static_arg_set else _cute_scalar(arg)
                    for i, arg in enumerate(scalar_args)
                ),
                fake_stream,
                options=f"--enable-tvm-ffi --gpu-arch {_gpu_arch_flag()}",
            )
            compiled_launch_cache[key] = compiled
        compiled(
            *tensor_args,
            *(_cute_scalar(arg) for i, arg in enumerate(scalar_args) if i not in static_arg_set),
            cuda.CUstream(torch.cuda.current_stream(tensor_args[0].device).cuda_stream),
        )

    return _run_compiled_launch


_run_compiled_launch = _make_compiled_launch_runner()


def _validate_rope_inputs(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, nope_dim: int, pos_dim: int
) -> Tuple[int, int, int]:
    """Validate common RoPE tensor dimensions and return flattened row metadata.

    Inputs:
        x: Tensor with first dimension as rows and remaining dims flattened per row.
        cos/sin: Rotary tables with first dimension as sequence position and at
            least ``pos_dim`` values per row.
        nope_dim/pos_dim: Per-head no-RoPE and RoPE dimensions.

    Output:
        ``(rows, row_width, head_dim)`` where ``head_dim = nope_dim + pos_dim``.
    """
    rows = x.shape[0]
    row_width = math.prod(x.shape[1:])
    head_dim = nope_dim + pos_dim
    if pos_dim <= 0 or pos_dim % 2 != 0:
        raise RuntimeError(f"DSv4 CP RoPE expects an even positive pos_dim, got {pos_dim}.")
    if head_dim <= 0 or row_width % head_dim != 0:
        raise RuntimeError(
            "DSv4 CP RoPE expects the flattened row width to be a multiple of "
            f"head_dim={head_dim}, got row_width={row_width}."
        )
    cos_width = math.prod(cos.shape[1:])
    sin_width = math.prod(sin.shape[1:])
    if cos_width < pos_dim or sin_width < pos_dim:
        raise RuntimeError(
            "DSv4 CP RoPE cos/sin tables are narrower than pos_dim: "
            f"cos={cos_width}, sin={sin_width}, pos_dim={pos_dim}."
        )
    return rows, row_width, head_dim


class ThdLocalRope(torch.autograd.Function):
    """Apply non-interleaved RoPE for THD rows owned by one or two CP chunks.

    Inputs:
        x: CUDA tensor, shape ``(local_rows, ...)``; each row flattens to
            ``num_heads * (nope_dim + pos_dim)`` and dtype is preserved.
        cos/sin: CUDA tensors, shape ``(max_seq_len, >= pos_dim)``.
        cu_seqlens_padded: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        chunk0_global_start/chunk1_global_start: global packed-token row starts.
        chunk_len: positive row count per chunk. ``local_rows <= 2 * chunk_len``.
        inverse/adjoint: select inverse RoPE or backward adjoint math.

    Output:
        Tensor with same shape and dtype as ``x``. The wrapper maps each local
        row back to a global THD token position, derives position within its
        sequence from ``cu_seqlens_padded``, and launches the fused RoPE kernel.
    """

    @staticmethod
    def _run(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        chunk0_global_start: int,
        chunk_len: int,
        nope_dim: int,
        pos_dim: int,
        chunk1_global_start: int = 0,
        inverse: bool = False,
        clamp_to_valid_token: bool = False,
        adjoint: bool = False,
    ) -> torch.Tensor:
        _require_cute(
            "DSv4 THD CP RoPE requires CUDA tensors and available CuTe kernels.",
            x,
            cos,
            sin,
            cu_seqlens_padded,
        )
        rows, row_width, head_dim = _validate_rope_inputs(x, cos, sin, nope_dim, pos_dim)
        chunk_len = int(chunk_len)
        if chunk_len <= 0:
            raise RuntimeError(f"DSv4 THD CP RoPE expects positive chunk_len, got {chunk_len}.")
        if rows > 2 * chunk_len:
            raise RuntimeError(
                "DSv4 THD CP RoPE supports at most two chunks: "
                f"rows={rows}, chunk_len={chunk_len}."
            )
        if rows == 0:
            return x
        out = x.clone()
        total_work = rows * (row_width // head_dim) * (int(pos_dim) // 2)
        _run_compiled_launch(
            _thd_local_rope_launch,
            (
                x.reshape(rows, row_width),
                out.reshape(rows, row_width),
                cos.reshape(cos.shape[0], math.prod(cos.shape[1:])),
                sin.reshape(sin.shape[0], math.prod(sin.shape[1:])),
                cu_seqlens_padded,
            ),
            (
                cu_seqlens_padded.shape[0] - 1,
                int(chunk0_global_start),
                int(chunk1_global_start),
                chunk_len,
                1 if clamp_to_valid_token else 0,
                row_width,
                head_dim,
                int(nope_dim),
                int(pos_dim),
                1 if inverse else 0,
                1 if adjoint else 0,
                total_work,
            ),
            key_arg_indices=(5,),
            static_arg_indices=(5,),
        )
        return out

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        chunk0_global_start: int,
        chunk_len: int,
        nope_dim: int,
        pos_dim: int,
        chunk1_global_start: int,
        inverse: bool,
        clamp_to_valid_token: bool,
    ) -> torch.Tensor:
        ctx.rope_args = (
            int(chunk0_global_start),
            int(chunk_len),
            int(nope_dim),
            int(pos_dim),
            int(chunk1_global_start),
            bool(inverse),
            bool(clamp_to_valid_token),
        )
        ctx.save_for_backward(cos, sin, cu_seqlens_padded)
        return ThdLocalRope._run(x, cos, sin, cu_seqlens_padded, *ctx.rope_args)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin, cu_seqlens_padded = ctx.saved_tensors
        grad_x = ThdLocalRope._run(
            grad_output.contiguous(), cos, sin, cu_seqlens_padded, *ctx.rope_args, adjoint=True
        )
        return (grad_x, *([None] * 10))


class ThdCompressedRope(torch.autograd.Function):
    """Apply RoPE to compressed rows using per-row compressed group ids.

    Inputs:
        x: CUDA tensor, shape ``(rows, ...)``; flattened row width must be a
            multiple of ``nope_dim + pos_dim``.
        cos/sin: CUDA tensors, shape ``(max_seq_len, >= pos_dim)``.
        comp_ids_local: int32 CUDA tensor, shape at least ``(rows,)``; -1 maps to 0.
        ratio: compression ratio, used as ``rope_position = comp_id * ratio``.

    Output:
        Tensor with same shape and dtype as ``x``. No-RoPE dims are copied; RoPE
        dims are rotated pairwise, with inverse/adjoint variants controlled by
        flags.
    """

    @staticmethod
    def _run(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        comp_ids_local: torch.Tensor,
        ratio: int,
        nope_dim: int,
        pos_dim: int,
        inverse: bool = False,
        adjoint: bool = False,
    ) -> torch.Tensor:
        _require_cute(
            "DSv4 CP compressed RoPE requires CUDA tensors and available CuTe kernels.",
            x,
            cos,
            sin,
            comp_ids_local,
        )
        rows, row_width, head_dim = _validate_rope_inputs(x, cos, sin, nope_dim, pos_dim)
        if comp_ids_local.shape[0] < rows:
            raise RuntimeError(
                "DSv4 CP compressed RoPE received too few comp ids: "
                f"comp_ids={comp_ids_local.shape[0]}, rows={rows}."
            )
        if rows == 0:
            return x
        out = torch.empty_like(x)
        total_work = rows * row_width
        _run_compiled_launch(
            _thd_compressed_rope_launch,
            (
                x.reshape(rows, row_width),
                out.reshape(rows, row_width),
                cos.reshape(cos.shape[0], math.prod(cos.shape[1:])),
                sin.reshape(sin.shape[0], math.prod(sin.shape[1:])),
                comp_ids_local[:rows],
            ),
            (
                int(ratio),
                row_width,
                head_dim,
                int(nope_dim),
                int(pos_dim),
                1 if inverse else 0,
                1 if adjoint else 0,
                total_work,
            ),
            key_arg_indices=(1,),
            static_arg_indices=(1,),
        )
        return out

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
        ctx.rope_args = (int(ratio), int(nope_dim), int(pos_dim), bool(inverse))
        ctx.save_for_backward(cos, sin, comp_ids_local)
        return ThdCompressedRope._run(x, cos, sin, comp_ids_local, *ctx.rope_args)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin, comp_ids_local = ctx.saved_tensors
        grad_x = ThdCompressedRope._run(
            grad_output.contiguous(), cos, sin, comp_ids_local, *ctx.rope_args, adjoint=True
        )
        return (grad_x, *([None] * 7))


class CompressorInputCompact(torch.autograd.Function):
    """Compact local and boundary hidden rows for compressor input.

    Inputs:
        hidden_local: CUDA tensor, shape ``(l_local, ...)``.
        boundary_hidden: CUDA tensor, shape ``(d_window, ...)``.
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start/l_local: local contiguous global row range.
        ratio/d_comp/d_window/c_cap: fixed compressor window and capacity.

    Outputs:
        ``hidden_compact`` shape ``(c_cap * ratio, ...)``, same dtype as hidden;
        ``cu_compact`` int32 shape ``(n_seq + 1,)``; ``seq_ids`` int32,
        ``comp_ids`` int32, and ``valid`` bool each shape ``(c_cap,)``.
        The kernel copies the ratio source tokens for each visible compressed
        group and emits metadata for the compressed rows.
    """

    @staticmethod
    def _forward_kernel(
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

        row_width = math.prod(hidden_local.shape[1:])
        hidden_local_flat = hidden_local.reshape(l_local, row_width)
        boundary_flat = boundary_hidden.reshape(d_window, row_width)
        compact_flat = hidden_compact.reshape(compact_len, row_width)
        total_work = max(compact_len * row_width, c_cap, cu_seqlens.shape[0])
        _run_compiled_launch(
            _compressor_input_compact_fwd_launch,
            (
                hidden_local_flat,
                boundary_flat,
                compact_flat,
                cu_seqlens,
                cu_compact,
                seq_ids,
                comp_ids,
                valid,
            ),
            (
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
            ),
            key_arg_indices=(8,),
            static_arg_indices=(8,),
        )
        return hidden_compact, cu_compact, seq_ids, comp_ids, valid

    @staticmethod
    def _backward_kernel(
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
        row_width = math.prod(hidden_shape[1:])
        grad_hidden.zero_()
        grad_boundary.zero_()

        total_work = compact_len * row_width
        if total_work > 0:
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
                    total_work,
                ),
                key_arg_indices=(7,),
                static_arg_indices=(7,),
            )
        return grad_hidden, grad_boundary

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
        ctx.compact_args = (int(global_start), int(l_local), int(ratio), int(d_comp), int(d_window))
        ctx.save_for_backward(cu_seqlens)
        return CompressorInputCompact._forward_kernel(
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            int(global_start),
            int(l_local),
            int(ratio),
            int(d_comp),
            int(d_window),
            int(c_cap),
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
        grad_hidden, grad_boundary = CompressorInputCompact._backward_kernel(
            grad_hidden_compact.contiguous(),
            ctx.hidden_shape,
            ctx.boundary_shape,
            cu_seqlens,
            *ctx.compact_args,
        )
        return (grad_hidden, grad_boundary, *([None] * 7))


class ThdFullKvPack(torch.autograd.Function):
    """Autograd wrapper for static-shape KV-full packing.

    Forward builds a physical ``kv_full`` row space from local window rows,
    boundary rows, and all-gathered compressed rows. Backward recomputes the
    same row mapping and scatters gradients back to those three sources.
    """

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
        rank_major_by_seq_major: torch.Tensor,
        cu_seqlens_compressed: torch.Tensor,
        global_start: int,
        l_local: int,
        d_window: int,
        ratio: int,
        kv_full_capacity: int,
        chunk0_start: int,
        chunk1_start: int,
        chunk_count: int,
        chunk_len: int,
        window_capacity_per_chunk: int,
    ) -> torch.Tensor:
        """Pack KV rows into the fixed physical sparse-attention row order.

        Inputs:
            kv_local: CUDA tensor, shape ``(local_rows, ...)``.
            boundary_kv: CUDA tensor, shape ``(d_window or 2*d_window, ...)``.
            compressed_rank_major: CUDA tensor, shape ``(compressed_rows, ...)``.
            seq_ids/comp_ids/valid: rank-major compressed metadata.
            cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
            optional compressed reverse-map metadata for contiguous CP.
            chunk fields select contiguous or two-chunk layout.

        Output:
            ``kv_full`` CUDA tensor, shape ``(kv_full_capacity, ...)``, same dtype
            as ``kv_local``. Invalid tail rows are zero-filled.
        """
        ctx.pack_shapes = (
            tuple(kv_local.shape),
            tuple(boundary_kv.shape),
            tuple(compressed_rank_major.shape),
        )
        ctx.kv_full_capacity = int(kv_full_capacity)

        output_shape = (int(kv_full_capacity),) + tuple(kv_local.shape[1:])
        kv_full = kv_local.new_empty(output_shape)
        row_width = math.prod(kv_local.shape[1:])
        local_rows = kv_local.shape[0]
        boundary_rows = boundary_kv.shape[0]
        kv_local_flat = kv_local.reshape(local_rows, row_width)
        boundary_flat = boundary_kv.reshape(boundary_rows, row_width)
        compressed_rows = compressed_rank_major.shape[0]
        compressed_flat = compressed_rank_major.reshape(compressed_rows, row_width)
        kv_full_flat = kv_full.reshape(int(kv_full_capacity), row_width)
        total_work = int(kv_full_capacity) * row_width
        use_seq_major_map = (
            int(chunk_count) == 0
            and ratio > 1
            and compressed_rows > 0
            and rank_major_by_seq_major.shape[0] > 1
        )
        _run_compiled_launch(
            _thd_full_kv_pack_launch,
            (
                kv_local_flat,
                boundary_flat,
                compressed_flat,
                kv_full_flat,
                seq_ids_rank_major,
                comp_ids_rank_major,
                valid_rank_major,
                cu_seqlens,
                cu_seqlens_compressed,
                rank_major_by_seq_major,
            ),
            (
                cu_seqlens.shape[0] - 1,
                int(global_start),
                int(l_local),
                int(chunk0_start),
                int(chunk1_start),
                int(chunk_count),
                int(chunk_len),
                int(window_capacity_per_chunk),
                int(d_window),
                int(ratio),
                compressed_rows,
                1 if use_seq_major_map else 0,
                rank_major_by_seq_major.shape[0] if use_seq_major_map else 0,
                int(kv_full_capacity),
                row_width,
                0,
                total_work,
            ),
            key_arg_indices=(14,),
            static_arg_indices=(14,),
        )
        ctx.save_for_backward(
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens,
            cu_seqlens_compressed,
            rank_major_by_seq_major,
        )
        ctx.pack_args = (
            int(global_start),
            int(l_local),
            int(chunk0_start),
            int(chunk1_start),
            int(chunk_count),
            int(chunk_len),
            int(window_capacity_per_chunk),
            int(d_window),
            int(ratio),
            compressed_rows,
            1 if use_seq_major_map else 0,
            rank_major_by_seq_major.shape[0] if use_seq_major_map else 0,
        )
        return kv_full

    @staticmethod
    def backward(ctx, grad_kv_full: torch.Tensor):
        """Scatter ``kv_full`` gradients back to local, boundary, and compressed KV."""
        local_shape, boundary_shape, compressed_shape = ctx.pack_shapes
        grad_kv_local = grad_kv_full.new_empty(local_shape)
        grad_boundary_kv = grad_kv_full.new_empty(boundary_shape)
        grad_compressed = grad_kv_full.new_empty(compressed_shape)

        compressed_rows = compressed_shape[0]
        kv_full_capacity = ctx.kv_full_capacity
        row_width = math.prod(local_shape[1:])
        (
            global_start,
            l_local,
            chunk0_start,
            chunk1_start,
            chunk_count,
            chunk_len,
            window_capacity_per_chunk,
            d_window,
            ratio,
            compressed_rows,
            use_seq_major_map,
            seq_major_rows,
        ) = ctx.pack_args
        grad_kv_local.zero_()
        grad_boundary_kv.zero_()
        grad_compressed.zero_()

        total_work = kv_full_capacity * row_width
        if total_work > 0:
            (
                seq_ids_rank_major,
                comp_ids_rank_major,
                valid_rank_major,
                cu_seqlens,
                cu_seqlens_compressed,
                rank_major_by_seq_major,
            ) = ctx.saved_tensors
            _run_compiled_launch(
                _thd_full_kv_pack_launch,
                (
                    grad_kv_local.reshape(local_shape[0], row_width),
                    grad_boundary_kv.reshape(boundary_shape[0], row_width),
                    grad_compressed.reshape(compressed_rows, row_width),
                    grad_kv_full.reshape(kv_full_capacity, row_width),
                    seq_ids_rank_major,
                    comp_ids_rank_major,
                    valid_rank_major,
                    cu_seqlens,
                    cu_seqlens_compressed,
                    rank_major_by_seq_major,
                ),
                (
                    cu_seqlens.shape[0] - 1,
                    global_start,
                    l_local,
                    chunk0_start,
                    chunk1_start,
                    chunk_count,
                    chunk_len,
                    window_capacity_per_chunk,
                    d_window,
                    ratio,
                    compressed_rows,
                    use_seq_major_map,
                    seq_major_rows,
                    kv_full_capacity,
                    row_width,
                    1,
                    total_work,
                ),
                key_arg_indices=(14,),
                static_arg_indices=(14,),
            )
        return (grad_kv_local, grad_boundary_kv, grad_compressed, *([None] * 16))


def repack_compressed_kv_to_seq_major(
    compressed_rank_major: torch.Tensor,
    seq_ids_rank_major: torch.Tensor,
    comp_ids_rank_major: torch.Tensor,
    valid_rank_major: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    seq_major_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repack all-gathered compressed rows from rank-major to seq-major order.

    Inputs:
        compressed_rank_major: CUDA tensor, shape ``(rank_major_rows, ...)``.
        seq_ids_rank_major/comp_ids_rank_major: int32 metadata, shape
            ``(rank_major_rows,)``.
        valid_rank_major: bool metadata, shape ``(rank_major_rows,)``.
        cu_seqlens_compressed: int32 prefixes, shape ``(n_seq + 1,)``.
        seq_major_rows: total compressed seq-major rows.

    Outputs:
        ``compressed_seq_major`` shape ``(seq_major_rows, ...)`` and
        ``rank_major_by_seq_major`` int32 shape ``(seq_major_rows,)``.
    """
    output_shape = (seq_major_rows,) + tuple(compressed_rank_major.shape[1:])
    compressed_seq_major = compressed_rank_major.new_empty(output_shape)
    rank_major_by_seq_major = torch.empty(
        (seq_major_rows,), dtype=torch.int32, device=compressed_rank_major.device
    )
    rank_major_rows = compressed_rank_major.shape[0]
    row_width = math.prod(compressed_rank_major.shape[1:])
    map_work = int(rank_major_rows)
    gather_work = int(seq_major_rows) * int(row_width)
    _run_compiled_launch(
        _repack_compressed_kv_init_rank_map_launch,
        (rank_major_by_seq_major,),
        (seq_major_rows, max(1, seq_major_rows)),
    )
    _run_compiled_launch(
        _repack_compressed_kv_rank_map_launch,
        (
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens_compressed,
            rank_major_by_seq_major,
        ),
        (rank_major_rows, seq_major_rows, max(1, map_work)),
    )
    _run_compiled_launch(
        _repack_compressed_kv_to_seq_major_launch,
        (
            compressed_rank_major.reshape(rank_major_rows, row_width),
            compressed_seq_major.reshape(seq_major_rows, row_width),
            rank_major_by_seq_major,
        ),
        (rank_major_rows, seq_major_rows, row_width, max(1, gather_work)),
        key_arg_indices=(2,),
        static_arg_indices=(2,),
    )
    return compressed_seq_major, rank_major_by_seq_major


def build_compressed_row_metadata(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
    *,
    c_cap_per_rank: Optional[int] = None,
    use_two_chunk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build rank-major compressed metadata for all CP ranks.

    Inputs:
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        cp_size: CP world size.
        l_local: rows per contiguous rank, or chunk length when ``use_two_chunk``.
        ratio/d_comp: compression parameters.
        c_cap: fixed compressed capacity per chunk.
        c_cap_per_rank: optional fixed compressed capacity per rank.
        use_two_chunk: whether each rank owns two chunks.

    Outputs:
        ``seq_ids`` int32, ``comp_ids`` int32, and ``valid`` bool, each shape
        ``(cp_size * c_cap_per_rank,)``. Rows are ordered rank-major and describe
        which compressed group each all-gathered row represents.
    """
    c_cap_per_rank = int(c_cap if c_cap_per_rank is None else c_cap_per_rank)
    total_rows = int(cp_size) * c_cap_per_rank
    seq_ids = torch.empty((total_rows,), dtype=torch.int32, device=cu_seqlens.device)
    comp_ids = torch.empty((total_rows,), dtype=torch.int32, device=cu_seqlens.device)
    valid = torch.empty((total_rows,), dtype=torch.bool, device=cu_seqlens.device)
    _run_compiled_launch(
        _build_compressed_row_metadata_launch,
        (cu_seqlens, seq_ids, comp_ids, valid),
        (
            cu_seqlens.shape[0] - 1,
            int(cp_size),
            int(l_local),
            int(ratio),
            int(d_comp),
            int(c_cap),
            c_cap_per_rank,
            1 if use_two_chunk else 0,
            total_rows,
        ),
        key_arg_indices=(),
    )
    return seq_ids, comp_ids, valid


def build_indexer_topk_metadata(
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build local trapezoid inputs for the indexer top-k kernel.

    Inputs:
        k_indexer_seq_major: CUDA tensor, shape ``(seq_major_rows, ...)``.
        cu_seqlens_q/cu_seqlens_compressed: int32 prefixes, shape ``(n_seq + 1,)``.
        global_start/l_local: local query range.
        ratio: compression ratio.

    Outputs:
        ``k_topk`` tensor with same shape as ``k_indexer_seq_major``;
        ``cu_q_topk`` and ``cu_k_topk`` int32 tensors, shape ``(n_seq + 2,)``;
        ``seq_lens`` int32 tensor, shape ``(l_local,)``. The metadata represents
        the local Q/K trapezoid that the indexer kernel consumes.
    """
    seq_major_rows = k_indexer_seq_major.shape[0]
    row_width = math.prod(k_indexer_seq_major.shape[1:])
    k_topk = k_indexer_seq_major.new_empty(k_indexer_seq_major.shape)
    cu_q_topk = torch.empty(
        (cu_seqlens_q.shape[0] + 1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device
    )
    cu_k_topk = torch.empty(
        (cu_seqlens_q.shape[0] + 1,), dtype=cu_seqlens_compressed.dtype, device=cu_seqlens_q.device
    )
    seq_lens = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens_q.device)
    n_seq = cu_seqlens_q.shape[0] - 1
    total_work = seq_major_rows * row_width
    _run_compiled_launch(
        _build_indexer_topk_metadata_launch,
        (
            k_indexer_seq_major.reshape(seq_major_rows, row_width),
            k_topk.reshape(seq_major_rows, row_width),
            cu_seqlens_q,
            cu_seqlens_compressed,
            cu_q_topk,
            cu_k_topk,
            seq_lens,
        ),
        (
            n_seq,
            global_start,
            l_local,
            ratio,
            row_width,
            seq_major_rows,
            total_work,
            max(1, total_work),
        ),
        key_arg_indices=(4,),
        static_arg_indices=(4,),
    )
    return k_topk, cu_q_topk, cu_k_topk, seq_lens


def build_attention_indices(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    indexer_topk_compressed_logical_ids: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    chunk_starts: Optional[Tuple[int, ...]] = None,
    chunk_len: int = 0,
    window_capacity_per_chunk: int = 0,
    shared_compressed_base: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build final sparse-attention indices in physical ``kv_full`` row space.

    Inputs:
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start/l_local: contiguous local query range, ignored for two-chunk mode.
        d_window/window_size: physical left-window capacity and per-query window width.
        ratio/compressed_width: compression mode and number of compressed columns.
        indexer_topk_compressed_logical_ids: optional int32 tensor, shape
            ``(l_local, compressed_width)``.
        cu_seqlens_compressed/rank_major_by_seq_major/chunk metadata: required for
            two-chunk compressed lowering.

    Outputs:
        ``topk_idxs`` int32, shape ``(l_local, window_size + compressed_width)``;
        ``topk_length`` int32, shape ``(l_local,)``. Window ids appear before
        compressed ids and are already lowered into ``kv_full`` rows.
    """
    chunk_count = 0
    chunk1_start = 0
    if chunk_starts is not None:
        _require_cute(
            "DSv4 two-chunk shared final idx requires CUDA tensors and CuTeDSL.",
            cu_seqlens_compressed,
            rank_major_by_seq_major,
        )
        if not chunk_starts or len(chunk_starts) > 2:
            raise RuntimeError(
                "DSv4 two-chunk shared final idx supports one or two chunks, "
                f"got {len(chunk_starts)}."
            )
        chunk_count = len(chunk_starts)
        l_local = int(chunk_count) * int(chunk_len)
        global_start = 0
        chunk1_start = int(chunk_starts[1]) if chunk_count > 1 else 0
    else:
        cu_seqlens_compressed = cu_seqlens
        rank_major_by_seq_major = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)

    total_width = window_size + compressed_width
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    topk_length = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens.device)
    if indexer_topk_compressed_logical_ids is None:
        dummy = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
        has_indexer = 0
    else:
        dummy = indexer_topk_compressed_logical_ids
        has_indexer = 1
    parallel_dense = 1 if has_indexer == 0 and chunk_count == 0 else 0
    launch_work = l_local * total_width if parallel_dense else l_local
    _run_compiled_launch(
        _build_attention_indices_launch,
        (cu_seqlens, cu_seqlens_compressed, dummy, rank_major_by_seq_major, topk_idxs, topk_length),
        (
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            int(chunk_starts[0]) if chunk_starts is not None else 0,
            chunk1_start,
            chunk_count,
            int(chunk_len),
            int(window_capacity_per_chunk),
            int(shared_compressed_base),
            d_window,
            window_size,
            ratio,
            compressed_width,
            rank_major_by_seq_major.shape[0],
            has_indexer,
            1 if ratio > 1 and compressed_width > 0 else 0,
            total_width,
            parallel_dense,
            launch_work,
        ),
        key_arg_indices=(),
    )
    return topk_idxs, topk_length


def build_indexer_loss_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    indexer_topk_compressed_logical_ids: torch.Tensor,
    indexer_rank_by_seq_major: torch.Tensor,
    chunk_starts: Optional[Tuple[int, ...]] = None,
    chunk_len: int = 0,
    window_capacity_per_chunk: int = 0,
    shared_compressed_base: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build compressed-first sparse-attention ids and indexer-loss rank-major ids.

    Inputs:
        cu_seqlens/cu_seqlens_compressed: int32 CUDA prefixes, shape ``(n_seq + 1,)``.
        global_start/l_local: contiguous local query range, ignored for two-chunk mode.
        d_window/window_size/ratio: final-id lowering parameters.
        indexer_topk_compressed_logical_ids: int32 tensor, shape
            ``(l_local, compressed_width)``.
        indexer_rank_by_seq_major: int32 reverse map, shape ``(seq_major_rows,)``.
        optional chunk metadata selects two-chunk lowering.

    Outputs:
        ``topk_idxs`` int32, shape ``(l_local, compressed_width + window_size)``,
        with compressed ids first; ``indexer_rank_major`` int32, shape
        ``(l_local, compressed_width)`` for indexer-loss K rows.
    """
    _require_cute(
        "DSv4 CP indexer-loss final idx generation requires CUDA tensors and CuTeDSL.",
        cu_seqlens,
        cu_seqlens_compressed,
        indexer_topk_compressed_logical_ids,
        indexer_rank_by_seq_major,
    )
    chunk_count = 0
    chunk1_start = 0
    if chunk_starts is not None:
        if not chunk_starts or len(chunk_starts) > 2:
            raise RuntimeError(
                "DSv4 two-chunk shared indexer-loss final idx supports one or two chunks, "
                f"got {len(chunk_starts)}."
            )
        chunk_count = len(chunk_starts)
        l_local = int(chunk_count) * int(chunk_len)
        global_start = 0
        chunk1_start = int(chunk_starts[1]) if chunk_count > 1 else 0

    compressed_width = indexer_topk_compressed_logical_ids.shape[-1]
    total_width = compressed_width + window_size
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    indexer_rank_major = torch.empty(
        (l_local, compressed_width), dtype=torch.int32, device=cu_seqlens.device
    )
    total_work = l_local * total_width
    if total_work == 0:
        return topk_idxs, indexer_rank_major
    _run_compiled_launch(
        _build_indexer_loss_indices_launch,
        (
            cu_seqlens,
            cu_seqlens_compressed,
            indexer_topk_compressed_logical_ids,
            indexer_rank_by_seq_major,
            topk_idxs,
            indexer_rank_major,
        ),
        (
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            int(chunk_starts[0]) if chunk_starts is not None else 0,
            chunk1_start,
            chunk_count,
            int(chunk_len),
            int(window_capacity_per_chunk),
            int(shared_compressed_base),
            d_window,
            window_size,
            ratio,
            compressed_width,
            indexer_rank_by_seq_major.shape[0],
            total_width,
            total_work,
        ),
        key_arg_indices=(),
    )
    return topk_idxs, indexer_rank_major

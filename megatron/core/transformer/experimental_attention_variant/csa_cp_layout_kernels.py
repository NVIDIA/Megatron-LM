# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CuTeDSL kernels for DSv4 THD context parallel layout work.

This module contains the few CuTeDSL kernels retained by the DSv4 CP path.
``csa_cp_utils.py`` owns partition semantics and calls the RoPE/compaction
autograd wrappers; ``csa.py`` calls the final-index builders directly.
"""

import math
import os
from typing import Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
    from cutlass.cutlass_dsl import T, dsl_user_op

    _CUTE_AVAILABLE = True
except ImportError:
    cuda = None
    cutlass = None
    cute = None
    llvm = None
    T = None
    dsl_user_op = None
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

    @dsl_user_op
    def _ptr_as_i64(
        tensor: cute.Tensor, offset: cutlass.Int32, *, loc=None, ip=None
    ) -> cutlass.Int64:
        elem_ptr = tensor.iterator + cutlass.Int32(offset)
        return cutlass.Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))

    @dsl_user_op
    def _ld_global_v4_u64(
        base_ptr: cutlass.Int64, *, loc=None, ip=None
    ) -> Tuple[cutlass.Uint64, cutlass.Uint64, cutlass.Uint64, cutlass.Uint64]:
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.i64(), T.i64(), T.i64(), T.i64()]),
            [cutlass.Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.L1::no_allocate.v4.u64 {$0, $1, $2, $3}, [$4];",
            "=l,=l,=l,=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        v0 = llvm.extractvalue(T.i64(), result, [0], loc=loc, ip=ip)
        v1 = llvm.extractvalue(T.i64(), result, [1], loc=loc, ip=ip)
        v2 = llvm.extractvalue(T.i64(), result, [2], loc=loc, ip=ip)
        v3 = llvm.extractvalue(T.i64(), result, [3], loc=loc, ip=ip)
        return cutlass.Uint64(v0), cutlass.Uint64(v1), cutlass.Uint64(v2), cutlass.Uint64(v3)

    @dsl_user_op
    def _st_global_v4_u64(
        base_ptr: cutlass.Int64,
        v0: cutlass.Uint64,
        v1: cutlass.Uint64,
        v2: cutlass.Uint64,
        v3: cutlass.Uint64,
        *,
        loc=None,
        ip=None,
    ):
        llvm.inline_asm(
            None,
            [
                cutlass.Int64(base_ptr).ir_value(loc=loc, ip=ip),
                cutlass.Uint64(v0).ir_value(loc=loc, ip=ip),
                cutlass.Uint64(v1).ir_value(loc=loc, ip=ip),
                cutlass.Uint64(v2).ir_value(loc=loc, ip=ip),
                cutlass.Uint64(v3).ir_value(loc=loc, ip=ip),
            ],
            "st.global.L1::evict_first.v4.u64 [$0], {$1, $2, $3, $4};",
            "l,l,l,l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

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

    def _launch_1d(kernel, name: str, args: Tuple, work: cutlass.Int32, stream: cuda.CUstream):
        """Launch a 1-D CuTe kernel with 128 threads per block."""
        _launch_named(
            kernel,
            name,
            args,
            grid=(cute.ceil_div(work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
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
        n_seq: cutlass.Constexpr,
        chunk0_global_start: cutlass.Int32,
        chunk1_global_start: cutlass.Int32,
        chunk_len: cutlass.Int32,
        clamp_to_valid_token: cutlass.Constexpr,
        row_width: cutlass.Constexpr,
        head_dim: cutlass.Constexpr,
        nope_dim: cutlass.Constexpr,
        pos_dim: cutlass.Constexpr,
        inverse: cutlass.Constexpr,
        adjoint: cutlass.Constexpr,
        head_group: cutlass.Constexpr,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx
        if row < total_work:
            n_heads = row_width // head_dim
            if cutlass.const_expr(
                x.element_type.width == 16 and nope_dim % 16 == 0 and head_dim % 16 == 0
            ):
                copy_vec_elems = cutlass.Int32(16)
                copy_vecs_per_head = nope_dim // 16
                copy_use_u256 = True
            elif cutlass.const_expr(
                x.element_type.width == 32 and nope_dim % 8 == 0 and head_dim % 8 == 0
            ):
                copy_vec_elems = cutlass.Int32(8)
                copy_vecs_per_head = nope_dim // 8
                copy_use_u256 = True
            elif cutlass.const_expr(
                (x.element_type.width == 16 and nope_dim % 4 == 0 and head_dim % 4 == 0)
                or (x.element_type.width == 32 and nope_dim % 2 == 0 and head_dim % 2 == 0)
            ):
                copy_use_u256 = False
                if cutlass.const_expr(x.element_type.width == 16):
                    copy_vec_elems = cutlass.Int32(4)
                    copy_vecs_per_head = nope_dim // 4
                else:
                    copy_vec_elems = cutlass.Int32(2)
                    copy_vecs_per_head = nope_dim // 2
            else:
                copy_vec_elems = cutlass.Int32(1)
                copy_vecs_per_head = nope_dim
                copy_use_u256 = False

            copy_work = tidx
            while copy_work < n_heads * copy_vecs_per_head:
                unit = copy_work % copy_vecs_per_head
                head = copy_work // copy_vecs_per_head
                offset = row * row_width + head * head_dim + unit * copy_vec_elems
                if cutlass.const_expr(
                    (x.element_type.width == 16 and nope_dim % 4 == 0 and head_dim % 4 == 0)
                    or (x.element_type.width == 32 and nope_dim % 2 == 0 and head_dim % 2 == 0)
                ):
                    if cutlass.const_expr(copy_use_u256):
                        src_ptr = _ptr_as_i64(x, offset)
                        dst_ptr = _ptr_as_i64(out, offset)
                        v0, v1, v2, v3 = _ld_global_v4_u64(src_ptr)
                        _st_global_v4_u64(dst_ptr, v0, v1, v2, v3)
                    else:
                        src_ptr = cute.recast_ptr(x.iterator + offset, dtype=cutlass.Int64)
                        dst_ptr = cute.recast_ptr(out.iterator + offset, dtype=cutlass.Int64)
                        value = cute.arch.load(
                            src_ptr.llvm_ptr,
                            cutlass.Int64,
                            level1_eviction_priority="evict_no_allocate",
                        )
                        cute.arch.store(
                            dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first"
                        )
                else:
                    col = head * head_dim + unit
                    out[row, col] = x[row, col]
                copy_work = copy_work + 128

            rope_pos = 0
            if tidx == 0 or tidx == 32 or tidx == 64 or tidx == 96:
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

                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    if global_token >= seq_start and global_token < seq_end:
                        rope_pos = global_token - seq_start

            rope_pos = cute.arch.shuffle_sync(rope_pos, 0, mask=-1, mask_and_clamp=31)

            pos_pairs = pos_dim // 2
            grouped_heads = n_heads // head_group
            work = tidx
            while work < grouped_heads * pos_pairs:
                pair = work % pos_pairs
                head_block = work // pos_pairs
                cos_left = cos[rope_pos, pair].to(cutlass.Float32)
                sin_left = sin[rope_pos, pair].to(cutlass.Float32)
                cos_right = cos[rope_pos, pos_pairs + pair].to(cutlass.Float32)
                sin_right = sin[rope_pos, pos_pairs + pair].to(cutlass.Float32)
                if inverse != 0:
                    sin_left = cutlass.Float32(0.0) - sin_left
                    sin_right = cutlass.Float32(0.0) - sin_right

                for slot in cutlass.range_constexpr(16):
                    if slot < head_group:
                        head = head_block * head_group + slot
                        col = head * head_dim + nope_dim + pair * 2
                        x1 = x[row, col].to(cutlass.Float32)
                        x2 = x[row, col + 1].to(cutlass.Float32)

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
                work = work + 128

    # Launch contract for _thd_local_rope_kernel.
    #   row_width is static; total_work is the local row count.
    @cute.jit
    def _thd_local_rope_launch(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Constexpr,
        chunk0_global_start: cutlass.Int32,
        chunk1_global_start: cutlass.Int32,
        chunk_len: cutlass.Int32,
        clamp_to_valid_token: cutlass.Constexpr,
        row_width: cutlass.Constexpr,
        head_dim: cutlass.Constexpr,
        nope_dim: cutlass.Constexpr,
        pos_dim: cutlass.Constexpr,
        inverse: cutlass.Constexpr,
        adjoint: cutlass.Constexpr,
        head_group: cutlass.Constexpr,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _launch_named(
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
                head_group,
                total_work,
            ),
            grid=(total_work, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    # Kernel contract:
    #   hidden_local: local hidden rows, shape (l_local, row_width), bf16/fp16/fp32.
    #   boundary_hidden: left boundary rows, shape (d_window, row_width).
    #   hidden_compact: output compact rows, shape (compact_len, row_width).
    #   comp_ids: original per-sequence compressed group ids, shape (c_cap,).
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
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        c_cap: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
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

        src_global = cutlass.Int32(-1)
        if row < compact_len:
            if tidx == 0 or tidx == 32:
                running_tokens = 0
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_start = seq_start
                    if local_seq_start < range_start:
                        local_seq_start = range_start
                    local_seq_end = seq_end
                    if local_seq_end > range_end:
                        local_seq_end = range_end

                    if local_seq_start < local_seq_end:
                        seq_full_group_count = (seq_end - seq_start) // ratio
                        first_visible_numer = first_range_group_start - seq_start
                        if first_visible_numer < 0:
                            first_visible_numer = 0
                        first_visible_group = 0
                        if first_visible_numer > 0:
                            first_visible_group = (first_visible_numer + ratio - 1) // ratio
                        stop_visible_group = (local_seq_end - seq_start) // ratio
                        if stop_visible_group > seq_full_group_count:
                            stop_visible_group = seq_full_group_count
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
                        running_tokens = running_tokens + visible_token_count

            src_global = cute.arch.shuffle_sync(src_global, 0, mask=-1, mask_and_clamp=31)

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

        if row < c_cap and tidx == 0:
            slot = row
            comp_ids[slot] = -1

            running = 0
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                local_seq_start = seq_start
                if local_seq_start < range_start:
                    local_seq_start = range_start
                local_seq_end = seq_end
                if local_seq_end > range_end:
                    local_seq_end = range_end

                if local_seq_start < local_seq_end:
                    seq_full_group_count = (seq_end - seq_start) // ratio
                    first_visible_numer = first_range_group_start - seq_start
                    if first_visible_numer < 0:
                        first_visible_numer = 0
                    first_visible_group = 0
                    if first_visible_numer > 0:
                        first_visible_group = (first_visible_numer + ratio - 1) // ratio
                    stop_visible_group = (local_seq_end - seq_start) // ratio
                    if stop_visible_group > seq_full_group_count:
                        stop_visible_group = seq_full_group_count
                    visible_group_count = stop_visible_group - first_visible_group
                    if visible_group_count < 0:
                        visible_group_count = 0

                    if slot >= running and slot < running + visible_group_count:
                        comp_id = first_visible_group + slot - running
                        comp_ids[slot] = comp_id
                    running = running + visible_group_count

    # Launch contract for _compressor_input_compact_fwd_kernel.
    #   total_work covers compact rows plus metadata rows.
    @cute.jit
    def _compressor_input_compact_fwd_launch(
        hidden_local: cute.Tensor,
        boundary_hidden: cute.Tensor,
        hidden_compact: cute.Tensor,
        cu_seqlens: cute.Tensor,
        comp_ids: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d_comp: cutlass.Constexpr,
        d_window: cutlass.Constexpr,
        c_cap: cutlass.Int32,
        compact_len: cutlass.Int32,
        row_width: cutlass.Constexpr,
        total_work: cutlass.Int32,
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
            grid=(total_work, 1, 1),
            block=(64, 1, 1),
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
        total_work: cutlass.Int32,
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
        if out_row < total_work:
            range_start = global_start
            range_end = global_start + l_local
            first_range_group_start = range_start - d_comp
            src_global = range_start + (out_row - d_window)
            dst_row = out_row - d_window
            if out_row < d_window:
                src_global = range_start - d_window + out_row
                dst_row = out_row

            if tidx == 0 or tidx == 32:
                running_tokens = 0
                for seq in range(n_seq):
                    seq_start = cu_seqlens[seq]
                    seq_end = cu_seqlens[seq + 1]
                    local_seq_start = seq_start
                    if local_seq_start < range_start:
                        local_seq_start = range_start
                    local_seq_end = seq_end
                    if local_seq_end > range_end:
                        local_seq_end = range_end

                    if local_seq_start < local_seq_end:
                        seq_full_group_count = (seq_end - seq_start) // ratio
                        first_visible_numer = first_range_group_start - seq_start
                        if first_visible_numer < 0:
                            first_visible_numer = 0
                        first_visible_group = 0
                        if first_visible_numer > 0:
                            first_visible_group = (first_visible_numer + ratio - 1) // ratio
                        stop_visible_group = (local_seq_end - seq_start) // ratio
                        if stop_visible_group > seq_full_group_count:
                            stop_visible_group = seq_full_group_count
                        visible_group_count = stop_visible_group - first_visible_group
                        if visible_group_count < 0:
                            visible_group_count = 0
                        visible_token_count = visible_group_count * ratio

                        if src_global >= seq_start and src_global < seq_end:
                            seq_offset = src_global - seq_start
                            comp_id = seq_offset // ratio
                            token_in_group = seq_offset - (seq_offset // ratio) * ratio
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
                        cute.arch.store(
                            dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first"
                        )
                    else:
                        dst_ptr = cute.recast_ptr(
                            grad_hidden_local.iterator + dst_offset, dtype=cutlass.Int64
                        )
                        cute.arch.store(
                            dst_ptr.llvm_ptr, value, level1_eviction_priority="evict_first"
                        )
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
                total_work,
            ),
            grid=(total_work, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _build_attention_indices_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        rank_row_for_seq_row: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
        has_indexer_topk: cutlass.Int32,
        has_compressed: cutlass.Int32,
        total_width: cutlass.Int32,
        parallel_dense: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx

        row = linear
        col = cutlass.Int32(0)
        if parallel_dense != 0:
            if linear < l_local * total_width:
                row = linear // total_width
                col = linear - row * total_width
            else:
                row = l_local

        if row < l_local:
            local_start = global_start
            row_in_chunk = row
            chunk_id = cutlass.Int32(0)
            boundary_rows = d_window
            if chunk_count != 0:
                chunk_id = row // chunk_len
                row_in_chunk = row - chunk_id * chunk_len
                local_start = chunk0_start
                if chunk_id == 1:
                    local_start = chunk1_start
                boundary_rows = chunk_count * d_window

            global_q = local_start + row_in_chunk
            seq_id = cutlass.Int32(-1)
            seq_start_found = cutlass.Int32(0)
            seq_comp_start = cutlass.Int32(0)
            seq_comp_len = cutlass.Int32(0)
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                if global_q >= seq_start and global_q < seq_end:
                    seq_id = seq
                    seq_start_found = seq_start
                    if has_compressed != 0:
                        seq_comp_start = cu_seqlens_compressed[seq]
                        seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start

            if parallel_dense != 0:
                topk_value = cutlass.Int32(-1)
                length = cutlass.Int32(0)
                if seq_id >= 0:
                    window_start = global_q - window_size + 1
                    if window_start < seq_start_found:
                        window_start = seq_start_found
                    window_count = global_q - window_start + 1
                    comp_count = cutlass.Int32(0)
                    if has_compressed != 0 and ratio > 1:
                        comp_count = (global_q - seq_start_found + 1) // ratio
                        if comp_count > compressed_width:
                            comp_count = compressed_width
                        if comp_count > seq_comp_len:
                            comp_count = seq_comp_len
                    length = window_count + comp_count
                    if col < window_count:
                        pos = window_start + col
                        if pos < local_start:
                            topk_value = chunk_id * d_window + pos - (local_start - d_window)
                        else:
                            topk_value = boundary_rows + chunk_id * chunk_len + pos - local_start
                    elif col < length:
                        seq_major_id = seq_comp_start + col - window_count
                        if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                            rank_major_id = rank_row_for_seq_row[seq_major_id]
                            if rank_major_id >= 0:
                                topk_value = compressed_base + rank_major_id
                elif total_width > 0:
                    length = 1
                    if col == 0:
                        topk_value = 0
                topk_idxs[row, col] = topk_value
                if col == 0:
                    topk_length[row] = length
            else:
                for out_col in range(total_width):
                    topk_idxs[row, out_col] = -1
                topk_length[row] = 0

                if seq_id >= 0:
                    write_col = cutlass.Int32(0)
                    window_start = global_q - window_size + 1
                    if window_start < seq_start_found:
                        window_start = seq_start_found
                    window_count = global_q - window_start + 1
                    for window_col in range(window_size):
                        if window_col < window_count:
                            pos = window_start + window_col
                            if pos < local_start:
                                topk_idxs[row, write_col] = (
                                    chunk_id * d_window + pos - (local_start - d_window)
                                )
                            else:
                                topk_idxs[row, write_col] = (
                                    boundary_rows + chunk_id * chunk_len + pos - local_start
                                )
                            write_col = write_col + 1

                    if has_compressed != 0 and ratio > 1:
                        for compressed_col in range(compressed_width):
                            comp_id = indexer_topk[row, compressed_col]
                            if comp_id >= 0 and comp_id < seq_comp_len:
                                seq_major_id = seq_comp_start + comp_id
                                if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                                    rank_major_id = rank_row_for_seq_row[seq_major_id]
                                    if rank_major_id >= 0:
                                        topk_idxs[row, write_col] = compressed_base + rank_major_id
                                        write_col = write_col + 1
                    topk_length[row] = write_col
                elif total_width > 0:
                    topk_idxs[row, 0] = 0
                    topk_length[row] = 1

    @cute.jit
    def _build_attention_indices_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        indexer_topk: cute.Tensor,
        rank_row_for_seq_row: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        chunk0_start: cutlass.Int32,
        chunk1_start: cutlass.Int32,
        chunk_count: cutlass.Int32,
        chunk_len: cutlass.Int32,
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
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
                rank_row_for_seq_row,
                topk_idxs,
                topk_length,
                n_seq,
                global_start,
                l_local,
                chunk0_start,
                chunk1_start,
                chunk_count,
                chunk_len,
                d_window,
                window_size,
                ratio,
                compressed_width,
                seq_major_rows,
                compressed_base,
                has_indexer_topk,
                has_compressed,
                total_width,
                parallel_dense,
            ),
            launch_work,
            stream,
        )

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
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
        total_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            row = linear // total_width
            col = linear - row * total_width
            topk_value = cutlass.Int32(-1)
            rank_major_value = cutlass.Int32(-1)

            local_start = global_start
            row_in_chunk = row
            chunk_id = cutlass.Int32(0)
            boundary_rows = d_window
            if chunk_count != 0:
                chunk_id = row // chunk_len
                row_in_chunk = row - chunk_id * chunk_len
                local_start = chunk0_start
                if chunk_id == 1:
                    local_start = chunk1_start
                boundary_rows = chunk_count * d_window

            global_q = local_start + row_in_chunk
            seq_id = cutlass.Int32(-1)
            seq_start_found = cutlass.Int32(0)
            seq_comp_start = cutlass.Int32(0)
            seq_comp_len = cutlass.Int32(0)
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                if global_q >= seq_start and global_q < seq_end:
                    seq_id = seq
                    seq_start_found = seq_start
                    seq_comp_start = cu_seqlens_compressed[seq]
                    seq_comp_len = cu_seqlens_compressed[seq + 1] - seq_comp_start

            if seq_id >= 0:
                if col < compressed_width:
                    comp_id = indexer_topk_logical[row, col]
                    if comp_id >= 0 and comp_id < seq_comp_len:
                        seq_major_id = seq_comp_start + comp_id
                        if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                            rank_major_id = indexer_rank_by_seq_major[seq_major_id]
                            if rank_major_id >= 0:
                                topk_value = compressed_base + rank_major_id
                                rank_major_value = rank_major_id
                else:
                    window_col = col - compressed_width
                    window_start = global_q - window_size + 1
                    if window_start < seq_start_found:
                        window_start = seq_start_found
                    window_count = global_q - window_start + 1
                    if window_col < window_count:
                        pos = window_start + window_col
                        if pos < local_start:
                            topk_value = chunk_id * d_window + pos - (local_start - d_window)
                        else:
                            topk_value = boundary_rows + chunk_id * chunk_len + pos - local_start

            topk_idxs[row, col] = topk_value
            if col < compressed_width:
                indexer_rank_major[row, col] = rank_major_value

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
        d_window: cutlass.Int32,
        window_size: cutlass.Int32,
        compressed_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        compressed_base: cutlass.Int32,
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
                d_window,
                window_size,
                compressed_width,
                seq_major_rows,
                compressed_base,
                total_width,
                total_work,
            ),
            total_work,
            stream,
        )


# =============================================================================
# Torch Wrapper Functions
# =============================================================================
# Torch-facing entry points called by csa.py and csa_cp_utils.py. They validate
# inputs, allocate outputs, and dispatch the CuTeDSL kernels defined above.
# =============================================================================


def _require_cute(message: str, *tensors: Optional[torch.Tensor]) -> None:
    """Raise ``RuntimeError`` when a wrapper cannot use CuTeDSL kernels."""
    if (
        not _CUTE_AVAILABLE
        or os.environ.get("DSV4_CP_DISABLE_CUTE_KERNELS")
        or not all(tensor is None or tensor.is_cuda for tensor in tensors)
    ):
        raise RuntimeError(message)


_COMPILED_LAUNCH_CACHE = {}


def _cute_scalar(value):
    if isinstance(value, bool):
        return cutlass.Int32(1 if value else 0)
    if isinstance(value, int):
        return cutlass.Int32(value)
    if isinstance(value, float):
        return cutlass.Float32(value)
    return value


def _run_compiled_launch(
    launch_fn,
    tensor_args: Tuple[torch.Tensor, ...],
    scalar_args: Tuple,
    key_arg_indices: Optional[Tuple[int, ...]] = None,
    static_arg_indices: Optional[Tuple[int, ...]] = None,
) -> None:
    """Compile/cache a CuTe launch function and invoke it on the current stream."""
    key_arg_indices = (
        tuple(range(len(scalar_args))) if key_arg_indices is None else tuple(key_arg_indices)
    )
    static_arg_indices = () if static_arg_indices is None else tuple(static_arg_indices)
    static_arg_set = set(static_arg_indices)
    key = (
        launch_fn.__name__,
        tuple(
            (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())) for tensor in tensor_args
        ),
        key_arg_indices,
        tuple(scalar_args[i] for i in key_arg_indices),
        static_arg_indices,
    )
    compiled = _COMPILED_LAUNCH_CACHE.get(key)
    if compiled is None:
        cap = torch.cuda.get_device_capability()
        arch = {(9, 0): "sm_90a", (10, 0): "sm_100a", (10, 3): "sm_103a"}.get(cap)
        if arch is None:
            raise RuntimeError(f"Unsupported GPU compute capability {cap} for CSA CP CuTe kernels.")
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
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=f"--enable-tvm-ffi --gpu-arch {arch}",
        )
        _COMPILED_LAUNCH_CACHE[key] = compiled
    compiled(
        *tensor_args,
        *(_cute_scalar(arg) for i, arg in enumerate(scalar_args) if i not in static_arg_set),
        cuda.CUstream(torch.cuda.current_stream(tensor_args[0].device).cuda_stream),
    )


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
        rows = x.shape[0]
        row_width = math.prod(x.shape[1:])
        head_dim = nope_dim + pos_dim
        if pos_dim <= 0 or pos_dim % 2 != 0:
            raise RuntimeError(f"DSv4 CP RoPE expects an even positive pos_dim, got {pos_dim}.")
        if head_dim <= 0 or row_width % head_dim != 0:
            raise RuntimeError(
                "DSv4 CP RoPE expects row width to be a multiple of "
                f"head_dim={head_dim}, got {row_width}."
            )
        if math.prod(cos.shape[1:]) < pos_dim or math.prod(sin.shape[1:]) < pos_dim:
            raise RuntimeError("DSv4 CP RoPE cos/sin tables are narrower than pos_dim.")
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
        out = torch.empty_like(x)
        n_heads = row_width // head_dim
        head_group = 1
        for candidate in (16, 8, 4, 2):
            if n_heads % candidate == 0:
                head_group = candidate
                break
        total_work = rows
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
                head_group,
                total_work,
            ),
            key_arg_indices=(0, 4, 5, 6, 7, 8, 9, 10, 11),
            static_arg_indices=(0, 4, 5, 6, 7, 8, 9, 10, 11),
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
        """Apply local THD CP RoPE and save metadata for backward."""
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
        """Apply the adjoint local THD CP RoPE transform to gradients."""
        cos, sin, cu_seqlens_padded = ctx.saved_tensors
        grad_x = ThdLocalRope._run(
            grad_output.contiguous(), cos, sin, cu_seqlens_padded, *ctx.rope_args, adjoint=True
        )
        return (grad_x, *([None] * 10))


class CompressorInputCompact(torch.autograd.Function):
    """Compact local and boundary hidden rows for compressor input.

    Inputs:
        hidden_local: CUDA tensor, shape ``(l_local, ...)``.
        boundary_hidden: CUDA tensor, shape ``(d_window, ...)``.
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start/l_local: local contiguous global row range.
        ratio/d_comp/d_window/c_cap: fixed compressor window and capacity.

    Outputs:
        ``hidden_compact`` shape ``(c_cap * ratio, ...)``, same dtype as hidden,
        and ``comp_ids`` int32 shape ``(c_cap,)``.
        The kernel copies the ratio source tokens for each visible compressed
        group and emits each row's compressed group id.
    """

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
        """Compact local and boundary hidden rows into compressor input rows."""
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.compact_args = (int(global_start), int(l_local), int(ratio), int(d_comp), int(d_window))
        ctx.save_for_backward(cu_seqlens)

        compact_len = int(c_cap) * int(ratio)
        hidden_compact = hidden_local.new_empty((compact_len,) + tuple(hidden_local.shape[1:]))
        comp_ids = torch.empty((int(c_cap),), dtype=torch.int32, device=hidden_local.device)
        row_width = math.prod(hidden_local.shape[1:])
        _run_compiled_launch(
            _compressor_input_compact_fwd_launch,
            (
                hidden_local.reshape(int(l_local), row_width),
                boundary_hidden.reshape(int(d_window), row_width),
                hidden_compact.reshape(compact_len, row_width),
                cu_seqlens,
                comp_ids,
            ),
            (
                cu_seqlens.shape[0] - 1,
                int(global_start),
                int(l_local),
                int(ratio),
                int(d_comp),
                int(d_window),
                int(c_cap),
                compact_len,
                row_width,
                max(compact_len, int(c_cap)),
            ),
            key_arg_indices=(3, 4, 5, 8),
            static_arg_indices=(3, 4, 5, 8),
        )
        return hidden_compact, comp_ids

    @staticmethod
    def backward(ctx, grad_hidden_compact: torch.Tensor, _grad_comp_ids: torch.Tensor):
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
            key_arg_indices=(3, 4, 5, 7),
            static_arg_indices=(3, 4, 5, 7),
        )
        return (grad_hidden, grad_boundary, *([None] * 7))


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
    rank_row_for_seq_row: Optional[torch.Tensor] = None,
    chunk_starts: Optional[Tuple[int, ...]] = None,
    chunk_len: int = 0,
    compressed_base: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build final sparse-attention indices in physical ``kv_full`` row space.

    Inputs:
        cu_seqlens: int32 CUDA tensor, shape ``(n_seq + 1,)``.
        global_start/l_local: contiguous local query range, ignored for two-chunk mode.
        d_window/window_size: physical left-window capacity and per-query window width.
        ratio/compressed_width: compression mode and number of compressed columns.
        indexer_topk_compressed_logical_ids: optional int32 tensor, shape
            ``(l_local, compressed_width)``.
        cu_seqlens_compressed/rank_row_for_seq_row: compressed row mapping.
        chunk metadata: optional two-chunk row layout.
        compressed_base: first compressed row in the raw KV tensor.

    Outputs:
        ``topk_idxs`` int32, shape ``(l_local, window_size + compressed_width)``;
        ``topk_length`` int32, shape ``(l_local,)``. Window ids appear before
        compressed ids and are already lowered into ``kv_full`` rows.
    """
    chunk_count = 0
    chunk1_start = 0
    if chunk_starts is not None:
        _require_cute(
            "DSv4 two-chunk final indices require CUDA tensors and CuTeDSL.",
            cu_seqlens_compressed,
            rank_row_for_seq_row,
        )
        if not chunk_starts or len(chunk_starts) > 2:
            raise RuntimeError(
                "DSv4 two-chunk final indices support one or two chunks, "
                f"got {len(chunk_starts)}."
            )
        chunk_count = len(chunk_starts)
        l_local = int(chunk_count) * int(chunk_len)
        global_start = 0
        chunk1_start = int(chunk_starts[1]) if chunk_count > 1 else 0
    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if rank_row_for_seq_row is None:
        rank_row_for_seq_row = torch.empty((1,), dtype=torch.int32, device=cu_seqlens.device)

    total_width = window_size + compressed_width
    topk_idxs = torch.empty((l_local, total_width), dtype=torch.int32, device=cu_seqlens.device)
    topk_length = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens.device)
    if indexer_topk_compressed_logical_ids is None:
        dummy = torch.empty((1, 1), dtype=torch.int32, device=cu_seqlens.device)
        has_indexer = 0
    else:
        dummy = indexer_topk_compressed_logical_ids
        has_indexer = 1
    parallel_dense = 1 if has_indexer == 0 else 0
    launch_work = l_local * total_width if parallel_dense else l_local
    _run_compiled_launch(
        _build_attention_indices_launch,
        (cu_seqlens, cu_seqlens_compressed, dummy, rank_row_for_seq_row, topk_idxs, topk_length),
        (
            cu_seqlens.shape[0] - 1,
            global_start,
            l_local,
            int(chunk_starts[0]) if chunk_starts is not None else 0,
            chunk1_start,
            chunk_count,
            int(chunk_len),
            d_window,
            window_size,
            ratio,
            compressed_width,
            rank_row_for_seq_row.shape[0],
            int(compressed_base),
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
    compressed_base: int = 0,
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
                "DSv4 two-chunk indexer-loss indices support one or two chunks, "
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
            d_window,
            window_size,
            compressed_width,
            indexer_rank_by_seq_major.shape[0],
            int(compressed_base),
            total_width,
            total_work,
        ),
        key_arg_indices=(),
    )
    return topk_idxs, indexer_rank_major

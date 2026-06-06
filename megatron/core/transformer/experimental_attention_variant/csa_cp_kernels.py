# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CuTe DSL kernels for DSv4 CSA context-parallel tensor layout work.

The functions in this module are intentionally narrow: they replace the CP
path's memory-layout helpers, not the communication ops or attention kernels.
The PyTorch reference implementations live in ``csa_cp_utils.py`` and are kept
for CPU tests and parity checks.
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


def can_use_cute_kernels(*tensors: Optional[torch.Tensor]) -> bool:
    """Return whether this process can run the fused CSA CP kernels."""

    if not _CUTE_AVAILABLE or os.environ.get("DSV4_CP_DISABLE_CUTE_KERNELS"):
        return False
    return all(tensor is None or tensor.is_cuda for tensor in tensors)


def _to_cute_tvm(tensor: torch.Tensor):
    cute_tensor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True)
    if tensor.ndim == 0:
        return cute_tensor
    return cute_tensor.mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _current_stream(device: torch.device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


@lru_cache(maxsize=None)
def _gpu_arch_flag() -> str:
    cap = torch.cuda.get_device_capability()
    arch_map = {
        (9, 0): "sm_90a",
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
    }
    arch = arch_map.get(cap)
    if arch is None:
        raise RuntimeError(f"Unsupported GPU compute capability {cap} for CSA CP CuTe kernels.")
    return arch


def _compile_options() -> str:
    return f"--enable-tvm-ffi --gpu-arch {_gpu_arch_flag()}"


_CUTE_COMPILED_LAUNCH_CACHE = {}


def _cute_scalar(value):
    if isinstance(value, bool):
        return cutlass.Int32(1 if value else 0)
    if isinstance(value, int):
        return cutlass.Int32(value)
    if isinstance(value, float):
        return cutlass.Float32(value)
    return value


def _tensor_desc(tensor: torch.Tensor):
    return tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())


def _run_compiled_launch(launch_fn, tensor_args: Tuple[torch.Tensor, ...], scalar_args: Tuple) -> None:
    key = (
        launch_fn.__name__,
        tuple(_tensor_desc(tensor) for tensor in tensor_args),
        tuple(scalar_args),
    )
    compiled = _CUTE_COMPILED_LAUNCH_CACHE.get(key)
    if compiled is None:
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        compiled = cute.compile(
            launch_fn,
            *(_to_cute_tvm(tensor) for tensor in tensor_args),
            *(_cute_scalar(arg) for arg in scalar_args),
            fake_stream,
            options=_compile_options(),
        )
        _CUTE_COMPILED_LAUNCH_CACHE[key] = compiled
    compiled(
        *tensor_args,
        *(_cute_scalar(arg) for arg in scalar_args),
        _current_stream(tensor_args[0].device),
    )


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
        _run_compiled_launch(
            _zero_2d_launch,
            (_flatten_rows(tensor, rows, row_width),),
            (row_width, total_work),
        )


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
        stream: cuda.CUstream,
    ):
        _zero_2d_kernel.set_name_prefix("dsv4_cp_zero_2d")
        _zero_2d_kernel(tensor, row_width, total_work).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1), stream=stream
        )


    @cute.kernel
    def _global_compressed_cu_kernel(
        cu_seqlens: cute.Tensor,
        cu_compressed: cute.Tensor,
        n_seq: cutlass.Int32,
        ratio: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row <= n_seq:
            running = 0
            for seq in range(n_seq):
                if seq < row:
                    seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq]
                    running = running + seq_len // ratio
            cu_compressed[row] = running


    @cute.jit
    def _global_compressed_cu_launch(
        cu_seqlens: cute.Tensor,
        cu_compressed: cute.Tensor,
        n_seq: cutlass.Int32,
        ratio: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _global_compressed_cu_kernel.set_name_prefix("dsv4_cp_global_compressed_cu")
        _global_compressed_cu_kernel(cu_seqlens, cu_compressed, n_seq, ratio).launch(
            grid=(cute.ceil_div(n_seq + 1, 128), 1, 1), block=(128, 1, 1), stream=stream
        )


    @cute.kernel
    def _overlap_transform_thd_kernel(
        tensor: cute.Tensor,
        is_first: cute.Tensor,
        out: cute.Tensor,
        n_groups: cutlass.Int32,
        ratio: cutlass.Int32,
        b_dim: cutlass.Int32,
        input_width: cutlass.Int32,
        head_dim: cutlass.Int32,
        fill_value: cutlass.Float32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            head_col = linear % head_dim
            tmp = linear // head_dim
            b = tmp % b_dim
            tmp = tmp // b_dim
            out_slot = tmp % (ratio * 2)
            row = tmp // (ratio * 2)

            value = fill_value
            if out_slot >= ratio:
                src_slot = out_slot - ratio
                src_col = head_dim + head_col
                if src_col < input_width:
                    value = tensor[row, src_slot, b, src_col].to(cutlass.Float32)
            else:
                if row > 0 and not is_first[row]:
                    value = tensor[row - 1, out_slot, b, head_col].to(cutlass.Float32)
            out[row, out_slot, b, head_col] = value.to(out.element_type)


    @cute.jit
    def _overlap_transform_thd_launch(
        tensor: cute.Tensor,
        is_first: cute.Tensor,
        out: cute.Tensor,
        n_groups: cutlass.Int32,
        ratio: cutlass.Int32,
        b_dim: cutlass.Int32,
        input_width: cutlass.Int32,
        head_dim: cutlass.Int32,
        fill_value: cutlass.Float32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _overlap_transform_thd_kernel.set_name_prefix("dsv4_cp_overlap_transform_thd")
        _overlap_transform_thd_kernel(
            tensor,
            is_first,
            out,
            n_groups,
            ratio,
            b_dim,
            input_width,
            head_dim,
            fill_value,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _overlap_transform_thd_backward_kernel(
        grad_output: cute.Tensor,
        is_first: cute.Tensor,
        grad_input: cute.Tensor,
        n_groups: cutlass.Int32,
        ratio: cutlass.Int32,
        b_dim: cutlass.Int32,
        input_width: cutlass.Int32,
        head_dim: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            col = linear % input_width
            tmp = linear // input_width
            b = tmp % b_dim
            tmp = tmp // b_dim
            in_slot = tmp % ratio
            row = tmp // ratio

            value = cutlass.Float32(0.0)
            if col < head_dim:
                next_row = row + 1
                if next_row < n_groups and not is_first[next_row]:
                    value = grad_output[next_row, in_slot, b, col].to(cutlass.Float32)
            elif col < head_dim * 2:
                value = grad_output[row, ratio + in_slot, b, col - head_dim].to(
                    cutlass.Float32
                )
            grad_input[row, in_slot, b, col] = value.to(grad_input.element_type)


    @cute.jit
    def _overlap_transform_thd_backward_launch(
        grad_output: cute.Tensor,
        is_first: cute.Tensor,
        grad_input: cute.Tensor,
        n_groups: cutlass.Int32,
        ratio: cutlass.Int32,
        b_dim: cutlass.Int32,
        input_width: cutlass.Int32,
        head_dim: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _overlap_transform_thd_backward_kernel.set_name_prefix(
            "dsv4_cp_overlap_transform_thd_backward"
        )
        _overlap_transform_thd_backward_kernel(
            grad_output,
            is_first,
            grad_input,
            n_groups,
            ratio,
            b_dim,
            input_width,
            head_dim,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _rope_thd_chunked_cp_kernel(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_row_base: cutlass.Int32,
        clamp_to_valid_token: cutlass.Int32,
        row_width: cutlass.Int32,
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

            global_token = global_row_base + row
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

            value_left = x1 * cos_left - x2 * sin_left
            value_right = x2 * cos_right + x1 * sin_right
            if adjoint != 0:
                value_left = x1 * cos_left + x2 * sin_right
                value_right = cutlass.Float32(0.0) - x1 * sin_left + x2 * cos_right
            out[row, col] = value_left.to(out.element_type)
            out[row, col + 1] = value_right.to(out.element_type)


    @cute.jit
    def _rope_thd_chunked_cp_launch(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        cu_seqlens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_row_base: cutlass.Int32,
        clamp_to_valid_token: cutlass.Int32,
        row_width: cutlass.Int32,
        head_dim: cutlass.Int32,
        nope_dim: cutlass.Int32,
        pos_dim: cutlass.Int32,
        inverse: cutlass.Int32,
        adjoint: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _rope_thd_chunked_cp_kernel.set_name_prefix("dsv4_thd_chunked_cp_rope")
        _rope_thd_chunked_cp_kernel(
            x,
            out,
            cos,
            sin,
            cu_seqlens,
            n_seq,
            global_row_base,
            clamp_to_valid_token,
            row_width,
            head_dim,
            nope_dim,
            pos_dim,
            inverse,
            adjoint,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _rope_compressed_kernel(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        comp_ids: cute.Tensor,
        ratio: cutlass.Int32,
        row_width: cutlass.Int32,
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

                value = x1 * cos_left - x2 * sin_left
                if pos_col - pair * 2 != 0:
                    value = x2 * cos_right + x1 * sin_right
                if adjoint != 0:
                    value = x1 * cos_left + x2 * sin_right
                    if pos_col - pair * 2 != 0:
                        value = cutlass.Float32(0.0) - x1 * sin_left + x2 * cos_right
                out[row, col] = value.to(out.element_type)


    @cute.jit
    def _rope_compressed_launch(
        x: cute.Tensor,
        out: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        comp_ids: cute.Tensor,
        ratio: cutlass.Int32,
        row_width: cutlass.Int32,
        head_dim: cutlass.Int32,
        nope_dim: cutlass.Int32,
        pos_dim: cutlass.Int32,
        inverse: cutlass.Int32,
        adjoint: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _rope_compressed_kernel.set_name_prefix("dsv4_cp_rope_compressed")
        _rope_compressed_kernel(
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
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
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
        stream: cuda.CUstream,
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
        ).launch(
            grid=(cute.ceil_div(total_rows, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


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
        stream: cuda.CUstream,
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
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


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
        stream: cuda.CUstream,
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
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _kv_pack_source_map_kernel(
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        out_row = bidx * 128 + tidx
        if out_row < kv_full_capacity:
            source_kind[out_row] = -1
            source_index[out_row] = 0

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
                                source_kind[out_row] = 1
                                source_index[out_row] = src_global - (global_start - d_window)
                            else:
                                source_kind[out_row] = 0
                                source_index[out_row] = src_global - global_start
                        else:
                            comp_id = inner - window_len
                            for comp_row in range(compressed_rows):
                                if (
                                    valid_rank_major[comp_row]
                                    and seq_ids_rank_major[comp_row] == seq
                                    and comp_ids_rank_major[comp_row] == comp_id
                                ):
                                    source_kind[out_row] = 2
                                    source_index[out_row] = comp_row
                    offset = offset + seq_total


    @cute.jit
    def _kv_pack_source_map_launch(
        seq_ids_rank_major: cute.Tensor,
        comp_ids_rank_major: cute.Tensor,
        valid_rank_major: cute.Tensor,
        cu_seqlens: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _kv_pack_source_map_kernel.set_name_prefix("dsv4_cp_kv_full_source_map")
        _kv_pack_source_map_kernel(
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens,
            source_kind,
            source_index,
            n_seq,
            global_start,
            l_local,
            d_window,
            ratio,
            compressed_rows,
            kv_full_capacity,
        ).launch(
            grid=(cute.ceil_div(kv_full_capacity, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _kv_pack_source_map_from_seq_major_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        out_row = bidx * 128 + tidx
        if out_row < kv_full_capacity:
            source_kind[out_row] = -1
            source_index[out_row] = 0

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
                    comp_start = cu_seqlens_compressed[seq]
                    comp_end = cu_seqlens_compressed[seq + 1]
                    comp_len = comp_end - comp_start
                    if ratio <= 1:
                        comp_len = 0
                    seq_total = window_len + comp_len

                    if out_row >= offset and out_row < offset + seq_total:
                        inner = out_row - offset
                        if inner < window_len:
                            src_global = window_start + inner
                            if src_global < global_start:
                                source_kind[out_row] = 1
                                source_index[out_row] = src_global - (global_start - d_window)
                            else:
                                source_kind[out_row] = 0
                                source_index[out_row] = src_global - global_start
                        else:
                            comp_id = inner - window_len
                            seq_major = comp_start + comp_id
                            if seq_major >= comp_start and seq_major < comp_end:
                                rank_major = rank_major_by_seq_major[seq_major]
                                if rank_major >= 0:
                                    source_kind[out_row] = 2
                                    source_index[out_row] = rank_major
                    offset = offset + seq_total


    @cute.jit
    def _kv_pack_source_map_from_seq_major_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        rank_major_by_seq_major: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        ratio: cutlass.Int32,
        kv_full_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _kv_pack_source_map_from_seq_major_kernel.set_name_prefix(
            "dsv4_cp_kv_full_source_map_seq_major"
        )
        _kv_pack_source_map_from_seq_major_kernel(
            cu_seqlens,
            cu_seqlens_compressed,
            rank_major_by_seq_major,
            source_kind,
            source_index,
            n_seq,
            global_start,
            l_local,
            d_window,
            ratio,
            kv_full_capacity,
        ).launch(
            grid=(cute.ceil_div(kv_full_capacity, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _kv_pack_direct_kernel(
        kv_local: cute.Tensor,
        boundary_kv: cute.Tensor,
        compressed_rank_major: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        kv_full: cute.Tensor,
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
            kind = source_kind[out_row]
            index = source_index[out_row]

            value = cutlass.Float32(0.0).to(kv_full.element_type)
            if kind == 0:
                value = kv_local[index, col]
            elif kind == 1:
                value = boundary_kv[index, col]
            elif kind == 2:
                value = compressed_rank_major[index, col]
            kv_full[out_row, col] = value


    @cute.jit
    def _kv_pack_direct_launch(
        kv_local: cute.Tensor,
        boundary_kv: cute.Tensor,
        compressed_rank_major: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        kv_full: cute.Tensor,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _kv_pack_direct_kernel.set_name_prefix("dsv4_cp_kv_full_pack")
        _kv_pack_direct_kernel(
            kv_local,
            boundary_kv,
            compressed_rank_major,
            source_kind,
            source_index,
            kv_full,
            kv_full_capacity,
            row_width,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _kv_pack_direct_backward_kernel(
        grad_kv_full: cute.Tensor,
        grad_kv_local: cute.Tensor,
        grad_boundary_kv: cute.Tensor,
        grad_compressed: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
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
            kind = source_kind[out_row]
            index = source_index[out_row]

            if kind == 0:
                grad_kv_local[index, col] = grad_kv_full[out_row, col]
            elif kind == 1:
                grad_boundary_kv[index, col] = grad_kv_full[out_row, col]
            elif kind == 2:
                grad_compressed[index, col] = grad_kv_full[out_row, col]


    @cute.jit
    def _kv_pack_direct_backward_launch(
        grad_kv_full: cute.Tensor,
        grad_kv_local: cute.Tensor,
        grad_boundary_kv: cute.Tensor,
        grad_compressed: cute.Tensor,
        source_kind: cute.Tensor,
        source_index: cute.Tensor,
        kv_full_capacity: cutlass.Int32,
        row_width: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _kv_pack_direct_backward_kernel.set_name_prefix("dsv4_cp_kv_full_pack_backward")
        _kv_pack_direct_backward_kernel(
            grad_kv_full,
            grad_kv_local,
            grad_boundary_kv,
            grad_compressed,
            source_kind,
            source_index,
            kv_full_capacity,
            row_width,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


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
        stream: cuda.CUstream,
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
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _indexer_topk_input_metadata_kernel(
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        cu_q_topk: cute.Tensor,
        cu_k_topk: cute.Tensor,
        seq_lens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
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


    @cute.jit
    def _indexer_topk_input_metadata_launch(
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        cu_q_topk: cute.Tensor,
        cu_k_topk: cute.Tensor,
        seq_lens: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        ratio: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _indexer_topk_input_metadata_kernel.set_name_prefix("dsv4_cp_indexer_topk_inputs_meta")
        _indexer_topk_input_metadata_kernel(
            cu_seqlens_q,
            cu_seqlens_compressed,
            cu_q_topk,
            cu_k_topk,
            seq_lens,
            n_seq,
            global_start,
            l_local,
            ratio,
        ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


    @cute.kernel
    def _indexer_topk_input_k_pack_kernel(
        k_seq_major: cute.Tensor,
        k_topk: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        cu_k_topk: cute.Tensor,
        n_seq: cutlass.Int32,
        row_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            dst_row = linear // row_width
            col = linear - dst_row * row_width
            valid_total = cu_k_topk[n_seq]
            if dst_row < valid_total:
                src_row = -1
                for seq in range(n_seq):
                    seg_start = cu_k_topk[seq]
                    seg_end = cu_k_topk[seq + 1]
                    if dst_row >= seg_start and dst_row < seg_end:
                        local_k = dst_row - seg_start
                        src_row = cu_seqlens_compressed[seq] + local_k
                if src_row >= 0 and src_row < seq_major_rows:
                    k_topk[dst_row, col] = k_seq_major[src_row, col]


    @cute.jit
    def _indexer_topk_input_k_pack_launch(
        k_seq_major: cute.Tensor,
        k_topk: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        cu_k_topk: cute.Tensor,
        n_seq: cutlass.Int32,
        row_width: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _indexer_topk_input_k_pack_kernel.set_name_prefix("dsv4_cp_indexer_topk_inputs_k_pack")
        _indexer_topk_input_k_pack_kernel(
            k_seq_major,
            k_topk,
            cu_seqlens_compressed,
            cu_k_topk,
            n_seq,
            row_width,
            seq_major_rows,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _pad_topk_indices_kernel(
        topk_in: cute.Tensor,
        topk_out: cute.Tensor,
        rows: cutlass.Int32,
        input_width: cutlass.Int32,
        output_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            row = linear // output_width
            col = linear - row * output_width
            value = -1
            if col < input_width:
                value = topk_in[row, col]
            topk_out[row, col] = value


    @cute.jit
    def _pad_topk_indices_launch(
        topk_in: cute.Tensor,
        topk_out: cute.Tensor,
        rows: cutlass.Int32,
        input_width: cutlass.Int32,
        output_width: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _pad_topk_indices_kernel.set_name_prefix("dsv4_cp_pad_topk_indices")
        _pad_topk_indices_kernel(
            topk_in,
            topk_out,
            rows,
            input_width,
            output_width,
            total_work,
        ).launch(
            grid=(cute.ceil_div(total_work, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


    @cute.kernel
    def _filter_indexer_topk_scores_kernel(
        scores: cute.Tensor,
        topk_in: cute.Tensor,
        topk_out: cute.Tensor,
        topk_length: cute.Tensor,
        rows: cutlass.Int32,
        score_width: cutlass.Int32,
        topk_width: cutlass.Int32,
        output_width: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row < rows:
            count = 0
            finite_floor = cutlass.Float32(-3.402823e38)
            for col in range(output_width):
                out = -1
                if col < topk_width:
                    idx = topk_in[row, col]
                    if idx >= 0 and idx < score_width:
                        score = scores[row, idx].to(cutlass.Float32)
                        if score == score and score > finite_floor:
                            out = idx
                            count = count + 1
                topk_out[row, col] = out
            topk_length[row] = count


    @cute.kernel
    def _filter_indexer_topk_scores_no_length_kernel(
        scores: cute.Tensor,
        topk_in: cute.Tensor,
        topk_out: cute.Tensor,
        rows: cutlass.Int32,
        score_width: cutlass.Int32,
        topk_width: cutlass.Int32,
        output_width: cutlass.Int32,
        total_work: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_work:
            row = linear // output_width
            col = linear - row * output_width
            out = -1
            finite_floor = cutlass.Float32(-3.402823e38)
            if col < topk_width:
                idx = topk_in[row, col]
                if idx >= 0 and idx < score_width:
                    score = scores[row, idx].to(cutlass.Float32)
                    if score == score and score > finite_floor:
                        out = idx
            topk_out[row, col] = out


    @cute.jit
    def _filter_indexer_topk_scores_launch(
        scores: cute.Tensor,
        topk_in: cute.Tensor,
        topk_out: cute.Tensor,
        topk_length: cute.Tensor,
        rows: cutlass.Int32,
        score_width: cutlass.Int32,
        topk_width: cutlass.Int32,
        output_width: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _filter_indexer_topk_scores_kernel.set_name_prefix("dsv4_cp_filter_indexer_topk")
        _filter_indexer_topk_scores_kernel(
            scores,
            topk_in,
            topk_out,
            topk_length,
            rows,
            score_width,
            topk_width,
            output_width,
        ).launch(grid=(cute.ceil_div(rows, 128), 1, 1), block=(128, 1, 1), stream=stream)


    @cute.jit
    def _filter_indexer_topk_scores_no_length_launch(
        scores: cute.Tensor,
        topk_in: cute.Tensor,
        topk_out: cute.Tensor,
        rows: cutlass.Int32,
        score_width: cutlass.Int32,
        topk_width: cutlass.Int32,
        output_width: cutlass.Int32,
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _filter_indexer_topk_scores_no_length_kernel.set_name_prefix(
            "dsv4_cp_filter_indexer_topk_no_length"
        )
        _filter_indexer_topk_scores_no_length_kernel(
            scores,
            topk_in,
            topk_out,
            rows,
            score_width,
            topk_width,
            output_width,
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1), stream=stream)


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
        stream: cuda.CUstream,
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
        ).launch(grid=(cute.ceil_div(l_local, 128), 1, 1), block=(128, 1, 1), stream=stream)


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

                if col < compressed_width:
                    comp_id = indexer_topk_logical[row, col]
                    if comp_id >= 0 and comp_id < seq_comp_len:
                        topk_value = seq_offset + seq_window_len + comp_id
                        seq_major_id = cu_seqlens_compressed[seq_id] + comp_id
                        if seq_major_id >= 0 and seq_major_id < seq_major_rows:
                            rank_major_id = indexer_rank_by_seq_major[seq_major_id]
                            if rank_major_id >= 0:
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

            topk_idxs[row, col] = topk_value
            if col < compressed_width:
                indexer_rank_major[row, col] = rank_major_value


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
        total_work: cutlass.Int32,
        stream: cuda.CUstream,
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
            total_work,
        ).launch(grid=(cute.ceil_div(total_work, 128), 1, 1), block=(128, 1, 1), stream=stream)


def _validate_rope_inputs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
) -> Tuple[int, int, int]:
    rows = x.shape[0]
    row_width = _row_width(tuple(x.shape))
    head_dim = nope_dim + pos_dim
    if pos_dim <= 0 or pos_dim % 2 != 0:
        raise RuntimeError(f"DSv4 CP RoPE expects an even positive pos_dim, got {pos_dim}.")
    if head_dim <= 0 or row_width % head_dim != 0:
        raise RuntimeError(
            "DSv4 CP RoPE expects the flattened row width to be a multiple of "
            f"head_dim={head_dim}, got row_width={row_width}."
        )
    cos_width = _row_width(tuple(cos.shape))
    sin_width = _row_width(tuple(sin.shape))
    if cos_width < pos_dim or sin_width < pos_dim:
        raise RuntimeError(
            "DSv4 CP RoPE cos/sin tables are narrower than pos_dim: "
            f"cos={cos_width}, sin={sin_width}, pos_dim={pos_dim}."
        )
    return rows, row_width, head_dim


def apply_thd_chunked_cp_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    global_row_base: int,
    nope_dim: int,
    pos_dim: int,
    inverse: bool = False,
    clamp_to_valid_token: bool = False,
    adjoint: bool = False,
) -> torch.Tensor:
    """Apply THD chunked-CP RoPE for rows mapped by ``cu_seqlens_padded``.

    ``global_row_base + row`` is resolved to a sequence-local RoPE position
    inside the CuTeDSL kernel. This is the production replacement for explicit
    per-token positions in the THD chunked-CP path.
    """
    if not can_use_cute_kernels(x, cos, sin, cu_seqlens_padded):
        raise RuntimeError("DSv4 THD chunked-CP RoPE requires CUDA tensors and available CuTe kernels.")
    rows, row_width, head_dim = _validate_rope_inputs(x, cos, sin, nope_dim, pos_dim)
    if rows == 0:
        return x
    out = x.clone()
    total_work = rows * (row_width // head_dim) * (int(pos_dim) // 2)
    _run_compiled_launch(
        _rope_thd_chunked_cp_launch,
        (
            _flatten_rows(x, rows, row_width),
            _flatten_rows(out, rows, row_width),
            _flatten_rows(cos, cos.shape[0], _row_width(tuple(cos.shape))),
            _flatten_rows(sin, sin.shape[0], _row_width(tuple(sin.shape))),
            cu_seqlens_padded,
        ),
        (
            cu_seqlens_padded.shape[0] - 1,
            int(global_row_base),
            1 if clamp_to_valid_token else 0,
            row_width,
            head_dim,
            int(nope_dim),
            int(pos_dim),
            1 if inverse else 0,
            1 if adjoint else 0,
            total_work,
        ),
    )
    return out


def build_global_compressed_cu_seqlens(
    cu_seqlens_padded: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Build seq-major compressed cumulative lengths without PyTorch cat/cumsum."""
    if not can_use_cute_kernels(cu_seqlens_padded):
        raise RuntimeError(
            "DSv4 CP compressed cu_seqlens requires CUDA tensors and available CuTe kernels."
        )
    out = torch.empty_like(cu_seqlens_padded)
    _run_compiled_launch(
        _global_compressed_cu_launch,
        (cu_seqlens_padded, out),
        (cu_seqlens_padded.shape[0] - 1, int(ratio)),
    )
    return out


def overlap_transform_thd(
    tensor: torch.Tensor,
    is_first_in_seg: torch.Tensor,
    head_dim: int,
    fill_value: float,
) -> torch.Tensor:
    """Apply CSA THD overlap transform without torch.roll or slice assignment."""
    if not can_use_cute_kernels(tensor, is_first_in_seg):
        raise RuntimeError(
            "DSv4 CP THD overlap transform requires CUDA tensors and available CuTe kernels."
        )
    if tensor.ndim != 4:
        raise RuntimeError(f"THD overlap transform expects a 4D tensor, got {tensor.shape}.")
    n_groups, ratio, b_dim, input_width = tensor.shape
    if is_first_in_seg.shape[0] != n_groups:
        raise RuntimeError(
            "THD overlap transform expects is_first_in_seg length to match group count: "
            f"mask={is_first_in_seg.shape[0]}, groups={n_groups}."
        )
    if input_width < head_dim * 2:
        raise RuntimeError(
            "THD overlap transform expects input_width >= 2 * head_dim: "
            f"input_width={input_width}, head_dim={head_dim}."
        )
    out = tensor.new_empty((n_groups, ratio * 2, b_dim, head_dim))
    total_work = n_groups * ratio * 2 * b_dim * head_dim
    if total_work > 0:
        _run_compiled_launch(
            _overlap_transform_thd_launch,
            (tensor, is_first_in_seg, out),
            (
                n_groups,
                ratio,
                b_dim,
                input_width,
                int(head_dim),
                float(fill_value),
                total_work,
            ),
        )
    return out


def overlap_transform_thd_backward(
    grad_output: torch.Tensor,
    is_first_in_seg: torch.Tensor,
    input_shape: Tuple[int, int, int, int],
    head_dim: int,
) -> torch.Tensor:
    """Backward for ``overlap_transform_thd``."""
    if not can_use_cute_kernels(grad_output, is_first_in_seg):
        raise RuntimeError(
            "DSv4 CP THD overlap transform backward requires CUDA tensors and CuTe kernels."
        )
    n_groups, ratio, b_dim, input_width = input_shape
    grad_input = grad_output.new_empty(input_shape)
    total_work = n_groups * ratio * b_dim * input_width
    if total_work > 0:
        _run_compiled_launch(
            _overlap_transform_thd_backward_launch,
            (grad_output, is_first_in_seg, grad_input),
            (
                n_groups,
                ratio,
                b_dim,
                input_width,
                int(head_dim),
                total_work,
            ),
        )
    return grad_input


def apply_compressed_rope(
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
    """Apply THD CP compressed RoPE using ``comp_ids_local[row] * ratio``."""
    if not can_use_cute_kernels(x, cos, sin, comp_ids_local):
        raise RuntimeError(
            "DSv4 CP compressed RoPE requires CUDA tensors and available CuTe kernels."
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
        _rope_compressed_launch,
        (
            _flatten_rows(x, rows, row_width),
            _flatten_rows(out, rows, row_width),
            _flatten_rows(cos, cos.shape[0], _row_width(tuple(cos.shape))),
            _flatten_rows(sin, sin.shape[0], _row_width(tuple(sin.shape))),
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
    )
    return out


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
    _run_compiled_launch(
        _rank_major_metadata_launch,
        (cu_seqlens, seq_ids, comp_ids, valid),
        (
            cu_seqlens.shape[0] - 1,
            cp_size,
            l_local,
            ratio,
            d_comp,
            c_cap,
            total_rows,
        ),
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
    _run_compiled_launch(
        _compressor_prep_launch,
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
        _run_compiled_launch(
            _compressor_prep_backward_launch,
            (
                _flatten_rows(grad_hidden_compact, compact_len, row_width),
                _flatten_rows(grad_hidden, l_local, row_width),
                _flatten_rows(grad_boundary, d_window, row_width),
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
        rank_major_by_seq_major: Optional[torch.Tensor] = None,
        cu_seqlens_compressed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx.local_shape = tuple(kv_local.shape)
        ctx.boundary_shape = tuple(boundary_kv.shape)
        ctx.compressed_shape = tuple(compressed_rank_major.shape)
        ctx.global_start = global_start
        ctx.l_local = l_local
        ctx.d_window = d_window
        ctx.ratio = ratio
        ctx.kv_full_capacity = kv_full_capacity

        output_shape = (kv_full_capacity,) + tuple(kv_local.shape[1:])
        kv_full = kv_local.new_empty(output_shape)
        row_width = _row_width(tuple(kv_local.shape))
        kv_local_flat = _flatten_rows(kv_local, l_local, row_width)
        boundary_flat = _flatten_rows(boundary_kv, d_window, row_width)
        compressed_rows = compressed_rank_major.shape[0]
        compressed_flat = _flatten_rows(compressed_rank_major, compressed_rows, row_width)
        kv_full_flat = _flatten_rows(kv_full, kv_full_capacity, row_width)
        total_work = kv_full_capacity * row_width

        # A row map removes repeated per-column sequence/compressed metadata
        # scans. DSv4 KV rows are wide enough that the extra metadata launch is
        # cheaper than redoing the scan for every column, including the
        # low-compressed-row ratio=128 path.
        source_kind = torch.empty((kv_full_capacity,), dtype=torch.int32, device=kv_local.device)
        source_index = torch.empty((kv_full_capacity,), dtype=torch.int32, device=kv_local.device)
        if (
            rank_major_by_seq_major is not None
            and cu_seqlens_compressed is not None
            and ratio > 1
            and compressed_rows > 0
        ):
            _run_compiled_launch(
                _kv_pack_source_map_from_seq_major_launch,
                (
                    cu_seqlens,
                    cu_seqlens_compressed,
                    rank_major_by_seq_major,
                    source_kind,
                    source_index,
                ),
                (
                    cu_seqlens.shape[0] - 1,
                    global_start,
                    l_local,
                    d_window,
                    ratio,
                    kv_full_capacity,
                ),
            )
        else:
            _run_compiled_launch(
                _kv_pack_source_map_launch,
                (
                    seq_ids_rank_major,
                    comp_ids_rank_major,
                    valid_rank_major,
                    cu_seqlens,
                    source_kind,
                    source_index,
                ),
                (
                    cu_seqlens.shape[0] - 1,
                    global_start,
                    l_local,
                    d_window,
                    ratio,
                    compressed_rows,
                    kv_full_capacity,
                ),
            )
        _run_compiled_launch(
            _kv_pack_direct_launch,
            (
                kv_local_flat,
                boundary_flat,
                compressed_flat,
                source_kind,
                source_index,
                kv_full_flat,
            ),
            (
                kv_full_capacity,
                row_width,
                total_work,
            ),
        )
        ctx.save_for_backward(source_kind, source_index)
        return kv_full

    @staticmethod
    def backward(ctx, grad_kv_full: torch.Tensor):
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
            source_kind, source_index = ctx.saved_tensors
            _run_compiled_launch(
                _kv_pack_direct_backward_launch,
                (
                    _flatten_rows(grad_kv_full, kv_full_capacity, row_width),
                    _flatten_rows(grad_kv_local, l_local, row_width),
                    _flatten_rows(grad_boundary_kv, d_window, row_width),
                    _flatten_rows(grad_compressed, compressed_rows, row_width),
                    source_kind,
                    source_index,
                ),
                (
                    kv_full_capacity,
                    row_width,
                    total_work,
                ),
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
            None,
            None,
        )


def kv_full_capacity(
    cu_seqlens: torch.Tensor, l_local: int, d_window: int, compressed_rows: int
) -> int:
    # Static upper bound matching the HTML layout: per active sequence window rows
    # plus compressed rows, followed by tail padding.
    n_seq = cu_seqlens.shape[0] - 1
    return max(1, l_local + d_window * n_seq + compressed_rows)


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
    rank_major_by_seq_major: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
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
        rank_major_by_seq_major,
        cu_seqlens_compressed,
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
    _run_compiled_launch(
        _repack_rank_major_launch,
        (
            _flatten_rows(compressed_rank_major, rank_major_rows, row_width),
            seq_ids_rank_major,
            comp_ids_rank_major,
            valid_rank_major,
            cu_seqlens_compressed,
            _flatten_rows(compressed_seq_major, seq_major_rows, row_width),
            rank_major_by_seq_major,
        ),
        (rank_major_rows, seq_major_rows, row_width, total_work),
    )
    return compressed_seq_major, rank_major_by_seq_major


def build_indexer_topk_inputs(
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_major_rows = k_indexer_seq_major.shape[0]
    row_width = _row_width(tuple(k_indexer_seq_major.shape))
    k_topk = k_indexer_seq_major.new_empty(k_indexer_seq_major.shape)
    cu_q_topk = torch.empty(
        (cu_seqlens_q.shape[0] + 1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device
    )
    cu_k_topk = torch.empty(
        (cu_seqlens_q.shape[0] + 1,), dtype=cu_seqlens_compressed.dtype, device=cu_seqlens_q.device
    )
    seq_lens = torch.empty((l_local,), dtype=torch.int32, device=cu_seqlens_q.device)
    n_seq = cu_seqlens_q.shape[0] - 1
    _run_compiled_launch(
        _indexer_topk_input_metadata_launch,
        (cu_seqlens_q, cu_seqlens_compressed, cu_q_topk, cu_k_topk, seq_lens),
        (n_seq, global_start, l_local, ratio),
    )
    _zero_rows(k_topk, seq_major_rows, row_width)
    total_work = seq_major_rows * row_width
    if total_work > 0:
        _run_compiled_launch(
            _indexer_topk_input_k_pack_launch,
            (
                _flatten_rows(k_indexer_seq_major, seq_major_rows, row_width),
                _flatten_rows(k_topk, seq_major_rows, row_width),
                cu_seqlens_compressed,
                cu_k_topk,
            ),
            (n_seq, row_width, seq_major_rows, total_work),
        )
    return k_topk, cu_q_topk, cu_k_topk, seq_lens


def pad_topk_indices(topk_indices: torch.Tensor, output_width: int) -> torch.Tensor:
    """Pad CP indexer top-k ids to a fixed width without torch.full + torch.cat."""
    if not can_use_cute_kernels(topk_indices):
        raise RuntimeError(
            "DSv4 CP top-k index padding requires CUDA tensors and available CuTe kernels."
        )
    if topk_indices.ndim != 2:
        raise RuntimeError(f"DSv4 CP top-k padding expects a 2D tensor, got {topk_indices.shape}.")
    if topk_indices.dtype != torch.int32:
        raise RuntimeError(
            f"DSv4 CP top-k padding expects int32 top-k ids, got {topk_indices.dtype}."
        )
    if not topk_indices.is_contiguous():
        raise RuntimeError("DSv4 CP top-k padding expects contiguous top-k ids.")

    rows, input_width = topk_indices.shape
    output_width = int(output_width)
    if output_width < input_width:
        raise RuntimeError(
            "DSv4 CP top-k padding cannot shrink the top-k width: "
            f"input={input_width}, output={output_width}."
        )
    if output_width == input_width:
        return topk_indices

    topk_out = torch.empty((rows, output_width), dtype=torch.int32, device=topk_indices.device)
    total_work = rows * output_width
    if total_work > 0:
        _run_compiled_launch(
            _pad_topk_indices_launch,
            (topk_indices, topk_out),
            (rows, input_width, output_width, total_work),
        )
    return topk_out


def filter_indexer_topk_scores(
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    output_width: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter top-k ids and optionally pad to a fixed width in one kernel."""
    if not can_use_cute_kernels(scores, topk_indices):
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering requires CUDA tensors and CuTe kernels."
        )
    if scores.ndim != 2 or topk_indices.ndim != 2:
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering expects 2D scores/top-k tensors, "
            f"got scores={scores.shape}, topk={topk_indices.shape}."
        )
    if topk_indices.dtype != torch.int32:
        raise RuntimeError(
            f"DSv4 CP indexer top-k filtering expects int32 ids, got {topk_indices.dtype}."
        )
    if scores.shape[0] != topk_indices.shape[0]:
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering expects matching row counts: "
            f"scores={scores.shape[0]}, topk={topk_indices.shape[0]}."
        )
    if not scores.is_contiguous() or not topk_indices.is_contiguous():
        raise RuntimeError("DSv4 CP indexer top-k score filtering expects contiguous tensors.")

    rows, score_width = scores.shape
    topk_width = topk_indices.shape[1]
    output_width = topk_width if output_width is None else int(output_width)
    if output_width < topk_width:
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering cannot shrink top-k width: "
            f"input={topk_width}, output={output_width}."
        )
    topk_out = torch.empty((rows, output_width), dtype=torch.int32, device=topk_indices.device)
    topk_length = torch.empty((rows,), dtype=torch.int32, device=topk_indices.device)
    if output_width == 0:
        _zero_rows(topk_length, rows, 1)
        return topk_out, topk_length

    _run_compiled_launch(
        _filter_indexer_topk_scores_launch,
        (scores, topk_indices, topk_out, topk_length),
        (rows, score_width, topk_width, output_width),
    )
    return topk_out, topk_length


def filter_indexer_topk_scores_no_length(
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    output_width: Optional[int] = None,
) -> torch.Tensor:
    """Filter top-k ids and pad to a fixed width when the valid length is unused."""
    if not can_use_cute_kernels(scores, topk_indices):
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering requires CUDA tensors and CuTe kernels."
        )
    if scores.ndim != 2 or topk_indices.ndim != 2:
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering expects 2D scores/top-k tensors, "
            f"got scores={scores.shape}, topk={topk_indices.shape}."
        )
    if topk_indices.dtype != torch.int32:
        raise RuntimeError(
            f"DSv4 CP indexer top-k filtering expects int32 ids, got {topk_indices.dtype}."
        )
    if scores.shape[0] != topk_indices.shape[0]:
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering expects matching row counts: "
            f"scores={scores.shape[0]}, topk={topk_indices.shape[0]}."
        )
    if not scores.is_contiguous() or not topk_indices.is_contiguous():
        raise RuntimeError("DSv4 CP indexer top-k score filtering expects contiguous tensors.")

    rows, score_width = scores.shape
    topk_width = topk_indices.shape[1]
    output_width = topk_width if output_width is None else int(output_width)
    if output_width < topk_width:
        raise RuntimeError(
            "DSv4 CP indexer top-k score filtering cannot shrink top-k width: "
            f"input={topk_width}, output={output_width}."
        )

    topk_out = torch.empty((rows, output_width), dtype=torch.int32, device=topk_indices.device)
    total_work = rows * output_width
    if total_work > 0:
        _run_compiled_launch(
            _filter_indexer_topk_scores_no_length_launch,
            (scores, topk_indices, topk_out),
            (rows, score_width, topk_width, output_width, total_work),
        )
    return topk_out


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
    _run_compiled_launch(
        _final_idx_launch,
        (cu_seqlens, dummy, topk_idxs, topk_length),
        (
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
        ),
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
    total_work = l_local * total_width
    if total_work == 0:
        return topk_idxs, indexer_rank_major
    _run_compiled_launch(
        _indexer_loss_final_idx_launch,
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
            d_window,
            window_size,
            ratio,
            compressed_width,
            indexer_rank_by_seq_major.shape[0],
            total_width,
            total_work,
        ),
    )
    return topk_idxs, indexer_rank_major

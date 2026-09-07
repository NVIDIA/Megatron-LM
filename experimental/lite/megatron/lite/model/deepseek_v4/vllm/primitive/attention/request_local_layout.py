"""mLite-owned request-local sparse-attention layout kernels."""

from __future__ import annotations

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_layout_kernels as _cp_layout,
)


if _cp_layout._CUTE_AVAILABLE:
    cuda = _cp_layout.cuda
    cutlass = _cp_layout.cutlass
    cute = _cp_layout.cute

    @cute.kernel
    def _request_workspace_map_kernel(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        workspace_row_map: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        compressed_base: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        physical_workspace_rows: cutlass.Int32,
        total_capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear = bidx * 128 + tidx
        if linear < total_capacity:
            source = physical_workspace_rows
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                comp_start = cu_seqlens_compressed[seq]
                comp_end = cu_seqlens_compressed[seq + 1]
                request_offset = seq_start + comp_start
                seq_len = seq_end - seq_start
                request_capacity = seq_len + comp_end - comp_start
                if (
                    linear >= request_offset
                    and linear < request_offset + request_capacity
                ):
                    request_row = linear - request_offset
                    if request_row < seq_len:
                        pos = seq_start + request_row
                        boundary_start = global_start - d_window
                        if pos >= boundary_start and pos < global_start:
                            source = pos - boundary_start
                        elif pos >= global_start and pos < global_start + l_local:
                            source = d_window + pos - global_start
                    else:
                        comp_id = request_row - seq_len
                        seq_major_id = comp_start + comp_id
                        if seq_major_id < seq_major_rows:
                            rank_major_id = seq_to_rank_row[seq_major_id]
                            if rank_major_id >= 0:
                                source = compressed_base + rank_major_id
            workspace_row_map[linear] = source

    @cute.jit
    def _request_workspace_map_launch(
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        seq_to_rank_row: cute.Tensor,
        workspace_row_map: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        compressed_base: cutlass.Int32,
        seq_major_rows: cutlass.Int32,
        physical_workspace_rows: cutlass.Int32,
        total_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _cp_layout._launch_named(
            _request_workspace_map_kernel,
            "mlite_dsv4_request_workspace_map",
            (
                cu_seqlens,
                cu_seqlens_compressed,
                seq_to_rank_row,
                workspace_row_map,
                n_seq,
                global_start,
                l_local,
                d_window,
                compressed_base,
                seq_major_rows,
                physical_workspace_rows,
                total_capacity,
            ),
            grid=(cute.ceil_div(total_capacity, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _remap_attention_indices_kernel(
        physical_indices: cute.Tensor,
        rank_to_seq_row: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        local_indices: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        compressed_base: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        width: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        if row < l_local:
            global_q = global_start + row
            seq_start_found = cutlass.Int32(-1)
            seq_end_found = cutlass.Int32(-1)
            comp_start_found = cutlass.Int32(0)
            comp_end_found = cutlass.Int32(0)
            for seq in range(n_seq):
                seq_start = cu_seqlens[seq]
                seq_end = cu_seqlens[seq + 1]
                if global_q >= seq_start and global_q < seq_end:
                    seq_start_found = seq_start
                    seq_end_found = seq_end
                    comp_start_found = cu_seqlens_compressed[seq]
                    comp_end_found = cu_seqlens_compressed[seq + 1]

            col = tidx
            while col < width:
                physical = physical_indices[row, col]
                local = cutlass.Int32(-1)
                if physical >= 0 and seq_start_found >= 0:
                    if physical < d_window:
                        position = global_start - d_window + physical
                        local = position - seq_start_found
                    elif physical < compressed_base:
                        position = global_start + physical - d_window
                        local = position - seq_start_found
                    else:
                        rank_row = physical - compressed_base
                        if rank_row < compressed_rows:
                            seq_row = rank_to_seq_row[rank_row]
                            if (
                                seq_row >= comp_start_found
                                and seq_row < comp_end_found
                            ):
                                local = (
                                    seq_end_found
                                    - seq_start_found
                                    + seq_row
                                    - comp_start_found
                                )
                local_indices[row, col] = local
                col = col + 128

    @cute.jit
    def _remap_attention_indices_launch(
        physical_indices: cute.Tensor,
        rank_to_seq_row: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_compressed: cute.Tensor,
        local_indices: cute.Tensor,
        n_seq: cutlass.Int32,
        global_start: cutlass.Int32,
        l_local: cutlass.Int32,
        d_window: cutlass.Int32,
        compressed_base: cutlass.Int32,
        compressed_rows: cutlass.Int32,
        width: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _cp_layout._launch_named(
            _remap_attention_indices_kernel,
            "mlite_dsv4_remap_attention_indices",
            (
                physical_indices,
                rank_to_seq_row,
                cu_seqlens,
                cu_seqlens_compressed,
                local_indices,
                n_seq,
                global_start,
                l_local,
                d_window,
                compressed_base,
                compressed_rows,
                width,
            ),
            grid=(l_local, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )


def build_request_local_layout(
    physical_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    seq_to_rank_row: torch.Tensor,
    *,
    global_start: int,
    l_local: int,
    d_window: int,
    physical_workspace_rows: int,
    total_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return remapped indices and one all-request physical workspace row map."""
    _cp_layout._require_cute(
        "mLite DS4 request-local layout requires CUDA tensors and CuTeDSL.",
        physical_indices,
        cu_seqlens,
        cu_seqlens_compressed,
        seq_to_rank_row,
    )
    flat_indices = physical_indices.reshape(l_local, -1)
    # Every output cell depends only on the physical index in that same cell,
    # so the CuTe remap is safe in-place and avoids a batch-sized int32 buffer.
    local_indices = flat_indices
    compressed_base = int(d_window) + int(l_local)
    compressed_rows = int(physical_workspace_rows) - compressed_base
    valid = (seq_to_rank_row >= 0) & (seq_to_rank_row < compressed_rows)
    rank_to_seq_row = torch.full(
        (compressed_rows,),
        -1,
        dtype=torch.int32,
        device=seq_to_rank_row.device,
    )
    seq_rows = torch.arange(
        seq_to_rank_row.numel(),
        dtype=torch.int32,
        device=seq_to_rank_row.device,
    )
    rank_to_seq_row.index_copy_(
        0,
        seq_to_rank_row[valid].to(torch.int64),
        seq_rows[valid],
    )
    _cp_layout._run_compiled_launch(
        _remap_attention_indices_launch,
        (
            flat_indices,
            rank_to_seq_row,
            cu_seqlens,
            cu_seqlens_compressed,
            local_indices,
        ),
        (
            cu_seqlens.shape[0] - 1,
            int(global_start),
            int(l_local),
            int(d_window),
            compressed_base,
            compressed_rows,
            flat_indices.shape[1],
        ),
    )

    if total_capacity < 1:
        raise ValueError("request-local workspace capacity must be positive")
    workspace_row_map = torch.empty(
        total_capacity,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    _cp_layout._run_compiled_launch(
        _request_workspace_map_launch,
        (
            cu_seqlens,
            cu_seqlens_compressed,
            seq_to_rank_row,
            workspace_row_map,
        ),
        (
            cu_seqlens.shape[0] - 1,
            int(global_start),
            int(l_local),
            int(d_window),
            compressed_base,
            seq_to_rank_row.shape[0],
            int(physical_workspace_rows),
            total_capacity,
        ),
    )
    return local_indices.reshape_as(physical_indices), workspace_row_map

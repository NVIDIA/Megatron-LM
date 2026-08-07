# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _y_scatter_kernel(
    src_ptr,  # (workspace_chunks * L, H, P) scratch, token-major within a chunk
    dst_ptr,  # (T, H, P) caller output
    chunk_token_base_ptr,
    chunk_valid_start_ptr,
    chunk_valid_end_ptr,
    stride_src_row,
    stride_dst_row,
    HP,
    BLOCK_D: tl.constexpr,
    L: tl.constexpr,
):
    # One program per (workspace token slot, feature block).
    slot = tl.program_id(0)
    c = slot // L
    i = slot % L
    token = tl.load(chunk_token_base_ptr + c) + i
    if (token < tl.load(chunk_valid_start_ptr + c)) | (token >= tl.load(chunk_valid_end_ptr + c)):
        return  # this lane belongs to another sequence (or is pad)

    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < HP
    vals = tl.load(src_ptr + slot * stride_src_row + offs, mask=mask)
    tl.store(dst_ptr + token * stride_dst_row + offs, vals, mask=mask)


def scatter_y_ragged(
    y_scratch: torch.Tensor,
    out: torch.Tensor,
    chunk_token_base: torch.Tensor,
    chunk_valid_start: torch.Tensor,
    chunk_valid_end: torch.Tensor,
    kernel_chunk_size: int,
) -> None:
    """Copy real tokens from the expanded Y scratch into the caller's output.

    Args:
        y_scratch: ``(workspace_chunks * L, H, P)`` kernel output, contiguous rows.
        out: ``(T, H, P)`` caller output, contiguous rows.
        chunk_token_base: ``(workspace_chunks,)`` first token of each workspace chunk.
        chunk_valid_start: ``(workspace_chunks,)`` first real token of the owning sequence.
        chunk_valid_end: ``(workspace_chunks,)`` one past its last real token.
        kernel_chunk_size: ``L``.
    """
    HP = y_scratch.shape[1] * y_scratch.shape[2]
    assert y_scratch.stride(2) == 1 and out.stride(2) == 1, "rows must be P-contiguous"
    BLOCK_D = min(1024, triton.next_power_of_2(HP))
    grid = (y_scratch.shape[0], triton.cdiv(HP, BLOCK_D))
    _y_scatter_kernel[grid](
        y_scratch,
        out,
        chunk_token_base,
        chunk_valid_start,
        chunk_valid_end,
        y_scratch.stride(0),
        out.stride(0),
        HP,
        BLOCK_D=BLOCK_D,
        L=kernel_chunk_size,
        num_warps=4,
    )
